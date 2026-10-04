"""局部 CMYK 修色（local CMYK retouch）。

以一个**已保存软打样任务**的分色结果为起点，在不超过 64×64 的选区内
搜索一组对全选区统一生效的**整数 CMYK 偏移** ``(dc, dm, dy, dk)``，
使选区像素沿**原任务的印刷 ICC** 回算出的平均 Lab 尽量逼近用户给定的
有限目标 Lab。

硬约束（任一像素违反即整组偏移不可行）：

* 通道取值保持字节范围：``0 <= C+dc <= 255``（M/Y/K 同理）；
* 每通道偏移绝对值不超过用户给定上限 ``max_shift``（0..8）；
* 每像素总墨量 ``(C+dc)+(M+dm)+(Y+dy)+(K+dk)`` 不超过 ``tac_limit``（0..1020）。

优选顺序（按序取最小）：

1. 选区平均 Lab（LittleCMS TYPE_Lab_DBL 回算）对目标 Lab 的 **ΔE00**；
2. 偏移绝对值之和 ``|dc|+|dm|+|dy|+|dk|``；
3. ``(C, M, Y, K)`` 偏移四元组字典序。

只有最优解的 ΔE00 不超过给定非负容差时才视为**可达**，否则不产出成品 TIFF。
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import colorscience, lcms_native

MAX_RECT_SIDE = 64
MAX_SHIFT_LIMIT = 8
# CMYK 四通道字节求和的理论最大值 4*255 = 1020。
MAX_TAC = 4 * 255


class RetouchError(ValueError):
    """请求参数非法或原任务无法作为修色起点（HTTP 422）。"""


@dataclass(frozen=True)
class RetouchRequest:
    job_id: str
    x: int
    y: int
    width: int
    height: int
    target_lab: tuple[float, float, float]
    tolerance_de: float
    max_shift: int
    tac_limit: int


def validate_request(raw: dict) -> RetouchRequest:
    """严格校验修色请求；任何不合法输入抛 RetouchError（不猜测、不截断）。"""
    if not isinstance(raw, dict):
        raise RetouchError("请求体必须是 JSON 对象")
    job_id = raw.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RetouchError("缺少已保存任务 job_id")

    def as_int(key: str, lo: int, hi: int) -> int:
        v = raw.get(key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise RetouchError(f"{key} 必须是 {lo}..{hi} 的整数")
        if not (lo <= v <= hi):
            raise RetouchError(f"{key}={v} 超出允许范围 {lo}..{hi}")
        return v

    x = as_int("x", 0, 10**9)
    y = as_int("y", 0, 10**9)
    width = as_int("width", 1, MAX_RECT_SIDE)
    height = as_int("height", 1, MAX_RECT_SIDE)

    target = raw.get("target_lab")
    if not isinstance(target, (list, tuple)) or len(target) != 3:
        raise RetouchError("target_lab 必须是 [L, a, b] 三个数值")
    try:
        L, a, b = (float(v) for v in target)
    except (TypeError, ValueError):
        raise RetouchError("target_lab 必须是 [L, a, b] 三个数值")
    if not np.isfinite([L, a, b]).all():
        raise RetouchError("target_lab 必须是有限数值")
    if not (0.0 <= L <= 100.0):
        raise RetouchError(f"目标 L*={L} 超出 0..100")
    if not (-128.0 <= a <= 127.0):
        raise RetouchError(f"目标 a*={a} 超出 -128..127")
    if not (-128.0 <= b <= 127.0):
        raise RetouchError(f"目标 b*={b} 超出 -128..127")

    tol = raw.get("tolerance_de")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise RetouchError("tolerance_de 必须是非负数值")
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise RetouchError("tolerance_de 必须是非负有限数值")

    max_shift = as_int("max_shift", 0, MAX_SHIFT_LIMIT)
    tac_limit = as_int("tac_limit", 0, MAX_TAC)

    return RetouchRequest(
        job_id=job_id, x=x, y=y, width=width, height=height,
        target_lab=(L, a, b), tolerance_de=tol,
        max_shift=max_shift, tac_limit=tac_limit,
    )


def load_job_context(job_dir: Path) -> dict:
    """读取原任务的分色起点与参数；缺文件即拒绝。"""
    try:
        meta = json.loads((job_dir / "meta.json").read_text("utf-8"))
    except FileNotFoundError:
        raise RetouchError("原任务不存在或已被删除")
    except (json.JSONDecodeError, OSError) as exc:
        raise RetouchError(f"原任务元数据不可读: {exc}")
    try:
        cmyk = np.load(job_dir / "cmyk.npy", mmap_mode="r")
    except FileNotFoundError:
        raise RetouchError("原任务分色结果 cmyk.npy 缺失，无法以其为修色起点")
    return {"meta": meta, "cmyk": np.asarray(cmyk, dtype=np.uint8)}


def verify_press_profile(press_bytes: bytes,
                         recorded_digest: str | None) -> None:
    """原任务印刷 ICC 必须可打开且 SHA-256 摘要与任务记录一致。

    ``recorded_digest`` 为任务创建时固化的全 64 位摘要（数据库 jobs 行 /
    meta.json）。摘要缺失或不符一律拒绝，绝不在被替换的配置上修色。
    """
    if not recorded_digest:
        raise RetouchError("原任务缺少印刷 ICC 摘要记录，拒绝修色")
    digest = hashlib.sha256(press_bytes).hexdigest()
    if digest != recorded_digest:
        raise RetouchError(
            "原任务印刷 ICC 摘要不符：任务记录为 "
            f"{recorded_digest}，当前文件为 {digest}；拒绝在被替换的配置上修色"
        )
    try:
        lcms_native.read_header(press_bytes)
    except Exception as exc:
        raise RetouchError(f"原任务印刷 ICC 已损坏，无法打开: {exc}")


def _feasible_offsets(roi: np.ndarray, max_shift: int, tac_limit: int) -> np.ndarray:
    """枚举出对选区**每个像素**都满足字节范围与总墨量约束的统一整数偏移。

    通道范围约束 ``min_ch+d >= 0`` 且 ``max_ch+d <= 255`` 只取决于该通道在
    选区内的最小/最大值；总墨量约束对任一像素为
    ``(C_p+M_p+Y_p+K_p) + (dc+dm+dy+dk) <= TAC``，其在选区内的最坏情况只
    取决于**原像素总墨量的最大值**——因此也是一个标量比较，无需逐像素展开。
    返回 (N, 4) int16 数组。
    """
    flat = roi.reshape(-1, 4).astype(np.int32)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    max_pixel_ink = int(flat.sum(axis=1).max())

    ranges = []
    for ch in range(4):
        lo = max(-max_shift, -int(mins[ch]))
        hi = min(max_shift, 255 - int(maxs[ch]))
        if lo > hi:
            return np.empty((0, 4), dtype=np.int16)
        ranges.append(np.arange(lo, hi + 1, dtype=np.int16))

    shape = tuple(r.size for r in ranges)
    grid = np.empty(shape + (4,), dtype=np.int16)
    for ch, r in enumerate(ranges):
        grid[..., ch] = r.reshape(
            tuple(r.size if i == ch else 1 for i in range(4))
        )
    candidates = grid.reshape(-1, 4)
    keep = candidates.astype(np.int32).sum(axis=1) <= (tac_limit - max_pixel_ink)
    return candidates[keep]


def _quad_keys(cmyk: np.ndarray) -> np.ndarray:
    """把 (..., 4) uint8 CMYK 数组编码为 (...,) uint64 四元组键。"""
    c = cmyk.astype(np.uint64)
    return c[..., 0] | (c[..., 1] << 8) | (c[..., 2] << 16) | (c[..., 3] << 24)


def _keys_to_quads(keys: np.ndarray) -> np.ndarray:
    k = keys.astype(np.uint64)
    return np.stack(
        (k & 255, (k >> 8) & 255, (k >> 16) & 255, (k >> 24) & 255), axis=-1
    ).astype(np.uint8)


def _mean_lab_for_offset(
    flat_cmyk: np.ndarray,
    offset: np.ndarray,
    lab_xf,
    lab_cache: dict[int, np.ndarray],
) -> np.ndarray:
    """偏移后选区 CMYK -> 像素加权平均 Lab（D50 double）。

    相同（偏移后）CMYK 四元组只让 LittleCMS 回算一次；相邻候选间大量重叠的
    油墨组合直接在 ``lab_cache`` 命中，使平坦/重复色区域的搜索非常便宜。
    查表/加权均为 NumPy 操作，无逐像素 Python 循环。
    """
    shifted = np.clip(
        flat_cmyk.astype(np.int32) + offset[None, :].astype(np.int32), 0, 255
    ).astype(np.uint8)

    keys = _quad_keys(shifted)
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )

    key_list = unique_keys.tolist()
    missing_idx = [i for i, k in enumerate(key_list) if k not in lab_cache]
    if missing_idx:
        miss = np.empty((len(missing_idx), 4), dtype=np.uint8)
        for row, i in enumerate(missing_idx):
            k = key_list[i]
            miss[row] = (k & 255, (k >> 8) & 255,
                         (k >> 16) & 255, (k >> 24) & 255)
        labs = lab_xf(miss)
        for i, lab in zip(missing_idx, labs):
            lab_cache[key_list[i]] = lab

    unique_lab = np.stack([lab_cache[k] for k in key_list])
    weights = (counts.astype(np.float64) / counts.sum())[:, None]
    return (unique_lab * weights).sum(axis=0)


# 分块向量化评估时，单块“候选数 × 选区内不同 CMYK 四元组数”的工作集上限。
# 取 ~2^19：典型情形内存占用为几十 MB，3000px 上限下 64×64 选区也安全。
_EVAL_CHUNK_CELLS = 1 << 19
# 全局四元组→Lab 缓存的条数上限（紧凑 ndarray 存储，每条 16 字节）；
# 命中即直接复用，超出上限后丢弃重建——缓存仅是可选记忆化，不影响结果，
# 保证即便选区内不同 CMYK 四元组极多也不会让 Python dict 无限膨胀。
_EVAL_CACHE_CAP = 400_000


def evaluate_candidate_mean_labs(
    flat_cmyk: np.ndarray,
    candidates: np.ndarray,
    lab_xf,
) -> np.ndarray:
    """一次性向量化评估所有候选偏移的**像素加权平均 Lab**。

    与逐个候选调用 :func:`_mean_lab_for_offset` 结果完全一致（同样的裁剪、
    同样的四元组去重、同样的 LittleCMS TYPE_Lab_DBL 回算），但以分块批量
    方式让 LittleCMS 成组转换，避免 17^4 量级候选时的逐次调用开销。
    """
    n_cand = candidates.shape[0]
    mean_labs = np.empty((n_cand, 3), dtype=np.float64)
    if n_cand == 0:
        return mean_labs

    # 选区内原始不同 CMYK 四元组及其像素占比（均值即占比加权和）。
    base_keys, counts = np.unique(_quad_keys(flat_cmyk), return_counts=True)
    base_quads = _keys_to_quads(base_keys).astype(np.int32)
    weights = (counts.astype(np.float64) / counts.sum())[None, :, None]
    n_quads = base_quads.shape[0]

    # 紧凑缓存：键与 Lab 各存一个 ndarray（无逐条 Python 对象开销）。
    cache_keys = np.empty(0, dtype=np.uint64)
    cache_labs = np.empty((0, 3), dtype=np.float64)

    chunk = max(1, _EVAL_CHUNK_CELLS // max(1, n_quads))
    for start in range(0, n_cand, chunk):
        offs = candidates[start:start + chunk].astype(np.int32)
        shifted = np.clip(
            base_quads[None, :, :] + offs[:, None, :], 0, 255
        ).astype(np.uint8)
        keys = _quad_keys(shifted)
        block_unique = np.unique(keys)

        # 与缓存求差集：仅未命中的四元组交给 LittleCMS 回算。
        if cache_keys.size:
            hit_idx = np.searchsorted(cache_keys, block_unique)
            hit_idx = np.clip(hit_idx, 0, cache_keys.size - 1)
            is_hit = cache_keys[hit_idx] == block_unique
            missing = block_unique[~is_hit]
        else:
            is_hit = np.zeros_like(block_unique, dtype=bool)
            missing = block_unique

        if missing.size:
            new_labs = lab_xf(np.ascontiguousarray(_keys_to_quads(missing)))
            if cache_keys.size + missing.size > _EVAL_CACHE_CAP:
                # 超上限：只保留本块用到的四元组（丢弃旧缓存，不影响正确性）。
                cache_keys = block_unique
                cache_labs = lab_xf(
                    np.ascontiguousarray(_keys_to_quads(block_unique)))
            else:
                cache_keys = np.concatenate([cache_keys, missing])
                order = np.argsort(cache_keys)
                cache_keys = cache_keys[order]
                cache_labs = np.concatenate([cache_labs, new_labs])[order]

        # 一次 searchsorted 取回本块全部键对应的 Lab，再按像素占比加权。
        flat_idx = np.clip(
            np.searchsorted(cache_keys, keys.ravel()), 0, cache_keys.size - 1)
        flat_lab = cache_labs[flat_idx]
        gathered = flat_lab.reshape(keys.shape[0], n_quads, 3)
        mean_labs[start:start + offs.shape[0]] = (gathered * weights).sum(axis=1)
    return mean_labs


def run_retouch(req: RetouchRequest, ctx: dict, press_bytes: bytes) -> dict:
    """执行搜索并返回可直接落盘/展示的修色结果字典。"""
    meta = ctx["meta"]
    cmyk_full = ctx["cmyk"]
    H, W = cmyk_full.shape[:2]
    if req.x >= W or req.y >= H:
        raise RetouchError("选区起点超出原任务图像范围")
    x2, y2 = min(W, req.x + req.width), min(H, req.y + req.height)
    rw, rh = x2 - req.x, y2 - req.y

    roi = cmyk_full[req.y:y2, req.x:x2]
    flat = np.ascontiguousarray(roi).reshape(-1, 4)

    intent = meta.get("source_intent", "relative_colorimetric")
    if intent not in lcms_native.INTENTS:
        raise RetouchError(f"原任务渲染意图无效: {intent}")

    candidates = _feasible_offsets(roi, req.max_shift, req.tac_limit)
    target = np.array(req.target_lab, dtype=np.float64)
    with lcms_native.lab_transform("CMYK", press_bytes, intent) as lab_xf:
        roi_lab_before = lab_xf(np.ascontiguousarray(flat))
        baseline_mean_lab = roi_lab_before.mean(axis=0)
        de_before_target = float(colorscience.ciede2000(
            target[None, None, :], baseline_mean_lab[None, None, :])[0, 0])

        n_cand = candidates.shape[0]
        mean_labs = evaluate_candidate_mean_labs(flat, candidates, lab_xf)

        # 一次性向量化计算全部候选对目标的 ΔE00（colorscience 与主流水线相同）。
        if n_cand:
            targets = np.repeat(target[None, :], n_cand, axis=0)
            de_all = colorscience.ciede2000(
                targets.reshape(n_cand, 1, 3),
                mean_labs.reshape(n_cand, 1, 3),
            ).reshape(n_cand)
        else:
            de_all = np.empty(0, dtype=np.float64)

        abs_sums = np.abs(candidates.astype(np.int32)).sum(axis=1)
        # 1) ΔE00 升序 2) 偏移绝对值和升序 3) (C,M,Y,K) 字典序；
        # 用 Schwartzian 变换只构造一次排序键。
        decorated = [
            (float(de_all[i]), int(abs_sums[i]),
             tuple(int(v) for v in candidates[i]), i)
            for i in range(n_cand)
        ]
        decorated.sort(key=lambda t: (t[0], t[1], t[2]))

        feasible_count = n_cand
        if feasible_count == 0:
            best = None
        else:
            de_b, abs_b, off_b, bi = decorated[0]
            best = (de_b, abs_b, off_b, mean_labs[bi])

        if best is None:
            reachable = False
            best_de = None
            best_offset = None
            best_mean_lab = None
        else:
            best_de, _abs_sum, best_offset, best_mean_lab = best
            reachable = best_de <= req.tolerance_de + 1e-9

        offset = list(best_offset) if best_offset is not None else [0, 0, 0, 0]

        # 构造修色后的整幅 CMYK（选区外字节严格不变）。
        adjusted_full = cmyk_full.copy()
        roi_adj = np.clip(
            roi.astype(np.int32) + np.array(offset, dtype=np.int32), 0, 255
        ).astype(np.uint8)
        adjusted_full[req.y:y2, req.x:x2] = roi_adj

        if reachable:
            roi_lab_after = lab_xf(np.ascontiguousarray(roi_adj.reshape(-1, 4)))
            de_residual = colorscience.ciede2000(
                roi_lab_before.reshape(rh, rw, 3),
                roi_lab_after.reshape(rh, rw, 3),
            )
            residual_stats = {
                "mean_de00_before_after": round(float(de_residual.mean()), 4),
                "max_de00_before_after": round(float(de_residual.max()), 4),
                "mean_lab_before": [round(float(v), 4) for v in baseline_mean_lab],
                "mean_lab_after": [round(float(v), 4) for v in best_mean_lab],
                "target_lab": [round(float(v), 4) for v in target],
                "de00_to_target": round(float(best_de), 4),
                "de00_before_to_target": round(float(de_before_target), 4),
            }
        else:
            residual_stats = {
                "mean_de00_before_after": None,
                "max_de00_before_after": None,
                "mean_lab_before": [round(float(v), 4) for v in baseline_mean_lab],
                "mean_lab_after": None,
                "target_lab": [round(float(v), 4) for v in target],
                "de00_to_target": (round(float(best_de), 4)
                                   if best_de is not None else None),
                "de00_before_to_target": round(float(de_before_target), 4),
            }

    return {
        "rect": {"x": int(req.x), "y": int(req.y),
                 "width": int(rw), "height": int(rh)},
        "pixel_count": int(rw * rh),
        "constraints": {
            "target_lab": [float(v) for v in target],
            "tolerance_de00": req.tolerance_de,
            "max_shift_per_channel": req.max_shift,
            "total_ink_limit": req.tac_limit,
        },
        "feasible_offset_count": int(feasible_count),
        "reachable": bool(reachable),
        "best_offset_cmyk": [int(v) for v in offset],
        "best_abs_shift_sum": (int(sum(abs(v) for v in offset))
                               if best_offset is not None else None),
        "residual": residual_stats,
        "_cmyk_adjusted": adjusted_full,
        "_roi_before": roi,
        "_roi_after": roi_adj,
        "_intent": intent,
    }


# ---------------------------------------------------------------------------
# 软打样可视化（选区修色前/后）与成品 TIFF
# ---------------------------------------------------------------------------

def render_proof_crops(result: dict, cmyk_full: np.ndarray,
                       press_bytes: bytes, display_bytes: bytes,
                       proof_intent: str) -> dict[str, bytes]:
    """生成修色前/后选区软打样 PNG（嵌入显示 ICC）。"""
    rect = result["rect"]
    x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
    before = np.ascontiguousarray(cmyk_full[y:y + h, x:x + w].reshape(-1, 4))
    after = np.ascontiguousarray(result["_roi_after"].reshape(-1, 4))

    with lcms_native.lab_transform("CMYK", press_bytes,
                                   result["_intent"]) as lab_xf:
        lab_before = lab_xf(before)
        lab_after = lab_xf(after)

    def to_png(lab: np.ndarray) -> bytes:
        rgb = lcms_native.lab_to_pixels(lab, display_bytes, "RGB", proof_intent)
        img = Image.fromarray(rgb.reshape(h, w, 3), "RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", icc_profile=display_bytes)
        return buf.getvalue()

    return {"before.png": to_png(lab_before), "after.png": to_png(lab_after)}


def export_retouched_tiff(cmyk_adjusted: np.ndarray, press_bytes: bytes) -> bytes:
    """整幅修色后 CMYK 存为嵌入原印刷 ICC 的 TIFF（选区外字节不变）。"""
    img = Image.fromarray(np.ascontiguousarray(cmyk_adjusted), "CMYK")
    buf = io.BytesIO()
    # 无压缩：字节确定、逐通道可核验；PIL 对 CMYK 写 4 个 8 位样例/像素。
    img.save(buf, format="TIFF", compression="raw", icc_profile=press_bytes)
    return buf.getvalue()


def _canon_float(v: float) -> float:
    """规范化浮点取值：-0.0 归一为 0.0，使数值相等的请求得到同一身份。"""
    v = float(v)
    return 0.0 if v == 0.0 else v


def request_fingerprint(meta: dict, req: RetouchRequest) -> dict:
    """完整修色请求的规范化指纹——修色结果身份与缓存核对的唯一依据。

    身份规则：一次修色由 **原任务（job_id 及其固化的渲染意图）+ 选区矩形
    + 目标 Lab + 容差 + 单通道偏移上限 + 总墨量上限** 的完整组合唯一确定。
    指纹为带字段名的字典，经 :func:`_canonical_json` 序列化后不存在字段间
    无分隔拼接的歧义（如 (x=1, y=23) 与 (x=12, y=3) 都拼出 "123"）；
    浮点按最短往返表示完整保留双精度取值，不做 ``%.6f`` 之类的截断，
    目标 Lab / 容差在任意小数位上的差异都会得到不同身份。
    """
    return {
        "job_id": meta["job_id"],
        "source_intent": meta.get("source_intent", ""),
        "proof_intent": meta.get("proof_intent", ""),
        "x": int(req.x),
        "y": int(req.y),
        "width": int(req.width),
        "height": int(req.height),
        "target_lab": [_canon_float(v) for v in req.target_lab],
        "tolerance_de": _canon_float(req.tolerance_de),
        "max_shift": int(req.max_shift),
        "tac_limit": int(req.tac_limit),
    }


def _canonical_json(obj) -> str:
    """逐字节确定的 JSON：键排序、紧凑分隔符；float 走最短往返表示（repr），
    完整保留双精度精度且 JSON 解析回原值（缓存核对可直接做字典相等比较）。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def compute_retouch_id(meta: dict, req: RetouchRequest) -> str:
    """retouch_id = 完整请求指纹的规范 JSON 之 SHA-256。"""
    return hashlib.sha256(
        _canonical_json(request_fingerprint(meta, req)).encode("utf-8")
    ).hexdigest()


def _reports_equivalent(a: object, b: object, *, _tol: float = 1e-9) -> bool:
    """结构相同即视为一致；浮点数按 ``_tol`` 容差比较（None 严格区分）。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= _tol
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _reports_equivalent(x, y, _tol=_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _reports_equivalent(a[k], b[k], _tol=_tol) for k in a)
    return a == b


def _expected_rect(meta: dict, req: RetouchRequest) -> dict:
    """本次请求在实际图像尺寸上裁剪后的真实选区（靠边缘时按实际选区核对）。"""
    H = int(meta.get("height", 0))
    W = int(meta.get("width", 0))
    x2, y2 = min(W, req.x + req.width), min(H, req.y + req.height)
    return {"x": int(req.x), "y": int(req.y),
            "width": int(x2 - req.x), "height": int(y2 - req.y)}


def _check_cached_structure(
    report: object,
    retouch_id: str,
    fingerprint: dict,
    meta: dict,
    req: RetouchRequest,
    press_summary: dict,
    display_summary: dict,
) -> bool:
    """不依赖重算的结构性绑定核对。

    逐项确认缓存报告里记录的实际选区、约束、原任务与 ICC 摘要确实对应当前
    请求（而不是只凭目录名/指纹文本/文件是否存在就复用）。任一项不符即
    拒绝复用。
    """
    if not isinstance(report, dict):
        return False
    if report.get("retouch_id") != retouch_id:
        return False
    # 完整请求指纹必须逐项相等（任务、意图、矩形、目标 Lab、容差、两类约束）。
    if report.get("request") != fingerprint:
        return False

    expected_rect = _expected_rect(meta, req)
    if report.get("region") != expected_rect:
        return False
    if report.get("pixel_count") != expected_rect["width"] * expected_rect["height"]:
        return False

    # 报告里另存的约束块必须是本次请求实际生效的约束（防止只改报告字段的
    # 脏缓存：request 不变但 constraints/region 被改写）。
    constraints = report.get("constraints")
    if not isinstance(constraints, dict):
        return False

    def _is_real_num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    cached_target = constraints.get("target_lab")
    if not isinstance(cached_target, list) or len(cached_target) != 3 \
            or not all(_is_real_num(v) for v in cached_target) \
            or any(abs(float(a) - float(b)) > 1e-9
                   for a, b in zip(cached_target, req.target_lab)):
        return False
    cached_tol = constraints.get("tolerance_de00")
    if not _is_real_num(cached_tol) or float(cached_tol) < 0.0 \
            or abs(float(cached_tol) - float(req.tolerance_de)) > 1e-9:
        return False
    if constraints.get("max_shift_per_channel") != int(req.max_shift) \
            or constraints.get("total_ink_limit") != int(req.tac_limit):
        return False

    # 原任务绑定：job_id、原图哈希、尺寸、模式、意图、阈值都必须与当前任务
    # 的 meta 一致（同 ID 目录被另一份有效任务数据替换时由此检出）。
    source_job = report.get("source_job")
    if not isinstance(source_job, dict):
        return False
    expected_job = {
        "job_id": meta.get("job_id"),
        "image_sha256": meta.get("image_sha256"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "image_mode": meta.get("mode"),
        "source_intent": meta.get("source_intent"),
        "proof_intent": meta.get("proof_intent"),
        "threshold_delta_e": meta.get("threshold"),
    }
    if any(source_job.get(k) != v for k, v in expected_job.items()):
        return False

    # ICC 摘要：修色报告必须绑定当前任务实际使用的印刷/显示 ICC 全摘要
    # （至少全 64 位 SHA-256 必须一致；调用方已在读取时校验过摘要）。
    profiles = report.get("profiles")
    if not isinstance(profiles, dict):
        return False
    for role, summary in (("press", press_summary), ("display", display_summary)):
        cached_summary = profiles.get(role)
        if not isinstance(cached_summary, dict):
            return False
        if cached_summary.get("sha256") != summary.get("sha256"):
            return False

    # reachable 与偏移字段必须自洽。
    if report.get("reachable") not in (True, False):
        return False
    off = report.get("best_offset_cmyk")
    if not isinstance(off, list) or len(off) != 4 \
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in off):
        return False
    if report.get("reachable") is True:
        if any(abs(v) > int(req.max_shift) for v in off):
            return False
        if report.get("best_abs_shift_sum") != sum(abs(v) for v in off):
            return False
        if not isinstance(report.get("residual"), dict) \
                or report["residual"].get("mean_lab_after") is None:
            return False
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    if bool(artifacts.get("tiff")) is not bool(report.get("reachable")):
        return False
    return True


def cache_entry_valid(report: object, retouch_id: str, fingerprint: dict,
                      base: Path, *, meta: dict, req: RetouchRequest,
                      cmyk_full: np.ndarray, press_bytes: bytes,
                      display_bytes: bytes,
                      press_summary: dict, display_summary: dict) -> bool:
    """命中缓存时完整核对报告、预览与成品与本次请求的绑定。

    核对分两层，任一不符即拒绝复用（调用方按本次请求重新搜索并覆盖报告
    与全部产物）：

    1. **结构绑定**（不重算）：``retouch_id`` 与完整请求指纹逐项一致；
       报告的实际选区 = 请求矩形在当前图像上**按边缘裁剪后**的真实选区；
       constraints 块与请求约束一致；原任务（job_id/原图哈希/尺寸/模式/
       意图/阈值）与 ICC 摘要（印刷+显示全 64 位 SHA-256）一致；
       reachable/偏移/产物字段自洽；
    2. **内容绑定**（向量化重算，平坦 64×64 选区也在亚秒级）：按当前任务
       的 ``cmyk.npy`` 与当前印刷 ICC **重新搜索**，得到的可达性、可行偏移
       数、最优偏移、绝对值和、残余色差必须与报告一致；再按报告/重算结果
       **重新渲染** before/after PNG 与（可达时）成品 TIFF，与磁盘文件
       逐字节比较。因此用另一份有效图像/TIFF 替换同名文件、或报告记录的
       选区/偏移/约束与产物不符，都无法命中复用。
    """
    try:
        if not _check_cached_structure(
            report, retouch_id, fingerprint, meta, req,
            press_summary, display_summary,
        ):
            return False

        # 内容重算：必须与当前任务分色起点、当前 ICC 上的真实搜索结果相同。
        result = run_retouch(req, {"meta": meta, "cmyk": cmyk_full}, press_bytes)
        if result["reachable"] != report["reachable"]:
            return False
        if result["best_offset_cmyk"] != report["best_offset_cmyk"]:
            return False
        if result["best_abs_shift_sum"] != report["best_abs_shift_sum"]:
            return False
        if result["feasible_offset_count"] != \
                report.get("search", {}).get("feasible_offset_count"):
            return False
        if not _reports_equivalent(result["residual"], report.get("residual")):
            return False

        proof_intent = meta.get("proof_intent", "relative_colorimetric")
        pngs = render_proof_crops(
            result, cmyk_full, press_bytes, display_bytes, proof_intent)
        if (base / "before.png").read_bytes() != pngs["before.png"]:
            return False
        if (base / "after.png").read_bytes() != pngs["after.png"]:
            return False

        tiff_path = base / "retouched.tif"
        if result["reachable"]:
            expected_tiff = export_retouched_tiff(
                result["_cmyk_adjusted"], press_bytes)
            if not tiff_path.is_file() \
                    or tiff_path.read_bytes() != expected_tiff:
                return False
        elif tiff_path.exists():
            # 不可达时不得残留成品 TIFF。
            return False
    except Exception:
        # 产物损坏、图像无法解码、色彩转换失败等任何意外：一律视为绑定不符，
        # 交由调用方重新计算并覆盖（重算路径会给出明确的 422 原因）。
        return False
    return True


def build_report(retouch_id: str, meta: dict, press_summary: dict,
                 display_summary: dict, req: RetouchRequest, result: dict) -> dict:
    """修色报告：记录原任务、ICC 摘要、全部约束与最终偏移。"""
    return {
        "retouch_id": retouch_id,
        # 完整请求指纹：缓存命中时据此核对报告与本次任务/矩形/约束的绑定。
        "request": request_fingerprint(meta, req),
        "source_job": {
            "job_id": meta["job_id"],
            "image_sha256": meta.get("image_sha256"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "image_mode": meta.get("mode"),
            "source_intent": meta.get("source_intent"),
            "proof_intent": meta.get("proof_intent"),
            "threshold_delta_e": meta.get("threshold"),
        },
        "profiles": {
            "press": press_summary,
            "display": display_summary,
        },
        "region": result["rect"],
        "pixel_count": result["pixel_count"],
        "constraints": result["constraints"],
        "search": {
            "max_rect_side": MAX_RECT_SIDE,
            "feasible_offset_count": result["feasible_offset_count"],
            "selection_order": [
                "mean Lab ΔE00 (CIEDE2000, LittleCMS TYPE_Lab_DBL 回算)",
                "offset abs sum |dC|+|dM|+|dY|+|dK|",
                "(dC, dM, dY, dK) lexicographic",
            ],
        },
        "reachable": result["reachable"],
        "best_offset_cmyk": result["best_offset_cmyk"],
        "best_abs_shift_sum": result["best_abs_shift_sum"],
        "residual": result["residual"],
        "artifacts": (
            {"tiff": "retouched.tif", "before_png": "before.png",
             "after_png": "after.png"}
            if result["reachable"]
            else {"tiff": None, "before_png": "before.png", "after_png": "after.png"}
        ),
        "note": (
            "修色以原任务分色 CMYK 为起点，按原印刷 ICC 搜索统一整数偏移；"
            "仅当最优平均 Lab ΔE00 不超过容差时标记可达并导出嵌入印刷 ICC 的 "
            "CMYK TIFF（选区外字节不变）。不可达时不生成成品。"
        ),
    }
