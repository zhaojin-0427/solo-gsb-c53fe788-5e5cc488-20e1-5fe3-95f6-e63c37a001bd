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

    keys = (
        shifted[:, 0].astype(np.uint64)
        | (shifted[:, 1].astype(np.uint64) << 8)
        | (shifted[:, 2].astype(np.uint64) << 16)
        | (shifted[:, 3].astype(np.uint64) << 24)
    )
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


def clip_region(cmyk_full: np.ndarray, req: RetouchRequest) -> tuple[int, int, int, int]:
    """按原任务实际尺寸裁剪选区，返回 ``(x, y, clipped_width, clipped_height)``。

    选区起点越界非法；宽高超出图像边缘时按实际交点裁剪（例如选区贴住右/下
    边）——缓存核对必须以**裁剪后的实际选区**为准，而不是请求字面值。
    """
    h, w = cmyk_full.shape[:2]
    if req.x >= w or req.y >= h:
        raise RetouchError("选区起点超出原任务图像范围")
    x2, y2 = min(w, req.x + req.width), min(h, req.y + req.height)
    return req.x, req.y, x2 - req.x, y2 - req.y


def compute_residual_stats(lab_xf, roi_before: np.ndarray, roi_after: np.ndarray,
                           target: np.ndarray, best_mean_lab, best_de,
                           de_before_target: float, reachable: bool,
                           rect_hw: tuple[int, int]) -> dict:
    """构造报告中的残余色差字典。

    搜索与缓存校验共用同一份实现，保证缓存命中时按当前 CMYK/ICC 复算出的
    数值与报告逐键一致；不可达时修色后相关字段一律为 ``None``。
    ``lab_xf`` 为已打开的 ``CMYK -> Lab_DBL`` 变换。
    """
    rh, rw = rect_hw
    flat_before = np.ascontiguousarray(roi_before).reshape(-1, 4)
    lab_before = lab_xf(flat_before)
    mean_lab_before = lab_before.mean(axis=0)
    common = {
        "mean_lab_before": [round(float(v), 4) for v in mean_lab_before],
        "target_lab": [round(float(v), 4) for v in target],
        "de00_before_to_target": round(float(de_before_target), 4),
    }
    if not reachable:
        return {
            "mean_de00_before_after": None,
            "max_de00_before_after": None,
            "mean_lab_after": None,
            "de00_to_target": (round(float(best_de), 4)
                               if best_de is not None else None),
            **common,
        }
    lab_after = lab_xf(np.ascontiguousarray(roi_after).reshape(-1, 4))
    de_residual = colorscience.ciede2000(
        lab_before.reshape(rh, rw, 3),
        lab_after.reshape(rh, rw, 3),
    )
    return {
        "mean_de00_before_after": round(float(de_residual.mean()), 4),
        "max_de00_before_after": round(float(de_residual.max()), 4),
        "mean_lab_after": [round(float(v), 4) for v in best_mean_lab],
        "de00_to_target": round(float(best_de), 4),
        **common,
    }


def run_retouch(req: RetouchRequest, ctx: dict, press_bytes: bytes) -> dict:
    """执行搜索并返回可直接落盘/展示的修色结果字典。"""
    meta = ctx["meta"]
    cmyk_full = ctx["cmyk"]

    intent = meta.get("source_intent", "relative_colorimetric")
    if intent not in lcms_native.INTENTS:
        raise RetouchError(f"原任务渲染意图无效: {intent}")

    x, y, rw, rh = clip_region(cmyk_full, req)
    x2, y2 = x + rw, y + rh

    roi = cmyk_full[y:y2, x:x2]
    flat = np.ascontiguousarray(roi).reshape(-1, 4)

    candidates = _feasible_offsets(roi, req.max_shift, req.tac_limit)
    target = np.array(req.target_lab, dtype=np.float64)
    with lcms_native.lab_transform("CMYK", press_bytes, intent) as lab_xf:
        roi_lab_before = lab_xf(np.ascontiguousarray(flat))
        baseline_mean_lab = roi_lab_before.mean(axis=0)
        de_before_target = float(colorscience.ciede2000(
            target[None, None, :], baseline_mean_lab[None, None, :])[0, 0])

        lab_cache: dict[int, np.ndarray] = {}
        mean_labs = np.empty((candidates.shape[0], 3), dtype=np.float64)
        for gi, off in enumerate(candidates):
            mean_labs[gi] = _mean_lab_for_offset(flat, off, lab_xf, lab_cache)

        # 一次性向量化计算全部候选对目标的 ΔE00（colorscience 与主流水线相同）。
        n_cand = candidates.shape[0]
        targets = np.repeat(target[None, :], n_cand, axis=0)
        de_all = colorscience.ciede2000(
            targets.reshape(n_cand, 1, 3),
            mean_labs.reshape(n_cand, 1, 3),
        ).reshape(n_cand)

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
        adjusted_full[y:y2, x:x2] = roi_adj

        residual_stats = compute_residual_stats(
            lab_xf=lab_xf,
            roi_before=roi,
            roi_after=roi_adj,
            target=target,
            best_mean_lab=best_mean_lab if reachable else None,
            best_de=best_de,
            de_before_target=de_before_target,
            reachable=reachable,
            rect_hw=(rh, rw),
        )

    return {
        "rect": {"x": int(x), "y": int(y),
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

def render_proof_pngs(roi_before: np.ndarray, roi_after: np.ndarray,
                      rect_hw: tuple[int, int], press_bytes: bytes,
                      display_bytes: bytes, intent: str,
                      proof_intent: str) -> dict[str, bytes]:
    """给定选区划 CMYK（修色前/后）生成嵌入显示 ICC 的软打样 PNG。

    搜索落盘与缓存内容核对共用同一渲染，字节完全确定，因此可以直接做
    字节相等比较。
    """
    rh, rw = rect_hw
    before = np.ascontiguousarray(roi_before).reshape(-1, 4)
    after = np.ascontiguousarray(roi_after).reshape(-1, 4)

    with lcms_native.lab_transform("CMYK", press_bytes, intent) as lab_xf:
        lab_before = lab_xf(before)
        lab_after = lab_xf(after)

    def to_png(lab: np.ndarray) -> bytes:
        rgb = lcms_native.lab_to_pixels(lab, display_bytes, "RGB", proof_intent)
        img = Image.fromarray(rgb.reshape(rh, rw, 3), "RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", icc_profile=display_bytes)
        return buf.getvalue()

    return {"before.png": to_png(lab_before), "after.png": to_png(lab_after)}


def render_proof_crops(result: dict, cmyk_full: np.ndarray,
                       press_bytes: bytes, display_bytes: bytes,
                       proof_intent: str) -> dict[str, bytes]:
    """生成修色前/后选区软打样 PNG（嵌入显示 ICC）。"""
    rect = result["rect"]
    x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
    return render_proof_pngs(
        cmyk_full[y:y + h, x:x + w],
        result["_roi_after"],
        (h, w),
        press_bytes, display_bytes,
        result["_intent"], proof_intent,
    )


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


def _same_lab_list(a, b, tol: float = 1e-4) -> bool:
    """两个报告中的 Lab/数值列表逐项比较（报告值 round 到 4 位）。"""
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)):
        return False
    return all(isinstance(x, (int, float)) and isinstance(y, (int, float))
               and abs(float(x) - float(y)) <= 1e-9 + tol
               for x, y in zip(a, b))


def _expected_constraints(req: RetouchRequest) -> dict:
    return {
        "target_lab": [float(v) for v in req.target_lab],
        "tolerance_de00": float(req.tolerance_de),
        "max_shift_per_channel": int(req.max_shift),
        "total_ink_limit": int(req.tac_limit),
    }


def _expected_source_job(meta: dict) -> dict:
    return {
        "job_id": meta["job_id"],
        "image_sha256": meta.get("image_sha256"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "image_mode": meta.get("mode"),
        "source_intent": meta.get("source_intent"),
        "proof_intent": meta.get("proof_intent"),
        "threshold_delta_e": meta.get("threshold"),
    }


def _offset_is_feasible(roi: np.ndarray, offset, max_shift: int,
                        tac_limit: int) -> bool:
    """统一偏移对选区**每个像素**满足字节范围 / 单通道上限 / 总墨量上限。"""
    off = np.asarray(offset, dtype=np.int32)
    if off.shape != (4,):
        return False
    if np.abs(off).max() > max_shift:
        return False
    shifted = roi.reshape(-1, 4).astype(np.int32) + off[None, :]
    if shifted.min() < 0 or shifted.max() > 255:
        return False
    return bool(shifted.sum(axis=1).max() <= tac_limit)


# 缓存命中时全量复算选优的候选上限；超过则“3^4 邻域 + 等距抽样”核对。
# 阈值取 4000：真实选区内墨色高度相关，lab 缓存命中率高、代价毫秒级；
# 只有“随机噪点级选区 + max_shift=8”这类病态输入会产生上万组低重叠候选，
# 其首次搜索本身就慢——缓存命中时抽样核对避免重复支付全量代价。
_FULL_CHECK_CANDIDATES = 4000
# 超大候选空间下的抽样规模。
_SAMPLE_CANDIDATES = 2000


def verify_cache(report: object, retouch_id: str, fingerprint: dict,
                 base: Path, ctx: dict, req: RetouchRequest,
                 press_bytes: bytes, display_bytes: bytes) -> bool:
    """逐项核对缓存报告/预览/成品与**本次请求**的实际绑定。

    不仅核对指纹、任务 ID 与文件存在，还核对：

    * 报告记录的实际选区 == 请求矩形按原任务尺寸**裁剪后**的选区
      （贴边的合法裁剪按实际交点核对）、约束 == 请求约束；
    * 报告绑定的原任务（原图哈希、尺寸、模式、双意图、阈值）与当前
      ``meta`` 一致，印刷/显示 ICC 的 SHA-256 摘要与当前字节一致；
    * before/after 预览 PNG 字节 == 按当前分色、报告偏移与当前 ICC
      **重新渲染**的结果（尺寸/嵌入显示 ICC/像素内容一并绑定）；
    * 可行偏移数量独立重算；报告偏移逐像素满足通道/总墨量约束，候选数在
      阈值内时全量、否则以 3^4 邻域 + 等距抽样复算选优，均不得优于报告
      偏移；残余色差按当前分色与印刷 ICC 复算一致；
    * 成品 TIFF 解码后逐像素核对（选区外逐字节相同、选区内恰为该统一
      偏移）并核对嵌入的印刷 ICC；可达时必须存在，不可达时不得残留。

    任一项不符返回 ``False``，调用方按本次请求重新搜索并覆盖全部产物。
    """
    try:
        if not isinstance(report, dict):
            return False
        if report.get("retouch_id") != retouch_id:
            return False
        if report.get("request") != fingerprint:
            return False

        meta = ctx["meta"]
        cmyk_full = ctx["cmyk"]
        x, y, rw, rh = clip_region(cmyk_full, req)
        x2, y2 = x + rw, y + rh

        # ---- 报告语义字段：实际选区 / 像素数 / 约束 --------------------
        if report.get("region") != {
                "x": int(x), "y": int(y),
                "width": int(rw), "height": int(rh)}:
            return False
        if report.get("pixel_count") != int(rw * rh):
            return False
        constraints = report.get("constraints")
        expected_constraints = _expected_constraints(req)
        if not isinstance(constraints, dict):
            return False
        if set(constraints) != set(expected_constraints):
            return False
        for k, v in expected_constraints.items():
            cv = constraints[k]
            if isinstance(v, list):
                if not _same_lab_list(cv, v, tol=0.0):
                    return False
            elif cv != v:
                return False

        # ---- 原任务绑定 ------------------------------------------------
        if report.get("source_job") != _expected_source_job(meta):
            return False

        # ---- ICC 摘要绑定 ----------------------------------------------
        profiles = report.get("profiles")
        if not isinstance(profiles, dict):
            return False
        press_summary = profiles.get("press")
        display_summary = profiles.get("display")
        if not isinstance(press_summary, dict) or \
                not isinstance(display_summary, dict):
            return False
        if press_summary.get("sha256") != hashlib.sha256(press_bytes).hexdigest():
            return False
        if display_summary.get("sha256") != hashlib.sha256(display_bytes).hexdigest():
            return False
        if press_summary.get("color_space") != "CMYK":
            return False

        # ---- 可达性与文件存在性一致 ------------------------------------
        reachable = report.get("reachable")
        if reachable not in (True, False):
            return False
        tiff_path = base / "retouched.tif"
        if reachable is True and not tiff_path.is_file():
            return False
        if reachable is False and tiff_path.exists():
            return False
        before_path, after_path = base / "before.png", base / "after.png"
        if not before_path.is_file() or not after_path.is_file():
            return False

        # ---- 报告偏移 / 搜索计数 / 选优 --------------------------------
        offset_raw = report.get("best_offset_cmyk")
        if not (isinstance(offset_raw, list) and len(offset_raw) == 4
                and all(isinstance(v, int) and not isinstance(v, bool)
                        for v in offset_raw)):
            return False
        offset = np.asarray(offset_raw, dtype=np.int32)
        abs_sum = int(np.abs(offset).sum())

        roi = cmyk_full[y:y2, x:x2]
        candidates = _feasible_offsets(roi, req.max_shift, req.tac_limit)
        search = report.get("search")
        if not isinstance(search, dict) or \
                search.get("feasible_offset_count") != int(candidates.shape[0]):
            return False
        if reachable and not _offset_is_feasible(
                roi, offset, req.max_shift, req.tac_limit):
            return False
        if candidates.shape[0] == 0:
            # 无可行偏移：报告偏移必须是零偏移（实现以零偏移构造“修色后”）。
            if offset_raw != [0, 0, 0, 0]:
                return False
            if report.get("best_abs_shift_sum") not in (None, 0):
                return False
        else:
            if not _offset_is_feasible(roi, offset, req.max_shift,
                                       req.tac_limit):
                return False
            if report.get("best_abs_shift_sum") != abs_sum:
                return False

        roi_after = np.clip(
            roi.astype(np.int32) + offset[None, None, :], 0, 255
        ).astype(np.uint8)
        target = np.array(req.target_lab, dtype=np.float64)
        intent = meta.get("source_intent", "relative_colorimetric")

        with lcms_native.lab_transform("CMYK", press_bytes, intent) as lab_xf:
            flat = np.ascontiguousarray(roi).reshape(-1, 4)
            lab_before = lab_xf(flat)
            mean_before = lab_before.mean(axis=0)
            de_before_target = float(colorscience.ciede2000(
                target[None, None, :], mean_before[None, None, :])[0, 0])
            mean_after = lab_xf(
                np.ascontiguousarray(roi_after).reshape(-1, 4)
            ).mean(axis=0)
            de_after_target = float(colorscience.ciede2000(
                target[None, None, :], mean_after[None, None, :])[0, 0])

            # 可达性必须与“报告偏移的实际 ΔE00 对容差”相符。
            if reachable != (de_after_target <= float(req.tolerance_de) + 1e-9):
                return False

            # 抽样/全量复算选优：不得有候选按 (ΔE00, 绝对值和, 字典序)
            # 严格优于报告偏移（与 run_retouch 完全相同的三级选优规则）。
            if candidates.shape[0]:
                if candidates.shape[0] <= _FULL_CHECK_CANDIDATES:
                    sample = candidates
                else:
                    # 超大候选空间：报告偏移的 3^4 邻域 + 等距抽样。
                    claimed_t = tuple(int(v) for v in offset)
                    near = np.all(
                        np.abs(candidates.astype(np.int32)
                               - np.asarray(claimed_t, dtype=np.int32)) <= 1,
                        axis=1,
                    )
                    step = max(1, candidates.shape[0] // _SAMPLE_CANDIDATES)
                    sample = np.unique(np.concatenate(
                        [candidates[near],
                         candidates[::step][:_SAMPLE_CANDIDATES]], axis=0
                    ), axis=0)
                lab_cache: dict[int, np.ndarray] = {}
                decorated = []
                for off in sample:
                    mean_lab = _mean_lab_for_offset(flat, off, lab_xf, lab_cache)
                    de_off = float(colorscience.ciede2000(
                        target[None, None, :], mean_lab[None, None, :])[0, 0])
                    decorated.append(
                        (de_off, int(np.abs(off.astype(np.int32)).sum()),
                         tuple(int(v) for v in off)))
                decorated.sort(key=lambda t: (t[0], t[1], t[2]))
                # 报告偏移的 ΔE00 在同一 transform 下按相同输入重算，与候选
                # 评估完全一致（同一 mean-Lab、同一 ΔE 公式），浮点结果按位
                # 相同，因此直接用三级键比较即可，无需容差。
                claimed_key = (de_after_target, abs_sum,
                               tuple(int(v) for v in offset))
                if decorated[0] < claimed_key:
                    return False

            # ---- 残余色差按当前分色 / ICC 复算一致 --------------------
            expected_residual = compute_residual_stats(
                lab_xf=lab_xf,
                roi_before=roi,
                roi_after=roi_after,
                target=target,
                best_mean_lab=mean_after if reachable else None,
                best_de=de_after_target if candidates.shape[0] else None,
                de_before_target=de_before_target,
                reachable=reachable,
                rect_hw=(rh, rw),
            )
        residual = report.get("residual")
        if not isinstance(residual, dict) or \
                set(residual) != set(expected_residual):
            return False
        for k, ev in expected_residual.items():
            rv = residual[k]
            if ev is None:
                if rv is not None:
                    return False
            elif isinstance(ev, list):
                if not _same_lab_list(rv, ev, tol=5e-5):
                    return False
            elif isinstance(ev, (int, float)):
                if not isinstance(rv, (int, float)) or abs(rv - ev) > 5e-5:
                    return False
            elif rv != ev:
                return False

        # ---- 预览 PNG：按当前数据/ICC 重渲染，字节必须一致 -------------
        proof_intent = meta.get("proof_intent", "relative_colorimetric")
        expected_pngs = render_proof_pngs(
            roi, roi_after, (rh, rw),
            press_bytes, display_bytes, intent, proof_intent,
        )
        if before_path.read_bytes() != expected_pngs["before.png"]:
            return False
        if after_path.read_bytes() != expected_pngs["after.png"]:
            return False

        # ---- 成品 TIFF：解码后逐像素核对（选区外逐字节相同、选区内恰为
        # 该统一偏移）并核对嵌入的印刷 ICC；不重新编码整幅 TIFF。 -------
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, dict):
            return False
        if reachable:
            adjusted_full = cmyk_full.copy()
            adjusted_full[y:y2, x:x2] = roi_after
            try:
                with Image.open(tiff_path) as im:
                    im.load()
                    if im.mode != "CMYK" or im.size != (
                            int(cmyk_full.shape[1]), int(cmyk_full.shape[0])):
                        return False
                    if im.info.get("icc_profile") != press_bytes:
                        return False
                    tiff_cmyk = np.asarray(im)
            except Exception:
                return False
            if tiff_cmyk.shape != adjusted_full.shape or \
                    not np.array_equal(tiff_cmyk, adjusted_full):
                return False
            if artifacts.get("tiff") != "retouched.tif":
                return False
        elif artifacts.get("tiff") is not None:
            return False
        if artifacts.get("before_png") != "before.png" or \
                artifacts.get("after_png") != "after.png":
            return False

        return True
    except (OSError, ValueError, KeyError, TypeError):
        # 缓存文件损坏 / 报告结构异常：一律拒绝复用。
        return False


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
