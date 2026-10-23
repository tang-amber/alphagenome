"""AlphaGenome Web 控制台 —— 后端服务。

AlphaGenome 官方 SDK 是基于 gRPC 的 Python 客户端（连接到
``dns:///gdmscience.googleapis.com:443``），浏览器无法直接调用，因此本服务
在本地把 SDK 包装成 HTTP 接口，供同目录下的前端页面使用。

启动::

    python server.py            # 默认 http://127.0.0.1:8000
    python server.py --port 9000

API Key 只保存在本进程内存中（可选持久化到本地 ``.apikey`` 文件），
永远不会写进日志，也不会明文回传给前端。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
KEY_FILE = BASE_DIR / ".apikey"

# --------------------------------------------------------------------------
# SDK 导入（可选）：未安装时服务仍可启动，前端会显示安装指引。
# --------------------------------------------------------------------------

SDK_IMPORT_ERROR: Optional[str] = None
genome = dna_client = variant_scorers = ontology = None

try:  # pragma: no cover - 取决于运行环境
    from alphagenome.data import genome as _genome
    from alphagenome.models.v1 import dna_client as _dna_client

    genome = _genome
    dna_client = _dna_client

    try:
        from alphagenome.models.v1 import variant_scorers as _vs
    except Exception:  # 旧版本路径
        from alphagenome.models import variant_scorers as _vs  # type: ignore
    variant_scorers = _vs

    try:
        from alphagenome.data import ontology as _onto

        ontology = _onto
    except Exception:
        ontology = None
except Exception:  # pragma: no cover
    SDK_IMPORT_ERROR = "AlphaGenome SDK 不可用，请检查本地安装"

SDK_AVAILABLE = dna_client is not None
MAX_TRACKS = 12
# SDK 0.10.0 的长度；正常运行始终使用安装版本的常量。
SDK_LENGTHS = dict(dna_client.SUPPORTED_SEQUENCE_LENGTHS) if SDK_AVAILABLE else {
    "SEQUENCE_LENGTH_16KB": 16384, "SEQUENCE_LENGTH_100KB": 131072,
    "SEQUENCE_LENGTH_500KB": 524288, "SEQUENCE_LENGTH_1MB": 1048576,
}


def require_sdk() -> None:
    if not SDK_AVAILABLE:
        raise HTTPException(503, "AlphaGenome SDK 不可用，请检查本地安装")


def sdk_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("alphagenome")
    except Exception:
        return None


# --------------------------------------------------------------------------
# OutputType -> Output 对象属性名
# --------------------------------------------------------------------------

OUTPUT_ATTR = {
    "RNA_SEQ": "rna_seq",
    "CAGE": "cage",
    "PROCAP": "procap",
    "DNASE": "dnase",
    "ATAC": "atac",
    "CHIP_HISTONE": "chip_histone",
    "CHIP_TF": "chip_tf",
    "SPLICE_SITES": "splice_sites",
    "SPLICE_SITE_USAGE": "splice_site_usage",
    "SPLICE_JUNCTIONS": "splice_junctions",
    "CONTACT_MAPS": "contact_maps",
}


def output_type_names() -> list[str]:
    """返回 SDK 支持的全部 OutputType 名称。"""
    if not SDK_AVAILABLE:
        return list(OUTPUT_ATTR)
    try:
        return [o.name for o in dna_client.OutputType]
    except Exception:
        return list(OUTPUT_ATTR)


def resolve_output_type(name: str):
    """把字符串转成 SDK 的 OutputType 枚举成员。"""
    name = name.strip().upper()
    if name not in output_type_names():
        raise HTTPException(400, "未知的 output type")
    require_sdk()
    return dna_client.OutputType[name]


def resolve_attr(output_obj: Any, type_name: str) -> Any:
    """从 Output 对象里取出对应模态的预测结果。"""
    attr = OUTPUT_ATTR.get(type_name.upper(), type_name.lower())
    val = getattr(output_obj, attr, None)
    if val is not None:
        return val
    # 回退：在 dataclass 字段里做模糊匹配
    if dataclasses.is_dataclass(output_obj):
        target = attr.replace("_", "")
        for f in dataclasses.fields(output_obj):
            if f.name.replace("_", "") == target:
                return getattr(output_obj, f.name)
    return None


# --------------------------------------------------------------------------
# API Key 管理
# --------------------------------------------------------------------------

_key_lock = threading.Lock()
_api_key: Optional[str] = os.environ.get("ALPHA_GENOME_API_KEY") or None
_client_cache: dict[str, Any] = {}


def _load_persisted_key() -> None:
    global _api_key
    if _api_key:
        return
    try:
        if KEY_FILE.exists():
            stored = KEY_FILE.read_text(encoding="utf-8").strip()
            if stored:
                _api_key = stored
    except Exception:
        pass


# 导入模块不读磁盘密钥；脚本启动才加载用户曾明确持久化的明文文件。


def key_preview(key: Optional[str]) -> Optional[str]:
    return "已配置" if key else None


def _close_client(client: Any) -> None:
    """0.10.0 无公开 close；已核实其持有可关闭的 grpc.Channel。"""
    channel = getattr(client, "_channel", None)
    if channel is not None:
        try:
            channel.close()
        except Exception:
            pass  # 不记录可能含认证信息的异常。


def _clear_clients() -> None:
    for client in _client_cache.values():
        _close_client(client)
    _client_cache.clear()


def get_client(api_key: Optional[str] = None, timeout: float = 60.0):
    """timeout 仅为 SDK 建连握手期限，不是 metadata/预测 RPC deadline。"""
    require_sdk()
    with _key_lock:
        key = api_key or _api_key
        if not key:
            raise HTTPException(401, "尚未配置 API Key，请先在页面顶部填写并保存。")
        digest = hashlib.sha256(key.encode()).hexdigest()
        client = _client_cache.get(digest)
        if client is None:
            client = run_guarded(dna_client.create, key, timeout=timeout)
            _client_cache[digest] = client
        return client


# --------------------------------------------------------------------------
# 序列化辅助：把 numpy 大数组降采样成前端可绘制的 JSON
# --------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """保持浮点精度，将所有非有限值转成 JSON null。"""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def downsample_1d(values: np.ndarray, max_points: int, agg: str = "mean") -> np.ndarray:
    """真实分组聚合；末组不补零，保留原始浮点精度。"""
    if max_points < 1 or agg not in ("mean", "max", "sum"):
        raise ValueError("Invalid downsampling parameters")
    n = values.shape[0]
    if n <= max_points:
        return values
    factor = (n + max_points - 1) // max_points
    starts = np.arange(0, n, factor)
    if agg == "max":
        return np.maximum.reduceat(values, starts, axis=0)
    with np.errstate(invalid="ignore", over="ignore"):
        sums = np.add.reduceat(values, starts, axis=0, dtype=np.float64)
        if agg == "sum":
            return sums
        counts = np.minimum(factor, n - starts).reshape((-1,) + (1,) * (values.ndim - 1))
        return sums / counts


def track_labels(metadata: Any) -> list[str]:
    """从 TrackData.metadata 生成可读的轨道标签。"""
    if metadata is None:
        return []
    try:
        df = metadata
        cols = list(getattr(df, "columns", []))
        names = (
            list(df["name"].astype(str))
            if "name" in cols
            else [f"track_{i}" for i in range(len(df))]
        )
        strands = list(df["strand"].astype(str)) if "strand" in cols else ["."] * len(names)
        labels = []
        for n, s in zip(names, strands):
            labels.append(n if s in (".", "", "nan", "None") else f"{n} [{s}]")
        return labels
    except Exception:
        return []


def serialize_trackdata(tdata: Any, max_points: int = 1500, agg: str = "mean") -> dict[str, Any]:
    """把 TrackData 转成前端绘图用的紧凑 JSON。

    SDK 的不变式是 ``values.shape[-1] == len(metadata)``，即**最后一轴恒为轨道数**。
    因此：

    * ndim == 1 → 无位置轴，每条轨道一个标量（``kind="scalars"``）
    * ndim == 2 → 常规轨迹 ``(位置, 轨道)``（``kind="tracks"``）
    * ndim >= 3 → contact map 等矩阵输出（``kind="matrix"``）
    """
    if tdata is None:
        return {"available": False, "reason": "SDK 未返回该模态"}

    values = np.asarray(getattr(tdata, "values", None))
    if values.size == 0:
        return {"available": False, "reason": "预测结果为空"}

    metadata = getattr(tdata, "metadata", None)
    if values.ndim >= 3:
        return serialize_matrix(tdata, max_side=max(64, min(512, max_points // 3)))

    n_tracks = int(values.shape[-1])
    shown = min(n_tracks, MAX_TRACKS)
    labels = preview_labels(metadata, shown)
    preview = {"total_tracks": n_tracks, "shown_tracks": shown,
               "tracks_truncated": n_tracks > shown}
    values = values[..., :shown]
    if values.ndim == 1:
        return {
            "available": True,
            "kind": "scalars",
            "shape": [n_tracks],
            **preview,
            "labels": labels,
            "values": json_safe(values),
        }

    interval = getattr(tdata, "interval", None)
    resolution = int(getattr(tdata, "resolution", 1) or 1)
    n_positions = int(values.shape[0])

    down = downsample_1d(values, max_points, agg)
    n_returned = int(down.shape[0])
    step = max(1, (n_positions + max_points - 1) // max_points)

    start = int(getattr(interval, "start", 0) or 0)
    x = [start + i * step * resolution for i in range(n_returned)]

    result: dict[str, Any] = {
        "available": True,
        "kind": "tracks",
        "shape": [n_positions, n_tracks],
        "returned_points": n_returned,
        "resolution_bp": resolution,
        "aggregation": agg,
        "downsample_factor": step,
        "chromosome": getattr(interval, "chromosome", None),
        "start": start,
        "end": int(getattr(interval, "end", 0) or 0),
        "labels": labels,
        **preview,
        "x": x,
        "series": json_safe(down.T),
    }

    # 仅预览所选轨道的元数据。
    try:
        if metadata is not None:
            result["metadata"] = json_safe(metadata.iloc[:shown].to_dict(orient="records"))
    except Exception:
        pass

    uns = getattr(tdata, "uns", None)
    if uns:
        try:
            result["uns_keys"] = [str(k) for k in uns.keys()]
        except Exception:
            pass
    return result


def preview_labels(metadata: Any, shown: int) -> list[str]:
    labels = track_labels(metadata.iloc[:shown] if metadata is not None else None)
    return [labels[i] if i < len(labels) else f"track_{i}" for i in range(shown)]


def interval_fields(interval: Any) -> dict[str, Any]:
    return {"chromosome": getattr(interval, "chromosome", None),
            "start": getattr(interval, "start", None),
            "end": getattr(interval, "end", None)}


def serialize_junctions(obj: Any) -> dict[str, Any]:
    """SDK JunctionData: 每个 junction 一行，坐标为 0-based 半开。"""
    values = obj.values
    total, n_tracks = values.shape
    shown = min(n_tracks, MAX_TRACKS)
    track_columns = [f"track_{i}" for i in range(shown)]
    rows = []
    for i, junction in enumerate(obj.junctions[:500]):
        row = {**interval_fields(junction), "strand": junction.strand}
        row.update(zip(track_columns, values[i, :shown]))
        rows.append(row)
    return json_safe({
        "available": True, "kind": "junctions",
        "columns": ["chromosome", "start", "end", "strand"] + track_columns,
        "rows": rows, "total_rows": total, "truncated": total > len(rows),
        "labels": preview_labels(obj.metadata, shown),
        "total_tracks": n_tracks, "shown_tracks": shown,
        "tracks_truncated": n_tracks > shown,
        "coordinate_system": "0-based-half-open", **interval_fields(obj.interval),
    })


def serialize_matrix(obj: Any, max_side: int = 256) -> dict[str, Any]:
    """先选第一轨道，再聚合；不复制全部轨道，也不补零。"""
    values = np.asarray(getattr(obj, "values", obj))
    if values.size == 0:
        return {"available": False, "reason": "预测结果为空"}
    if values.ndim not in (2, 3):
        raise ValueError("Unsupported matrix shape")
    n_tracks = int(values.shape[-1]) if values.ndim == 3 else 1
    out = values[..., 0] if values.ndim == 3 else values
    shape = list(out.shape)
    out = downsample_1d(out, max_side)
    out = downsample_1d(out.T, max_side).T
    label = preview_labels(getattr(obj, "metadata", None), 1)[0]
    return {
        "available": True, "kind": "matrix", "shape": shape,
        "returned_points": int(out.shape[0]), "n_tracks": n_tracks,
        "track_index": 0, "selected_track_label": label,
        "total_tracks": n_tracks, "shown_tracks": 1,
        "tracks_truncated": n_tracks > 1,
        "values": json_safe(out),
    }


def serialize_prediction(output_obj: Any, requested: list[str], max_points: int, agg: str) -> dict[str, Any]:
    """把 Output 对象按请求的模态逐个序列化。"""
    modalities: dict[str, Any] = {}
    for name in requested:
        obj = resolve_attr(output_obj, name)
        if obj is None:
            modalities[name] = {"available": False, "reason": "SDK 未返回该模态"}
            continue
        try:
            modalities[name] = (serialize_junctions(obj) if name.upper() == "SPLICE_JUNCTIONS"
                                else serialize_trackdata(obj, max_points=max_points, agg=agg))
        except Exception:
            modalities[name] = {"available": False, "reason": "预测结果序列化失败"}
    return modalities


# --------------------------------------------------------------------------
# 请求 / 响应模型
# --------------------------------------------------------------------------


class SafeRequest(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")


class KeyRequest(SafeRequest):
    api_key: str = Field(..., repr=False, description="AlphaGenome API Key")
    persist: bool = Field(False, description="是否明文写入本地 .apikey（不是加密存储）")


class CommonParams(SafeRequest):
    """各预测接口共用的参数。"""

    outputs: list[str] = Field(default_factory=lambda: ["RNA_SEQ"])
    ontology_terms: list[str] = Field(default_factory=list)
    organism: str = Field("HOMO_SAPIENS", description="HOMO_SAPIENS 或 MUS_MUSCULUS")
    max_points: int = Field(1500, ge=50, le=20000)
    aggregation: str = Field("mean", pattern="^(mean|max|sum)$")


class IntervalRequest(CommonParams):
    chromosome: str = Field(..., examples=["chr22"])
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)


class SequenceRequest(CommonParams):
    sequence: str = Field(..., min_length=1)
    interval_width: int = 1048576
    pad_to_length: bool = False


class VariantRequest(CommonParams):
    chromosome: str = Field(..., examples=["chr22"])
    position: int = Field(..., ge=1, description="1-based 位置")
    reference_bases: str = Field(..., min_length=1)
    alternate_bases: str = Field(..., min_length=1)
    interval_width: Optional[int] = Field(None, description="留空则用 SEQUENCE_LENGTH_1MB")


class ScoreRequest(SafeRequest):
    chromosome: str
    position: int = Field(..., ge=1)
    reference_bases: str = Field(..., min_length=1)
    alternate_bases: str = Field(..., min_length=1)
    scorers: list[str] = Field(default_factory=lambda: ["RNA_SEQ"])
    organism: str = Field("HOMO_SAPIENS")
    interval_width: Optional[int] = None
    max_rows: int = Field(500, ge=1, le=20000)


def resolve_organism(name: Optional[str]):
    """本地校验先于 SDK / Key / 网络。"""
    normalized = (name or "").strip().upper()
    if normalized not in ("HOMO_SAPIENS", "MUS_MUSCULUS"):
        raise HTTPException(400, "organism 必须为 HOMO_SAPIENS 或 MUS_MUSCULUS")
    return dna_client.Organism[normalized] if SDK_AVAILABLE else None


def common_kwargs(req: CommonParams, types: Optional[list[Any]] = None) -> dict[str, Any]:
    names = [name.strip().upper() for name in req.outputs]
    if not 1 <= len(names) <= 3:
        raise HTTPException(400, "必须选择 1-3 个输出模态")
    if any(name not in output_type_names() for name in names):
        raise HTTPException(400, "未知的 output type")
    terms = [term.strip() for term in req.ontology_terms]
    if len(terms) > 8 or (not terms and names != ["SPLICE_SITES"]):
        raise HTTPException(400, "必须指定 1-8 个 ontology terms（仅 SPLICE_SITES 可不指定）")
    for term in terms:
        if not re.fullmatch(r"(?:CLO|UBERON|CL|EFO|NTR):[0-9]+", term):
            raise HTTPException(400, "ontology term 必须是受支持的 CURIE")
        try:
            if ontology is not None:
                ontology.from_curie(term).to_proto()
        except Exception:
            raise HTTPException(400, "ontology term 不合法") from None
    org = resolve_organism(req.organism)
    return {"requested_outputs": types if types is not None else [resolve_output_type(n) for n in names],
            "ontology_terms": terms or None, "organism": org}


def build_variant(req: VariantRequest | ScoreRequest):
    """构造 genome.Variant，并做本地校验以便给出清晰的错误信息。"""
    chrom = (req.chromosome or "").strip()
    ref = (req.reference_bases or "").strip().upper()
    alt = (req.alternate_bases or "").strip().upper()

    if not chrom:
        raise HTTPException(400, "chromosome 不能为空（例如 chr22）")
    if int(req.position) < 1:
        raise HTTPException(400, "position 是 1-based，必须 >= 1")
    if not ref or not alt:
        raise HTTPException(400, "reference_bases 与 alternate_bases 都不能为空")
    if not set(ref + alt) <= set("ACGTN"):
        raise HTTPException(400, "碱基只能是 A/C/G/T/N")
    if ref == alt:
        raise HTTPException(400, "REF 与 ALT 不能相同")
    require_sdk()
    try:
        return genome.Variant(
            chromosome=chrom, position=int(req.position),
            reference_bases=ref, alternate_bases=alt,
        )
    except Exception:
        raise HTTPException(400, "变异参数不合法") from None


def build_interval(req: VariantRequest | ScoreRequest, variant: Any) -> Any:
    """由变异推导输入区间；未指定宽度时用 1MB，并校验是否为受支持的长度。"""
    width = req.interval_width
    if width is None:
        width = getattr(dna_client, "SEQUENCE_LENGTH_1MB", 2**20)
    width = int(width)

    validate_width(width)
    try:
        return variant.reference_interval.resize(width)
    except Exception:
        raise HTTPException(400, "区间构造失败") from None


def validate_width(width: int) -> None:
    allowed = set(supported_lengths()["supported"].values())
    if width not in allowed:
        raise HTTPException(400, f"序列/区间宽度必须为 {sorted(allowed)} 之一")


def supported_lengths() -> dict[str, Any]:
    sup = dna_client.SUPPORTED_SEQUENCE_LENGTHS if SDK_AVAILABLE else SDK_LENGTHS
    return {"supported": {str(k): int(v) for k, v in sup.items()}, "default_1mb": 1048576}


def run_guarded(fn, *args, **kwargs):
    """统一捕获 SDK 异常，把 gRPC 错误转成可读的 HTTP 错误。"""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        # 只用于分类，绝不将上游文本/认证元数据写入响应或日志。
        lowered = str(exc).lower()
        if any(s in lowered for s in ("unauthenticated", "invalid api key", "permission_denied")):
            raise HTTPException(401, "API Key 无效或无权限") from None
        if any(s in lowered for s in ("unavailable", "failed to connect", "deadline")) or isinstance(exc, TimeoutError):
            raise HTTPException(502, "无法连接 AlphaGenome 服务或服务请求超时") from None
        if "resource_exhausted" in lowered or "quota" in lowered:
            raise HTTPException(429, "服务配额已用尽") from None
        if "invalidargument" in lowered or "invalid argument" in lowered:
            raise HTTPException(400, "参数被模型服务拒绝") from None
        raise HTTPException(500, "模型调用失败，请检查服务状态") from None


# --------------------------------------------------------------------------
# FastAPI 应用
# --------------------------------------------------------------------------

app = FastAPI(title="AlphaGenome Web Console", version="1.0.0")


def _local_origin(value: str) -> Optional[tuple[str, str, int]]:
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1")
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment):
            return None
        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None


@app.middleware("http")
async def local_requests_only(request: Request, call_next):
    hosts = request.headers.getlist("host")
    target = _local_origin(f"{request.url.scheme}://{hosts[0]}") if len(hosts) == 1 else None
    if target is None:
        return JSONResponse(status_code=400, content={"detail": "仅允许 localhost / 127.0.0.1 Host"})
    origins = request.headers.getlist("origin")
    if (origins and (len(origins) != 1 or _local_origin(origins[0]) != target)
            or request.headers.get("sec-fetch-site") == "cross-site"):
        return JSONResponse(status_code=403, content={"detail": "仅允许本地页面同源请求"})
    try:
        return await call_next(request)
    except Exception:
        # 在 ASGI ServerErrorMiddleware 之前消化异常，避免服务日志打印 traceback。
        return JSONResponse(status_code=500, content={"detail": "服务内部错误"})


@app.exception_handler(RequestValidationError)
async def invalid_request_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": "请求参数不合法，请检查字段类型与范围"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    """服务与 SDK 状态，前端启动时调用。"""
    return {
        "service": "ok",
        "sdk_available": SDK_AVAILABLE,
        "sdk_import_error": SDK_IMPORT_ERROR,
        "sdk_version": sdk_version(),
        "api_key_configured": bool(_api_key),
        "api_key_preview": key_preview(_api_key),
        "key_source": "environment" if os.environ.get("ALPHA_GENOME_API_KEY") else "manual",
        "output_types": output_type_names(),
        "sequence_lengths": supported_lengths(),
        "scorers": list_scorer_names(),
    }


def list_scorer_names() -> list[str]:
    """可用的推荐变异打分器名称。

    注意：SDK 的 ``RECOMMENDED_VARIANT_SCORERS`` 是 ``immutabledict``，它并不
    继承 ``dict``，因此这里用鸭子类型判断而不是 isinstance。
    """
    if variant_scorers is None:
        return list(OUTPUT_ATTR)

    table = getattr(variant_scorers, "RECOMMENDED_VARIANT_SCORERS", None)
    if _is_mapping(table):
        return [str(k) for k in table.keys()]

    fn = getattr(variant_scorers, "get_recommended_scorers", None)
    if callable(fn):
        # get_recommended_scorers 需要传入 protobuf 的 organism 枚举
        for org_name in ("HOMO_SAPIENS",):
            try:
                res = fn(dna_client.Organism[org_name])
            except Exception:
                continue
            if _is_mapping(res):
                return [str(k) for k in res.keys()]
            if isinstance(res, (list, tuple)):
                return [str(getattr(r, "requested_output", r)) for r in res]
    return list(OUTPUT_ATTR)


def _is_mapping(obj: Any) -> bool:
    """判断对象是否可按 key 取值（兼容 immutabledict）。"""
    return obj is not None and hasattr(obj, "keys") and hasattr(obj, "__getitem__") and len(obj) > 0


def get_scorer(name: str):
    if variant_scorers is None:
        raise HTTPException(503, "variant_scorers 模块不可用")

    key = name.strip().upper()

    table = getattr(variant_scorers, "RECOMMENDED_VARIANT_SCORERS", None)
    if _is_mapping(table):
        for k in table.keys():
            if str(k).upper() == key:
                return table[k]
        raise HTTPException(
            400, f"未知的打分器: {name}。可选值: {', '.join(list_scorer_names())}"
        )

    fn = getattr(variant_scorers, "get_recommended_scorers", None)
    if callable(fn):
        try:
            res = fn(dna_client.Organism["HOMO_SAPIENS"])
        except Exception:
            res = None
        if _is_mapping(res):
            for k in res.keys():
                if str(k).upper() == key:
                    return res[k]
        elif isinstance(res, (list, tuple)):
            for v in res:
                if str(getattr(v, "requested_output", "")).upper() == key:
                    return v

    raise HTTPException(400, f"未知的打分器: {name}。可选值: {', '.join(list_scorer_names())}")


@app.post("/api/key")
def set_key(req: KeyRequest) -> dict[str, Any]:
    """默认仅内存；可选文件持久化是明文，不是 Windows 加密存储。"""
    global _api_key
    key = req.api_key.strip()
    if not key:
        raise HTTPException(400, "API Key 不能为空")
    client = None
    try:
        if SDK_AVAILABLE:
            # timeout 仅握手；output_metadata 无公开 RPC deadline。
            client = run_guarded(dna_client.create, key, timeout=30.0)
            run_guarded(client.output_metadata)
        with _key_lock:
            try:
                if req.persist:
                    KEY_FILE.write_text(key, encoding="utf-8")
                else:
                    KEY_FILE.unlink(missing_ok=True)
            except OSError:
                raise HTTPException(500, "密钥文件写入或删除失败，未更新内存密钥") from None
            _clear_clients()
            _api_key = key
            if client is not None:
                _client_cache[hashlib.sha256(key.encode()).hexdigest()] = client
    except Exception:
        if client is not None:
            _close_client(client)
        raise
    return {"ok": True, "api_key_preview": key_preview(key),
            "persisted": req.persist, "validated": SDK_AVAILABLE,
            "storage": "plaintext_file" if req.persist else "memory",
            "storage_warning": "本地 .apikey 为明文，未加密" if req.persist else None}


@app.delete("/api/key")
def clear_key() -> dict[str, Any]:
    global _api_key
    with _key_lock:
        _api_key = None
        _clear_clients()
        try:
            KEY_FILE.unlink(missing_ok=True)
        except OSError:
            raise HTTPException(500, "内存密钥已清除，但 .apikey 删除失败；磁盘密钥仍可能存在") from None
    return {"ok": True, "api_key_configured": False}


@app.get("/api/metadata")
def metadata(organism: str = "HOMO_SAPIENS") -> dict[str, Any]:
    """拉取模型元数据：各模态的轨道数、可用的本体术语（组织/细胞类型）。"""
    org_enum = resolve_organism(organism)
    client = get_client()

    def _fetch():
        return client.output_metadata() if org_enum is None else client.output_metadata(org_enum)

    meta = run_guarded(_fetch)

    per_type: dict[str, Any] = {}
    ontology_index: dict[str, set[str]] = {}

    for type_name in output_type_names():
        # OutputMetadata 的每个字段本身就是 pandas.DataFrame（或 None）
        df = getattr(meta, OUTPUT_ATTR.get(type_name, type_name.lower()), None)
        if df is None or not hasattr(df, "columns"):
            continue
        try:
            n_tracks = int(len(df))
        except Exception:
            n_tracks = 0
        info: dict[str, Any] = {"n_tracks": n_tracks}
        try:
            cols = [str(c) for c in df.columns]
            info["columns"] = cols
            if "ontology_curie" in cols:
                info["ontology_terms"] = df["ontology_curie"].astype(str).unique().tolist()
                label_col = next(
                    (c for c in ("biosample_name", "name", "gtex_tissue", "Assay title") if c in cols),
                    None,
                )
                if label_col:
                    pairs = (
                        df[["ontology_curie", label_col]].astype(str).drop_duplicates().values.tolist()
                    )
                    for curie, label in pairs:
                        ontology_index.setdefault(curie, set()).add(label)
        except Exception:
            pass
        per_type[type_name] = info

    terms = []
    for curie in sorted(ontology_index):
        labels = sorted(ontology_index[curie])
        terms.append({"curie": curie, "label": labels[0], "n_labels": len(labels), "aliases": labels[:6]})

    return {
        "organism": organism.strip().upper(),
        "output_types": output_type_names(),
        "per_type": per_type,
        "ontology_terms": terms,
        "sequence_lengths": supported_lengths(),
        "scorers": list_scorer_names(),
    }


@app.post("/api/predict/interval")
def predict_interval(req: IntervalRequest) -> dict[str, Any]:
    """对参考基因组区间做预测（DnaClient.predict_interval）。"""
    # 先做本地参数校验，再要求 API Key，避免用错误提示掩盖真正的参数问题
    if req.end <= req.start:
        raise HTTPException(400, "end 必须大于 start（区间为 0-based 半开）")
    if not req.chromosome.strip():
        raise HTTPException(400, "chromosome 不能为空")
    validate_width(req.end - req.start)
    kwargs = common_kwargs(req)
    types = kwargs["requested_outputs"]
    interval = genome.Interval(chromosome=req.chromosome.strip(), start=req.start, end=req.end)
    client = get_client()
    names = [t.name for t in types]
    output = run_guarded(client.predict_interval, interval=interval, **kwargs)
    return {
        "ok": True,
        "call": "predict_interval",
        "interval": str(interval),
        "requested_outputs": names,
        "ontology_terms": req.ontology_terms,
        "modalities": serialize_prediction(output, names, req.max_points, req.aggregation),
    }


@app.post("/api/predict/sequence")
def predict_sequence(req: SequenceRequest) -> dict[str, Any]:
    """对自定义 DNA 序列做预测（DnaClient.predict_sequence）。"""
    seq = "".join(req.sequence.split()).upper()
    if not seq or not set(seq) <= set("ACGTN"):
        raise HTTPException(400, "序列去除空白后必须非空，且只能包含 A/C/G/T/N")
    original_length = len(seq)
    validate_width(req.interval_width)
    padding_left = 0
    if req.pad_to_length:
        if len(seq) > req.interval_width:
            raise HTTPException(400, "序列长于指定填充宽度")
        padding = req.interval_width - len(seq)
        padding_left = padding // 2
        seq = "N" * padding_left + seq + "N" * (padding - padding_left)
    else:
        validate_width(len(seq))
    padded = len(seq) != original_length
    kwargs = common_kwargs(req)
    types = kwargs["requested_outputs"]
    client = get_client()
    names = [t.name for t in types]
    output = run_guarded(client.predict_sequence, sequence=seq, **kwargs)
    return {
        "ok": True,
        "call": "predict_sequence",
        "original_length": original_length,
        "input_length": len(seq),
        "padded_with_N": padded,
        "padding_left_bp": padding_left,
        "requested_outputs": names,
        "modalities": serialize_prediction(output, names, req.max_points, req.aggregation),
    }


@app.post("/api/predict/variant")
def predict_variant(req: VariantRequest) -> dict[str, Any]:
    """对变异的 REF / ALT 两条序列分别预测（DnaClient.predict_variant）。"""
    variant = build_variant(req)
    interval = build_interval(req, variant)
    kwargs = common_kwargs(req)
    types = kwargs["requested_outputs"]
    client = get_client()
    names = [t.name for t in types]
    out = run_guarded(
        client.predict_variant,
        interval=interval,
        variant=variant,
        **kwargs,
    )
    ref = getattr(out, "reference", None)
    alt = getattr(out, "alternate", None)

    return {
        "ok": True,
        "call": "predict_variant",
        "variant": str(variant),
        "interval": str(interval),
        "requested_outputs": names,
        "reference": serialize_prediction(ref, names, req.max_points, req.aggregation) if ref is not None else {},
        "alternate": serialize_prediction(alt, names, req.max_points, req.aggregation) if alt is not None else {},
        "variant_position": int(req.position),
    }


@app.post("/api/score/variant")
def score_variant(req: ScoreRequest) -> dict[str, Any]:
    """变异效应打分（DnaClient.score_variant），返回展平后的分数表。"""
    variant = build_variant(req)
    interval = build_interval(req, variant)
    if not 1 <= len(req.scorers) <= 3:
        raise HTTPException(400, "必须选择 1-3 个打分器")
    scorers = [get_scorer(s) for s in req.scorers]

    kwargs: dict[str, Any] = {"interval": interval, "variant": variant, "variant_scorers": scorers}
    org = resolve_organism(req.organism)
    if org is not None:
        kwargs["organism"] = org

    client = get_client()
    results = list(run_guarded(client.score_variant, **kwargs) or [])
    if not results:
        return {
            "ok": True,
            "call": "score_variant",
            "variant": str(variant),
            "interval": str(interval),
            "scorers_requested": req.scorers,
            "tables": [],
            "note": "服务端返回了空的打分结果列表",
        }

    # tidy_scores 接收的是 AnnData 的**序列**（不是单个对象），一次调用即得到
    # 所有打分器合并后的长表，含 variant_scorer / raw_score 列。
    tidy = None
    tidy_error = False
    try:
        tidy = variant_scorers.tidy_scores(results)
    except Exception:
        tidy_error = True

    tables: list[dict[str, Any]] = []
    if tidy_error:
        # 只在异常时回退；整个响应最多 max_rows 行，不构造完整 gene × track 表。
        remaining = req.max_rows
        for idx, ad in enumerate(results):
            table = _anndata_to_table(ad, remaining, idx, req.scorers)
            tables.append(table)
            remaining -= len(table["rows"])
    elif tidy is not None and len(tidy):
        tables.append(_tidy_to_table(tidy, req.max_rows, title="全部打分器（tidy_scores 合并）"))

    resp = {
        "ok": True,
        "call": "score_variant",
        "variant": str(variant),
        "interval": str(interval),
        "scorers_requested": req.scorers,
        "n_anndata": len(results),
        "tables": tables,
    }
    if tidy_error:
        resp["tidy_error"] = "tidy_scores 处理失败"
        resp["note"] = "tidy_scores 异常，已回退到原始 AnnData 预览；未应用 tidy 的基因过滤"
    elif not tables:
        resp["note"] = "tidy_scores 过滤后无结果，未回退到未过滤分数"
    return resp


def _df_to_records(df: Any, max_rows: int) -> dict[str, Any]:
    """DataFrame → 前端表格 JSON，剔除全空列并把 NaN/Inf 转成 null。"""
    total = int(len(df))
    slim = df.head(max_rows).copy()

    # 去掉整列皆空的字段（如非 junction 打分器时的 junction_Start/End），减少噪音
    try:
        allnull = slim.isna().all()
        drop_cols = [c for c in slim.columns if bool(allnull.get(c, False))]
        # 同时剔除字符串 "None" 占满整列的情况
        for c in list(slim.columns):
            if c in drop_cols:
                continue
            try:
                vals = slim[c].astype(str).unique()
                if len(vals) == 1 and vals[0] in ("None", "nan", "<NA>"):
                    drop_cols.append(c)
            except Exception:
                pass
        if drop_cols and len(drop_cols) < len(slim.columns):
            slim = slim.drop(columns=drop_cols)
    except Exception:
        drop_cols = []

    # pandas.to_json 默认仅 10 位小数；to_dict + 标准 JSON 保留 float64 精度。
    records = json_safe(slim.to_dict(orient="records"))

    return {
        "columns": [str(c) for c in slim.columns],
        "rows": records,
        "total_rows": total,
        "truncated": total > len(records),
        "dropped_empty_columns": [str(c) for c in drop_cols],
    }


def _tidy_to_table(tidy: Any, max_rows: int, title: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"index": 0, "scorer": title, "source": "tidy_scores"}
    try:
        entry["shape"] = [int(tidy.shape[0]), int(tidy.shape[1])]
    except Exception:
        pass
    entry.update(_df_to_records(tidy, max_rows))
    # 让 raw_score 排在前面，便于默认按它排序
    cols = entry.get("columns", [])
    if "raw_score" in cols:
        entry["columns"] = ["raw_score"] + [c for c in cols if c != "raw_score"]
    return entry


def _anndata_to_table(ad: Any, max_rows: int, idx: int, requested: list[str]) -> dict[str, Any]:
    """回退路径：直接把 AnnData 的 X / obs / var 展平成 gene × track × score。"""
    entry: dict[str, Any] = {
        "index": idx,
        "source": "anndata_fallback",
        "scorer": requested[idx] if idx < len(requested) else f"scorer_{idx}",
    }
    try:
        uns = getattr(ad, "uns", None) or {}
        sc = uns.get("variant_scorer") or uns.get("scorer")
        if sc is not None:
            entry["scorer"] = str(sc)
    except Exception:
        pass
    try:
        entry["shape"] = [int(ad.X.shape[0]), int(ad.X.shape[1])]
    except Exception:
        pass
    try:
        x = ad.X  # 保持稀疏/惰性矩阵，不执行全表 np.asarray / toarray。
        obs, var = getattr(ad, "obs", None), getattr(ad, "var", None)
        n_genes, n_tracks = x.shape
        total = int(n_genes) * int(n_tracks)
        flat: list[dict[str, Any]] = []
        for linear in range(min(total, max_rows)):
            gi, ti = divmod(linear, n_tracks)
            g = obs["gene_name"].iloc[gi] if obs is not None and "gene_name" in obs.columns else f"row_{gi}"
            t = var["name"].iloc[ti] if var is not None and "name" in var.columns else f"track_{ti}"
            flat.append({"gene_name": json_safe(g), "track_name": json_safe(t),
                         "raw_score": json_safe(x[gi, ti])})
        entry["columns"] = ["raw_score", "gene_name", "track_name"]
        entry["rows"] = flat
        entry["total_rows"] = total
        entry["truncated"] = total > len(flat)
    except Exception:
        entry["error"] = "无法解析打分结果"
        entry["columns"] = []
        entry["rows"] = []
        entry["total_rows"] = 0
        entry["truncated"] = False
    return entry


# --------------------------------------------------------------------------
# 静态前端
# --------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "前端页面缺失: static/index.html")
    return FileResponse(page)


@app.get("/favicon.ico")
def favicon() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.exception_handler(Exception)
def unhandled_exception_handler(request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": "服务内部错误"})


def main() -> None:
    _load_persisted_key()
    parser = argparse.ArgumentParser(description="AlphaGenome Web 控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    print("=" * 64)
    print("AlphaGenome Web 控制台")
    print(f"  地址      : http://{args.host}:{args.port}")
    print(f"  SDK 可用  : {SDK_AVAILABLE}" + (f" ({sdk_version()})" if SDK_AVAILABLE else ""))
    if not SDK_AVAILABLE:
        print(f"  SDK 错误  : {SDK_IMPORT_ERROR}")
        print("  安装命令  : pip install alphagenome")
    print(f"  已有 Key  : {key_preview(_api_key)}")
    print("=" * 64)

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
