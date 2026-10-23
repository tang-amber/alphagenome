"""用合成数据验证后端序列化管线（不需要真实 API Key）。

真实 SDK 返回的 TrackData / AnnData 结构与此处构造的一致，
因此可以确认降采样、轨道标签、矩阵热图、打分表展平这些环节正确。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 导入也不得使用真实环境 Key；server 导入不读取持久化文件。
with patch.dict(os.environ, {"ALPHA_GENOME_API_KEY": ""}):
    import server  # noqa: E402
from alphagenome.data import genome, junction_data, track_data  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def make_trackdata(n_pos: int, n_tracks: int, resolution: int, interval) -> track_data.TrackData:
    rng = np.random.default_rng(42)
    values = (rng.random((n_pos, n_tracks)).astype(np.float32) * 10)
    # 造一个尖峰，验证 max 聚合能捕获到它
    values[n_pos // 2, 0] = 999.0
    meta = pd.DataFrame({
        "name": [f"UBERON:0002048 track{i}" for i in range(n_tracks)],
        "strand": ["+", "-", "."][:n_tracks] if n_tracks <= 3 else ["."] * n_tracks,
        "ontology_curie": ["UBERON:0002048"] * n_tracks,
    })
    return track_data.TrackData(
        values=values, metadata=meta, resolution=resolution, interval=interval
    )


print("\n=== 1. TrackData 降采样（1MB × 3 轨道，1bp 分辨率）===")
iv = genome.Interval(chromosome="chr22", start=35677410, end=35677410 + 1048576)
td = make_trackdata(1048576, 3, 1, iv)
out = server.serialize_trackdata(td, max_points=1500, agg="mean")
check("标记为可用", out["available"] is True)
check("原始形状正确", out["shape"] == [1048576, 3], str(out["shape"]))
check("降采样到 <=1500 点", out["returned_points"] <= 1500, str(out["returned_points"]))
check("series 轨道数 = 3", len(out["series"]) == 3, str(len(out["series"])))
check("每条 series 长度一致", all(len(s) == out["returned_points"] for s in out["series"]))
check("轨道标签带链信息", out["labels"][0].endswith("[+]"), out["labels"][0])
check("坐标起点正确", out["start"] == 35677410, str(out["start"]))
check("x 轴单调递增", all(b > a for a, b in zip(out["x"], out["x"][1:])))
payload = len(json.dumps(out))
check("JSON 体积可控 (<1.5MB)", payload < 1_500_000, f"{payload/1024:.0f} KB")
check("mean 聚合下尖峰被平均掉", max(out["series"][0]) < 999, f"max={max(out['series'][0]):.3f}")

out_max = server.serialize_trackdata(td, max_points=1500, agg="max")
check("max 聚合能保留尖峰", max(out_max["series"][0]) == 999.0, f"max={max(out_max['series'][0])}")
check("max 聚合记录了聚合方式", out_max["aggregation"] == "max")

print("\n=== 2. 小数据不被降采样 ===")
iv_small = genome.Interval(chromosome="chr1", start=1000, end=1512)
td_small = make_trackdata(4, 3, 128, iv_small)
o2 = server.serialize_trackdata(td_small, max_points=1500)
check("原样返回 4 个点", o2["returned_points"] == 4, str(o2["returned_points"]))
check("分辨率记录为 128bp", o2["resolution_bp"] == 128, str(o2["resolution_bp"]))

print("\n=== 3. 无位置轴数据（ndim=1 → 每轨道一个标量）===")
td_1d = track_data.TrackData(
    values=np.arange(10, dtype=np.float32),
    metadata=pd.DataFrame({
        "name": [f"scorer_t{i}" for i in range(10)],
        "strand": ["."] * 10,
    }),
)
o3 = server.serialize_trackdata(td_1d, max_points=1500)
check("识别为 scalars", o3["kind"] == "scalars", o3["kind"])
check("shape = [10] 个轨道", o3["shape"] == [10], str(o3["shape"]))
check("数值保真", o3["values"][:3] == [0.0, 1.0, 2.0], str(o3["values"][:3]))
check("标签数量匹配", len(o3["labels"]) == 10, str(len(o3["labels"])))

print("\n=== 3b. 3D 数据（contact map）自动走矩阵分支 ===")
td_3d = track_data.TrackData(
    values=np.random.default_rng(3).random((64, 64, 2)).astype(np.float32),
    metadata=pd.DataFrame({"name": ["cm0", "cm1"], "strand": [".", "."]}),
    resolution=2048,
    interval=genome.Interval(chromosome="chr1", start=0, end=64 * 2048),
)
o3b = server.serialize_trackdata(td_3d, max_points=1500)
check("识别为 matrix", o3b["kind"] == "matrix", o3b["kind"])
check("记录了轨道数 2", o3b["n_tracks"] == 2, str(o3b.get("n_tracks")))
check("矩阵为正方形 64×64", len(o3b["values"]) == 64 and len(o3b["values"][0]) == 64)

print("\n=== 4. contact map 矩阵序列化 ===")
mat = np.random.default_rng(1).random((512, 512, 1)).astype(np.float32)


class _FakeContact:
    values = mat


o4 = server.serialize_matrix(_FakeContact(), max_side=128)
check("kind = matrix", o4["kind"] == "matrix")
check("原始形状 512×512", o4["shape"] == [512, 512], str(o4["shape"]))
check("降采样到 <=128", o4["returned_points"] <= 128, str(o4["returned_points"]))
check("矩阵为正方形", len(o4["values"]) == len(o4["values"][0]))

print("\n=== 5. Output 对象按模态分发 ===")


class _FakeOutput:
    def __init__(self):
        self.rna_seq = td
        self.dnase = make_trackdata(1048576, 1, 1, iv)
        self.contact_maps = _FakeContact()
        self.cage = None


mods = server.serialize_prediction(_FakeOutput(), ["RNA_SEQ", "DNASE", "CONTACT_MAPS", "CAGE"], 800, "mean")
check("返回全部 4 个请求模态", set(mods) == {"RNA_SEQ", "DNASE", "CONTACT_MAPS", "CAGE"}, str(list(mods)))
check("RNA_SEQ 为轨迹图", mods["RNA_SEQ"]["kind"] == "tracks")
check("CONTACT_MAPS 为热图", mods["CONTACT_MAPS"]["kind"] == "matrix")
check("CAGE 缺失被优雅处理", mods["CAGE"]["available"] is False, str(mods["CAGE"]))
check("未知模态不抛异常", "available" in server.serialize_prediction(_FakeOutput(), ["ATAC"], 100, "mean")["ATAC"])

print("\n=== 6. AnnData 打分结果展平 ===")
import anndata  # noqa: E402

REC = server.variant_scorers.RECOMMENDED_VARIANT_SCORERS
var = genome.Variant(chromosome="chr22", position=36201698, reference_bases="A", alternate_bases="C")


def mk_anndata(genes, ntracks, scorer):
    X = np.random.default_rng(0).random((len(genes), ntracks)).astype(np.float32)
    X[0, 0] = np.nan  # 验证 NaN 被清洗成 null
    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame({
            "gene_id": [f"ENSG{i}" for i in range(len(genes))],
            "strand": ["-"] * len(genes),
            "gene_name": genes,
            "gene_type": ["protein_coding"] * len(genes),
        }),
        var=pd.DataFrame({
            "name": [f"UBERON:0001157 t{i}" for i in range(ntracks)],
            "strand": ["-"] * ntracks,
            "ontology_curie": ["UBERON:0001157"] * ntracks,
        }),
        uns={"interval": iv, "variant": var, "variant_scorer": scorer},
    )


genes = ["RBFOX2", "APOL4", "APOL1"]
a1 = mk_anndata(genes, 4, REC["RNA_SEQ"])
a2 = mk_anndata(genes, 4, REC["RNA_SEQ_ACTIVE"])

tidy = server.variant_scorers.tidy_scores([a1, a2])
check("tidy_scores 接收序列并合并", len(tidy) == 24, str(tidy.shape))
check("含 raw_score 列", "raw_score" in tidy.columns, str(list(tidy.columns)))
check("含 variant_scorer 列", "variant_scorer" in tidy.columns)

tbl = server._tidy_to_table(tidy, max_rows=500, title="test")
check("表格列已生成", len(tbl["columns"]) > 0, str(len(tbl["columns"])))
check("raw_score 被排到首列", tbl["columns"][0] == "raw_score", tbl["columns"][0])
check("行数 = 24", tbl["total_rows"] == 24, str(tbl["total_rows"]))
check("未截断", tbl["truncated"] is False)
check("全空列被剔除", len(tbl["dropped_empty_columns"]) >= 2, str(tbl["dropped_empty_columns"]))
blob = json.dumps(tbl["rows"])
check("NaN 已清洗为 null（JSON 无 NaN 字面量）", "NaN" not in blob)
check("可被 json.loads 还原", json.loads(blob) is not None)

check("max_rows 截断生效", server._tidy_to_table(tidy, max_rows=5, title="t")["truncated"] is True)
check("截断后行数 = 5", len(server._tidy_to_table(tidy, max_rows=5, title="t")["rows"]) == 5)

print("\n=== 6b. 非基因型打分器（CenterMaskScorer，无 obs）===")
a3 = anndata.AnnData(
    X=np.random.default_rng(1).random((1, 5)).astype(np.float32),
    var=pd.DataFrame({"name": [f"ATAC t{i}" for i in range(5)], "strand": ["."] * 5}),
    uns={"interval": iv, "variant": var, "variant_scorer": REC["ATAC"]},
)
tidy3 = server.variant_scorers.tidy_scores([a3])
t3 = server._tidy_to_table(tidy3, 500, "atac")
check("1×5 → 5 行", t3["total_rows"] == 5, str(t3["total_rows"]))
check("gene 相关列被剔除", "gene_name" not in t3["columns"], str(t3["columns"]))

print("\n=== 6c. tidy_scores 失败时的 AnnData 回退路径 ===")
fb = server._anndata_to_table(a1, max_rows=500, idx=0, requested=["RNA_SEQ"])
check("回退表生成成功", "error" not in fb, str(fb.get("error")))
check("回退行数 = 3×4", fb["total_rows"] == 12, str(fb["total_rows"]))
check("scorer 取自 uns", "GeneMaskLFCScorer" in fb["scorer"], fb["scorer"])
check("列顺序为 score 优先", fb["columns"][0] == "raw_score", str(fb["columns"]))

bad = server._anndata_to_table(object(), max_rows=10, idx=0, requested=["X"])
check("无法解析时返回 error 而非抛异常", "error" in bad, str(bad.get("error"))[:70])

print("\n=== 7. 区间宽度校验 ===")
var = genome.Variant(chromosome="chr22", position=36201698, reference_bases="A", alternate_bases="C")


class _Req:
    interval_width = None


check("默认宽度 = 1MB",
      server.build_interval(_Req(), var).width == 1048576, str(server.build_interval(_Req(), var).width))


class _ReqBad:
    interval_width = 99999


try:
    server.build_interval(_ReqBad(), var)
    check("非法宽度被拒绝", False, "没有抛出异常")
except Exception as e:
    check("非法宽度被拒绝", getattr(e, "status_code", None) == 400, str(e))


class _ReqOk:
    interval_width = 16384


check("16KB 宽度被接受", server.build_interval(_ReqOk(), var).width == 16384)

print("\n=== 8. 变异/区间构造与字符串表示 ===")


def _vreq(**kw):
    base = {"chromosome": "chr3", "position": 10000,
            "reference_bases": "A", "alternate_bases": "C"}
    base.update(kw)
    return type("R", (), base)()


v2 = server.build_variant(_vreq(reference_bases="a", alternate_bases="cgtcaat"))
check("碱基自动大写", "CGTCAAT" in str(v2), str(v2))
check("reference_interval 可推导", v2.reference_interval.width > 0, str(v2.reference_interval))

for label, kwargs in [
    ("空 reference_bases", {"reference_bases": ""}),
    ("空 alternate_bases", {"alternate_bases": ""}),
    ("position < 1", {"position": 0}),
    ("空 chromosome", {"chromosome": "  "}),
    ("非法碱基字符", {"alternate_bases": "CXG"}),
]:
    try:
        server.build_variant(_vreq(**kwargs))
        check(f"{label} 被拒绝", False, "没有抛出异常")
    except Exception as e:
        check(f"{label} 被拒绝", getattr(e, "status_code", None) == 400, str(e)[:70])

check("合法 indel 被接受",
      server.build_variant(_vreq(reference_bases="AGGGATC", alternate_bases="C")) is not None)

print("\n=== 9. 打分器名称解析 ===")
names = server.list_scorer_names()
check("识别到 19 个推荐打分器", len(names) == 19, f"{len(names)}: {names[:4]}…")
for n in ("RNA_SEQ", "rna_seq", "POLYADENYLATION", "ATAC_ACTIVE"):
    sc = server.get_scorer(n)
    check(f"解析打分器 {n}", sc is not None, type(sc).__name__)
try:
    server.get_scorer("BOGUS")
    check("非法打分器被拒绝", False, "没有抛出异常")
except Exception as e:
    check("非法打分器被拒绝", getattr(e, "status_code", None) == 400, str(e)[:60])

print("\n=== 10. OutputType 解析 ===")
check("RNA_SEQ 可解析", server.resolve_output_type("rna_seq").name == "RNA_SEQ")
check("CONTACT_MAPS 可解析", server.resolve_output_type("CONTACT_MAPS").name == "CONTACT_MAPS")
try:
    server.resolve_output_type("NOT_A_TYPE")
    check("非法 OutputType 被拒绝", False, "没有抛出异常")
except Exception as e:
    check("非法 OutputType 被拒绝", getattr(e, "status_code", None) == 400, str(e)[:60])

print("\n=== 11. 真实末组、坐标、非有限值及预览上限 ===")
negative = -np.arange(1, 11, dtype=np.float64).reshape(-1, 1)
for agg, expected in [("mean", -9.5), ("max", -9.0), ("sum", -19.0)]:
    reduced = server.downsample_1d(negative, 3, agg)
    check(f"末组 {agg} 不补零", reduced[-1, 0] == expected)
coord_td = track_data.TrackData(
    values=np.ones((7, 13), dtype=np.float64),
    metadata=pd.DataFrame({"name": [f"t{i}" for i in range(13)], "strand": ["."] * 13}),
    resolution=128, interval=genome.Interval("chr1", 100, 996),
)
coord_out = server.serialize_trackdata(coord_td, 3)
check("x 与真实 factor=3 对应", coord_out["downsample_factor"] == 3 and coord_out["x"] == [100, 484, 868])
check("轨迹仅预览12条且保留原始计数", coord_out["total_tracks"] == 13 and coord_out["shown_tracks"] == 12
      and coord_out["tracks_truncated"] and len(coord_out["series"]) == len(coord_out["labels"]) == len(coord_out["metadata"]) == 12)
finite_td = track_data.TrackData(values=np.array([[np.nan], [np.inf], [-np.inf]]),
                                metadata=pd.DataFrame({"name": ["finite"], "strand": ["."]}))
check("轨迹非有限值全为 null", server.serialize_trackdata(finite_td)["series"] == [[None, None, None]])
scalar_td = track_data.TrackData(values=np.array([np.inf] + list(range(12)), dtype=float),
                                metadata=pd.DataFrame({"name": [f"s{i}" for i in range(13)], "strand": ["."] * 13}))
scalar_out = server.serialize_trackdata(scalar_td)
check("标量截断且非有限值为 null", len(scalar_out["values"]) == 12 and scalar_out["values"][0] is None and scalar_out["total_tracks"] == 13)
check("矩阵明确选择第一轨道", o3b["track_index"] == 0 and o3b["selected_track_label"] == "cm0" and o3b["n_tracks"] == 2)
negative_matrix = SimpleNamespace(values=-np.ones((5, 5, 2)))
check("矩阵末组不补零", server.serialize_matrix(negative_matrix, 2)["values"] == [[-1., -1.], [-1., -1.]])
nonfinite_matrix = server.serialize_matrix(SimpleNamespace(values=np.array([[[np.nan], [np.inf]], [[-np.inf], [1.0]]])))
json.dumps([coord_out, scalar_out, nonfinite_matrix], allow_nan=False)
check("严格 JSON 可序列化全部非有限值", True)

print("\n=== 12. SDK JunctionData 实例与独立序列化 ===")
junctions = np.array([genome.Interval("chr1", 100 + i, 200 + i, "+" if i % 2 == 0 else "-")
                      for i in range(503)], dtype=object)
jd = junction_data.JunctionData(junctions=junctions, values=np.ones((503, 14), dtype=np.float64),
    metadata=pd.DataFrame({"name": [f"j{i}" for i in range(14)]}), interval=genome.Interval("chr1", 0, 16384))
jd.values[0, 0] = np.inf
junction_out = server.serialize_prediction(SimpleNamespace(splice_junctions=jd), ["SPLICE_JUNCTIONS"], 100, "mean")["SPLICE_JUNCTIONS"]
check("junctions 专用表格且500行上限", junction_out["kind"] == "junctions" and len(junction_out["rows"]) == 500
      and junction_out["total_rows"] == 503 and junction_out["truncated"])
check("junctions 12轨道标签与列一一对应", junction_out["shown_tracks"] == 12 and junction_out["total_tracks"] == 14
      and len(junction_out["columns"]) == 16 and len(junction_out["labels"]) == 12)
check("junctions 坐标及链保持", {k: junction_out["rows"][1][k] for k in ("chromosome", "start", "end", "strand")}
      == {"chromosome": "chr1", "start": 101, "end": 201, "strand": "-"})
check("junctions 非有限值转 null", junction_out["rows"][0]["track_0"] is None)
empty_jd = junction_data.JunctionData(junctions=np.array([], dtype=object), values=np.empty((0, 1)), metadata=pd.DataFrame({"name": ["j"]}))
check("空 junctions 保持表格契约", server.serialize_junctions(empty_jd)["total_rows"] == 0)
json.dumps(junction_out, allow_nan=False)

print("\n=== 13. float64 精度与有界手动回退 ===")
precise = 0.12345678901234566
precision_out = server._df_to_records(pd.DataFrame({"raw_score": [precise, np.inf, -np.inf, np.nan]}), 4)
check("表格 float64 不经10位小数截断", precision_out["rows"][0]["raw_score"] == precise)
check("表格非有限值为 null", all(r["raw_score"] is None for r in precision_out["rows"][1:]))


class BoundedMatrix:
    shape = (1_000_000, 10_000)
    reads = 0

    def __array__(self, *args, **kwargs):
        raise AssertionError("不得物化完整矩阵")

    def __getitem__(self, index):
        self.reads += 1
        if self.reads > 7:
            raise AssertionError("读取量超过 max_rows")
        return np.float64(precise)


bounded = BoundedMatrix()
bounded_out = server._anndata_to_table(SimpleNamespace(X=bounded), 7, 0, ["RNA_SEQ"])
check("百亿行 fallback 只读取7个元素", bounded.reads == 7 and len(bounded_out["rows"]) == 7
      and bounded_out["total_rows"] == 10_000_000_000 and bounded_out["truncated"])
check("fallback 保留浮点精度", bounded_out["rows"][0]["raw_score"] == precise)

print("\n=== 14. TestClient 本地安全、Key 生命周期、参数及预测路径 ===")
# 所有 Key 文件操作限制在临时目录；禁止底层网络，即使遗漏 mock 也不能联网。
with tempfile.TemporaryDirectory() as tmp, \
     patch.object(server, "KEY_FILE", Path(tmp) / ".apikey"), \
     patch.object(server, "_api_key", None), \
     patch.object(server, "_client_cache", {}), \
     patch("grpc.secure_channel", side_effect=AssertionError("测试禁止网络")), \
     TestClient(server.app, base_url="http://127.0.0.1:8000", raise_server_exceptions=False) as web:
    fake_client = Mock()
    fake_client.output_metadata.return_value = SimpleNamespace()
    fake_client.predict_sequence.return_value = SimpleNamespace(rna_seq=finite_td, splice_sites=td_small)
    fake_client.predict_interval.return_value = SimpleNamespace(splice_junctions=jd, contact_maps=td_3d)
    fake_client.predict_variant.return_value = SimpleNamespace(reference=SimpleNamespace(splice_junctions=jd), alternate=SimpleNamespace(splice_junctions=jd))
    fake_client.score_variant.return_value = [a1, a2]
    secret = "synthetic-test-only-key-never-real"
    with patch.object(server.dna_client, "create", return_value=fake_client) as create:
        for origin in ("http://evil.example", "null", "http://localhost:8000", "http://127.0.0.1:9000", "https://127.0.0.1:8000"):
            check(f"拒绝非同源 {origin}", web.post("/api/key", json={"api_key": secret}, headers={"origin": origin}).status_code == 403)
        check("Host 阻止 DNS rebinding", web.get("/api/health", headers={"host": "evil.example"}).status_code == 400)
        check("不接受跨站无Origin浏览器请求", web.get("/api/health", headers={"sec-fetch-site": "cross-site"}).status_code == 403)
        check("跨源拒绝未建连", create.call_count == 0)
        response = web.post("/api/key", json={"api_key": secret}, headers={"origin": "http://127.0.0.1:8000"})
        check("同源Key默认仅内存", response.status_code == 200 and response.json()["storage"] == "memory" and not server.KEY_FILE.exists())
        check("预览只有已配置", response.json()["api_key_preview"] == "已配置" and secret not in response.text)
        check("无 wildcard CORS", "access-control-allow-origin" not in response.headers)
        check("health 不泄露片段", web.get("/api/health").json()["api_key_preview"] == "已配置")
        response = web.post("/api/key", json={"api_key": secret, "persist": True})
        check("持久化明确明文", response.status_code == 200 and response.json()["storage"] == "plaintext_file"
              and "明文" in response.json()["storage_warning"] and server.KEY_FILE.read_text() == secret)
        response = web.post("/api/key", json={"api_key": "new-synthetic-key"})
        check("切换仅内存删除旧文件", response.status_code == 200 and not server.KEY_FILE.exists())
        server.KEY_FILE.write_text("old-synthetic-key")
        with patch.object(Path, "unlink", side_effect=PermissionError(secret)):
            response = web.post("/api/key", json={"api_key": "not-committed"})
            check("删除失败不得假保存", response.status_code == 500 and server._api_key == "new-synthetic-key" and secret not in response.text)
            response = web.delete("/api/key")
            check("clear报告磁盘失败并清除内存", response.status_code == 500 and server._api_key is None and secret not in response.text)
        check("clear成功确实删除磁盘", web.delete("/api/key").status_code == 200 and not server.KEY_FILE.exists())
        check("clear关闭旧channel", fake_client._channel.close.called)
        with patch.object(Path, "write_text", side_effect=PermissionError(secret)):
            response = web.post("/api/key", json={"api_key": secret, "persist": True})
            check("写入失败不提交Key", response.status_code == 500 and server._api_key is None and secret not in response.text)
        fake_client.output_metadata.side_effect = RuntimeError("unauthenticated " + secret)
        response = web.post("/api/key", json={"api_key": secret})
        check("验证Key失败不缓存不泄露", response.status_code == 401 and server._api_key is None and not server._client_cache and secret not in response.text)
        fake_client.output_metadata.side_effect = None
        create.side_effect = RuntimeError("unavailable " + secret)
        response = web.post("/api/key", json={"api_key": secret})
        check("create失败脱敏", response.status_code == 502 and secret not in response.text)
        create.side_effect = None
        for invalid in ({"api_key": {"secret": secret}}, {"api_key": secret, "persist": secret}, {"api_key": secret, "unexpected": secret}):
            response = web.post("/api/key", json=invalid)
            check("pydantic错误禁止输入回显", response.status_code == 400 and secret not in response.text and "input" not in response.json())
        response = web.post("/api/key", content='{"api_key":"' + secret, headers={"content-type": "application/json"})
        check("无效JSON不回显原文", response.status_code == 400 and secret not in response.text)
        terms = ["UBERON:0002048"]
        interval_payload = {"chromosome": "chr1", "start": 0, "end": 16384, "ontology_terms": terms}
        seq_payload = {"sequence": "A" * 16384, "ontology_terms": terms}
        variant_payload = {"chromosome": "chr1", "position": 20000, "reference_bases": "A", "alternate_bases": "C", "ontology_terms": terms}
        score_payload = {k: v for k, v in variant_payload.items() if k != "ontology_terms"}
        before = create.call_count
        invalid_requests = [
            ("interval", {**interval_payload, "end": 100}),
            ("interval", {**interval_payload, "chromosome": " "}),
            ("interval", {**interval_payload, "ontology_terms": []}),
            ("interval", {**interval_payload, "ontology_terms": terms * 9}),
            ("interval", {**interval_payload, "ontology_terms": ["INVALID:123"]}),
            ("interval", {**interval_payload, "ontology_terms": ["CL:999999999999999999999999"]}),
            ("interval", {**interval_payload, "outputs": []}),
            ("interval", {**interval_payload, "outputs": ["RNA_SEQ", "ATAC", "CAGE", "DNASE"]}),
            ("interval", {**interval_payload, "outputs": ["BOGUS"]}),
            ("interval", {**interval_payload, "organism": "BAD"}),
            ("interval", {**interval_payload, "max_points": 1}),
            ("sequence", {**seq_payload, "sequence": " \t\r\n\u2003"}),
            ("sequence", {**seq_payload, "sequence": "ACGT"}),
            ("sequence", {**seq_payload, "sequence": "<script>"}),
            ("sequence", {**seq_payload, "interval_width": 16384, "pad_to_length": True, "sequence": "A" * 16385}),
            ("sequence", {**seq_payload, "interval_width": 123}),
            ("variant", {**variant_payload, "alternate_bases": "a"}),
            ("variant", {**variant_payload, "alternate_bases": "*"}),
            ("variant", {**variant_payload, "alternate_bases": "."}),
            ("variant", {**variant_payload, "alternate_bases": "<DEL>"}),
            ("variant", {**variant_payload, "interval_width": 999}),
        ]
        for route, payload in invalid_requests:
            response = web.post(f"/api/predict/{route}", json=payload)
            check(f"{route} 非法参数先400", response.status_code == 400, str(response.status_code))
        check("非法参数无SDK建连", create.call_count == before)
        check("合法参数无Key为401", web.post("/api/predict/sequence", json=seq_payload).status_code == 401)
        with patch.object(server, "SDK_AVAILABLE", False), patch.object(server, "dna_client", None):
            response = web.post("/api/predict/sequence", json=seq_payload)
            check("SDK缺失为503且无堆栈", response.status_code == 503 and "Traceback" not in response.text)
        web.post("/api/key", json={"api_key": secret})
        for length in server.dna_client.SUPPORTED_SEQUENCE_LENGTHS.values():
            response = web.post("/api/predict/sequence", json={**seq_payload, "sequence": "a" * length})
            data = response.json()
            check(f"受支持长度{length}不自动填充", response.status_code == 200 and data["original_length"] == data["input_length"] == length
                  and not data["padded_with_N"] and data["padding_left_bp"] == 0
                  and fake_client.predict_sequence.call_args.kwargs["sequence"] == "A" * length)
        response = web.post("/api/predict/sequence", json={**seq_payload, "sequence": " a\tc\r\ng\u2003", "pad_to_length": True, "interval_width": 16384})
        data = response.json()
        check("去除全部空白后居中N填充", response.status_code == 200 and data["original_length"] == 3 and data["input_length"] == 16384
              and data["padded_with_N"] and data["padding_left_bp"] == 8190
              and fake_client.predict_sequence.call_args.kwargs["sequence"] == "N" * 8190 + "ACG" + "N" * 8191)
        response = web.post("/api/predict/sequence", json={"sequence": "A" * 16384, "outputs": ["SPLICE_SITES"]})
        check("SPLICE_SITES单独可不选terms", response.status_code == 200 and fake_client.predict_sequence.call_args.kwargs["ontology_terms"] is None)
        response = web.post("/api/predict/interval", json={**interval_payload, "outputs": ["SPLICE_JUNCTIONS", "CONTACT_MAPS"]})
        check("路由分发真实junction与matrix", response.status_code == 200 and response.json()["modalities"]["SPLICE_JUNCTIONS"]["kind"] == "junctions"
              and response.json()["modalities"]["CONTACT_MAPS"]["track_index"] == 0)
        response = web.post("/api/predict/variant", json={**variant_payload, "outputs": ["SPLICE_JUNCTIONS"]})
        check("REF ALT 专用junction序列化", response.status_code == 200 and all(response.json()[allele]["SPLICE_JUNCTIONS"]["kind"] == "junctions" for allele in ("reference", "alternate")))
        response = web.post("/api/metadata")
        check("原GET metadata契约不变", response.status_code == 405 and web.get("/api/metadata").status_code == 200)
        for result in (None, pd.DataFrame()):
            with patch.object(server.variant_scorers, "tidy_scores", return_value=result), patch.object(server, "_anndata_to_table") as fallback:
                response = web.post("/api/score/variant", json=score_payload)
                check("tidy空结果绝不回退", response.status_code == 200 and response.json()["tables"] == [] and not fallback.called)
        with patch.object(server.variant_scorers, "tidy_scores", side_effect=ValueError(secret)):
            response = web.post("/api/score/variant", json={**score_payload, "max_rows": 5})
            check("tidy异常可回退且有note/全局行数上限", response.status_code == 200 and "note" in response.json()
                  and sum(len(t["rows"]) for t in response.json()["tables"]) == 5 and secret not in response.text)
        with patch.object(server.variant_scorers, "tidy_scores", return_value=pd.DataFrame({"raw_score": [precise, np.inf]})):
            response = web.post("/api/score/variant", json=score_payload)
            check("score路由精度/null", response.status_code == 200 and response.json()["tables"][0]["rows"] == [{"raw_score": precise}, {"raw_score": None}])
        for message, code in [("unauthenticated", 401), ("quota", 429), ("deadline", 502), ("invalid argument", 400), ("unknown", 500)]:
            fake_client.predict_sequence.side_effect = RuntimeError(message + " " + secret)
            response = web.post("/api/predict/sequence", json=seq_payload)
            check(f"上游{code}错误脱敏", response.status_code == code and secret not in response.text and "RuntimeError" not in response.text)
        fake_client.predict_sequence.side_effect = None
        with patch.object(server, "serialize_prediction", side_effect=RuntimeError(secret)):
            response = web.post("/api/predict/sequence", json=seq_payload)
            check("未捕获异常无堆栈/Key", response.status_code == 500 and secret not in response.text and "Traceback" not in response.text)

print("\n" + "=" * 60)
if FAILURES:
    print(f"失败 {len(FAILURES)} / {CHECKS} 项：")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print(f"全部通过：{CHECKS} 项检查（离线，无真实 Key）")
