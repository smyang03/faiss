from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import faiss
import numpy as np

from fp_finder.curation import (
    CurationReportConfig,
    SimilarityReductionConfig,
    build_curation_report,
    build_similarity_reduction_plan,
    export_reduced_dataset,
    export_similarity_reduction_plan,
)
from fp_finder.feature_clustering import build_feature_clusters
from fp_finder.video import collect_video_detections, read_video_frame
from fp_finder.yolo_dataset import crop_from_record, index_records_ready, open_record_store
from fp_finder.yolo_feature_index import YoloFeatureIndex


CheckFn = Callable[[], dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke/health checks for the YOLO feature search app.")
    parser.add_argument("--index-dir", default="artifacts/yolo_feature_index_svms")
    parser.add_argument("--weights-path", default="model/SIAV2_Detector_YOLOV7_SafeEnv_V8.0.0_FP32_260616.pt")
    parser.add_argument("--repo-path", default="external/yolov7")
    parser.add_argument("--video-dir", default="video")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/health_checks/latest")
    parser.add_argument("--include-detector", action="store_true")
    parser.add_argument("--cluster-sample", type=int, default=300)
    parser.add_argument("--curation-sample", type=int, default=100)
    return parser.parse_args()


def timed(name: str, fn: CheckFn) -> dict:
    start = time.time()
    try:
        detail = fn()
        status = "PASS"
        error = ""
    except Exception as exc:
        detail = {}
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        detail["traceback"] = traceback.format_exc()
    elapsed = time.time() - start
    row = {"name": name, "status": status, "elapsed_sec": round(elapsed, 3), "error": error, "detail": detail}
    print(f"[{status}] {name} {elapsed:.3f}s {error}", flush=True)
    return row


def first_video(video_dir: str) -> Path | None:
    root = Path(video_dir)
    if not root.exists():
        return None
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        found = sorted(root.glob(ext))
        if found:
            return found[0]
    return None


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_dir = Path(args.index_dir)
    records_cache = {}

    def load_records():
        if "records" not in records_cache:
            records_cache["records"] = open_record_store(index_dir)
        return records_cache["records"]

    checks: List[tuple[str, CheckFn]] = []

    def check_index_files() -> dict:
        required = ["config.json", "features.npy", "index.faiss"]
        missing = [name for name in required if not (index_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")
        if not index_records_ready(index_dir):
            raise FileNotFoundError("records metadata is not ready")
        with (index_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        return {
            "index_dir": str(index_dir),
            "dim": int(config.get("dim", 0) or 0),
            "num_records": int(config.get("num_records", 0) or 0),
            "faiss_type": config.get("faiss_type", ""),
        }

    checks.append(("index_files", check_index_files))

    def check_records() -> dict:
        records = load_records()
        total = len(records)
        indices = [0, max(0, total // 2), max(0, total - 1)]
        rows = []
        for idx in indices:
            record = records[idx]
            rows.append(
                {
                    "idx": int(idx),
                    "record_id": int(record.record_id),
                    "class_id": int(record.class_id),
                    "class_name": str(record.class_name),
                    "file_name": Path(record.image_path).name,
                }
            )
        return {"records": int(total), "samples": rows}

    checks.append(("record_store", check_records))

    def check_faiss_search() -> dict:
        features = np.load(str(index_dir / "features.npy"), mmap_mode="r")
        index = faiss.read_index(str(index_dir / "index.faiss"))
        if int(features.shape[0]) != int(index.ntotal):
            raise ValueError(f"features/index count mismatch: {features.shape[0]} != {index.ntotal}")
        scores, ids = index.search(np.asarray(features[[0]], dtype=np.float32), 10)
        return {
            "features_shape": [int(features.shape[0]), int(features.shape[1])],
            "index_ntotal": int(index.ntotal),
            "top_ids": [int(value) for value in ids[0].tolist()],
            "top_scores": [round(float(value), 5) for value in scores[0].tolist()],
        }

    checks.append(("faiss_search", check_faiss_search))

    def check_yolo_index_search() -> dict:
        yolo_index = YoloFeatureIndex.load(str(index_dir), device=args.device)
        try:
            record = yolo_index.records[0]
            record_results = yolo_index.search_record(record, top_k=5, exclude_self=True)
            crop = crop_from_record(record)
            crop_results = yolo_index.search_crop(crop, top_k=3)
            return {
                "records": int(len(yolo_index.records)),
                "record_results": len(record_results),
                "crop_results": len(crop_results),
                "record_top": [
                    [int(item["rank"]), round(float(item["score"]), 5), int(item["record"].class_id)]
                    for item in record_results[:3]
                ],
                "crop_top": [
                    [int(item["rank"]), round(float(item["score"]), 5), int(item["record"].class_id)]
                    for item in crop_results[:3]
                ],
            }
        finally:
            yolo_index.extractor.close()

    checks.append(("yolo_feature_search", check_yolo_index_search))

    def check_clustering() -> dict:
        results = []
        configs = [
            ("all", "", "global", "minibatch_kmeans"),
            ("class0", "0", "global", "minibatch_kmeans"),
            ("per_class", "", "per_class", "bisecting_kmeans"),
            ("class_size", "", "class_size", "birch"),
            ("hdbscan", "", "global", "hdbscan"),
        ]
        for label, class_filter, scope, method in configs:
            result = build_feature_clusters(
                str(index_dir),
                max_points=int(args.cluster_sample),
                n_clusters=8,
                seed=42,
                class_filter=class_filter,
                clustering_scope=scope,
                clustering_method=method,
            )
            if result["df"].empty:
                raise ValueError(f"empty clustering result for {label}")
            results.append(
                {
                    "label": label,
                    "method": method,
                    "scope": scope,
                    "sample_size": int(result["sample_size"]),
                    "clusters": int(result["n_clusters"]),
                    "summary_rows": int(len(result["summary"])),
                }
            )
        return {"runs": results}

    checks.append(("clustering", check_clustering))

    def check_curation_export() -> dict:
        report_dir = output_dir / "curation_report"
        export_dir = output_dir / "reduced_dataset"
        result = build_curation_report(
            CurationReportConfig(
                index_dir=str(index_dir),
                output_dir=str(report_dir),
                max_query_records=int(args.curation_sample),
                top_k=10,
                rerank_k=30,
                batch_size=64,
            )
        )
        export_summary = export_reduced_dataset(str(report_dir), str(export_dir), mode="manifest")
        required = [
            "summary.json",
            "curation_recommendations.csv",
            "image_recommendations.csv",
            "near_duplicates.csv",
            "cross_class_overlap.csv",
        ]
        missing = [name for name in required if not (report_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing curation outputs: {missing}")
        return {
            "sampled_records": int(result["summary"]["sampled_records"]),
            "near_duplicate_edges": int(result["summary"]["near_duplicate_edges"]),
            "cross_class_edges": int(result["summary"]["cross_class_edges"]),
            "export": export_summary,
        }

    checks.append(("curation_export", check_curation_export))

    def check_similarity_reduction() -> dict:
        plan_dir = output_dir / "similarity_reduction_plan"
        export_dir = output_dir / "similarity_reduction_export"
        result = build_similarity_reduction_plan(
            SimilarityReductionConfig(
                index_dir=str(index_dir),
                output_dir=str(plan_dir),
                max_query_records=int(args.curation_sample),
                top_k=10,
                rerank_k=30,
                tight_threshold=0.98,
                protect_cross_class_threshold=0.90,
                batch_size=64,
            )
        )
        export_summary = export_similarity_reduction_plan(str(plan_dir), str(export_dir), mode="manifest")
        required = [
            "reduction_summary.json",
            "reduction_groups.csv",
            "reduction_group_members.csv",
            "reduction_image_plan.csv",
            "reduction_recommendations.csv",
        ]
        missing = [name for name in required if not (plan_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing reduction outputs: {missing}")
        return {
            "planned_records": int(result["summary"]["planned_records"]),
            "tight_groups": int(result["summary"]["tight_groups"]),
            "drop_record_candidates": int(result["summary"]["drop_record_candidates"]),
            "record_reduction_pct": round(float(result["summary"]["record_reduction_pct_of_planned"]), 4),
            "export": export_summary,
        }

    checks.append(("similarity_reduction", check_similarity_reduction))

    def check_calibration() -> dict:
        import app

        rows = []
        for class_filter in ("", "0"):
            result = app.cached_similarity_calibration(str(index_dir), 100, 5, 42, class_filter, 0.02)
            detail = result.get("detail")
            if detail is None or detail.empty:
                raise ValueError(f"empty calibration detail for class_filter={class_filter!r}")
            rows.append(
                {
                    "class_filter": class_filter or "all",
                    "detail_rows": int(len(detail)),
                    "bins": int(len(result.get("bins"))),
                    "thresholds": int(len(result.get("thresholds"))),
                    "candidate_mode": str(result.get("candidate_mode", "")),
                }
            )
        return {"runs": rows}

    checks.append(("calibration", check_calibration))

    def check_plotly_payload() -> dict:
        import app

        cluster = build_feature_clusters(str(index_dir), max_points=1200, n_clusters=12, seed=11)
        df = cluster["df"].copy()
        df["area_pct"] = df["area_ratio"] * 100.0
        df["file_name"] = df["image_path"].map(lambda value: Path(str(value)).name)
        fig = app.build_cluster_hover_figure(df, "cluster_label", seed=11, projection="3D", max_points=800)
        first_points = len(fig.data[0].x) if fig.data else 0
        if len(fig.data) > 25:
            raise ValueError(f"too many plot traces: {len(fig.data)}")
        return {"traces": int(len(fig.data)), "first_trace_points": int(first_points)}

    checks.append(("plotly_payload", check_plotly_payload))

    def check_video_frame() -> dict:
        video_path = first_video(args.video_dir)
        if video_path is None:
            return {"skipped": True, "reason": "no video files"}
        frame = read_video_frame(str(video_path), 0)
        return {"video": str(video_path), "frame_size": [int(frame.size[0]), int(frame.size[1])]}

    checks.append(("video_frame", check_video_frame))

    if args.include_detector:
        def check_detector_video() -> dict:
            from fp_finder.detector_yolov7 import YoloV7Detector

            video_path = first_video(args.video_dir)
            if video_path is None:
                return {"skipped": True, "reason": "no video files"}
            detector = YoloV7Detector(
                weights_path=args.weights_path,
                repo_path=args.repo_path,
                device=args.device,
                img_size=640,
                conf_thres=0.25,
            )
            detections = collect_video_detections(
                str(video_path),
                detector,
                frame_stride=999999,
                max_frames=1,
                max_detections=10,
            )
            return {
                "video": str(video_path),
                "detections": int(len(detections)),
                "first": [
                    {
                        "class_id": int(det.class_id),
                        "confidence": round(float(det.confidence), 5),
                        "bbox": [int(v) for v in det.bbox_xyxy],
                    }
                    for det in detections[:3]
                ],
            }

        checks.append(("detector_video", check_detector_video))

    rows = [timed(name, fn) for name, fn in checks]
    summary = {
        "total": len(rows),
        "passed": sum(1 for row in rows if row["status"] == "PASS"),
        "failed": sum(1 for row in rows if row["status"] == "FAIL"),
        "checks": rows,
    }
    with (output_dir / "health_check_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"passed": summary["passed"], "failed": summary["failed"], "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
