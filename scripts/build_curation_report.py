from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fp_finder.curation import CurationReportConfig, build_curation_report


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS kNN curation reports from a YOLO feature index.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-query-records", type=int, default=50000, help="0 uses all records. Use a sample for fast review.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rerank-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-filter", default="")
    parser.add_argument("--size-bucket", default="")
    parser.add_argument("--duplicate-threshold", type=float, default=0.98)
    parser.add_argument("--cross-class-threshold", type=float, default=0.90)
    parser.add_argument("--rare-group-max", type=int, default=20)
    parser.add_argument("--boundary-per-group", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    last_print = {"t": 0.0}

    def progress(done: int, total: int, message: str) -> None:
        now = time.time()
        if now - last_print["t"] < 5 and done < total:
            return
        last_print["t"] = now
        pct = 0.0 if total <= 0 else done / max(1, total) * 100.0
        elapsed = now - start
        eta_text = ""
        if done > 0 and total > 0 and done < total:
            rate = done / max(elapsed, 1e-6)
            eta_text = f" eta={format_duration((total - done) / max(rate, 1e-6))}"
        print(f"{pct:6.2f}% | {message} | elapsed={format_duration(elapsed)}{eta_text}", flush=True)

    config = CurationReportConfig(
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        max_query_records=int(args.max_query_records),
        top_k=int(args.top_k),
        rerank_k=int(args.rerank_k),
        seed=int(args.seed),
        class_filter=str(args.class_filter),
        size_bucket=str(args.size_bucket),
        duplicate_threshold=float(args.duplicate_threshold),
        cross_class_threshold=float(args.cross_class_threshold),
        rare_group_max=int(args.rare_group_max),
        boundary_per_group=int(args.boundary_per_group),
        batch_size=int(args.batch_size),
    )
    result = build_curation_report(config, progress=progress)
    summary = result["summary"]
    print(f"done output={result['output_dir']} elapsed={format_duration(time.time() - start)}", flush=True)
    print(
        "summary "
        f"sampled={summary['sampled_records']:,} "
        f"near_edges={summary['near_duplicate_edges']:,} "
        f"duplicate_groups={summary['duplicate_groups']:,} "
        f"cross_edges={summary['cross_class_edges']:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

