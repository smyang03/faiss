from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fp_finder.curation import SimilarityReductionConfig, build_similarity_reduction_plan


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a natural feature-similarity dataset reduction plan.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-query-records", type=int, default=50000, help="0 uses all records for a full safe plan.")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rerank-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-filter", default="")
    parser.add_argument("--size-bucket", default="")
    parser.add_argument("--tight-threshold", type=float, default=0.985)
    parser.add_argument("--protect-cross-class-threshold", type=float, default=0.90)
    parser.add_argument("--same-class-only", default="true")
    parser.add_argument("--same-size-only", default="true")
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--representatives-per-group", type=int, default=1)
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

    config = SimilarityReductionConfig(
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        max_query_records=int(args.max_query_records),
        top_k=int(args.top_k),
        rerank_k=int(args.rerank_k),
        seed=int(args.seed),
        class_filter=str(args.class_filter),
        size_bucket=str(args.size_bucket),
        tight_threshold=float(args.tight_threshold),
        protect_cross_class_threshold=float(args.protect_cross_class_threshold),
        same_class_only=parse_bool(args.same_class_only),
        same_size_only=parse_bool(args.same_size_only),
        min_group_size=int(args.min_group_size),
        representatives_per_group=int(args.representatives_per_group),
        batch_size=int(args.batch_size),
    )
    result = build_similarity_reduction_plan(config, progress=progress)
    summary = result["summary"]
    print(f"done output={result['output_dir']} elapsed={format_duration(time.time() - start)}", flush=True)
    print(
        "summary "
        f"sampled={summary['sampled_query_records']:,} "
        f"groups={summary['tight_groups']:,} "
        f"drop_records={summary['drop_record_candidates']:,} "
        f"record_reduction={summary['record_reduction_pct_of_planned']:.2f}% "
        f"safe_drop_images={summary['safe_image_drop_candidates']:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
