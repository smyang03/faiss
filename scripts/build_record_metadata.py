from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fp_finder.feature_clustering import (
    SIZE_BUCKET_ORDER,
    load_or_build_record_meta_arrays,
)
from fp_finder.yolo_dataset import open_record_store


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def print_progress(done: int, total: int, message: str, start_time: float) -> None:
    pct = 0.0 if total <= 0 else done / total * 100.0
    elapsed = time.time() - start_time
    if done > 0 and total > 0:
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        print(
            f"{pct:6.2f}% | {message} | elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
            flush=True,
        )
    else:
        print(f"{pct:6.2f}% | {message} | elapsed={format_duration(elapsed)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build record class/size metadata cache for a YOLO feature index.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"index dir not found: {index_dir}")

    start = time.time()
    print_progress(0, 1, "Opening record store", start)
    records = open_record_store(index_dir)
    total = len(records)
    print(f"records={total}", flush=True)

    last_print = {"t": 0.0}

    def progress(done: int, total_count: int, message: str) -> None:
        now = time.time()
        if now - last_print["t"] >= 5 or done >= total_count:
            last_print["t"] = now
            print_progress(int(done), int(total_count), message, start)

    class_ids, size_codes = load_or_build_record_meta_arrays(str(index_dir), records, total, progress=progress)
    print_progress(total, total, "Summarizing metadata counts", start)

    class_values, class_counts = np.unique(class_ids[:total], return_counts=True)
    size_values, size_counts = np.unique(size_codes[:total], return_counts=True)
    summary = {
        "index_dir": str(index_dir),
        "record_meta_cache": str(index_dir / "record_meta_cache.npz"),
        "total_records": int(total),
        "class_counts": {
            str(int(class_id)): int(count)
            for class_id, count in zip(class_values.tolist(), class_counts.tolist())
            if int(class_id) >= 0
        },
        "size_counts": {
            SIZE_BUCKET_ORDER[int(size_code)]: int(count)
            for size_code, count in zip(size_values.tolist(), size_counts.tolist())
            if 0 <= int(size_code) < len(SIZE_BUCKET_ORDER)
        },
        "elapsed_sec": round(time.time() - start, 3),
    }
    for bucket in SIZE_BUCKET_ORDER:
        summary["size_counts"].setdefault(bucket, 0)

    summary_json = Path(args.summary_json) if args.summary_json else index_dir / "record_meta_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_progress(total, total, "Record metadata cache ready", start)
    print(json.dumps({"summary_json": str(summary_json), "elapsed_sec": summary["elapsed_sec"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
