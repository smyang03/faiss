from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fp_finder.yolo_dataset import (
    iter_yolo_records,
    load_class_names,
    load_yolo_records,
    parse_class_ids,
    record_to_dict,
    records_to_json,
)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def eta_ready(done: int, total: int, elapsed: float) -> bool:
    if total <= 0 or done <= 0:
        return False
    if done >= total:
        return True
    min_done = min(100, max(10, total // 100))
    return done >= min_done and elapsed >= 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO txt CropRecord metadata as records.json.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", default="")
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--shard-root", default="", help="Optional output root for pre-sharded records.jsonl files.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--expand", type=float, default=0.08)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--class-ids", default="", help="Optional comma/space separated class ids to include, e.g. 0,3,4.")
    parser.add_argument("--image-size-cache", default="", help="Optional SQLite cache for image width/height by path+mtime+size.")
    parser.add_argument(
        "--dataset-layout",
        default="single",
        choices=["single", "nested_jpegimages_labels", "nested_image_labels"],
        help="single: one images dir + one labels dir. nested_image_labels: root/*/(JPEGImages|images) + sibling labels.",
    )
    return parser.parse_args()


def write_sharded_records(args: argparse.Namespace, class_names, progress) -> int:
    shard_root = Path(args.shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, int(args.num_shards))
    files = []
    offsets = [[] for _idx in range(num_shards)]
    counts = [0 for _idx in range(num_shards)]
    try:
        for shard_idx in range(num_shards):
            path = shard_root / f"records_shard_{shard_idx}.jsonl"
            files.append(path.open("wb"))

        for record in iter_yolo_records(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            class_names=class_names,
            expand=args.expand,
            max_records=args.max_records or None,
            class_ids=parse_class_ids(args.class_ids),
            image_size_cache_path=args.image_size_cache or None,
            progress=progress,
            dataset_layout=args.dataset_layout,
        ):
            digest = hashlib.blake2b(str(record.image_path).encode("utf-8"), digest_size=8).digest()
            shard_idx = int.from_bytes(digest, byteorder="little", signed=False) % num_shards
            f = files[shard_idx]
            offsets[shard_idx].append(f.tell())
            line = json.dumps(record_to_dict(record), ensure_ascii=False, separators=(",", ":"))
            f.write(line.encode("utf-8"))
            f.write(b"\n")
            counts[shard_idx] += 1
    finally:
        for f in files:
            f.close()

    for shard_idx, shard_offsets in enumerate(offsets):
        np.save(str(shard_root / f"record_offsets_shard_{shard_idx}.npy"), np.asarray(shard_offsets, dtype=np.int64))

    summary = {
        "num_shards": num_shards,
        "total_records": int(sum(counts)),
        "counts": counts,
        "records": [
            {
                "shard": idx,
                "records_jsonl": str(shard_root / f"records_shard_{idx}.jsonl"),
                "offsets": str(shard_root / f"record_offsets_shard_{idx}.npy"),
                "count": int(counts[idx]),
            }
            for idx in range(num_shards)
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return int(sum(counts))


def main() -> None:
    args = parse_args()
    start = time.time()
    class_names = load_class_names(args.data_yaml) or {0: "person"}
    print(f"images_dir={args.images_dir}", flush=True)
    print(f"labels_dir={args.labels_dir}", flush=True)
    print(f"dataset_layout={args.dataset_layout}", flush=True)
    print(f"class_ids={args.class_ids or 'all'}", flush=True)
    print(f"max_records={args.max_records or 'all'}", flush=True)
    print(f"image_size_cache={args.image_size_cache or 'off'}", flush=True)
    print(f"output_json={args.output_json}", flush=True)

    last_print = {"t": 0.0}

    def progress(done: int, total: int, message: str) -> None:
        now = time.time()
        if now - last_print["t"] >= 5 or done >= total:
            last_print["t"] = now
            pct = 0.0 if total <= 0 else done / total * 100.0
            elapsed = now - start
            if eta_ready(done, total, elapsed):
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(
                    f"{pct:6.2f}% | {message} | elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                    flush=True,
                )
            else:
                print(f"{pct:6.2f}% | {message} | elapsed={format_duration(elapsed)}", flush=True)

    if args.shard_root:
        total_records = write_sharded_records(args, class_names, progress)
        print(f"records={total_records} elapsed_sec={time.time() - start:.1f}", flush=True)
    else:
        records = load_yolo_records(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            class_names=class_names,
            expand=args.expand,
            max_records=args.max_records or None,
            class_ids=parse_class_ids(args.class_ids),
            image_size_cache_path=args.image_size_cache or None,
            progress=progress,
            dataset_layout=args.dataset_layout,
        )
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        records_to_json(records, str(output))
        print(f"records={len(records)} elapsed_sec={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
