from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fp_finder.yolo_dataset import load_class_names, load_yolo_records, parse_class_ids, records_from_json
from fp_finder.yolo_feature_index import build_yolo_feature_index


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
    parser = argparse.ArgumentParser(description="Build/save YOLO ROI feature vectors for YOLO txt DB.")
    parser.add_argument("--images-dir", default="db/data_cogress2/JPEGImages")
    parser.add_argument("--labels-dir", default="db/data_cogress2/labels")
    parser.add_argument("--data-yaml", default="db/data_cogress2/data.yaml")
    parser.add_argument("--records-json", default="", help="Precomputed records.json. Skips image/label scanning.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, 0, 1, cuda:0 ...")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--expand", type=float, default=0.08)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--class-ids", default="", help="Optional comma/space separated class ids to include, e.g. 0,3,4.")
    parser.add_argument("--feature-batch-size", type=int, default=0, help="0=auto, 1=single image, N=batch images per YOLO forward.")
    parser.add_argument(
        "--dataset-layout",
        default="single",
        choices=["single", "nested_jpegimages_labels", "nested_image_labels"],
        help="single: one images dir + one labels dir. nested_image_labels: root/*/(JPEGImages|images) + sibling labels.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-faiss", action="store_true", help="Only save features.npy/records/config.")
    parser.add_argument("--no-features", action="store_true", help="Do not save features.npy, only FAISS index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    start = time.time()
    class_names = load_class_names(args.data_yaml) or {0: "person"}
    print(f"repo_path={args.repo_path}", flush=True)
    print(f"weights_path={args.weights_path}", flush=True)
    print(f"index_dir={args.index_dir}", flush=True)
    print(f"device={args.device}", flush=True)
    print(f"feature_batch_size_arg={args.feature_batch_size}", flush=True)
    if args.records_json:
        print(f"loading precomputed records: {args.records_json}", flush=True)
        records = records_from_json(args.records_json)
        class_ids = parse_class_ids(args.class_ids)
        if class_ids is not None:
            records = [record for record in records if int(record.class_id) in class_ids]
        if args.max_records:
            records = records[: args.max_records]
    else:
        print("loading YOLO records...", flush=True)
        records = load_yolo_records(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            class_names=class_names,
            expand=args.expand,
            max_records=args.max_records or None,
            class_ids=parse_class_ids(args.class_ids),
            dataset_layout=args.dataset_layout,
        )
    if args.num_shards > 1:
        records = [r for idx, r in enumerate(records) if idx % args.num_shards == args.shard_index]
        for new_id, record in enumerate(records):
            record.record_id = new_id
        print(f"shard={args.shard_index}/{args.num_shards}", flush=True)
    print(f"records={len(records)}", flush=True)

    last_print = {"t": 0.0}
    feature_start = time.time()

    def progress(done: int, total: int, message: str) -> None:
        now = time.time()
        if now - last_print["t"] >= 5 or done >= total:
            last_print["t"] = now
            pct = 0.0 if total <= 0 else done / total * 100.0
            elapsed = now - feature_start
            if eta_ready(done, total, elapsed):
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(
                    f"{pct:6.2f}% | {message} | elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                    flush=True,
                )
            else:
                print(f"{pct:6.2f}% | {message} | elapsed={format_duration(elapsed)}", flush=True)

    build_yolo_feature_index(
        records=records,
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        index_dir=args.index_dir,
        device=args.device,
        img_size=args.img_size,
        progress=progress,
        save_features=not args.no_features,
        build_faiss=not args.no_faiss,
        feature_batch_size=int(args.feature_batch_size),
    )
    print(f"done elapsed_sec={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
