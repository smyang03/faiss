from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fp_finder.faiss_index import build_faiss_index
from fp_finder.yolo_dataset import load_class_names, load_yolo_records, parse_class_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for YOLO txt dataset.")
    parser.add_argument("--images-dir", default="db/data_cogress2/JPEGImages")
    parser.add_argument("--labels-dir", default="db/data_cogress2/labels")
    parser.add_argument("--data-yaml", default="db/data_cogress2/data.yaml")
    parser.add_argument("--index-dir", default="artifacts/faiss_index_data_cogress2")
    parser.add_argument("--encoder", default="resnet18", choices=["resnet18", "resnet50", "dinov2_vits14"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expand", type=float, default=0.08)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--class-ids", default="", help="Optional comma/space separated class ids to include, e.g. 0,3,4.")
    parser.add_argument(
        "--dataset-layout",
        default="single",
        choices=["single", "nested_jpegimages_labels", "nested_image_labels"],
        help="single: one images dir + one labels dir. nested_image_labels: root/*/(JPEGImages|images) + sibling labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    class_names = load_class_names(args.data_yaml) or {0: "person"}
    print(f"images_dir={args.images_dir}", flush=True)
    print(f"labels_dir={args.labels_dir}", flush=True)
    print(f"index_dir={args.index_dir}", flush=True)
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
    print(f"records={len(records)}", flush=True)

    last_print = {"t": 0.0}

    def progress(done: int, total: int, message: str) -> None:
        now = time.time()
        if now - last_print["t"] >= 5 or done >= total:
            last_print["t"] = now
            pct = 0.0 if total <= 0 else done / total * 100.0
            print(f"{pct:6.2f}% | {message}", flush=True)

    build_faiss_index(
        records=records,
        encoder_name=args.encoder,
        index_dir=args.index_dir,
        device=args.device,
        batch_size=args.batch_size,
        progress=progress,
    )
    elapsed = time.time() - start
    print(f"done elapsed_sec={elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
