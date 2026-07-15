from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fp_finder.curation import export_similarity_reduction_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a reduced YOLO dataset from a similarity reduction plan.")
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-root", default="")
    parser.add_argument("--labels-root", default="")
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--mode", choices=["manifest", "copy", "hardlink"], default="manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_similarity_reduction_plan(
        plan_dir=args.plan_dir,
        output_dir=args.output_dir,
        images_root=args.images_root,
        labels_root=args.labels_root,
        data_yaml=args.data_yaml,
        mode=args.mode,
    )
    print(
        "done "
        f"mode={summary['mode']} "
        f"kept_images={summary['kept_images']:,} "
        f"drop_image_candidates={summary['drop_image_candidates']:,} "
        f"drop_record_candidates={summary['drop_record_candidates']:,} "
        f"output={summary['output_dir']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
