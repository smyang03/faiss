from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a YOLO feature index project in the background.")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", default="")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--expand", type=float, default=0.08)
    parser.add_argument(
        "--dataset-layout",
        default="single",
        choices=["single", "nested_jpegimages_labels", "nested_image_labels"],
        help="single: one images dir + one labels dir. nested_image_labels: root/*/(JPEGImages|images) + sibling labels.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=0, help="Max shard processes running at once. 0 uses number of devices.")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--class-ids", default="")
    parser.add_argument("--feature-batch-size", type=int, default=0, help="0=auto, 1=single image, N=batch images per YOLO forward.")
    parser.add_argument("--image-size-cache", default="", help="Optional SQLite cache for records preparation image sizes.")
    parser.add_argument("--faiss-type", default="ivfpq", choices=["flat", "ivfpq"])
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--pq-m", type=int, default=64)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=200000)
    parser.add_argument("--rerank-k", type=int, default=500)
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--faiss-gpu-device", type=int, default=0)
    parser.add_argument("--faiss-gpu-required", action="store_true")
    parser.add_argument("--records-json", default="")
    parser.add_argument("--shard-root", default="")
    parser.add_argument("--log-root", default="")
    parser.add_argument("--force-prepare", action="store_true", help="Recreate records.json even if it already exists.")
    return parser.parse_args()


def log(log_root: Path, message: str) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
    print(line, flush=True)
    with (log_root / "launcher.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    args = parse_args()
    project_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.project_name).strip("_")
    if not project_slug:
        project_slug = "project"

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    data_yaml = Path(args.data_yaml)
    repo_path = Path(args.repo_path)
    weights_path = Path(args.weights_path)
    index_dir = Path(args.index_dir)
    log_root = Path(args.log_root) if args.log_root else ROOT / "artifacts" / "project_build_logs" / project_slug
    shard_root = Path(args.shard_root) if args.shard_root else ROOT / "artifacts" / "project_feature_shards" / project_slug
    records_json = Path(args.records_json) if args.records_json else ROOT / "artifacts" / "project_records" / f"{project_slug}.json"
    image_size_cache = (
        Path(args.image_size_cache)
        if args.image_size_cache
        else ROOT / "artifacts" / "image_size_cache" / f"{project_slug}.sqlite"
    )

    required_paths = [
        ("images_dir", images_dir),
        ("data_yaml", data_yaml),
        ("repo_path", repo_path),
        ("weights_path", weights_path),
    ]
    if args.dataset_layout == "single":
        required_paths.insert(1, ("labels_dir", labels_dir))

    log_root.mkdir(parents=True, exist_ok=True)
    for label, path in required_paths:
        if not path.exists():
            log(log_root, f"preflight failed: {label} not found: {path}")
            raise FileNotFoundError(f"{label} not found: {path}")

    shard_root.mkdir(parents=True, exist_ok=True)
    records_json.parent.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )

    if args.force_prepare and records_json.exists():
        records_json.unlink()
        log(log_root, f"removed existing records json because --force-prepare was set: {records_json}")
    if args.force_prepare:
        for old_file in shard_root.glob("records_shard_*.jsonl"):
            old_file.unlink(missing_ok=True)
        for old_file in shard_root.glob("record_offsets_shard_*.npy"):
            old_file.unlink(missing_ok=True)

    if not records_json.exists():
        log(log_root, f"preparing records json: {records_json}")
        prep_cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "prepare_yolo_records.py"),
            "--images-dir",
            str(images_dir),
            "--labels-dir",
            str(labels_dir),
            "--data-yaml",
            str(data_yaml),
            "--output-json",
            str(records_json),
            "--expand",
            str(args.expand),
            "--dataset-layout",
            args.dataset_layout,
            "--max-records",
            str(args.max_records),
            "--class-ids",
            str(args.class_ids),
            "--image-size-cache",
            str(image_size_cache),
            "--num-shards",
            str(args.num_shards),
            "--shard-root",
            str(shard_root),
        ]
        with (log_root / "prepare_records.out.log").open("w", encoding="utf-8") as out_f, (
            log_root / "prepare_records.err.log"
        ).open("w", encoding="utf-8") as err_f:
            code = subprocess.call(prep_cmd, cwd=str(ROOT), stdout=out_f, stderr=err_f, env=env)
        log(log_root, f"prepare records finished exit_code={code}")
        if code != 0:
            return int(code)
    else:
        log(log_root, f"using existing records json: {records_json}")

    device_values = [value.strip() for value in str(args.device).split(",") if value.strip()]
    if not device_values:
        device_values = ["cpu"]

    max_workers = int(args.max_workers or len(device_values) or 1)
    max_workers = max(1, min(max_workers, int(args.num_shards)))
    processes = []
    failed = []
    next_shard_idx = 0

    def start_shard(shard_idx: int):
        shard_device = device_values[shard_idx % len(device_values)]
        shard_dir = shard_root / f"shard_{shard_idx}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        out_log = log_root / f"shard_{shard_idx}.out.log"
        err_log = log_root / f"shard_{shard_idx}.err.log"
        shard_records = shard_root / f"records_shard_{shard_idx}.jsonl"
        if not shard_records.exists():
            raise FileNotFoundError(f"pre-sharded records not found: {shard_records}")

        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "build_yolo_feature_index.py"),
            "--images-dir",
            str(images_dir),
            "--labels-dir",
            str(labels_dir),
            "--data-yaml",
            str(data_yaml),
            "--records-json",
            str(shard_records),
            "--repo-path",
            str(repo_path),
            "--weights-path",
            str(weights_path),
            "--index-dir",
            str(shard_dir),
            "--device",
            shard_device,
            "--img-size",
            str(args.img_size),
            "--expand",
            str(args.expand),
            "--dataset-layout",
            args.dataset_layout,
            "--num-shards",
            "1",
            "--shard-index",
            "0",
            "--feature-batch-size",
            str(args.feature_batch_size),
            "--no-faiss",
        ]

        out_f = out_log.open("w", encoding="utf-8")
        err_f = err_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out_f, stderr=err_f, env=env)
        (log_root / f"shard_{shard_idx}.pid").write_text(str(proc.pid), encoding="ascii")
        log(log_root, f"started shard {shard_idx}/{args.num_shards} pid={proc.pid} device={shard_device}")
        return shard_idx, proc, out_f, err_f

    log(
        log_root,
        f"starting {args.num_shards} shard processes on devices={','.join(device_values)} max_workers={max_workers}",
    )
    while next_shard_idx < args.num_shards and len(processes) < max_workers:
        processes.append(start_shard(next_shard_idx))
        next_shard_idx += 1

    while processes:
        still_running = []
        for shard_idx, proc, out_f, err_f in processes:
            code = proc.poll()
            if code is None:
                still_running.append((shard_idx, proc, out_f, err_f))
                continue
            out_f.close()
            err_f.close()
            log(log_root, f"shard {shard_idx} finished exit_code={code}")
            if code != 0:
                failed.append((shard_idx, code))
            if next_shard_idx < args.num_shards and not failed:
                still_running.append(start_shard(next_shard_idx))
                next_shard_idx += 1
        processes = still_running
        if processes:
            time.sleep(5)

    if failed:
        log(log_root, f"not merging because shard failures occurred: {failed}")
        return 1

    shard_dirs = [str(shard_root / f"shard_{idx}") for idx in range(args.num_shards)]
    merge_cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "merge_yolo_feature_shards.py"),
        "--shard-dirs",
        *shard_dirs,
        "--output-dir",
        str(index_dir),
        "--faiss-type",
        args.faiss_type,
        "--nlist",
        str(args.nlist),
        "--nprobe",
        str(args.nprobe),
        "--pq-m",
        str(args.pq_m),
        "--pq-nbits",
        str(args.pq_nbits),
        "--train-size",
        str(args.train_size),
        "--rerank-k",
        str(args.rerank_k),
    ]
    if args.faiss_gpu:
        merge_cmd.append("--faiss-gpu")
        merge_cmd.extend(["--faiss-gpu-device", str(args.faiss_gpu_device)])
    if args.faiss_gpu_required:
        merge_cmd.append("--faiss-gpu-required")
    log(log_root, "starting merge")
    with (log_root / "merge.out.log").open("w", encoding="utf-8") as out_f, (
        log_root / "merge.err.log"
    ).open("w", encoding="utf-8") as err_f:
        code = subprocess.call(merge_cmd, cwd=str(ROOT), stdout=out_f, stderr=err_f, env=env)
    log(log_root, f"merge finished exit_code={code}")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
