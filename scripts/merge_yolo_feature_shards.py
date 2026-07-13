from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fp_finder.yolo_dataset import record_to_dict, records_from_json, records_to_json


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
    parser = argparse.ArgumentParser(description="Merge YOLO feature shard directories into one FAISS index.")
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--faiss-type", default="ivfpq", choices=["flat", "ivfpq"])
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--pq-m", type=int, default=64)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=200000)
    parser.add_argument("--rerank-k", type=int, default=500)
    parser.add_argument("--add-batch-size", type=int, default=50000)
    parser.add_argument("--faiss-gpu", action="store_true", help="Use FAISS GPU for index train/add when faiss-gpu is available.")
    parser.add_argument("--faiss-gpu-device", type=int, default=0)
    parser.add_argument("--faiss-gpu-required", action="store_true", help="Fail instead of falling back to CPU when FAISS GPU is unavailable.")
    parser.add_argument("--write-records-json", action="store_true", help="Also write legacy records.json. Not recommended for very large DBs.")
    return parser.parse_args()


def compatible_pq_m(dim: int, requested: int) -> int:
    for value in range(min(int(requested), int(dim)), 0, -1):
        if dim % value == 0:
            return value
    return 1


def comparable_config_value(key: str, value):
    if key in {"repo_path", "weights_path"}:
        return str(value or "").replace("\\", "/").rstrip("/")
    return value


def faiss_gpu_status() -> tuple[bool, str]:
    required_names = ("StandardGpuResources", "index_cpu_to_gpu", "index_gpu_to_cpu", "get_num_gpus")
    missing = [name for name in required_names if not hasattr(faiss, name)]
    if missing:
        return False, f"faiss gpu api missing: {', '.join(missing)}"
    try:
        num_gpus = int(faiss.get_num_gpus())
    except Exception as exc:
        return False, f"faiss get_num_gpus failed: {exc}"
    if num_gpus <= 0:
        return False, "faiss reports 0 gpu"
    return True, f"faiss reports {num_gpus} gpu(s)"


def maybe_gpu_index(index, args: argparse.Namespace):
    if not bool(args.faiss_gpu):
        return index, {"faiss_gpu_requested": False, "faiss_gpu_used": False}, None

    available, reason = faiss_gpu_status()
    if not available:
        message = f"FAISS GPU requested but unavailable; using CPU. reason={reason}"
        if bool(args.faiss_gpu_required):
            raise RuntimeError(message)
        print(message, flush=True)
        return index, {
            "faiss_gpu_requested": True,
            "faiss_gpu_used": False,
            "faiss_gpu_reason": reason,
        }, None

    resources = faiss.StandardGpuResources()
    try:
        gpu_index = faiss.index_cpu_to_gpu(resources, int(args.faiss_gpu_device), index)
    except Exception as exc:
        message = f"FAISS GPU clone failed; using CPU. reason={exc}"
        if bool(args.faiss_gpu_required):
            raise RuntimeError(message) from exc
        print(message, flush=True)
        return index, {
            "faiss_gpu_requested": True,
            "faiss_gpu_used": False,
            "faiss_gpu_reason": str(exc),
        }, None

    print(f"FAISS GPU enabled device={int(args.faiss_gpu_device)}", flush=True)
    return gpu_index, {
        "faiss_gpu_requested": True,
        "faiss_gpu_used": True,
        "faiss_gpu_device": int(args.faiss_gpu_device),
        "faiss_gpu_reason": reason,
    }, resources


def index_to_cpu(index, gpu_meta: dict):
    if not gpu_meta.get("faiss_gpu_used"):
        return index
    return faiss.index_gpu_to_cpu(index)


def set_index_nprobe(index, nprobe: int) -> None:
    try:
        index.nprobe = int(nprobe)
        return
    except Exception:
        pass
    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", int(nprobe))
    except Exception:
        pass


def build_faiss_index(features_path: Path, total: int, dim: int, args: argparse.Namespace, start_time: float):
    features = np.load(str(features_path), mmap_mode="r")
    if args.faiss_type == "flat" or total < 10000:
        index, gpu_meta, _resources = maybe_gpu_index(faiss.IndexFlatIP(dim), args)
        for start in range(0, total, int(args.add_batch_size)):
            end = min(total, start + int(args.add_batch_size))
            index.add(np.asarray(features[start:end], dtype=np.float32))
            print_progress(end, total, f"Building FAISS index {end}/{total}", start_time)
        return index_to_cpu(index, gpu_meta), {"faiss_type": "flat", "rerank_k": 0, **gpu_meta}

    train_size = min(int(args.train_size), int(total))
    sample_indices = np.linspace(0, total - 1, num=train_size, dtype=np.int64)
    train = np.asarray(features[sample_indices], dtype=np.float32)
    nlist = min(int(args.nlist), max(1, int(train.shape[0]) // 40))
    nlist = max(1, nlist)
    pq_m = compatible_pq_m(dim, int(args.pq_m))
    quantizer = faiss.IndexFlatIP(dim)
    index, gpu_meta, _resources = maybe_gpu_index(
        faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, int(args.pq_nbits), faiss.METRIC_INNER_PRODUCT),
        args,
    )
    print(
        f"training IVFPQ train={train.shape} nlist={nlist} pq_m={pq_m} pq_nbits={args.pq_nbits}",
        flush=True,
    )
    index.train(train)
    actual_nprobe = min(int(args.nprobe), nlist)
    set_index_nprobe(index, actual_nprobe)
    for start in range(0, total, int(args.add_batch_size)):
        end = min(total, start + int(args.add_batch_size))
        index.add(np.asarray(features[start:end], dtype=np.float32))
        print_progress(end, total, f"Building FAISS index {end}/{total}", start_time)
    return index_to_cpu(index, gpu_meta), {
        "faiss_type": "ivfpq",
        "nlist": int(nlist),
        "nprobe": int(actual_nprobe),
        "pq_m": int(pq_m),
        "pq_nbits": int(args.pq_nbits),
        "train_size": int(train.shape[0]),
        "rerank_k": int(args.rerank_k),
        **gpu_meta,
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    shard_infos = []
    base_config = None
    total = 0
    dim = 0

    for shard_dir in args.shard_dirs:
        root = Path(shard_dir)
        features_path = root / "features.npy"
        records_path = root / "records.json"
        config_path = root / "config.json"
        if not features_path.exists() or not records_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Shard is missing required files: {root}")

        features = np.load(str(features_path), mmap_mode="r")
        records = records_from_json(str(records_path))
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        if base_config is None:
            base_config = config
        else:
            for key in ("repo_path", "weights_path", "img_size", "layer_indices", "dim"):
                if comparable_config_value(key, base_config.get(key)) != comparable_config_value(key, config.get(key)):
                    raise ValueError(f"Shard config mismatch for {key}: {root}")

        if int(features.shape[0]) != len(records):
            raise ValueError(f"Feature/record count mismatch in {root}: features={features.shape[0]} records={len(records)}")
        dim = int(features.shape[1])
        shard_infos.append((root, features_path, records_path, int(features.shape[0])))
        total += int(features.shape[0])
        print(f"scanned {root}: features={features.shape} records={len(records)}", flush=True)

    if total <= 0 or dim <= 0:
        raise ValueError("No shard features to merge.")

    merged_features_path = output / "features.npy"
    merged_features = np.lib.format.open_memmap(
        str(merged_features_path),
        mode="w+",
        dtype="float32",
        shape=(int(total), int(dim)),
    )
    offsets = np.lib.format.open_memmap(
        str(output / "record_offsets.npy"),
        mode="w+",
        dtype="int64",
        shape=(int(total),),
    )
    records_jsonl = output / "records.jsonl"
    legacy_records = [] if args.write_records_json else None

    cursor = 0
    with records_jsonl.open("wb") as record_f:
        for root, features_path, records_path, count in shard_infos:
            features = np.load(str(features_path), mmap_mode="r")
            merged_features[cursor : cursor + count] = np.asarray(features, dtype=np.float32)
            records = records_from_json(str(records_path))
            for local_idx, record in enumerate(records):
                record.record_id = cursor + local_idx
                offsets[cursor + local_idx] = record_f.tell()
                line = json.dumps(record_to_dict(record), ensure_ascii=False, separators=(",", ":"))
                record_f.write(line.encode("utf-8"))
                record_f.write(b"\n")
                if legacy_records is not None:
                    legacy_records.append(record)
            print(f"merged {root}: {cursor + count}/{total}", flush=True)
            print_progress(cursor + count, total, f"Merging shard features {cursor + count}/{total}", start_time)
            cursor += count
    merged_features.flush()
    offsets.flush()
    if legacy_records is not None:
        records_to_json(legacy_records, str(output / "records.json"))

    merged_config = dict(base_config or {})
    merged_config["num_records"] = int(total)
    merged_config["dim"] = int(dim)
    merged_config["merged_from"] = [str(Path(p)) for p in args.shard_dirs]
    merged_config["records_format"] = "jsonl_offsets"
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(merged_config, f, ensure_ascii=False, indent=2)

    if not args.no_faiss:
        index, faiss_config = build_faiss_index(merged_features_path, int(total), int(dim), args, start_time)
        merged_config.update(faiss_config)
        with (output / "config.json").open("w", encoding="utf-8") as f:
            json.dump(merged_config, f, ensure_ascii=False, indent=2)
        faiss.write_index(index, str(output / "index.faiss"))

    print(f"done output={output} features=({total}, {dim})", flush=True)


if __name__ == "__main__":
    main()
