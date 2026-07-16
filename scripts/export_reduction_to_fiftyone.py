from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "artifacts" / "fiftyone_exports" / "fiftyone_export.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a similarity reduction plan to a native FiftyOne dataset.")
    parser.add_argument("--plan-dir", required=True, help="Reduction plan directory containing reduction_group_members.csv")
    parser.add_argument("--dataset-name", default="", help="FiftyOne dataset name. Defaults to plan folder name.")
    parser.add_argument("--output-dir", default="", help="Crop export directory. Defaults under artifacts/fiftyone_exports")
    parser.add_argument("--max-records", type=int, default=5000, help="0 exports all matching records")
    parser.add_argument(
        "--action-filter",
        default="All",
        choices=["All", "Drop candidates", "Representatives", "Protected keeps", "Other keeps"],
    )
    parser.add_argument("--include-embeddings", action="store_true", help="Attach YOLO feature vectors to samples")
    parser.add_argument("--compute-visualization", action="store_true", help="Compute FiftyOne Brain visualization")
    parser.add_argument("--visualization-method", default="pca", choices=["pca", "umap", "tsne"])
    parser.add_argument("--launch", action="store_true", help="Launch the FiftyOne App after export")
    parser.add_argument("--wait", action="store_true", help="Keep the process alive while the FiftyOne App is open")
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument("--address", default="localhost")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing FiftyOne dataset with the same name")
    return parser.parse_args()


def action_group(action: Any) -> str:
    text = str(action or "").upper()
    if text.startswith("DROP"):
        return "Drop candidates"
    if "REPRESENTATIVE" in text:
        return "Representatives"
    if "PROTECTED" in text:
        return "Protected keeps"
    return "Other keeps"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def parse_bbox(value: Any) -> List[int]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = [int(part) for part in re.findall(r"-?\d+", value)[:4]]
    else:
        parsed = list(value)
    if len(parsed) < 4:
        return [0, 0, 1, 1]
    return [int(float(v)) for v in parsed[:4]]


def load_summary(plan_dir: Path) -> Dict:
    path = plan_dir / "reduction_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def filtered_members(plan_dir: Path, action_filter: str, max_records: int) -> pd.DataFrame:
    path = plan_dir / "reduction_group_members.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    members = pd.read_csv(path)
    if members.empty:
        return members
    members["_action_group"] = members["action"].map(action_group) if "action" in members.columns else "Other keeps"
    if action_filter != "All":
        members = members[members["_action_group"].astype(str) == str(action_filter)].copy()
    if "similarity_to_primary" in members.columns:
        members["_sim"] = members["similarity_to_primary"].map(lambda value: safe_float(value, 0.0))
    else:
        members["_sim"] = 0.0
    if "record_idx" in members.columns:
        members["_record_idx_int"] = members["record_idx"].map(lambda value: safe_int(value, -1))
    else:
        members["_record_idx_int"] = range(len(members))
    members = members.sort_values(["_action_group", "_sim", "_record_idx_int"], ascending=[True, False, True])
    if max_records > 0:
        members = members.head(int(max_records)).copy()
    return members


def crop_path_for_row(output_dir: Path, row: pd.Series) -> Path:
    class_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(row.get("class_name", "class")))
    action = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(row.get("_action_group", "action")))
    group_id = safe_int(row.get("reduction_group_id", -1), -1)
    record_idx = safe_int(row.get("_record_idx_int", row.get("record_idx", -1)), -1)
    source_name = Path(str(row.get("image_path", row.get("file_name", "sample")))).stem
    filename = f"g{group_id}_r{record_idx}_{class_name}_{action}_{source_name}.jpg"
    return output_dir / action / class_name / filename


def write_crop(row: pd.Series, crop_path: Path) -> bool:
    if crop_path.exists():
        return True
    image_path = Path(str(row.get("image_path", "")))
    if not image_path.exists():
        return False
    bbox = parse_bbox(row.get("bbox_xyxy", [0, 0, 1, 1]))
    try:
        with Image.open(image_path) as image:
            crop = image.convert("RGB").crop(tuple(bbox))
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, quality=92)
        return True
    except Exception:
        return False


def feature_matrix_for_rows(rows: pd.DataFrame, summary: Dict) -> Optional[np.ndarray]:
    index_dir = str(summary.get("index_dir") or summary.get("reduction_config", {}).get("index_dir") or "")
    feature_path = ROOT / index_dir / "features.npy" if index_dir and not Path(index_dir).is_absolute() else Path(index_dir) / "features.npy"
    if not feature_path.exists():
        return None
    features = np.load(str(feature_path), mmap_mode="r")
    record_indices = rows["_record_idx_int"].astype(int).to_numpy()
    valid = (record_indices >= 0) & (record_indices < int(features.shape[0]))
    vectors = np.zeros((len(rows), int(features.shape[1])), dtype=np.float32)
    if valid.any():
        vectors[np.flatnonzero(valid)] = np.asarray(features[record_indices[valid]], dtype=np.float32)
    return vectors


def ensure_fiftyone():
    try:
        import fiftyone as fo
        import fiftyone.brain as fob
    except Exception as exc:
        raise RuntimeError(
            "FiftyOne import failed. Fix the FiftyOne environment before launching native UI. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return fo, fob


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except Exception:
        if sys.platform.startswith("win"):
            try:
                import ctypes

                process_query_limited_information = 0x1000
                still_active = 259
                handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
                if not handle:
                    return False
                code = ctypes.c_ulong()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                ctypes.windll.kernel32.CloseHandle(handle)
                return bool(ok) and int(code.value) == still_active
            except Exception:
                return False
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: Optional[int] = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            pid = safe_int(payload.get("pid", -1), -1)
            if process_is_running(pid):
                raise RuntimeError(
                    f"Another FiftyOne export/launcher is already running: pid={pid}. "
                    f"Wait for it to finish or remove stale lock only after confirming it is stopped: {self.path}"
                )
            try:
                self.path.unlink()
            except Exception:
                pass
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        self.fd = os.open(str(self.path), flags)
        payload = {"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "script": str(Path(__file__).name)}
        os.write(self.fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        os.close(self.fd)
        self.fd = None
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
        except Exception:
            pass
        try:
            self.path.unlink()
        except Exception:
            pass
        return False


def make_sample(fo, row: pd.Series, crop_path: Path, vector: Optional[np.ndarray] = None):
    sample = fo.Sample(filepath=str(crop_path))
    action = str(row.get("action", ""))
    action_group_value = str(row.get("_action_group", action_group(action)))
    bbox = parse_bbox(row.get("bbox_xyxy", [0, 0, 1, 1]))
    sample["action"] = action
    sample["action_group"] = action_group_value
    sample["class_id"] = safe_int(row.get("class_id", 0), 0)
    sample["class_name"] = str(row.get("class_name", ""))
    sample["reduction_group_id"] = safe_int(row.get("reduction_group_id", -1), -1)
    sample["record_idx"] = safe_int(row.get("_record_idx_int", row.get("record_idx", -1)), -1)
    sample["record_id"] = safe_int(row.get("record_id", -1), -1)
    sample["similarity_to_primary"] = safe_float(row.get("similarity_to_primary", 0.0), 0.0)
    sample["size_bucket"] = str(row.get("size_bucket", ""))
    sample["original_image_path"] = str(row.get("image_path", ""))
    sample["label_path"] = str(row.get("label_path", ""))
    sample["bbox_xyxy"] = json.dumps(bbox)
    sample["is_representative"] = action_group_value == "Representatives"
    sample["is_protected"] = action_group_value == "Protected keeps"
    sample.tags = [
        action_group_value.replace(" ", "_").lower(),
        str(row.get("class_name", "")).replace(" ", "_").lower(),
    ]
    if vector is not None:
        sample["yolo_feature"] = vector.astype(float).tolist()
    return sample


def build_dataset(args: argparse.Namespace):
    fo, fob = ensure_fiftyone()
    plan_dir = Path(args.plan_dir)
    if not plan_dir.is_absolute():
        plan_dir = ROOT / plan_dir
    plan_dir = plan_dir.resolve()
    summary = load_summary(plan_dir)
    dataset_name = args.dataset_name.strip() or f"reduction_{plan_dir.parent.name}_{plan_dir.name}"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "fiftyone_exports" / dataset_name / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = filtered_members(plan_dir, args.action_filter, int(args.max_records))
    if rows.empty:
        raise RuntimeError("No reduction rows matched the export filters")

    if fo.dataset_exists(dataset_name):
        if args.overwrite:
            fo.delete_dataset(dataset_name)
        else:
            dataset = fo.load_dataset(dataset_name)
            print(f"Loaded existing FiftyOne dataset: {dataset_name} ({len(dataset)} samples)", flush=True)
            return dataset

    vectors = feature_matrix_for_rows(rows, summary) if args.include_embeddings or args.compute_visualization else None
    samples = []
    failed = 0
    started = time.time()
    for pos, (_, row) in enumerate(rows.iterrows(), start=1):
        crop_path = crop_path_for_row(output_dir, row)
        if not write_crop(row, crop_path):
            failed += 1
            continue
        vector = vectors[pos - 1] if vectors is not None and args.include_embeddings else None
        samples.append(make_sample(fo, row, crop_path, vector=vector))
        if pos % 500 == 0 or pos == len(rows):
            elapsed = time.time() - started
            print(f"prepared={pos:,}/{len(rows):,} samples={len(samples):,} failed={failed:,} elapsed={elapsed:.1f}s", flush=True)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    dataset.info["source_plan_dir"] = str(plan_dir)
    dataset.info["action_filter"] = args.action_filter
    dataset.info["max_records"] = int(args.max_records)
    dataset.add_samples(samples)
    print(f"Created FiftyOne dataset: {dataset_name} samples={len(dataset):,} failed_crops={failed:,}", flush=True)

    if args.compute_visualization and vectors is not None and len(dataset) > 1:
        vectors_for_samples = vectors[: len(samples)]
        print(f"Computing FiftyOne visualization: method={args.visualization_method}", flush=True)
        try:
            fob.compute_visualization(
                dataset,
                embeddings=vectors_for_samples,
                brain_key="yolo_feature_viz",
                method=args.visualization_method,
            )
            print("Computed brain_key=yolo_feature_viz", flush=True)
        except Exception as exc:
            print(f"Visualization failed: {type(exc).__name__}: {exc}", flush=True)
    return dataset


def main() -> int:
    args = parse_args()
    with FileLock(LOCK_PATH):
        dataset = build_dataset(args)
        if args.launch:
            import fiftyone as fo

            session = fo.launch_app(dataset, address=args.address, port=int(args.port), auto=False)
            print(f"FiftyOne App: http://{args.address}:{int(args.port)} dataset={dataset.name}", flush=True)
            if args.wait:
                session.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
