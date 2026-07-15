from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", message=r"\s*Found Intel OpenMP.*", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.cluster import Birch, BisectingKMeans, HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

from .yolo_dataset import CropRecord, index_record_files, index_records_ready, open_record_store


ProgressCallback = Optional[Callable[[int, int, str], None]]
warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"sklearn\.cluster\._birch")


SIZE_BUCKET_ORDER = ["tiny", "small", "medium", "large", "huge"]
SIZE_BUCKET_LABELS = {
    "tiny": "tiny <0.5%",
    "small": "small 0.5-2%",
    "medium": "medium 2-8%",
    "large": "large 8-20%",
    "huge": "huge >=20%",
}
SIZE_BUCKET_TO_CODE = {name: idx for idx, name in enumerate(SIZE_BUCKET_ORDER)}
SIZE_CODE_TO_BUCKET = {idx: name for name, idx in SIZE_BUCKET_TO_CODE.items()}


def load_feature_index_metadata(index_dir: str) -> Dict:
    root = Path(index_dir)
    with (root / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def bbox_stats(record: CropRecord) -> Dict:
    x1, y1, x2, y2 = record.bbox_xyxy
    box_w = max(1, int(x2) - int(x1))
    box_h = max(1, int(y2) - int(y1))
    image_w = max(1, int(record.image_width or 0), int(x2), box_w)
    image_h = max(1, int(record.image_height or 0), int(y2), box_h)
    area_ratio = float((box_w * box_h) / max(1, image_w * image_h))
    aspect_ratio = float(box_w / max(1, box_h))
    return {
        "bbox_width": box_w,
        "bbox_height": box_h,
        "bbox_area": int(box_w * box_h),
        "image_width": image_w,
        "image_height": image_h,
        "area_ratio": area_ratio,
        "size_bucket": size_bucket_from_area_ratio(area_ratio),
        "aspect_ratio": aspect_ratio,
        "aspect_bucket": aspect_bucket_from_ratio(aspect_ratio),
    }


def size_bucket_from_area_ratio(area_ratio: float) -> str:
    if area_ratio < 0.005:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.08:
        return "medium"
    if area_ratio < 0.20:
        return "large"
    return "huge"


def size_bucket_code(size_bucket: str) -> int:
    return int(SIZE_BUCKET_TO_CODE.get(str(size_bucket), -1))


def class_id_filter_value(class_filter: Optional[str]) -> Optional[int]:
    value = str(class_filter or "").strip()
    if not value or value.lower() == "all":
        return None
    if ":" in value:
        value = value.split(":", 1)[0].strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return None


def bbox_stats_from_dict(item: Dict) -> Dict:
    x1, y1, x2, y2 = [int(value) for value in item.get("bbox_xyxy", [0, 0, 1, 1])]
    box_w = max(1, int(x2) - int(x1))
    box_h = max(1, int(y2) - int(y1))
    image_w = max(1, int(item.get("image_width", 0) or 0), int(x2), box_w)
    image_h = max(1, int(item.get("image_height", 0) or 0), int(y2), box_h)
    area_ratio = float((box_w * box_h) / max(1, image_w * image_h))
    return {"size_bucket": size_bucket_from_area_ratio(area_ratio)}


def aspect_bucket_from_ratio(aspect_ratio: float) -> str:
    if aspect_ratio >= 1.8:
        return "wide"
    if aspect_ratio <= 0.56:
        return "tall"
    return "balanced"


def record_matches_class(record: CropRecord, class_filter: Optional[str]) -> bool:
    value = str(class_filter or "").strip()
    if not value or value.lower() == "all":
        return True
    if ":" in value:
        value = value.split(":", 1)[0].strip()
    if value.lstrip("-").isdigit():
        return int(record.class_id) == int(value)
    return record.class_name.lower() == value.lower()


def load_cluster_metadata(index_dir: str) -> Dict:
    root = Path(index_dir)
    records = open_record_store(root)
    class_counts: Dict[str, int] = {}
    size_counts: Dict[str, int] = {key: 0 for key in SIZE_BUCKET_ORDER}
    for record in records:
        class_key = f"{record.class_id}: {record.class_name}"
        class_counts[class_key] = class_counts.get(class_key, 0) + 1
        stats = bbox_stats(record)
        size_counts[stats["size_bucket"]] = size_counts.get(stats["size_bucket"], 0) + 1
    return {
        "total_records": len(records),
        "class_counts": class_counts,
        "size_counts": size_counts,
        "size_bucket_order": SIZE_BUCKET_ORDER,
        "size_bucket_labels": SIZE_BUCKET_LABELS,
    }


def _record_source(index_dir: Path) -> Tuple[Optional[Path], str]:
    records_json, records_jsonl, _offsets = index_record_files(index_dir)
    if records_jsonl is not None:
        return records_jsonl, "jsonl"
    if records_json is not None:
        return records_json, "json"
    return None, ""


def _meta_cache_valid(cache_path: Path, source_path: Path, total: int) -> bool:
    if not cache_path.exists() or not source_path.exists():
        return False
    try:
        source_stat = source_path.stat()
        with np.load(str(cache_path), allow_pickle=False) as data:
            return (
                int(data["total"]) == int(total)
                and int(data["source_size"]) == int(source_stat.st_size)
                and int(data["source_mtime_ns"]) == int(source_stat.st_mtime_ns)
                and int(data["class_ids"].shape[0]) >= int(total)
                and int(data["size_codes"].shape[0]) >= int(total)
            )
    except Exception:
        return False


def load_or_build_record_meta_arrays(
    index_dir: str,
    records,
    total: int,
    progress: ProgressCallback = None,
) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(index_dir)
    cache_path = root / "record_meta_cache.npz"
    source_path, source_type = _record_source(root)
    if source_path is None:
        raise FileNotFoundError(f"Missing records metadata in {root}")

    if _meta_cache_valid(cache_path, source_path, total):
        if progress:
            progress(1, 1, "Loading record filter cache")
        with np.load(str(cache_path), allow_pickle=False) as data:
            return np.asarray(data["class_ids"][:total], dtype=np.int32), np.asarray(data["size_codes"][:total], dtype=np.int16)

    if progress:
        progress(0, total, "Building record filter cache")
    class_ids = np.empty(total, dtype=np.int32)
    size_codes = np.empty(total, dtype=np.int16)

    if source_type == "jsonl":
        with source_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= total:
                    break
                item = json.loads(line)
                class_ids[idx] = int(item.get("class_id", -1))
                size_codes[idx] = size_bucket_code(bbox_stats_from_dict(item)["size_bucket"])
                if progress and (idx == 0 or (idx + 1) % 50000 == 0 or idx + 1 == total):
                    progress(idx + 1, total, f"Building record filter cache {idx + 1:,}/{total:,}")
    else:
        for idx in range(total):
            record = records[idx]
            class_ids[idx] = int(record.class_id)
            size_codes[idx] = size_bucket_code(bbox_stats(record)["size_bucket"])
            if progress and (idx == 0 or (idx + 1) % 50000 == 0 or idx + 1 == total):
                progress(idx + 1, total, f"Building record filter cache {idx + 1:,}/{total:,}")

    source_stat = source_path.stat()
    np.savez_compressed(
        str(cache_path),
        class_ids=class_ids,
        size_codes=size_codes,
        total=np.asarray(total, dtype=np.int64),
        source_size=np.asarray(source_stat.st_size, dtype=np.int64),
        source_mtime_ns=np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
    )
    if progress:
        progress(total, total, "Record filter cache ready")
    return class_ids, size_codes


def sample_filtered_indices_fast(
    records,
    total: int,
    max_points: int,
    seed: int,
    class_filter: Optional[str],
    size_bucket: str,
    progress: ProgressCallback = None,
) -> np.ndarray:
    if max_points <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    target = min(int(max_points), int(total))
    max_attempts = min(int(total), max(target * 80, target + 5000))
    if max_attempts <= 0:
        return np.empty(0, dtype=np.int64)
    probe_indices = rng.choice(np.arange(total, dtype=np.int64), size=max_attempts, replace=False)
    selected: List[int] = []
    for pos, idx in enumerate(probe_indices, start=1):
        record = records[int(idx)]
        stats = bbox_stats(record)
        if progress and (pos == 1 or pos % 2000 == 0 or pos == max_attempts):
            progress(pos, max_attempts, f"Fast filtering sampled records; selected={len(selected):,}/{target:,}")
        if not record_matches_class(record, class_filter):
            continue
        if size_bucket and stats["size_bucket"] != size_bucket:
            continue
        selected.append(int(idx))
        if len(selected) >= target:
            break
    if progress:
        progress(min(pos, max_attempts), max_attempts, f"Fast filtering complete; selected={len(selected):,}/{target:,}")
    return np.sort(np.asarray(selected, dtype=np.int64))


def _empty_result(
    total: int,
    message: str,
    n_clusters: int,
    explained: Optional[Sequence[float]] = None,
) -> Dict:
    return {
        "df": pd.DataFrame(),
        "summary": pd.DataFrame(),
        "total_records": int(total),
        "sample_size": 0,
        "n_clusters": int(n_clusters),
        "explained_variance_ratio": list(explained or [0.0, 0.0, 0.0]),
        "message": message,
    }


def _sample_indices(indices: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    sample_size = min(int(max_points), int(indices.size))
    if sample_size < indices.size:
        return np.sort(rng.choice(indices, size=sample_size, replace=False))
    return indices


def _pca_3d(features: np.ndarray, seed: int) -> Tuple[np.ndarray, List[float]]:
    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 3), dtype=np.float32), [0.0, 0.0, 0.0]

    components = min(3, features.shape[0], features.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=int(seed))
    reduced = pca.fit_transform(features)
    coords = np.zeros((features.shape[0], 3), dtype=np.float32)
    coords[:, :components] = reduced[:, :components]
    explained = [float(v) for v in pca.explained_variance_ratio_]
    while len(explained) < 3:
        explained.append(0.0)
    return coords, explained[:3]


def _fit_cluster(features: np.ndarray, n_clusters: int, seed: int, method: str = "minibatch_kmeans") -> np.ndarray:
    if features.shape[0] <= 1:
        return np.zeros(features.shape[0], dtype=np.int32)
    k = max(1, min(int(n_clusters), int(features.shape[0])))
    if k == 1:
        return np.zeros(features.shape[0], dtype=np.int32)
    method = str(method or "minibatch_kmeans").lower()
    if method == "bisecting_kmeans":
        model = BisectingKMeans(n_clusters=k, random_state=int(seed), n_init=3)
        return model.fit_predict(features).astype(np.int32)
    if method == "birch":
        model = Birch(n_clusters=k, threshold=0.5)
        return model.fit_predict(features).astype(np.int32)
    if method == "hdbscan":
        min_cluster_size = max(5, min(100, int(features.shape[0]) // max(2, k)))
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(2, min_cluster_size // 2))
        return model.fit_predict(features).astype(np.int32)

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(seed),
        batch_size=min(4096, max(256, features.shape[0])),
        n_init=3,
        max_iter=80,
    )
    return model.fit_predict(features).astype(np.int32)


def _label_maps(labels: np.ndarray) -> Tuple[Dict[int, int], List[int]]:
    unique_labels = sorted(int(value) for value in np.unique(labels).tolist())
    return {label: pos for pos, label in enumerate(unique_labels)}, unique_labels


def _cluster_label_text(prefix: str, label: int) -> str:
    label = int(label)
    if label < 0:
        return f"{prefix}noise"
    return f"{prefix}C{label:02d}"


def _cluster_by_scope(
    features: np.ndarray,
    rows: List[Dict],
    n_clusters: int,
    seed: int,
    scope: str,
    method: str = "minibatch_kmeans",
    progress: ProgressCallback = None,
) -> Tuple[List[str], List[int], int]:
    scope = str(scope or "global")
    if scope == "global":
        if progress:
            progress(0, 1, f"Clustering sampled features with {method}")
        labels = _fit_cluster(features, n_clusters=n_clusters, seed=seed, method=method)
        label_map, unique_labels = _label_maps(labels)
        if progress:
            progress(1, 1, f"Clustered sampled features with {method}")
        return [
            "noise" if int(label) < 0 else f"G{int(label):02d}"
            for label in labels
        ], [int(label_map[int(v)]) for v in labels], int(len(unique_labels))

    if scope == "per_class":
        group_key = lambda row: str(row["class_name"])
    elif scope == "class_size":
        group_key = lambda row: f"{row['class_name']} / {row['size_bucket']}"
    else:
        group_key = lambda row: str(row["class_name"])

    labels_text = [""] * len(rows)
    labels_num = [-1] * len(rows)
    next_cluster_id = 0
    groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(group_key(row), []).append(idx)

    group_names = sorted(groups)
    for group_pos, group_name in enumerate(group_names, start=1):
        positions = groups[group_name]
        local_features = features[np.asarray(positions, dtype=np.int64)]
        if progress:
            progress(group_pos - 1, len(group_names), f"Clustering group {group_pos}/{len(group_names)}: {group_name}")
        local_labels = _fit_cluster(local_features, n_clusters=n_clusters, seed=seed, method=method)
        label_map, unique_labels = _label_maps(local_labels)
        for position, label in zip(positions, local_labels):
            labels_text[position] = f"{group_name} / {_cluster_label_text('', int(label))}"
            labels_num[position] = next_cluster_id + int(label_map[int(label)])
        next_cluster_id += max(1, len(unique_labels))
    if progress:
        progress(len(group_names), len(group_names), f"Clustered {len(group_names)} groups with {method}")

    return labels_text, labels_num, next_cluster_id


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for cluster_label, group in df.groupby("cluster_label"):
        class_counts = group["class_name"].value_counts()
        size_counts = group["size_bucket"].value_counts()
        dominant_class = str(class_counts.index[0]) if not class_counts.empty else ""
        dominant_size = str(size_counts.index[0]) if not size_counts.empty else ""
        count = int(len(group))
        rows.append(
            {
                "cluster_label": cluster_label,
                "count": count,
                "n_classes": int(class_counts.shape[0]),
                "dominant_class": dominant_class,
                "class_purity": round(float(class_counts.iloc[0] / count), 4) if count else 0.0,
                "class_mix": ", ".join(f"{idx}:{int(val)}" for idx, val in class_counts.head(5).items()),
                "n_size_buckets": int(size_counts.shape[0]),
                "dominant_size": dominant_size,
                "size_purity": round(float(size_counts.iloc[0] / count), 4) if count else 0.0,
                "size_mix": ", ".join(f"{idx}:{int(val)}" for idx, val in size_counts.head(5).items()),
                "avg_area_pct": round(float(group["area_ratio"].mean() * 100.0), 3),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["class_purity", "n_classes", "size_purity", "count"],
        ascending=[True, False, True, False],
    )


def build_feature_clusters(
    index_dir: str,
    max_points: int = 10000,
    n_clusters: int = 24,
    seed: int = 42,
    class_name: Optional[str] = None,
    class_filter: Optional[str] = None,
    size_bucket: Optional[str] = None,
    clustering_scope: str = "global",
    clustering_method: str = "minibatch_kmeans",
    progress: ProgressCallback = None,
) -> Dict:
    root = Path(index_dir)
    features_path = root / "features.npy"
    records_path = root / "records.json"
    config_path = root / "config.json"
    if not features_path.exists() or not index_records_ready(root) or not config_path.exists():
        raise FileNotFoundError(f"Missing features.npy/records metadata/config.json in {root}")

    if progress:
        progress(0, 1, "Loading feature matrix and record metadata")
    features = np.load(str(features_path), mmap_mode="r")
    records = open_record_store(root)
    total = min(len(records), int(features.shape[0]))
    if total <= 0:
        return _empty_result(total, "No records/features available for clustering.", n_clusters)
    if progress:
        progress(1, 1, f"Loaded records={total:,}, feature_dim={int(features.shape[1])}")

    class_filter = class_filter if class_filter is not None else class_name
    size_bucket = str(size_bucket or "").strip()
    if size_bucket.lower() == "all":
        size_bucket = ""

    record_stats: Dict[int, Dict] = {}
    class_filter_text = str(class_filter or "").strip()
    filter_required = bool(class_filter_text and class_filter_text.lower() != "all") or bool(size_bucket)
    if filter_required:
        candidate_indices_arr = sample_filtered_indices_fast(
            records,
            total=total,
            max_points=int(max_points),
            seed=int(seed),
            class_filter=class_filter,
            size_bucket=size_bucket,
            progress=progress,
        )
        if int(candidate_indices_arr.size) > 0:
            if progress:
                progress(
                    int(candidate_indices_arr.size),
                    min(int(max_points), total),
                    f"Using fast filtered sample; candidates={int(candidate_indices_arr.size):,}",
                )
        else:
            if progress:
                progress(0, 1, "Fast filtering found no candidates; falling back to exact filter cache")

    exact_target = min(int(max_points), int(total))
    if filter_required and int(candidate_indices_arr.size) < exact_target:
        class_id_filter = class_id_filter_value(class_filter)
        if class_filter_text and class_id_filter is None:
            candidate_indices = []
            for idx in range(total):
                record = records[idx]
                stats = bbox_stats(record)
                record_stats[idx] = stats
                if not record_matches_class(record, class_filter):
                    continue
                if size_bucket and stats["size_bucket"] != size_bucket:
                    continue
                candidate_indices.append(idx)
                if progress and (idx == 0 or (idx + 1) % 10000 == 0 or idx + 1 == total):
                    progress(idx + 1, total, f"Filtering records; candidates={len(candidate_indices):,}")
            candidate_indices_arr = np.asarray(candidate_indices, dtype=np.int64)
        else:
            class_ids, size_codes = load_or_build_record_meta_arrays(str(root), records, total, progress=progress)
            mask = np.ones(total, dtype=bool)
            if class_id_filter is not None:
                mask &= class_ids[:total] == int(class_id_filter)
            if size_bucket:
                mask &= size_codes[:total] == size_bucket_code(size_bucket)
            candidate_indices_arr = np.flatnonzero(mask).astype(np.int64)
            if progress:
                progress(total, total, f"Filtered records with cache; candidates={int(candidate_indices_arr.size):,}")
    elif not filter_required:
        candidate_indices_arr = np.arange(total, dtype=np.int64)
        if progress:
            progress(total, total, "No class/size filter; using all record indices without full metadata scan")

    if int(candidate_indices_arr.size) == 0:
        return _empty_result(
            total,
            f"No records found for class='{class_filter or 'all'}', size='{size_bucket or 'all'}'.",
            n_clusters,
        )

    if progress:
        progress(0, 1, f"Sampling up to {int(max_points):,} points from {int(candidate_indices_arr.size):,} candidates")
    sample_indices = _sample_indices(candidate_indices_arr, max_points=max_points, seed=seed)
    if progress:
        progress(1, 1, f"Sampled {int(sample_indices.size):,} points")

    if progress:
        progress(0, 1, "Loading sampled feature vectors")
    sample_features = np.asarray(features[sample_indices], dtype=np.float32)
    sample_size = int(sample_features.shape[0])
    if sample_size <= 0:
        return _empty_result(total, "No sampled records available for clustering.", n_clusters)
    if progress:
        progress(1, 1, f"Loaded sampled feature matrix {sample_features.shape}")

    if progress:
        progress(0, 1, "Running PCA projection")
    coords, explained = _pca_3d(sample_features, seed=seed)
    if progress:
        progress(1, 1, f"PCA projection complete; explained={sum(explained) * 100:.1f}%")

    rows: List[Dict] = []
    for point_id, (record_idx, coord) in enumerate(zip(sample_indices, coords)):
        record = records[int(record_idx)]
        stats = record_stats.get(int(record_idx))
        if stats is None:
            stats = bbox_stats(record)
            record_stats[int(record_idx)] = stats
        rows.append(
            {
                "point_id": int(point_id),
                "record_id": int(record.record_id),
                "record_idx": int(record_idx),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
                "class_id": int(record.class_id),
                "class_name": record.class_name,
                "image_path": record.image_path,
                "label_path": record.label_path,
                "bbox_xyxy": list(record.bbox_xyxy),
                "annotation_line": int(record.annotation_line),
                **stats,
            }
        )
        if progress and (point_id == 0 or (point_id + 1) % 1000 == 0 or point_id + 1 == sample_size):
            progress(point_id + 1, sample_size, f"Preparing point metadata {point_id + 1:,}/{sample_size:,}")

    cluster_labels, cluster_nums, actual_clusters = _cluster_by_scope(
        sample_features,
        rows,
        n_clusters=n_clusters,
        seed=seed,
        scope=clustering_scope,
        method=clustering_method,
        progress=progress,
    )
    for row, label, cluster_num in zip(rows, cluster_labels, cluster_nums):
        row["cluster_label"] = label
        row["cluster"] = int(cluster_num)

    if progress:
        progress(0, 1, "Building cluster summary tables")
    df = pd.DataFrame(rows)
    summary = summarize_clusters(df)
    if progress:
        progress(1, 1, "Cluster summary complete")
    return {
        "df": df,
        "summary": summary,
        "total_records": int(total),
        "sample_size": int(sample_size),
        "n_clusters": int(actual_clusters),
        "clustering_method": str(clustering_method),
        "explained_variance_ratio": explained,
        "message": "",
    }
