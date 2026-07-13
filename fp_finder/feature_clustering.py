from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", message=r"\s*Found Intel OpenMP.*", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .yolo_dataset import CropRecord, index_records_ready, open_record_store


SIZE_BUCKET_ORDER = ["tiny", "small", "medium", "large", "huge"]
SIZE_BUCKET_LABELS = {
    "tiny": "tiny <0.5%",
    "small": "small 0.5-2%",
    "medium": "medium 2-8%",
    "large": "large 8-20%",
    "huge": "huge >=20%",
}


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


def _fit_kmeans(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if features.shape[0] <= 1:
        return np.zeros(features.shape[0], dtype=np.int32)
    k = max(1, min(int(n_clusters), int(features.shape[0])))
    if k == 1:
        return np.zeros(features.shape[0], dtype=np.int32)
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(seed),
        batch_size=min(4096, max(256, features.shape[0])),
        n_init=3,
        max_iter=80,
    )
    return kmeans.fit_predict(features).astype(np.int32)


def _cluster_by_scope(
    features: np.ndarray,
    rows: List[Dict],
    n_clusters: int,
    seed: int,
    scope: str,
) -> Tuple[List[str], List[int], int]:
    scope = str(scope or "global")
    if scope == "global":
        labels = _fit_kmeans(features, n_clusters=n_clusters, seed=seed)
        return [f"G{int(label):02d}" for label in labels], [int(v) for v in labels], int(labels.max() + 1 if labels.size else 0)

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

    for group_name in sorted(groups):
        positions = groups[group_name]
        local_features = features[np.asarray(positions, dtype=np.int64)]
        local_labels = _fit_kmeans(local_features, n_clusters=n_clusters, seed=seed)
        for position, label in zip(positions, local_labels):
            labels_text[position] = f"{group_name} / C{int(label):02d}"
            labels_num[position] = next_cluster_id + int(label)
        next_cluster_id += int(local_labels.max() + 1 if local_labels.size else 1)

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
) -> Dict:
    root = Path(index_dir)
    features_path = root / "features.npy"
    records_path = root / "records.json"
    config_path = root / "config.json"
    if not features_path.exists() or not index_records_ready(root) or not config_path.exists():
        raise FileNotFoundError(f"Missing features.npy/records metadata/config.json in {root}")

    features = np.load(str(features_path), mmap_mode="r")
    records = open_record_store(root)
    total = min(len(records), int(features.shape[0]))
    if total <= 0:
        return _empty_result(total, "No records/features available for clustering.", n_clusters)

    class_filter = class_filter if class_filter is not None else class_name
    size_bucket = str(size_bucket or "").strip()
    if size_bucket.lower() == "all":
        size_bucket = ""

    candidate_indices = []
    record_stats: Dict[int, Dict] = {}
    for idx in range(total):
        record = records[idx]
        stats = bbox_stats(record)
        record_stats[idx] = stats
        if not record_matches_class(record, class_filter):
            continue
        if size_bucket and stats["size_bucket"] != size_bucket:
            continue
        candidate_indices.append(idx)

    if not candidate_indices:
        return _empty_result(
            total,
            f"No records found for class='{class_filter or 'all'}', size='{size_bucket or 'all'}'.",
            n_clusters,
        )

    candidate_indices_arr = np.asarray(candidate_indices, dtype=np.int64)
    sample_indices = _sample_indices(candidate_indices_arr, max_points=max_points, seed=seed)
    sample_features = np.asarray(features[sample_indices], dtype=np.float32)
    sample_size = int(sample_features.shape[0])
    if sample_size <= 0:
        return _empty_result(total, "No sampled records available for clustering.", n_clusters)

    coords, explained = _pca_3d(sample_features, seed=seed)

    rows: List[Dict] = []
    for point_id, (record_idx, coord) in enumerate(zip(sample_indices, coords)):
        record = records[int(record_idx)]
        stats = record_stats[int(record_idx)]
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

    cluster_labels, cluster_nums, actual_clusters = _cluster_by_scope(
        sample_features,
        rows,
        n_clusters=n_clusters,
        seed=seed,
        scope=clustering_scope,
    )
    for row, label, cluster_num in zip(rows, cluster_labels, cluster_nums):
        row["cluster_label"] = label
        row["cluster"] = int(cluster_num)

    df = pd.DataFrame(rows)
    summary = summarize_clusters(df)
    return {
        "df": df,
        "summary": summary,
        "total_records": int(total),
        "sample_size": int(sample_size),
        "n_clusters": int(actual_clusters),
        "explained_variance_ratio": explained,
        "message": "",
    }
