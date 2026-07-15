from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import yaml

from .feature_clustering import bbox_stats, record_matches_class
from .projects import slugify
from .yolo_dataset import CropRecord, index_records_ready, open_record_store


ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass
class CurationReportConfig:
    index_dir: str
    output_dir: str
    max_query_records: int = 50000
    top_k: int = 50
    rerank_k: int = 200
    seed: int = 42
    class_filter: str = ""
    size_bucket: str = ""
    duplicate_threshold: float = 0.98
    cross_class_threshold: float = 0.90
    rare_group_max: int = 20
    boundary_per_group: int = 2
    batch_size: int = 256


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, value: int) -> int:
        value = int(value)
        if value not in self.parent:
            self.parent[value] = value
            return value
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)

    def groups(self) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = {}
        for value in list(self.parent):
            root = self.find(value)
            grouped.setdefault(root, []).append(value)
        return {root: sorted(values) for root, values in grouped.items() if len(values) > 1}


def record_size_bucket(record: CropRecord) -> str:
    return str(bbox_stats(record)["size_bucket"])


def record_to_row(record: CropRecord, record_idx: int) -> Dict:
    stats = bbox_stats(record)
    return {
        "record_idx": int(record_idx),
        "record_id": int(record.record_id),
        "class_id": int(record.class_id),
        "class_name": str(record.class_name),
        "size_bucket": str(stats["size_bucket"]),
        "area_pct": float(stats["area_ratio"] * 100.0),
        "bbox_width": int(stats["bbox_width"]),
        "bbox_height": int(stats["bbox_height"]),
        "aspect_bucket": str(stats["aspect_bucket"]),
        "image_path": str(record.image_path),
        "label_path": str(record.label_path),
        "file_name": Path(record.image_path).name,
        "bbox_xyxy": list(record.bbox_xyxy),
        "annotation_line": int(record.annotation_line),
    }


def _load_index(index_dir: Path):
    config_path = index_dir / "config.json"
    index_path = index_dir / "index.faiss"
    features_path = index_dir / "features.npy"
    if not index_path.exists() or not features_path.exists() or not config_path.exists() or not index_records_ready(index_dir):
        raise FileNotFoundError(f"Missing index.faiss/features.npy/config/records in {index_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    index = faiss.read_index(str(index_path))
    nprobe = int(config.get("nprobe", 0) or 0)
    if nprobe > 0:
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
        except Exception:
            pass
    features = np.load(str(features_path), mmap_mode="r")
    records = open_record_store(index_dir)
    total = min(len(records), int(features.shape[0]), int(getattr(index, "ntotal", features.shape[0]) or features.shape[0]))
    return config, index, features, records, int(total)


def _candidate_indices(
    records,
    total: int,
    max_query_records: int,
    seed: int,
    class_filter: str,
    size_bucket: str,
    progress: ProgressCallback = None,
) -> Tuple[np.ndarray, Dict[int, CropRecord]]:
    all_indices = np.arange(total, dtype=np.int64)
    max_query_records = int(max_query_records or 0)
    if max_query_records > 0 and max_query_records < total:
        rng = np.random.default_rng(int(seed))
        candidate_indices = np.sort(rng.choice(all_indices, size=max_query_records, replace=False))
    else:
        candidate_indices = all_indices

    selected: List[int] = []
    record_cache: Dict[int, CropRecord] = {}
    class_filter = str(class_filter or "").strip()
    size_bucket = str(size_bucket or "").strip()
    if size_bucket.lower() == "all":
        size_bucket = ""

    for pos, idx in enumerate(candidate_indices, start=1):
        record = records[int(idx)]
        if class_filter and not record_matches_class(record, class_filter):
            continue
        if size_bucket and record_size_bucket(record) != size_bucket:
            continue
        selected.append(int(idx))
        record_cache[int(idx)] = record
        if progress and (pos == 1 or pos % 10000 == 0 or pos == len(candidate_indices)):
            progress(pos, len(candidate_indices), f"Filtering records selected={len(selected):,}")

    return np.asarray(selected, dtype=np.int64), record_cache


def _exact_neighbors(
    index,
    features: np.ndarray,
    query_indices: np.ndarray,
    total: int,
    top_k: int,
    rerank_k: int,
    batch_size: int,
    progress: ProgressCallback = None,
) -> Iterable[Tuple[int, List[Tuple[int, float]]]]:
    top_k = max(1, int(top_k))
    search_k = min(int(total), max(top_k + 1, int(rerank_k or 0), top_k * 4))
    batch_size = max(1, int(batch_size or 1))
    for start in range(0, len(query_indices), batch_size):
        end = min(len(query_indices), start + batch_size)
        batch_indices = query_indices[start:end]
        query_vectors = np.asarray(features[batch_indices], dtype=np.float32)
        _scores, candidate_matrix = index.search(query_vectors, search_k)
        for local_pos, query_idx in enumerate(batch_indices):
            query_idx_int = int(query_idx)
            raw_ids = []
            seen = set()
            for candidate_id in candidate_matrix[local_pos]:
                candidate_id = int(candidate_id)
                if candidate_id < 0 or candidate_id >= total or candidate_id == query_idx_int or candidate_id in seen:
                    continue
                seen.add(candidate_id)
                raw_ids.append(candidate_id)
            if raw_ids:
                candidate_vectors = np.asarray(features[np.asarray(raw_ids, dtype=np.int64)], dtype=np.float32)
                query_vector = np.asarray(query_vectors[local_pos], dtype=np.float32)
                scores = candidate_vectors @ query_vector
                order = np.argsort(-scores)[:top_k]
                neighbors = [(int(raw_ids[int(pos)]), float(scores[int(pos)])) for pos in order]
            else:
                neighbors = []
            yield query_idx_int, neighbors
        if progress:
            progress(end, len(query_indices), f"Searching feature kNN {end:,}/{len(query_indices):,}")


def _group_medoid(features: np.ndarray, member_indices: Sequence[int]) -> int:
    members = [int(value) for value in member_indices]
    if len(members) == 1:
        return members[0]
    matrix = np.asarray(features[np.asarray(members, dtype=np.int64)], dtype=np.float32)
    centroid = matrix.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    scores = matrix @ centroid
    return int(members[int(np.argmax(scores))])


def _group_boundary(features: np.ndarray, member_indices: Sequence[int], count: int) -> List[int]:
    members = [int(value) for value in member_indices]
    if count <= 0 or len(members) <= 1:
        return []
    matrix = np.asarray(features[np.asarray(members, dtype=np.int64)], dtype=np.float32)
    centroid = matrix.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    scores = matrix @ centroid
    order = np.argsort(scores)[: min(int(count), len(members))]
    return [int(members[int(pos)]) for pos in order]


def build_curation_report(config: CurationReportConfig, progress: ProgressCallback = None) -> Dict:
    index_dir = Path(config.index_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_config, index, features, records, total = _load_index(index_dir)
    query_indices, record_cache = _candidate_indices(
        records,
        total=total,
        max_query_records=int(config.max_query_records),
        seed=int(config.seed),
        class_filter=str(config.class_filter or ""),
        size_bucket=str(config.size_bucket or ""),
        progress=progress,
    )
    if len(query_indices) == 0:
        raise ValueError("No records matched the curation filters.")

    def get_record(idx: int) -> CropRecord:
        idx = int(idx)
        record = record_cache.get(idx)
        if record is None:
            record = records[idx]
            record_cache[idx] = record
        return record

    query_rows: List[Dict] = []
    near_edges: List[Dict] = []
    cross_edges: List[Dict] = []
    neighbor_summary: List[Dict] = []
    duplicate_uf = UnionFind()
    cross_review_ids = set()
    duplicate_ids = set()

    query_set = set(int(value) for value in query_indices.tolist())
    for query_idx, neighbors in _exact_neighbors(
        index,
        features,
        query_indices=query_indices,
        total=total,
        top_k=int(config.top_k),
        rerank_k=int(config.rerank_k),
        batch_size=int(config.batch_size),
        progress=progress,
    ):
        query_record = get_record(query_idx)
        query_stats = bbox_stats(query_record)
        same_class_count = 0
        cross_class_count = 0
        same_size_count = 0
        top1_idx = -1
        top1_score = float("nan")
        top1_class = ""
        top1_same_class = False
        for rank, (neighbor_idx, score) in enumerate(neighbors, start=1):
            neighbor_record = get_record(neighbor_idx)
            neighbor_stats = bbox_stats(neighbor_record)
            same_class = int(query_record.class_id) == int(neighbor_record.class_id)
            same_size = str(query_stats["size_bucket"]) == str(neighbor_stats["size_bucket"])
            if rank == 1:
                top1_idx = int(neighbor_idx)
                top1_score = float(score)
                top1_class = f"{int(neighbor_record.class_id)} {neighbor_record.class_name}"
                top1_same_class = bool(same_class)
            if same_class:
                same_class_count += 1
            else:
                cross_class_count += 1
            if same_size:
                same_size_count += 1

            if same_class and same_size and float(score) >= float(config.duplicate_threshold):
                left, right = sorted((int(query_idx), int(neighbor_idx)))
                near_edges.append(
                    {
                        "record_idx": left,
                        "neighbor_idx": right,
                        "similarity": float(score),
                        "class_id": int(query_record.class_id),
                        "class_name": str(query_record.class_name),
                        "size_bucket": str(query_stats["size_bucket"]),
                        "record_file": Path(get_record(left).image_path).name,
                        "neighbor_file": Path(get_record(right).image_path).name,
                    }
                )
                duplicate_uf.union(left, right)
                duplicate_ids.update([left, right])

            if (not same_class) and float(score) >= float(config.cross_class_threshold):
                cross_edges.append(
                    {
                        "record_idx": int(query_idx),
                        "neighbor_idx": int(neighbor_idx),
                        "similarity": float(score),
                        "class_id": int(query_record.class_id),
                        "class_name": str(query_record.class_name),
                        "neighbor_class_id": int(neighbor_record.class_id),
                        "neighbor_class_name": str(neighbor_record.class_name),
                        "size_bucket": str(query_stats["size_bucket"]),
                        "neighbor_size_bucket": str(neighbor_stats["size_bucket"]),
                        "record_file": Path(query_record.image_path).name,
                        "neighbor_file": Path(neighbor_record.image_path).name,
                        "image_path": str(query_record.image_path),
                        "neighbor_image_path": str(neighbor_record.image_path),
                    }
                )
                if int(query_idx) in query_set:
                    cross_review_ids.add(int(query_idx))
                if int(neighbor_idx) in query_set:
                    cross_review_ids.add(int(neighbor_idx))

        neighbor_summary.append(
            {
                "record_idx": int(query_idx),
                "top1_idx": int(top1_idx),
                "top1_similarity": top1_score,
                "top1_class": top1_class,
                "top1_same_class": bool(top1_same_class),
                "same_class_neighbors": int(same_class_count),
                "cross_class_neighbors": int(cross_class_count),
                "same_size_neighbors": int(same_size_count),
            }
        )
        query_rows.append(record_to_row(query_record, query_idx))

    records_df = pd.DataFrame(query_rows)
    neighbor_df = pd.DataFrame(neighbor_summary)
    near_df = pd.DataFrame(near_edges).drop_duplicates(["record_idx", "neighbor_idx"]) if near_edges else pd.DataFrame()
    cross_df = pd.DataFrame(cross_edges).drop_duplicates(["record_idx", "neighbor_idx"]) if cross_edges else pd.DataFrame()
    if not near_df.empty:
        near_df = near_df.sort_values("similarity", ascending=False)
    if not cross_df.empty:
        cross_df = cross_df.sort_values("similarity", ascending=False)

    representative_ids = set()
    boundary_ids = set()
    rare_ids = set()
    group_rows = []
    duplicate_group_rows = []
    duplicate_representatives = set()
    for group_key, group in records_df.groupby(["class_id", "class_name", "size_bucket"]):
        member_indices = [int(value) for value in group["record_idx"].tolist()]
        representative = _group_medoid(features, member_indices)
        boundaries = _group_boundary(features, member_indices, int(config.boundary_per_group))
        representative_ids.add(int(representative))
        boundary_ids.update(int(value) for value in boundaries)
        if len(member_indices) <= int(config.rare_group_max):
            rare_ids.update(member_indices)
        group_rows.append(
            {
                "class_id": int(group_key[0]),
                "class_name": str(group_key[1]),
                "size_bucket": str(group_key[2]),
                "count": int(len(member_indices)),
                "representative_idx": int(representative),
                "boundary_indices": " ".join(str(value) for value in boundaries),
                "is_rare_group": bool(len(member_indices) <= int(config.rare_group_max)),
            }
        )

    duplicate_groups = duplicate_uf.groups()
    for group_id, (_root, members) in enumerate(sorted(duplicate_groups.items()), start=1):
        representative = _group_medoid(features, members)
        duplicate_representatives.add(int(representative))
        for member in members:
            record = get_record(member)
            duplicate_group_rows.append(
                {
                    "duplicate_group_id": int(group_id),
                    "record_idx": int(member),
                    "representative_idx": int(representative),
                    "is_representative": bool(int(member) == int(representative)),
                    "class_id": int(record.class_id),
                    "class_name": str(record.class_name),
                    "size_bucket": record_size_bucket(record),
                    "image_path": str(record.image_path),
                    "file_name": Path(record.image_path).name,
                }
            )

    recommendation_rows = []
    for row in records_df.to_dict("records"):
        record_idx = int(row["record_idx"])
        reasons = []
        action = "KEEP"
        if record_idx in cross_review_ids:
            action = "REVIEW_CROSS_CLASS"
            reasons.append("high similarity to another class")
        if record_idx in rare_ids:
            action = "KEEP_RARE" if action == "KEEP" else action
            reasons.append("rare class/size group")
        if record_idx in representative_ids or record_idx in duplicate_representatives:
            action = "KEEP_REPRESENTATIVE" if action == "KEEP" else action
            reasons.append("group representative")
        if record_idx in boundary_ids:
            action = "KEEP_BOUNDARY" if action == "KEEP" else action
            reasons.append("feature boundary sample")
        if record_idx in duplicate_ids and record_idx not in duplicate_representatives and action == "KEEP":
            action = "DROP_NEAR_DUPLICATE"
            reasons.append("same-class same-size duplicate")

        recommendation_rows.append({**row, "action": action, "reason": "; ".join(reasons)})

    recommendations_df = pd.DataFrame(recommendation_rows)
    if not neighbor_df.empty:
        recommendations_df = recommendations_df.merge(neighbor_df, on="record_idx", how="left")

    image_rows = []
    partial_report = int(config.max_query_records or 0) > 0 and int(config.max_query_records) < int(total)
    for image_path, group in recommendations_df.groupby("image_path"):
        actions = set(str(value) for value in group["action"].tolist())
        keep_like = [action for action in actions if not action.startswith("DROP")]
        image_action = "DROP_IMAGE_CANDIDATE" if not keep_like else "KEEP_IMAGE"
        if any(action.startswith("REVIEW") for action in actions):
            image_action = "REVIEW_IMAGE"
        drop_safety = "SAFE_ONLY_IN_FULL_REPORT"
        if image_action == "DROP_IMAGE_CANDIDATE" and partial_report:
            drop_safety = "SAMPLE_ONLY_DO_NOT_DELETE"
        elif image_action != "DROP_IMAGE_CANDIDATE":
            drop_safety = "KEEP_OR_REVIEW"
        image_rows.append(
            {
                "image_path": str(image_path),
                "label_path": str(group["label_path"].iloc[0]),
                "file_name": Path(str(image_path)).name,
                "image_action": image_action,
                "drop_safety": drop_safety,
                "sampled_records": int(len(group)),
                "actions": ", ".join(sorted(actions)),
                "classes": ", ".join(sorted(set(str(value) for value in group["class_name"].tolist()))),
            }
        )
    image_df = pd.DataFrame(image_rows).sort_values(["image_action", "image_path"])

    representatives_df = records_df[records_df["record_idx"].astype(int).isin(representative_ids)].copy()
    boundary_df = records_df[records_df["record_idx"].astype(int).isin(boundary_ids)].copy()
    rare_df = records_df[records_df["record_idx"].astype(int).isin(rare_ids)].copy()
    duplicate_groups_df = pd.DataFrame(duplicate_group_rows)
    group_summary_df = pd.DataFrame(group_rows).sort_values(["class_name", "size_bucket"])

    outputs = {
        "records_sample.csv": records_df,
        "neighbor_summary.csv": neighbor_df,
        "near_duplicates.csv": near_df,
        "cross_class_overlap.csv": cross_df,
        "duplicate_groups.csv": duplicate_groups_df,
        "group_summary.csv": group_summary_df,
        "representatives.csv": representatives_df,
        "boundary_samples.csv": boundary_df,
        "rare_samples.csv": rare_df,
        "curation_recommendations.csv": recommendations_df,
        "image_recommendations.csv": image_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    summary = {
        "index_dir": str(index_dir),
        "output_dir": str(output_dir),
        "total_records": int(total),
        "sampled_records": int(len(query_indices)),
        "feature_dim": int(features.shape[1]),
        "top_k": int(config.top_k),
        "rerank_k": int(config.rerank_k),
        "duplicate_threshold": float(config.duplicate_threshold),
        "cross_class_threshold": float(config.cross_class_threshold),
        "near_duplicate_edges": int(len(near_df)),
        "duplicate_groups": int(len(duplicate_groups)),
        "cross_class_edges": int(len(cross_df)),
        "recommendation_counts": recommendations_df["action"].value_counts().to_dict(),
        "image_action_counts": image_df["image_action"].value_counts().to_dict(),
        "partial_report": bool(partial_report),
        "index_config": index_config,
        "report_config": asdict(config),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"summary": summary, "outputs": {key: str(output_dir / key) for key in outputs}, "output_dir": str(output_dir)}


def latest_curation_report_dir(project_name: str) -> Path:
    root = Path("artifacts") / "curation_reports" / slugify(project_name)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_relative(path: Path, root: Optional[Path], fallback_prefix: str) -> Path:
    try:
        if root:
            return path.resolve().relative_to(root.resolve())
    except Exception:
        pass
    parts = [part.replace(":", "") for part in path.parts if part not in (path.anchor, "\\", "/")]
    if len(parts) > 6:
        parts = parts[-6:]
    return Path(fallback_prefix) / Path(*parts)


def export_reduced_dataset(
    report_dir: str,
    output_dir: str,
    images_root: str = "",
    labels_root: str = "",
    data_yaml: str = "",
    mode: str = "manifest",
) -> Dict:
    report_root = Path(report_dir)
    image_recs_path = report_root / "image_recommendations.csv"
    if not image_recs_path.exists():
        raise FileNotFoundError(f"Missing image_recommendations.csv in {report_root}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_recs = pd.read_csv(image_recs_path)
    drop_mask = image_recs["image_action"].astype(str).str.startswith("DROP")
    if "drop_safety" in image_recs.columns:
        drop_mask = drop_mask & (image_recs["drop_safety"].astype(str) != "SAMPLE_ONLY_DO_NOT_DELETE")
    keep_df = image_recs[~drop_mask].copy()
    drop_df = image_recs[drop_mask].copy()

    keep_images = [str(value) for value in keep_df["image_path"].dropna().tolist()]
    keep_labels = [str(value) for value in keep_df["label_path"].dropna().tolist()]
    (output / "keep_images.txt").write_text("\n".join(keep_images) + ("\n" if keep_images else ""), encoding="utf-8")
    (output / "keep_labels.txt").write_text("\n".join(keep_labels) + ("\n" if keep_labels else ""), encoding="utf-8")
    drop_df.to_csv(output / "drop_image_candidates.csv", index=False, encoding="utf-8-sig")

    copied_images = 0
    copied_labels = 0
    mode = str(mode or "manifest").lower()
    if mode in {"copy", "hardlink"}:
        img_root = Path(images_root) if images_root else None
        lbl_root = Path(labels_root) if labels_root else None
        image_out_root = output / "images"
        label_out_root = output / "labels"
        for row in keep_df.itertuples(index=False):
            image_path = Path(str(row.image_path))
            label_path = Path(str(row.label_path))
            if image_path.exists():
                rel = _safe_relative(image_path, img_root, "external_images")
                target = image_out_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if mode == "hardlink":
                    try:
                        if not target.exists():
                            target.hardlink_to(image_path)
                    except Exception:
                        shutil.copy2(image_path, target)
                else:
                    shutil.copy2(image_path, target)
                copied_images += 1
            if label_path.exists():
                rel = _safe_relative(label_path, lbl_root, "external_labels")
                target = label_out_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if mode == "hardlink":
                    try:
                        if not target.exists():
                            target.hardlink_to(label_path)
                    except Exception:
                        shutil.copy2(label_path, target)
                else:
                    shutil.copy2(label_path, target)
                copied_labels += 1

    if data_yaml:
        source_yaml = Path(data_yaml)
        if source_yaml.exists():
            with source_yaml.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            reduced_yaml = {
                "path": str(output.resolve()),
                "train": "images",
                "val": "images",
                "names": data.get("names", {}),
            }
            with (output / "reduced_data.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(reduced_yaml, f, allow_unicode=True, sort_keys=False)

    summary = {
        "report_dir": str(report_root),
        "output_dir": str(output),
        "mode": mode,
        "kept_images": int(len(keep_df)),
        "drop_image_candidates": int(len(drop_df)),
        "copied_images": int(copied_images),
        "copied_labels": int(copied_labels),
    }
    with (output / "reduced_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
