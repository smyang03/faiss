from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
import torch
from PIL import Image

from .yolo_dataset import CropRecord, index_records_ready, open_record_store, records_to_json
from .yolo_features import ProgressCallback, YoloFeatureExtractor, extract_record_features


def auto_feature_batch_size(extractor: YoloFeatureExtractor) -> int:
    device = extractor.backend.get("device")
    if getattr(device, "type", "cpu") != "cuda" or not torch.cuda.is_available():
        return 1
    try:
        props = torch.cuda.get_device_properties(device)
        total_gb = float(props.total_memory) / (1024 ** 3)
    except Exception:
        total_gb = 0.0
    if total_gb >= 23:
        return 8
    if total_gb >= 15:
        return 6
    if total_gb >= 10:
        return 4
    return 2


class YoloFeatureIndex:
    def __init__(
        self,
        index: faiss.Index,
        records: Sequence[CropRecord],
        extractor: YoloFeatureExtractor,
        config: Dict,
        features: Optional[np.ndarray] = None,
    ) -> None:
        self.index = index
        self.records = records
        self.extractor = extractor
        self.config = config
        self.features = features

    @classmethod
    def load(cls, index_dir: str, device: str = "cpu") -> "YoloFeatureIndex":
        root = Path(index_dir)
        index_path = root / "index.faiss"
        config_path = root / "config.json"
        if not index_path.exists() or not index_records_ready(root) or not config_path.exists():
            raise FileNotFoundError(f"Missing YOLO feature index files in {root}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        extractor = YoloFeatureExtractor(
            repo_path=config["repo_path"],
            weights_path=config["weights_path"],
            device=device,
            img_size=int(config.get("img_size", 640)),
            layer_indices=config.get("layer_indices"),
        )
        index = faiss.read_index(str(index_path))
        nprobe = int(config.get("nprobe", 0) or 0)
        if nprobe > 0:
            try:
                faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
            except Exception:
                pass
        records = open_record_store(root)
        features_path = root / "features.npy"
        features = np.load(str(features_path), mmap_mode="r") if features_path.exists() else None
        return cls(index=index, records=records, extractor=extractor, config=config, features=features)

    def search_image_bbox(
        self,
        image: Image.Image,
        bbox_xyxy: Sequence[float],
        top_k: int = 20,
    ) -> List[Dict]:
        query = self.extractor.encode_image_bboxes(image, [bbox_xyxy])
        return self.search_vector(query, top_k=top_k)

    def search_crop(self, image: Image.Image, top_k: int = 20) -> List[Dict]:
        width, height = image.size
        return self.search_image_bbox(image, (0, 0, width, height), top_k=top_k)

    def search_record(self, record: CropRecord, top_k: int = 20, exclude_self: bool = False) -> List[Dict]:
        record_id = int(record.record_id)
        search_k = top_k + 1 if exclude_self else top_k
        if 0 <= record_id < len(self.records) and self._record_matches(record_id, record):
            try:
                if self.features is not None and record_id < int(self.features.shape[0]):
                    vector = np.asarray(self.features[record_id], dtype="float32").reshape(1, -1)
                else:
                    vector = self.index.reconstruct(record_id).reshape(1, -1).astype("float32")
                results = self.search_vector(vector, top_k=search_k)
                return self._exclude_record(results, record, top_k) if exclude_self else results
            except Exception:
                pass

        with Image.open(record.image_path) as img:
            image = img.convert("RGB")
        query = self.extractor.encode_image_bboxes(image, [record.bbox_xyxy])
        results = self.search_vector(query, top_k=search_k)
        return self._exclude_record(results, record, top_k) if exclude_self else results

    def _record_matches(self, record_id: int, record: CropRecord) -> bool:
        candidate = self.records[record_id]
        return (
            candidate.image_path == record.image_path
            and tuple(candidate.bbox_xyxy) == tuple(record.bbox_xyxy)
            and int(candidate.annotation_line) == int(record.annotation_line)
        )

    def _same_record(self, left: CropRecord, right: CropRecord) -> bool:
        return (
            int(left.record_id) == int(right.record_id)
            or (
                left.image_path == right.image_path
                and tuple(left.bbox_xyxy) == tuple(right.bbox_xyxy)
                and int(left.annotation_line) == int(right.annotation_line)
            )
        )

    def _exclude_record(self, results: List[Dict], query_record: CropRecord, top_k: int) -> List[Dict]:
        filtered = [item for item in results if not self._same_record(item["record"], query_record)]
        for rank, item in enumerate(filtered[:top_k], start=1):
            item["rank"] = rank
        return filtered[:top_k]

    def search_vector(self, query: np.ndarray, top_k: int = 20) -> List[Dict]:
        query = np.asarray(query, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if top_k <= 0 or query.size == 0:
            return []

        faiss_type = str(self.config.get("faiss_type", "") or "").lower()
        rerank_k = int(self.config.get("rerank_k", 500) or 0)
        should_rerank = self.features is not None and faiss_type == "ivfpq" and rerank_k > top_k
        search_k = top_k
        if should_rerank:
            ntotal = int(getattr(self.index, "ntotal", len(self.records)) or len(self.records))
            search_k = min(ntotal, max(top_k, rerank_k, top_k * 20))

        scores, indices = self.index.search(query, search_k)
        if should_rerank:
            reranked = self._exact_rerank(query[0], indices[0], top_k)
            if reranked:
                return reranked

        results: List[Dict] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.records):
                continue
            results.append({"rank": rank, "score": float(score), "record": self.records[int(idx)]})
        return results

    def _exact_rerank(self, query: np.ndarray, indices: np.ndarray, top_k: int) -> List[Dict]:
        if self.features is None:
            return []
        candidate_ids: List[int] = []
        seen = set()
        for idx in indices:
            record_id = int(idx)
            if record_id < 0 or record_id >= len(self.records) or record_id in seen:
                continue
            if record_id >= int(self.features.shape[0]):
                continue
            seen.add(record_id)
            candidate_ids.append(record_id)
        if not candidate_ids:
            return []

        query_vec = np.asarray(query, dtype="float32")
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm > 0:
            query_vec = query_vec / query_norm

        candidate_vectors = np.asarray(self.features[candidate_ids], dtype="float32")
        norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        candidate_vectors = candidate_vectors / np.maximum(norms, 1e-12)
        exact_scores = candidate_vectors @ query_vec
        order = np.argsort(-exact_scores)[:top_k]

        results: List[Dict] = []
        for rank, pos in enumerate(order, start=1):
            record_id = int(candidate_ids[int(pos)])
            results.append(
                {
                    "rank": rank,
                    "score": float(exact_scores[int(pos)]),
                    "record": self.records[record_id],
                    "reranked": True,
                }
            )
        return results


def build_yolo_feature_index(
    records: Sequence[CropRecord],
    repo_path: str,
    weights_path: str,
    index_dir: str,
    device: str = "cpu",
    img_size: int = 640,
    progress: ProgressCallback = None,
    save_features: bool = True,
    build_faiss: bool = True,
    feature_batch_size: int = 0,
) -> Optional[YoloFeatureIndex]:
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)

    extractor = YoloFeatureExtractor(
        repo_path=repo_path,
        weights_path=weights_path,
        device=device,
        img_size=img_size,
    )
    resolved_batch_size = int(feature_batch_size or 0)
    if resolved_batch_size <= 0:
        resolved_batch_size = auto_feature_batch_size(extractor)
    print(f"feature_batch_size={resolved_batch_size}", flush=True)
    features = extract_record_features(
        records,
        extractor,
        progress=progress,
        feature_batch_size=resolved_batch_size,
    )
    if features.size == 0:
        raise ValueError("No YOLO features extracted.")

    config = {
        "repo_path": repo_path,
        "weights_path": weights_path,
        "img_size": int(img_size),
        "layer_indices": extractor.layer_indices,
        "num_records": len(records),
        "dim": int(features.shape[1]),
        "feature_type": "yolo_roi_pooled_p3p4p5",
        "feature_batch_size": int(resolved_batch_size),
        "faiss_type": "flat",
        "rerank_k": 0,
    }

    records_to_json(records, str(root / "records.json"))
    with (root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if save_features:
        np.save(str(root / "features.npy"), features.astype("float32"))

    if not build_faiss:
        extractor.close()
        return None

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features.astype("float32"))
    faiss.write_index(index, str(root / "index.faiss"))
    return YoloFeatureIndex(index=index, records=records, extractor=extractor, config=config)
