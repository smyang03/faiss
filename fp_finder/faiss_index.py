from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from PIL import Image

from .embeddings import ImageEncoder
from .yolo_dataset import CropRecord, crop_from_record, records_from_json, records_to_json


ProgressCallback = Optional[Callable[[int, int, str], None]]


class SimilarityIndex:
    def __init__(
        self,
        index: faiss.Index,
        records: Sequence[CropRecord],
        encoder_name: str,
        encoder: Optional[ImageEncoder] = None,
    ) -> None:
        self.index = index
        self.records = list(records)
        self.encoder_name = encoder_name
        self.encoder = encoder

    @classmethod
    def load(cls, index_dir: str, device: str = "cpu") -> "SimilarityIndex":
        root = Path(index_dir)
        index_path = root / "index.faiss"
        records_path = root / "records.json"
        config_path = root / "config.json"

        if not index_path.exists() or not records_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Missing FAISS index files in {root}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        index = faiss.read_index(str(index_path))
        records = records_from_json(str(records_path))
        encoder = ImageEncoder(config["encoder_name"], device=device)
        return cls(index=index, records=records, encoder_name=config["encoder_name"], encoder=encoder)

    def search_image(self, image: Image.Image, top_k: int = 20) -> List[Dict]:
        if self.encoder is None:
            self.encoder = ImageEncoder(self.encoder_name)
        query = self.encoder.encode([image], batch_size=1)
        scores, indices = self.index.search(query, top_k)

        results: List[Dict] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.records):
                continue
            record = self.records[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "record": record,
                }
            )
        return results


def build_faiss_index(
    records: Sequence[CropRecord],
    encoder_name: str,
    index_dir: str,
    device: str = "cpu",
    batch_size: int = 32,
    progress: ProgressCallback = None,
) -> SimilarityIndex:
    if not records:
        raise ValueError("No crop records found. Check image/label paths.")

    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)

    encoder = ImageEncoder(encoder_name, device=device)
    all_vectors: List[np.ndarray] = []
    total = len(records)

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        if progress:
            progress(start, total, f"Encoding crops {start + 1}-{end} / {total}")
        crops = []
        for record in records[start:end]:
            try:
                crops.append(crop_from_record(record))
            except Exception:
                crops.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))

        vectors = encoder.encode(crops, batch_size=batch_size)
        all_vectors.append(vectors)

    if progress:
        progress(total, total, "Building FAISS index")

    features = np.vstack(all_vectors).astype("float32")
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)

    faiss.write_index(index, str(root / "index.faiss"))
    records_to_json(records, str(root / "records.json"))
    with (root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "encoder_name": encoder_name,
                "num_records": len(records),
                "dim": int(features.shape[1]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return SimilarityIndex(index=index, records=records, encoder_name=encoder_name, encoder=encoder)
