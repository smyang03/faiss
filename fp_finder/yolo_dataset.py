from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from PIL import Image


IMAGE_EXT_ORDER = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IMAGE_EXTS = set(IMAGE_EXT_ORDER)
DATASET_LAYOUT_SINGLE = "single"
DATASET_LAYOUT_NESTED_JPEGIMAGES_LABELS = "nested_jpegimages_labels"
DATASET_LAYOUT_NESTED_IMAGE_LABELS = "nested_image_labels"
NESTED_LAYOUT_ALIASES = {
    DATASET_LAYOUT_NESTED_JPEGIMAGES_LABELS,
    DATASET_LAYOUT_NESTED_IMAGE_LABELS,
    "nested",
    "auto",
    "auto_jpegimages_labels",
    "auto_image_labels",
    "image_labels",
    "jpegimages_labels",
}
NESTED_IMAGE_DIR_NAMES = {"JPEGImages", "images"}


@dataclass
class CropRecord:
    record_id: int
    image_path: str
    label_path: str
    class_id: int
    class_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    image_width: int
    image_height: int
    annotation_line: int


class ImageSizeCache:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_size_cache (
                image_path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            )
            """
        )
        self.pending_writes = 0

    def get(self, image_path: Path) -> Optional[Tuple[int, int]]:
        try:
            stat = image_path.stat()
        except OSError:
            return None
        row = self.conn.execute(
            "SELECT mtime_ns, file_size, width, height FROM image_size_cache WHERE image_path = ?",
            (str(image_path),),
        ).fetchone()
        if not row:
            return None
        mtime_ns, file_size, width, height = row
        if int(mtime_ns) == int(stat.st_mtime_ns) and int(file_size) == int(stat.st_size):
            return int(width), int(height)
        return None

    def set(self, image_path: Path, width: int, height: int) -> None:
        try:
            stat = image_path.stat()
        except OSError:
            return
        self.conn.execute(
            """
            INSERT OR REPLACE INTO image_size_cache
                (image_path, mtime_ns, file_size, width, height)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(image_path), int(stat.st_mtime_ns), int(stat.st_size), int(width), int(height)),
        )
        self.pending_writes += 1
        if self.pending_writes >= 1000:
            self.conn.commit()
            self.pending_writes = 0

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def read_image_size(image_path: Path, cache: Optional[ImageSizeCache] = None) -> Optional[Tuple[int, int]]:
    if cache is not None:
        cached = cache.get(image_path)
        if cached is not None:
            return cached
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        return None
    if cache is not None:
        cache.set(image_path, width, height)
    return int(width), int(height)


def load_class_names(data_yaml_path: Optional[str]) -> Dict[int, str]:
    if not data_yaml_path:
        return {}

    path = Path(data_yaml_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return {}


def parse_class_ids(value: Optional[str]) -> Optional[set[int]]:
    text = str(value or "").strip()
    if not text:
        return None
    class_ids: set[int] = set()
    for part in re_split_class_ids(text):
        try:
            class_ids.add(int(float(part)))
        except ValueError:
            continue
    return class_ids or None


def re_split_class_ids(text: str) -> List[str]:
    return [part for part in re.split(r"[\s,;]+", text.strip()) if part]


def find_images(images_dir: str) -> List[Path]:
    root = Path(images_dir)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def iter_label_files(labels_dir: Path) -> Iterable[Path]:
    if not labels_dir.exists():
        return
    for dirpath, dirnames, filenames in os.walk(labels_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() == ".txt":
                yield Path(dirpath) / filename


def count_label_files(labels_dir: Path) -> int:
    total = 0
    for _label_path in iter_label_files(labels_dir):
        total += 1
    return total


def image_lookup_for_dir(image_dir: Path) -> Dict[str, Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        return {}
    lookup: Dict[str, Path] = {}
    try:
        entries = sorted(image_dir.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return {}
    for path in entries:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            lookup.setdefault(path.stem.lower(), path)
    return lookup


def image_path_for_label(
    label_path: Path,
    labels_dir: Path,
    images_dir: Path,
    image_dir_cache: Dict[Path, Dict[str, Path]],
) -> Optional[Path]:
    try:
        rel = label_path.relative_to(labels_dir)
    except ValueError:
        rel = Path(label_path.name)

    rel_no_suffix = rel.with_suffix("")
    image_dir = images_dir / rel_no_suffix.parent
    lookup = image_dir_cache.get(image_dir)
    if lookup is None:
        lookup = image_lookup_for_dir(image_dir)
        image_dir_cache[image_dir] = lookup

    image_path = lookup.get(rel_no_suffix.name.lower())
    if image_path:
        return image_path

    for suffix in IMAGE_EXT_ORDER:
        candidate = image_dir / f"{rel_no_suffix.name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return labels_dir / rel.with_suffix(".txt")


def discover_nested_image_label_pairs(dataset_root: str) -> List[Tuple[Path, Path]]:
    root = Path(dataset_root)
    if not root.exists():
        return []

    pairs: List[Tuple[Path, Path]] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        names = set(dirnames)
        if "labels" in names:
            for image_dir_name in sorted(NESTED_IMAGE_DIR_NAMES):
                if image_dir_name in names:
                    pairs.append((Path(dirpath) / image_dir_name, Path(dirpath) / "labels"))

        # Do not descend into heavy image/label payload folders while discovering pairs.
        dirnames[:] = [
            name
            for name in dirnames
            if name not in NESTED_IMAGE_DIR_NAMES and name != "labels"
        ]
    return pairs


def discover_nested_jpegimages_label_pairs(dataset_root: str) -> List[Tuple[Path, Path]]:
    return discover_nested_image_label_pairs(dataset_root)


def image_label_sources(images_dir: str, labels_dir: str, dataset_layout: str = DATASET_LAYOUT_SINGLE) -> List[Tuple[Path, Path, Path]]:
    layout = str(dataset_layout or DATASET_LAYOUT_SINGLE).strip().lower()
    if layout in NESTED_LAYOUT_ALIASES:
        sources = []
        for image_root, label_root in discover_nested_image_label_pairs(images_dir):
            for image_path in find_images(str(image_root)):
                sources.append((image_path, image_root, label_root))
        return sources

    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    return [(image_path, images_root, labels_root) for image_path in find_images(str(images_root))]


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    expand: float = 0.0,
) -> Tuple[int, int, int, int]:
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height

    if expand:
        box_w = x2 - x1
        box_h = y2 - y1
        x1 -= box_w * expand / 2.0
        x2 += box_w * expand / 2.0
        y1 -= box_h * expand / 2.0
        y2 += box_h * expand / 2.0

    x1 = int(max(0, min(image_width - 1, round(x1))))
    y1 = int(max(0, min(image_height - 1, round(y1))))
    x2 = int(max(0, min(image_width, round(x2))))
    y2 = int(max(0, min(image_height, round(y2))))
    return x1, y1, x2, y2


def iter_yolo_records(
    images_dir: str,
    labels_dir: str,
    class_names: Optional[Dict[int, str]] = None,
    expand: float = 0.0,
    min_box_size: int = 2,
    max_records: Optional[int] = None,
    class_ids: Optional[set[int]] = None,
    image_size_cache_path: Optional[str] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    dataset_layout: str = DATASET_LAYOUT_SINGLE,
) -> Iterable[CropRecord]:
    class_names = class_names or {}
    class_filter = set(class_ids) if class_ids else None
    record_count = 0
    layout = str(dataset_layout or DATASET_LAYOUT_SINGLE).strip().lower()
    if progress:
        progress(0, 0, "Discovering image/label folders")

    if layout in NESTED_LAYOUT_ALIASES:
        pairs = discover_nested_image_label_pairs(images_dir)
    else:
        pairs = [(Path(images_dir), Path(labels_dir))]

    if progress:
        progress(0, max(1, len(pairs)), f"Discovered {len(pairs)} image/label folder pairs")

    if progress:
        progress(0, 0, "Counting label files")
    total_labels = sum(count_label_files(label_root) for _image_root, label_root in pairs)
    if progress:
        progress(0, 0, f"Found {total_labels:,} label files")

    image_dir_cache: Dict[Path, Dict[str, Path]] = {}
    image_size_cache = ImageSizeCache(image_size_cache_path) if image_size_cache_path else None
    missing_images = 0
    unreadable_images = 0
    unreadable_labels = 0
    label_idx = 0

    try:
        for images_root, labels_root in pairs:
            for label_path in iter_label_files(labels_root):
                label_idx += 1
                if progress and (label_idx == 1 or label_idx % 1000 == 0 or label_idx == total_labels):
                    progress(
                        label_idx,
                        total_labels,
                        f"Scanning labels {label_idx}/{total_labels}; records={record_count:,}; missing_images={missing_images:,}",
                    )

                try:
                    with label_path.open("r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                except OSError:
                    unreadable_labels += 1
                    continue

                parsed_rows = []
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        x_center, y_center, width, height = [float(v) for v in parts[1:5]]
                    except ValueError:
                        continue
                    if class_filter is not None and class_id not in class_filter:
                        continue
                    parsed_rows.append((line_idx, class_id, x_center, y_center, width, height))

                if not parsed_rows:
                    continue

                image_path = image_path_for_label(label_path, labels_root, images_root, image_dir_cache)
                if image_path is None:
                    missing_images += 1
                    continue

                image_size = read_image_size(image_path, cache=image_size_cache)
                if image_size is None:
                    unreadable_images += 1
                    continue
                image_width, image_height = image_size

                for line_idx, class_id, x_center, y_center, width, height in parsed_rows:
                    bbox = yolo_to_xyxy(
                        x_center,
                        y_center,
                        width,
                        height,
                        image_width,
                        image_height,
                        expand=expand,
                    )
                    x1, y1, x2, y2 = bbox
                    if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
                        continue

                    yield CropRecord(
                        record_id=record_count,
                        image_path=str(image_path),
                        label_path=str(label_path),
                        class_id=class_id,
                        class_name=class_names.get(class_id, str(class_id)),
                        bbox_xyxy=bbox,
                        image_width=image_width,
                        image_height=image_height,
                        annotation_line=line_idx,
                    )
                    record_count += 1

                    if max_records and record_count >= max_records:
                        return
    finally:
        if image_size_cache is not None:
            image_size_cache.close()

    if progress:
        progress(
            label_idx,
            total_labels,
            "Finished labels "
            f"{label_idx}/{total_labels}; records={record_count:,}; "
            f"missing_images={missing_images:,}; unreadable_images={unreadable_images:,}; "
            f"unreadable_labels={unreadable_labels:,}",
        )


def load_yolo_records(
    images_dir: str,
    labels_dir: str,
    class_names: Optional[Dict[int, str]] = None,
    expand: float = 0.0,
    min_box_size: int = 2,
    max_records: Optional[int] = None,
    class_ids: Optional[set[int]] = None,
    image_size_cache_path: Optional[str] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    dataset_layout: str = DATASET_LAYOUT_SINGLE,
) -> List[CropRecord]:
    records = list(
        iter_yolo_records(
            images_dir=images_dir,
            labels_dir=labels_dir,
            class_names=class_names,
            expand=expand,
            min_box_size=min_box_size,
            max_records=max_records,
            class_ids=class_ids,
            image_size_cache_path=image_size_cache_path,
            progress=progress,
            dataset_layout=dataset_layout,
        )
    )
    return records


def crop_from_record(record: CropRecord) -> Image.Image:
    with Image.open(record.image_path) as img:
        rgb = img.convert("RGB")
        return rgb.crop(record.bbox_xyxy)


def crop_from_xyxy(image: Image.Image, bbox_xyxy: Sequence[float]) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))
    return rgb.crop((x1, y1, x2, y2))


def records_to_json(records: Sequence[CropRecord], path: str) -> None:
    output = [record_to_dict(record) for record in records]
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def record_to_dict(record: CropRecord) -> Dict:
    data = asdict(record)
    data["bbox_xyxy"] = list(record.bbox_xyxy)
    return data


def record_from_dict(item: Dict) -> CropRecord:
    data = dict(item)
    data["bbox_xyxy"] = tuple(data["bbox_xyxy"])
    return CropRecord(**data)


def records_from_json(path: str) -> List[CropRecord]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records = []
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    records.append(record_from_dict(json.loads(text)))
        return records

    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [record_from_dict(item) for item in data]


def records_to_jsonl_with_offsets(
    records: Iterable[CropRecord],
    jsonl_path: Union[str, Path],
    offsets_path: Union[str, Path],
) -> int:
    jsonl = Path(jsonl_path)
    offsets = Path(offsets_path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    offset_values: List[int] = []
    with jsonl.open("wb") as f:
        for record in records:
            offset_values.append(f.tell())
            line = json.dumps(record_to_dict(record), ensure_ascii=False, separators=(",", ":"))
            f.write(line.encode("utf-8"))
            f.write(b"\n")
    np.save(str(offsets), np.asarray(offset_values, dtype=np.int64))
    return len(offset_values)


def jsonl_record_count(jsonl_path: Union[str, Path]) -> int:
    path = Path(jsonl_path)
    if not path.exists():
        return 0
    total = 0
    with path.open("rb") as f:
        for _line in f:
            total += 1
    return total


class JsonlRecordStore:
    def __init__(self, jsonl_path: Union[str, Path], offsets_path: Union[str, Path]) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.offsets_path = Path(offsets_path)
        self.offsets = np.load(str(self.offsets_path), mmap_mode="r")

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def __getitem__(self, idx: int) -> CropRecord:
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        with self.jsonl_path.open("rb") as f:
            f.seek(int(self.offsets[idx]))
            line = f.readline().decode("utf-8")
        return record_from_dict(json.loads(line))


def index_record_files(index_dir: Union[str, Path]) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    root = Path(index_dir)
    records_json = root / "records.json"
    records_jsonl = root / "records.jsonl"
    offsets = root / "record_offsets.npy"
    return (
        records_json if records_json.exists() else None,
        records_jsonl if records_jsonl.exists() else None,
        offsets if offsets.exists() else None,
    )


def index_records_ready(index_dir: Union[str, Path]) -> bool:
    records_json, records_jsonl, offsets = index_record_files(index_dir)
    return bool(records_json or (records_jsonl and offsets))


def open_record_store(index_dir: Union[str, Path]) -> Union[List[CropRecord], JsonlRecordStore]:
    records_json, records_jsonl, offsets = index_record_files(index_dir)
    if records_json:
        return records_from_json(str(records_json))
    if records_jsonl and offsets:
        return JsonlRecordStore(records_jsonl, offsets)
    raise FileNotFoundError(f"Missing records.json or records.jsonl+record_offsets.npy in {index_dir}")


def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))
