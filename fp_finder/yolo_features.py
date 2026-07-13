from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .detector_yolov7 import YoloV7Detector
from .yolo_dataset import CropRecord


ProgressCallback = Optional[Callable[[int, int, str], None]]


def _int_stride(value) -> int:
    try:
        if hasattr(value, "max"):
            return int(value.max())
        return int(value)
    except Exception:
        return 32


def _img_size_hw(value) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            return int(value[0]), int(value[0])
    return int(value), int(value)


def expected_letterbox_shape(width: int, height: int, extractor: "YoloFeatureExtractor") -> Tuple[int, int]:
    new_h, new_w = _img_size_hw(extractor.backend["img_size"])
    stride = _int_stride(extractor.backend["stride"])
    auto = True if extractor.backend["backend_type"] == "legacy_yolov7" else bool(extractor.backend.get("pt", False))
    if width <= 0 or height <= 0:
        return new_h, new_w

    ratio = min(new_h / float(height), new_w / float(width))
    unpad_w = int(round(width * ratio))
    unpad_h = int(round(height * ratio))
    pad_w = new_w - unpad_w
    pad_h = new_h - unpad_h
    if auto:
        pad_w = int(np.mod(pad_w, stride))
        pad_h = int(np.mod(pad_h, stride))
    return int(unpad_h + pad_h), int(unpad_w + pad_w)


class YoloFeatureExtractor:
    """Extract YOLO-internal ROI features from Detect input feature maps."""

    def __init__(
        self,
        repo_path: str,
        weights_path: str,
        device: str = "cpu",
        img_size: int = 640,
        class_names: Optional[Dict[int, str]] = None,
        layer_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.detector = YoloV7Detector(
            repo_path=repo_path,
            weights_path=weights_path,
            device=device,
            img_size=img_size,
            conf_thres=0.25,
            class_names=class_names or {0: "person"},
        )
        self.backend = self.detector.local_backend
        if self.backend is None:
            raise RuntimeError("YOLO feature extraction requires a local YOLO repo backend.")

        self.model = self.backend["model"]
        self.sequence = self._model_sequence(self.model)
        self.layer_indices = list(layer_indices or self._detect_input_layers())
        self.captures: Dict[int, torch.Tensor] = {}
        self.handles = []
        self._register_hooks()

    def _model_sequence(self, model):
        target = getattr(model, "model", model)
        return getattr(target, "model", target)

    def _detect_input_layers(self) -> List[int]:
        last = self.sequence[-1]
        layers = getattr(last, "f", None)
        if isinstance(layers, int):
            return [layers]
        if isinstance(layers, (list, tuple)):
            return [int(v) for v in layers]
        raise RuntimeError("Cannot infer YOLO Detect input feature layers.")

    def _register_hooks(self) -> None:
        for layer_idx in self.layer_indices:
            module = self.sequence[layer_idx]

            def make_hook(idx: int):
                def hook(_module, _inputs, output):
                    if isinstance(output, torch.Tensor):
                        self.captures[idx] = output.detach()

                return hook

            self.handles.append(module.register_forward_hook(make_hook(layer_idx)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _preprocess(self, image: Image.Image):
        rgb = image.convert("RGB")
        im0 = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        letterbox = self.backend["letterbox"]
        stride = self.backend["stride"]
        img_size = self.backend["img_size"]

        if self.backend["backend_type"] == "legacy_yolov7":
            im, ratio, pad = letterbox(im0, img_size, stride=stride)
        else:
            im, ratio, pad = letterbox(im0, img_size, stride=stride, auto=self.backend["pt"])

        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        tensor = torch.from_numpy(im).to(self.backend["device"])
        tensor = tensor.float() / 255.0
        if len(tensor.shape) == 3:
            tensor = tensor[None]
        return rgb, im0, tensor, ratio, pad

    @torch.no_grad()
    def encode_image_bboxes(
        self,
        image: Image.Image,
        bboxes_xyxy: Sequence[Sequence[float]],
    ) -> np.ndarray:
        if not bboxes_xyxy:
            return np.empty((0, 0), dtype="float32")

        rgb, _im0, tensor, ratio, pad = self._preprocess(image)
        self.captures.clear()

        if self.backend["backend_type"] == "legacy_yolov7":
            _ = self.model(tensor, augment=False)[0]
        else:
            _ = self.model(tensor, augment=False, visualize=False)

        feature_maps = [self.captures[idx] for idx in self.layer_indices if idx in self.captures]
        if not feature_maps:
            raise RuntimeError(f"No feature maps captured for layers {self.layer_indices}")

        vectors = []
        for bbox in bboxes_xyxy:
            per_level = [
                self._pool_bbox_feature(fmap, bbox, rgb.size, tensor.shape[2:], ratio, pad)
                for fmap in feature_maps
            ]
            vector = torch.cat(per_level, dim=0)
            vector = torch.nn.functional.normalize(vector, p=2, dim=0)
            vectors.append(vector.cpu().numpy().astype("float32"))

        return np.vstack(vectors).astype("float32")

    @torch.no_grad()
    def encode_image_batch_bboxes(
        self,
        images: Sequence[Image.Image],
        bboxes_by_image: Sequence[Sequence[Sequence[float]]],
    ) -> List[np.ndarray]:
        if len(images) != len(bboxes_by_image):
            raise ValueError("images and bboxes_by_image must have the same length")
        if not images:
            return []

        preprocessed = [self._preprocess(image) for image in images]
        by_shape: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, (_rgb, _im0, tensor, _ratio, _pad) in enumerate(preprocessed):
            by_shape[(int(tensor.shape[2]), int(tensor.shape[3]))].append(idx)

        outputs: List[Optional[np.ndarray]] = [None] * len(images)
        for image_indices in by_shape.values():
            batch_tensor = torch.cat([preprocessed[idx][2] for idx in image_indices], dim=0)
            self.captures.clear()

            if self.backend["backend_type"] == "legacy_yolov7":
                _ = self.model(batch_tensor, augment=False)[0]
            else:
                _ = self.model(batch_tensor, augment=False, visualize=False)

            feature_maps = [self.captures[idx] for idx in self.layer_indices if idx in self.captures]
            if not feature_maps:
                raise RuntimeError(f"No feature maps captured for layers {self.layer_indices}")

            for batch_pos, image_idx in enumerate(image_indices):
                rgb, _im0, tensor, ratio, pad = preprocessed[image_idx]
                vectors = []
                for bbox in bboxes_by_image[image_idx]:
                    per_level = [
                        self._pool_bbox_feature(
                            fmap,
                            bbox,
                            rgb.size,
                            tensor.shape[2:],
                            ratio,
                            pad,
                            batch_index=batch_pos,
                        )
                        for fmap in feature_maps
                    ]
                    vector = torch.cat(per_level, dim=0)
                    vector = torch.nn.functional.normalize(vector, p=2, dim=0)
                    vectors.append(vector.cpu().numpy().astype("float32"))
                outputs[image_idx] = np.vstack(vectors).astype("float32") if vectors else np.empty((0, 0), dtype="float32")

        return [output if output is not None else np.empty((0, 0), dtype="float32") for output in outputs]

    def encode_crop(self, image: Image.Image) -> np.ndarray:
        width, height = image.size
        return self.encode_image_bboxes(image, [(0, 0, width, height)])

    def _pool_bbox_feature(
        self,
        feature_map: torch.Tensor,
        bbox_xyxy: Sequence[float],
        image_size: Tuple[int, int],
        tensor_hw: Sequence[int],
        ratio,
        pad,
        batch_index: int = 0,
    ) -> torch.Tensor:
        fmap = feature_map[int(batch_index)].float()
        _channels, fmap_h, fmap_w = fmap.shape
        tensor_h, tensor_w = int(tensor_hw[0]), int(tensor_hw[1])

        ratio_x, ratio_y = ratio if isinstance(ratio, tuple) else (ratio, ratio)
        pad_x, pad_y = pad
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]

        x1_l = x1 * ratio_x + pad_x
        x2_l = x2 * ratio_x + pad_x
        y1_l = y1 * ratio_y + pad_y
        y2_l = y2 * ratio_y + pad_y

        fx1 = int(np.floor(x1_l / max(1, tensor_w) * fmap_w))
        fx2 = int(np.ceil(x2_l / max(1, tensor_w) * fmap_w))
        fy1 = int(np.floor(y1_l / max(1, tensor_h) * fmap_h))
        fy2 = int(np.ceil(y2_l / max(1, tensor_h) * fmap_h))

        fx1 = max(0, min(fmap_w - 1, fx1))
        fy1 = max(0, min(fmap_h - 1, fy1))
        fx2 = max(fx1 + 1, min(fmap_w, fx2))
        fy2 = max(fy1 + 1, min(fmap_h, fy2))

        region = fmap[:, fy1:fy2, fx1:fx2]
        return region.mean(dim=(1, 2))


def extract_record_features(
    records: Sequence[CropRecord],
    extractor: YoloFeatureExtractor,
    progress: ProgressCallback = None,
    feature_batch_size: int = 1,
) -> np.ndarray:
    if not records:
        return np.empty((0, 0), dtype="float32")

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[record.image_path].append(idx)

    features = None
    total_images = len(groups)
    total_records = len(records)
    done_images = 0
    done_records = 0
    batch_size = max(1, int(feature_batch_size or 1))
    def sort_key(item):
        indices = item[1]
        record = records[indices[0]]
        shape_h, shape_w = expected_letterbox_shape(record.image_width, record.image_height, extractor)
        return shape_h, shape_w, str(item[0])

    group_items = sorted(groups.items(), key=sort_key)

    for start in range(0, total_images, batch_size):
        batch_items = group_items[start : start + batch_size]
        images = []
        boxes_by_image = []
        indices_by_image = []
        for image_path, indices in batch_items:
            last_error = None
            for attempt in range(4):
                try:
                    with Image.open(image_path) as img:
                        image = img.convert("RGB")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        time.sleep(2.0)
            if last_error is not None:
                raise last_error

            images.append(image)
            boxes_by_image.append([records[i].bbox_xyxy for i in indices])
            indices_by_image.append(indices)

        batch_outputs = extractor.encode_image_batch_bboxes(images, boxes_by_image)

        if features is None:
            first = next((output for output in batch_outputs if output.size > 0), None)
            if first is None:
                continue
            features = np.zeros((len(records), first.shape[1]), dtype="float32")

        for indices, batch_features in zip(indices_by_image, batch_outputs):
            if batch_features.size == 0:
                continue
            features[indices, :] = batch_features
            done_records += len(indices)

        done_images += len(batch_items)
        if progress:
            progress(
                done_images,
                total_images,
                f"Encoding YOLO features: {done_images}/{total_images}; records={done_records:,}/{total_records:,}; batch_size={batch_size}",
            )

    if features is None:
        return np.empty((0, 0), dtype="float32")
    return features
