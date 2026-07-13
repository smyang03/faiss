from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class Detection:
    det_id: int
    frame_index: int
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    crop: Image.Image
    image_width: int = 0
    image_height: int = 0


class YoloV7Detector:
    def __init__(
        self,
        weights_path: str,
        repo_path: Optional[str] = None,
        device: str = "cpu",
        img_size: int = 640,
        conf_thres: float = 0.25,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.weights_path = weights_path
        self.repo_path = repo_path
        self.device = self._normalize_device(device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.class_names = class_names or {}
        self.local_backend = None
        self.model = self._load_model()
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _normalize_device(self, device: str) -> str:
        value = str(device or "cpu").lower()
        if value == "cpu" or not torch.cuda.is_available():
            return "cpu"
        if value == "cuda":
            return "0"
        if value.startswith("cuda:"):
            return value.split(":", 1)[1]
        return value

    def _load_model(self):
        weights = str(Path(self.weights_path))
        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLOv7 weights not found: {weights}")

        if self.repo_path:
            repo = str(Path(self.repo_path))
            if not Path(repo).exists():
                raise FileNotFoundError(f"YOLOv7 repo not found: {repo}")
            if (Path(repo) / "models" / "common.py").exists() and (
                Path(repo) / "utils" / "general.py"
            ).exists():
                return self._load_local_yolo_repo(Path(repo), weights)
            try:
                return torch.hub.load(
                    repo,
                    "custom",
                    path_or_model=weights,
                    source="local",
                    trust_repo=True,
                ).to(self.device)
            except TypeError:
                return torch.hub.load(
                    repo,
                    "custom",
                    path=weights,
                    source="local",
                ).to(self.device)

        try:
            return torch.hub.load(
                "WongKinYiu/yolov7",
                "custom",
                path_or_model=weights,
                trust_repo=True,
            ).to(self.device)
        except TypeError:
            return torch.hub.load(
                "WongKinYiu/yolov7",
                "custom",
                path=weights,
            ).to(self.device)

    def _load_local_yolo_repo(self, repo: Path, weights: str):
        import sys

        repo = repo.resolve()
        repo_str = str(repo)
        if repo_str in sys.path:
            sys.path.remove(repo_str)
        sys.path.insert(0, repo_str)

        # YOLO repos use top-level "models" and "utils" packages. Clear cached
        # modules before switching between yolov7/yolov9 repos in one process.
        for module_name in list(sys.modules):
            if module_name == "models" or module_name.startswith("models."):
                del sys.modules[module_name]
            if module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]

        try:
            from models.common import DetectMultiBackend
        except ImportError:
            return self._load_legacy_yolov7_repo(repo, weights)

        from utils.dataloaders import letterbox
        from utils.general import check_img_size
        from utils.general import non_max_suppression
        from utils.general import scale_boxes
        from utils.torch_utils import select_device

        device = select_device(self.device if self.device != "cpu" else "cpu")
        model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        img_size = check_img_size((self.img_size, self.img_size), s=stride)
        model.warmup(imgsz=(1, 3, *img_size))
        self.local_backend = {
            "repo": repo,
            "stride": stride,
            "names": names,
            "pt": pt,
            "img_size": img_size,
            "device": device,
            "model": model,
            "letterbox": letterbox,
            "non_max_suppression": non_max_suppression,
            "scale_fn": scale_boxes,
            "backend_type": "detect_multi_backend",
        }
        return model

    def _load_legacy_yolov7_repo(self, repo: Path, weights: str):
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import check_img_size
        from utils.general import non_max_suppression
        from utils.general import scale_coords
        from utils.general import set_logging
        from utils.torch_utils import select_device

        set_logging()
        device = select_device(self.device if self.device != "cpu" else "cpu")
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        img_size = check_img_size(self.img_size, s=stride)
        names = model.module.names if hasattr(model, "module") else model.names

        if device.type != "cpu":
            model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

        self.local_backend = {
            "repo": repo,
            "stride": stride,
            "names": names,
            "pt": False,
            "img_size": img_size,
            "device": device,
            "model": model,
            "letterbox": letterbox,
            "non_max_suppression": non_max_suppression,
            "scale_fn": scale_coords,
            "backend_type": "legacy_yolov7",
        }
        return model

    def detect(self, image: Image.Image, frame_index: int = 0) -> List[Detection]:
        if self.local_backend is not None:
            return self._detect_local_repo(image, frame_index=frame_index)

        rgb = image.convert("RGB")
        results = self.model(rgb, size=self.img_size)
        rows = self._extract_rows(results)
        detections: List[Detection] = []
        width, height = rgb.size

        for row in rows:
            x1, y1, x2, y2, conf, cls_id = row[:6]
            if float(conf) < self.conf_thres:
                continue
            x1 = int(max(0, min(width - 1, round(float(x1)))))
            y1 = int(max(0, min(height - 1, round(float(y1)))))
            x2 = int(max(0, min(width, round(float(x2)))))
            y2 = int(max(0, min(height, round(float(y2)))))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(cls_id)
            detections.append(
                Detection(
                    det_id=len(detections),
                    frame_index=frame_index,
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                    crop=rgb.crop((x1, y1, x2, y2)),
                    image_width=width,
                    image_height=height,
                )
            )
        return detections

    @torch.no_grad()
    def _detect_local_repo(self, image: Image.Image, frame_index: int = 0) -> List[Detection]:
        rgb = image.convert("RGB")
        im0 = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        backend = self.local_backend
        model = backend["model"]
        img_size = backend["img_size"]
        stride = backend["stride"]
        pt = backend["pt"]

        letterbox = backend["letterbox"]
        if backend["backend_type"] == "legacy_yolov7":
            im = letterbox(im0, img_size, stride=stride)[0]
        else:
            im = letterbox(im0, img_size, stride=stride, auto=pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        tensor = torch.from_numpy(im).to(backend["device"])
        tensor = tensor.float()
        tensor /= 255.0
        if len(tensor.shape) == 3:
            tensor = tensor[None]

        if backend["backend_type"] == "legacy_yolov7":
            pred = model(tensor, augment=False)[0]
            pred = backend["non_max_suppression"](
                pred,
                self.conf_thres,
                0.45,
                classes=None,
                agnostic=False,
            )
        else:
            pred = model(tensor, augment=False, visualize=False)
            pred = backend["non_max_suppression"](
                pred,
                self.conf_thres,
                0.45,
                None,
                False,
                max_det=1000,
            )

        names = backend.get("names") or self.class_names
        detections: List[Detection] = []
        width, height = rgb.size
        det = pred[0]
        if len(det):
            det[:, :4] = backend["scale_fn"](tensor.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det.tolist():
                x1, y1, x2, y2 = xyxy
                x1 = int(max(0, min(width - 1, round(float(x1)))))
                y1 = int(max(0, min(height - 1, round(float(y1)))))
                x2 = int(max(0, min(width, round(float(x2)))))
                y2 = int(max(0, min(height, round(float(y2)))))
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(cls)
                if isinstance(names, dict):
                    class_name = str(names.get(class_id, self.class_names.get(class_id, str(class_id))))
                else:
                    try:
                        class_name = str(names[class_id])
                    except Exception:
                        class_name = self.class_names.get(class_id, str(class_id))
                detections.append(
                    Detection(
                        det_id=len(detections),
                        frame_index=frame_index,
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=class_id,
                        class_name=class_name,
                        crop=rgb.crop((x1, y1, x2, y2)),
                        image_width=width,
                        image_height=height,
                    )
                )
        return detections

    def _extract_rows(self, results) -> np.ndarray:
        if hasattr(results, "xyxy"):
            det = results.xyxy[0]
            if hasattr(det, "detach"):
                det = det.detach().cpu().numpy()
            return np.asarray(det, dtype=np.float32)

        if isinstance(results, torch.Tensor):
            arr = results.detach().cpu().numpy()
            return self._coerce_prediction_array(arr)

        if isinstance(results, (list, tuple)) and results:
            first = results[0]
            if isinstance(first, torch.Tensor):
                arr = first.detach().cpu().numpy()
                return self._coerce_prediction_array(arr)
            return np.asarray(first, dtype=np.float32)

        return np.empty((0, 6), dtype=np.float32)

    def _coerce_prediction_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape[-1] < 6:
            return np.empty((0, 6), dtype=np.float32)
        return arr[:, :6].astype(np.float32)
