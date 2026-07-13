from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


class ImageEncoder:
    def __init__(self, name: str = "resnet18", device: str = "cpu") -> None:
        self.name = name
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model, self.preprocess, self.dim = self._load_model(name)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, name: str):
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            base = models.resnet18(weights=weights)
            model = torch.nn.Sequential(*(list(base.children())[:-1]))
            return model, weights.transforms(), 512

        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
            model = torch.nn.Sequential(*(list(base.children())[:-1]))
            return model, weights.transforms(), 2048

        if name == "dinov2_vits14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            preprocess = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
            return model, preprocess, 384

        raise ValueError(f"Unsupported encoder: {name}")

    @torch.no_grad()
    def encode(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            tensor = torch.stack([self.preprocess(img.convert("RGB")) for img in batch_images])
            tensor = tensor.to(self.device)
            feats = self.model(tensor)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.reshape(feats.shape[0], -1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            vectors.append(feats.cpu().numpy().astype("float32"))

        if not vectors:
            return np.empty((0, self.dim), dtype="float32")
        return np.vstack(vectors).astype("float32")
