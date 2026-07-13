from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import cv2
from PIL import Image

from .detector_yolov7 import Detection, YoloV7Detector


ProgressCallback = Optional[Callable[[int, int, str], None]]


def read_video_frame(video_path: str, frame_index: int) -> Image.Image:
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def collect_video_detections(
    video_path: str,
    detector: YoloV7Detector,
    frame_stride: int = 15,
    max_frames: int = 300,
    max_detections: int = 200,
    progress: ProgressCallback = None,
) -> List[Detection]:
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detections: List[Detection] = []
    frame_index = 0
    processed = 0

    try:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % max(1, frame_stride) != 0:
                frame_index += 1
                continue

            processed += 1
            if progress:
                progress(
                    processed,
                    max_frames,
                    f"Detecting frame {frame_index} ({processed}/{max_frames})",
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            frame_dets = detector.detect(image, frame_index=frame_index)
            for det in frame_dets:
                det.det_id = len(detections)
                detections.append(det)
                if len(detections) >= max_detections:
                    return detections

            frame_index += 1
            if total_frames and frame_index >= total_frames:
                break
    finally:
        cap.release()

    return detections
