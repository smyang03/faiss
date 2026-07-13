from __future__ import annotations

import base64
import html
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", message=r"\s*Found Intel OpenMP.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*", category=UserWarning)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

from fp_finder.detector_yolov7 import Detection, YoloV7Detector
from fp_finder.projects import (
    PROJECTS_PATH,
    delete_project,
    get_project,
    index_ready,
    load_projects,
    project_log_root,
    project_records_json,
    project_shard_root,
    slugify,
    upsert_project,
)
from fp_finder.feature_clustering import (
    SIZE_BUCKET_LABELS,
    SIZE_BUCKET_ORDER,
    build_feature_clusters,
    load_cluster_metadata,
    size_bucket_from_area_ratio,
)
from fp_finder.video import collect_video_detections, read_video_frame
from fp_finder.yolo_feature_index import YoloFeatureIndex
from fp_finder.yolo_dataset import (
    DATASET_LAYOUT_NESTED_JPEGIMAGES_LABELS,
    DATASET_LAYOUT_NESTED_IMAGE_LABELS,
    DATASET_LAYOUT_SINGLE,
    CropRecord,
    crop_from_record,
    discover_nested_image_label_pairs,
    index_records_ready,
    load_class_names,
    open_record_store,
    records_from_json,
)


st.set_page_config(
    page_title="YOLOv7 FP Sample Finder",
    page_icon="",
    layout="wide",
)

FIREDB_IMAGES_DIR = r"V:\00.영상파트\08_fireDB\01_fireDB_v1\08_data\00_2차라벨링데이터\images"
FIREDB_LABELS_DIR = r"V:\00.영상파트\08_fireDB\01_fireDB_v1\08_data\00_2차라벨링데이터\labels"
FIREDB_DATA_YAML = "firedb_v1_data.yaml"
FIREDB_YOLO_FEATURE_INDEX_DIR = "artifacts/yolo_feature_index_fire_8class_w122"


def config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #07111f !important;
            color: #e5f3ff !important;
            overflow-x: hidden !important;
        }
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] .main,
        [data-testid="stHeader"] {
            background: #07111f !important;
            color: #e5f3ff !important;
            overflow-x: hidden !important;
            max-width: 100vw !important;
        }
        [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="column"] {
            max-width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
        }
        [data-testid="stMainBlockContainer"] {
            width: 1280px !important;
            max-width: min(1280px, calc(100vw - 320px)) !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        @media (max-width: 1200px) {
            [data-testid="stMainBlockContainer"] {
                width: calc(100vw - 48px) !important;
                max-width: calc(100vw - 48px) !important;
            }
        }
        [data-testid="column"] > div {
            min-width: 0 !important;
            max-width: 100% !important;
        }
        [data-testid="stDataFrame"] {
            width: 100% !important;
            max-width: 100% !important;
            overflow: hidden !important;
        }
        [data-testid="stDataFrame"] * {
            max-width: 100% !important;
        }
        [data-testid="stToolbar"],
        [data-testid="stToolbar"] * {
            color: #dbeafe !important;
        }
        [data-testid="stSidebar"] {
            background: #081827 !important;
            border-right: 1px solid #17324d !important;
        }
        [data-testid="stSidebar"] * {
            color: #dbeafe !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
            letter-spacing: 0;
        }
        p, label, span, div, small, li, td, th {
            color: #dbeafe !important;
            letter-spacing: 0;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] *,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stText"],
        [data-testid="stText"] * {
            color: #dbeafe !important;
        }
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] *,
        [data-testid="stCheckbox"],
        [data-testid="stCheckbox"] *,
        [data-testid="stRadio"],
        [data-testid="stRadio"] *,
        [data-testid="stSelectbox"],
        [data-testid="stSelectbox"] *,
        [data-testid="stMultiSelect"],
        [data-testid="stMultiSelect"] *,
        [data-testid="stSlider"],
        [data-testid="stSlider"] *,
        [data-testid="stNumberInput"],
        [data-testid="stNumberInput"] *,
        [data-testid="stTextInput"],
        [data-testid="stTextInput"] *,
        [data-testid="stTextArea"],
        [data-testid="stTextArea"] * {
            color: #e5f3ff !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #071827 !important;
            border: 1px dashed #2b5c86 !important;
            color: #e5f3ff !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #e5f3ff !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #17324d !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: #0b2035 !important;
            border: 1px solid #1d3a57 !important;
            border-radius: 8px 8px 0 0;
            color: #bfdbfe !important;
            height: 40px;
            padding: 0 14px;
        }
        .stTabs [data-baseweb="tab"] * {
            color: #bfdbfe !important;
        }
        .stTabs [aria-selected="true"] {
            background: #12395a !important;
            color: #ffffff !important;
        }
        .stTabs [aria-selected="true"] * {
            color: #ffffff !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stMetric"],
        div[data-testid="stExpander"],
        div[data-testid="stForm"] {
            border: 1px solid #1b3855 !important;
            border-radius: 8px;
            background: #0a1b2d !important;
            color: #e5f3ff !important;
        }
        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] *,
        div[data-testid="stMetric"] *,
        div[data-testid="stExpander"] *,
        div[data-testid="stForm"] * {
            color: #e5f3ff !important;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            border: 1px solid #2b5c86 !important;
            background: #0d2b45 !important;
            color: #e0f2fe !important;
            min-height: 34px;
        }
        .stButton > button *,
        .stDownloadButton > button * {
            color: #e0f2fe !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #38bdf8 !important;
            background: #123f63 !important;
            color: #ffffff !important;
        }
        button:disabled,
        button[disabled],
        .stButton > button:disabled,
        .stDownloadButton > button:disabled {
            background: #0b2035 !important;
            border-color: #1b3855 !important;
            color: #8fb4d2 !important;
            opacity: 1 !important;
        }
        button:disabled *,
        button[disabled] * {
            color: #8fb4d2 !important;
        }
        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="select"],
        div[data-baseweb="base-input"] {
            background: #071827 !important;
            border-color: #254766 !important;
            color: #e5f3ff !important;
        }
        div[data-baseweb="input"] *,
        div[data-baseweb="textarea"] *,
        div[data-baseweb="select"] *,
        div[data-baseweb="base-input"] * {
            color: #e5f3ff !important;
            -webkit-text-fill-color: #e5f3ff !important;
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div {
            background: #071827 !important;
            border-color: #254766 !important;
            color: #e5f3ff !important;
            -webkit-text-fill-color: #e5f3ff !important;
        }
        .stTextInput input:disabled,
        .stNumberInput input:disabled,
        .stTextArea textarea:disabled {
            color: #b7d4ea !important;
            -webkit-text-fill-color: #b7d4ea !important;
            opacity: 1 !important;
        }
        input::placeholder,
        textarea::placeholder {
            color: #8fb4d2 !important;
            -webkit-text-fill-color: #8fb4d2 !important;
            opacity: 1 !important;
        }
        .stSelectbox div[data-baseweb="select"] *,
        div[data-baseweb="popover"] *,
        ul[role="listbox"] *,
        div[role="option"] * {
            color: #e5f3ff !important;
        }
        div[data-baseweb="popover"],
        ul[role="listbox"],
        div[role="option"] {
            background: #071827 !important;
        }
        div[role="option"]:hover {
            background: #123f63 !important;
        }
        [data-testid="stAlert"],
        [data-testid="stAlert"] * {
            background: #0b2035 !important;
            color: #e5f3ff !important;
            border-color: #2b5c86 !important;
        }
        code, pre {
            background: #061422 !important;
            color: #dbeafe !important;
        }
        .stSlider [data-testid="stTickBar"] {
            background: #17324d !important;
        }
        [data-testid="stImageCaption"],
        [data-testid="stImageCaption"] *,
        .caption,
        .caption * {
            color: #bfdbfe !important;
        }
        svg text {
            fill: #dbeafe !important;
        }
        .thumb-card {
            height: 430px;
            border: 1px solid #1d3a57 !important;
            background: #0a1b2d !important;
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 10px;
            overflow: hidden;
        }
        .thumb-title {
            color: #e5f3ff !important;
            font-size: 12px;
            font-weight: 600;
            line-height: 1.25;
            min-height: 30px;
            overflow: hidden;
            overflow-wrap: anywhere;
        }
        .thumb-meta {
            color: #9cc7e8 !important;
            font-size: 11px;
            line-height: 1.2;
            min-height: 26px;
            overflow-wrap: anywhere;
        }
        .thumb-img-frame {
            width: 100%;
            max-width: 240px;
            aspect-ratio: 1 / 1;
            margin: 7px auto 8px auto;
            border: 1px solid #183653;
            border-radius: 6px;
            background: #061422;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        .thumb-img-frame img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .thumb-badge {
            position: absolute;
            top: 7px;
            left: 7px;
            max-width: calc(100% - 14px);
            padding: 3px 7px;
            border-radius: 999px;
            background: rgba(6, 20, 34, 0.86);
            border: 1px solid rgba(125, 211, 252, 0.55);
            color: #f8fafc !important;
            font-size: 12px;
            font-weight: 700;
            line-height: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .thumb-card-compact {
            height: 398px;
        }
        .thumb-card-compact .thumb-img-frame {
            max-width: 260px;
            margin-top: 0;
        }
        .query-preview-title {
            color: #f8fafc !important;
            font-size: 13px;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 3px;
        }
        .query-preview-meta {
            color: #9cc7e8 !important;
            font-size: 11px;
            line-height: 1.2;
            overflow-wrap: anywhere;
            margin-bottom: 6px;
        }
        .group-band {
            margin: 14px 0 8px 0;
            padding: 7px 10px;
            border: 1px solid #1d3a57;
            border-radius: 7px;
            background: #092035;
            color: #e5f3ff !important;
            font-size: 13px;
            font-weight: 600;
        }
        a {
            color: #7dd3fc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def progress_with_eta(done: int, total: int, message: str, start_time: float) -> str:
    elapsed = time.time() - start_time
    if done <= 0 or total <= 0:
        return f"{message} | elapsed {format_duration(elapsed)}"

    ratio = min(1.0, done / total)
    rate = done / max(elapsed, 1e-6)
    remaining = max(0.0, (total - done) / max(rate, 1e-6))
    eta_time = datetime.now() + timedelta(seconds=remaining)
    return (
        f"{message} | {ratio * 100:5.1f}% | "
        f"elapsed {format_duration(elapsed)} | "
        f"ETA {format_duration(remaining)} ({eta_time:%H:%M:%S})"
    )


def init_state() -> None:
    defaults = {
        "video_detections": [],
        "last_results": [],
        "last_query_image": None,
        "last_video_path": None,
        "yolo_feature_index": None,
        "yolo_feature_index_dir": None,
        "yolo_detector": None,
        "yolo_detector_key": None,
        "active_project_name": None,
        "active_project": None,
        "pending_db_neighbor_record": None,
        "pending_db_neighbor_top_k": 20,
        "db_neighbor_results": [],
        "db_neighbor_query_record": None,
        "db_neighbor_error": "",
        "preview_image": None,
        "preview_caption": "",
        "thumb_uri_cache": {},
        "selected_data_paths": {},
        "selection_generation": 0,
        "cluster_request": None,
        "cluster_result": None,
        "cluster_result_request": None,
        "cluster_result_elapsed": 0.0,
        "cluster_compare_points": [],
        "calibration_request": None,
        "calibration_result": None,
        "calibration_result_request": None,
        "calibration_result_elapsed": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar_config() -> Dict:
    st.sidebar.header("Runtime")
    default_device_index = 0
    try:
        import torch

        default_device_index = 1 if torch.cuda.is_available() else 0
    except Exception:
        default_device_index = 0
    device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=default_device_index, key="cfg_device")

    return {
        "device": device,
    }


def model_files() -> List[Path]:
    return sorted(Path("model").glob("*.pt")) if Path("model").exists() else []


def default_fire_project() -> Optional[Dict]:
    matches = [path for path in model_files() if is_fire_w122_model(str(path))]
    if not matches or not Path(FIREDB_YOLO_FEATURE_INDEX_DIR, "config.json").exists():
        return None

    weights_path = str(matches[0])
    return {
        "name": "fire_8class_w122",
        "description": "FireDB v1 / pj_fire_8class_v2 / YOLOv7 feature index",
        "dataset_layout": DATASET_LAYOUT_SINGLE,
        "images_dir": FIREDB_IMAGES_DIR,
        "labels_dir": FIREDB_LABELS_DIR,
        "data_yaml": FIREDB_DATA_YAML,
        "weights_path": weights_path,
        "repo_path": infer_repo_path_for_model(weights_path),
        "feature_index_dir": FIREDB_YOLO_FEATURE_INDEX_DIR,
        "img_size": 640,
        "expand": 0.08,
    }


def ensure_default_projects() -> None:
    if get_project("fire_8class_w122") is not None:
        return
    project = default_fire_project()
    if project is not None:
        upsert_project(project)


def active_project() -> Optional[Dict]:
    project = st.session_state.get("active_project")
    if project:
        return project

    projects = load_projects()
    if not projects:
        return None

    preferred_name = st.session_state.get("active_project_name")
    selected = next((item for item in projects if item.get("name") == preferred_name), projects[0])
    st.session_state["active_project"] = selected
    st.session_state["active_project_name"] = selected.get("name")
    return selected


def set_active_project(project: Dict) -> None:
    previous_project = st.session_state.get("active_project") or {}
    previous_project_dir = previous_project.get("feature_index_dir")
    next_dir = project.get("feature_index_dir")
    loaded_dir = st.session_state.get("yolo_feature_index_dir")
    if loaded_dir is not None and loaded_dir != next_dir:
        st.session_state["yolo_feature_index"] = None
        st.session_state["yolo_feature_index_dir"] = None
    if previous_project_dir is not None and previous_project_dir != next_dir:
        clear_db_neighbor_state()
    st.session_state["active_project"] = project
    st.session_state["active_project_name"] = project.get("name")


def project_to_row(project: Dict) -> Dict:
    return {
        "name": project.get("name", ""),
        "ready": "yes" if index_ready(project) else "no",
        "model": Path(str(project.get("weights_path", ""))).name,
        "layout": project.get("dataset_layout", DATASET_LAYOUT_SINGLE),
        "class_ids": project.get("class_ids", "") or "all",
        "max_records": int(project.get("max_records", 0) or 0),
        "feature_batch_size": int(project.get("feature_batch_size", 0) or 0),
        "faiss_type": project.get("faiss_type", "ivfpq"),
        "faiss_gpu": "yes" if config_bool(project.get("faiss_gpu", False)) else "no",
        "feature_index_dir": project.get("feature_index_dir", ""),
        "images_dir": project.get("images_dir", ""),
        "updated_at": project.get("updated_at", ""),
    }


def clear_db_neighbor_state() -> None:
    st.session_state["pending_db_neighbor_record"] = None
    st.session_state["pending_db_neighbor_top_k"] = 20
    st.session_state["db_neighbor_results"] = []
    st.session_state["db_neighbor_query_record"] = None
    st.session_state["db_neighbor_error"] = ""


def record_to_state(record: CropRecord) -> Dict:
    data = asdict(record)
    data["bbox_xyxy"] = list(record.bbox_xyxy)
    return data


def record_from_state(data: Dict) -> CropRecord:
    item = dict(data)
    item["bbox_xyxy"] = tuple(item["bbox_xyxy"])
    return CropRecord(**item)


def request_db_neighbor_search(record: CropRecord, top_k: int) -> None:
    st.session_state["pending_db_neighbor_record"] = record_to_state(record)
    st.session_state["pending_db_neighbor_top_k"] = int(max(5, min(100, top_k)))
    st.session_state["db_neighbor_error"] = ""


def set_preview_image(image: Image.Image, caption: str) -> None:
    st.session_state["preview_image"] = image.copy()
    st.session_state["preview_caption"] = caption


def image_to_data_uri(image: Image.Image, thumb_size: int = 240) -> str:
    rgb = image.convert("RGB")
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    thumb_size = int(thumb_size)
    width, height = rgb.size
    longest_side = max(1, width, height)
    scale = thumb_size / float(longest_side)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    thumb = rgb.resize(resized_size, resampling)
    canvas = Image.new("RGB", (thumb_size, thumb_size), (6, 20, 34))
    x = (thumb_size - thumb.width) // 2
    y = (thumb_size - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=78, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_thumb_uri(uri: str, alt: str = "", badge: str = "", wrapper_class: str = "") -> None:
    badge_html = f'<div class="thumb-badge">{html.escape(str(badge))}</div>' if badge else ""
    class_name = f"thumb-img-frame {html.escape(str(wrapper_class))}".strip()
    st.markdown(
        f'<div class="{class_name}"><img src="{uri}" alt="{html.escape(alt)}">{badge_html}</div>',
        unsafe_allow_html=True,
    )


def render_thumb_image(
    image: Image.Image,
    alt: str = "",
    cache_key: Optional[str] = None,
    badge: str = "",
    wrapper_class: str = "",
) -> None:
    if cache_key:
        cache = st.session_state.setdefault("thumb_uri_cache", {})
        uri = cache.get(cache_key)
        if uri is None:
            uri = image_to_data_uri(image)
            cache[cache_key] = uri
    else:
        uri = image_to_data_uri(image)
    render_thumb_uri(uri, alt, badge=badge, wrapper_class=wrapper_class)


def file_mtime_ns(path: str) -> int:
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data(show_spinner=False, max_entries=4096)
def cached_record_thumb_uri(
    image_path: str,
    bbox_xyxy: tuple,
    record_id: int,
    annotation_line: int,
    mtime_ns: int,
    thumb_size: int = 240,
) -> str:
    del record_id, annotation_line, mtime_ns
    with Image.open(image_path) as img:
        crop = img.convert("RGB").crop(tuple(int(v) for v in bbox_xyxy))
    return image_to_data_uri(crop, thumb_size=thumb_size)


def render_record_thumb(record: CropRecord, badge: str = "", wrapper_class: str = "") -> bool:
    try:
        uri = cached_record_thumb_uri(
            record.image_path,
            tuple(int(v) for v in record.bbox_xyxy),
            int(record.record_id),
            int(record.annotation_line),
            file_mtime_ns(record.image_path),
        )
        render_thumb_uri(uri, f"{record.class_name} {record.record_id}", badge=badge, wrapper_class=wrapper_class)
        return True
    except Exception:
        return False


def open_card(title: str = "", meta: str = "", compact: bool = False) -> None:
    class_name = "thumb-card thumb-card-compact" if compact else "thumb-card"
    title_html = f'<div class="thumb-title">{html.escape(str(title))}</div>' if title else ""
    meta_html = f'<div class="thumb-meta">{html.escape(str(meta))}</div>' if meta else ""
    st.markdown(
        f"""
        <div class="{class_name}">
          {title_html}
          {meta_html}
        """,
        unsafe_allow_html=True,
    )


def close_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def open_data_location(path: str) -> None:
    target = Path(path)
    if not target.exists():
        st.warning(f"File not found: {path}")
        return
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer.exe", f"/select,{str(target)}"])
    else:
        subprocess.Popen(["xdg-open", str(target.parent)])


def selected_path_rows() -> List[Dict]:
    selected = st.session_state.setdefault("selected_data_paths", {})
    return list(selected.values())


def render_path_selector(path: str, record: Optional[CropRecord], key: str) -> None:
    selected = st.session_state.setdefault("selected_data_paths", {})
    initial = path in selected
    generation = int(st.session_state.get("selection_generation", 0))
    checked = st.checkbox("Select", value=initial, key=f"{key}_select_{generation}")
    if checked:
        selected[path] = {
            "image_path": path,
            "class_name": record.class_name if record is not None else "",
            "class_id": record.class_id if record is not None else "",
            "record_id": record.record_id if record is not None else "",
            "bbox_xyxy": list(record.bbox_xyxy) if record is not None else "",
        }
    else:
        selected.pop(path, None)


def render_selected_paths_panel(key_prefix: str = "selected_paths") -> None:
    rows = selected_path_rows()
    if not rows:
        return

    st.divider()
    st.subheader(f"Selected Data Paths ({len(rows)})")
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "file_name": Path(str(row["image_path"])).name,
                "class_id": row.get("class_id", ""),
                "class_name": row.get("class_name", ""),
                "record_id": row.get("record_id", ""),
                "bbox_xyxy": row.get("bbox_xyxy", ""),
            }
        )
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True, key=f"{key_prefix}_table")
    paths_text = "\n".join(row["image_path"] for row in rows)
    st.text_area(
        "Selected image paths",
        value=paths_text,
        height=140,
        key=f"{key_prefix}_text",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Paths TXT",
            paths_text.encode("utf-8-sig"),
            "selected_image_paths.txt",
            "text/plain",
            key=f"{key_prefix}_download_txt",
            use_container_width=True,
        )
    with col2:
        if st.button("Clear Selected", key=f"{key_prefix}_clear", use_container_width=True):
            st.session_state["selected_data_paths"] = {}
            st.session_state["selection_generation"] = int(st.session_state.get("selection_generation", 0)) + 1
            st.rerun()


def render_preview_image() -> None:
    image = st.session_state.get("preview_image")
    if image is None:
        return
    st.divider()
    st.subheader("Preview")
    col_img, col_meta = st.columns([1, 1])
    with col_img:
        st.image(image, caption=st.session_state.get("preview_caption", ""), width=520)
    with col_meta:
        width, height = image.size
        st.caption(f"size={width}x{height}")
        if st.button("Close preview", key="btn_close_preview"):
            st.session_state["preview_image"] = None
            st.session_state["preview_caption"] = ""
            st.rerun()


def result_rows(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in results:
        record = item["record"]
        rows.append(
            {
                "rank": item["rank"],
                "similarity": round(item["score"], 5),
                "class_id": record.class_id,
                "class_name": record.class_name,
                "image_path": record.image_path,
                "bbox_xyxy": list(record.bbox_xyxy),
                "annotation_line": record.annotation_line,
            }
        )
    return pd.DataFrame(rows)


def result_display_rows(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in results:
        record = item["record"]
        rows.append(
            {
                "rank": item["rank"],
                "similarity": round(item["score"], 5),
                "class_id": record.class_id,
                "class_name": record.class_name,
                "file_name": Path(record.image_path).name,
                "bbox_xyxy": list(record.bbox_xyxy),
            }
        )
    return pd.DataFrame(rows)


def detection_size_info(det: Detection) -> Dict:
    x1, y1, x2, y2 = det.bbox_xyxy
    box_w = max(1, int(x2) - int(x1))
    box_h = max(1, int(y2) - int(y1))
    image_w = max(1, int(getattr(det, "image_width", 0) or 0), int(x2), box_w)
    image_h = max(1, int(getattr(det, "image_height", 0) or 0), int(y2), box_h)
    area_ratio = float((box_w * box_h) / max(1, image_w * image_h))
    return {
        "bbox_width": box_w,
        "bbox_height": box_h,
        "area_ratio": area_ratio,
        "area_pct": area_ratio * 100.0,
        "size_bucket": size_bucket_from_area_ratio(area_ratio),
    }


def detection_group_name(det: Detection, group_mode: str) -> str:
    info = detection_size_info(det)
    size_label = SIZE_BUCKET_LABELS.get(info["size_bucket"], info["size_bucket"])
    if group_mode == "Class":
        return str(det.class_name)
    if group_mode == "Size":
        return size_label
    if group_mode == "Class + Size":
        return f"{det.class_name} / {size_label}"
    return "All crops"


def show_results(results: List[Dict], columns: int = 4, key_prefix: str = "results") -> None:
    if not results:
        st.info("검색 결과가 없습니다.")
        return

    full_df = result_rows(results)
    display_df = result_display_rows(results)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=280,
        key=f"{key_prefix}_table",
    )
    st.download_button(
        "Download CSV",
        full_df.to_csv(index=False).encode("utf-8-sig"),
        "similar_samples.csv",
        "text/csv",
        key=f"{key_prefix}_download_csv",
    )

    grid = st.columns(columns)
    for idx, item in enumerate(results):
        record = item["record"]
        with grid[idx % columns]:
            thumb_ok = render_record_thumb(
                record,
                badge=f"{item['score']:.3f} | {record.class_id} {record.class_name}",
            )
            if not thumb_ok:
                st.caption(f"#{item['rank']} crop load failed")
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if thumb_ok and st.button(
                    "View",
                    key=f"{key_prefix}_preview_{idx}_{record.record_id}",
                    use_container_width=True,
                ):
                    crop = crop_from_record(record)
                    set_preview_image(
                        crop,
                        f"#{item['rank']} {record.class_name} sim={item['score']:.3f} | {Path(record.image_path).name}",
                    )
                    st.rerun()
            with action_col2:
                if st.button(
                    "Data",
                    key=f"{key_prefix}_open_data_{idx}_{record.record_id}",
                    use_container_width=True,
                ):
                    open_data_location(record.image_path)
            render_path_selector(
                record.image_path,
                record,
                key=f"{key_prefix}_path_{idx}_{record.record_id}",
            )
            if st.button(
                "Neighbors",
                key=f"{key_prefix}_db_requery_{idx}_{record.record_id}",
                use_container_width=True,
            ):
                request_db_neighbor_search(record, max(5, min(100, len(results))))
                st.rerun()

    render_selected_paths_panel(key_prefix=f"{key_prefix}_selected_paths")


def search_yolo_feature_bbox(
    image: Image.Image,
    bbox_xyxy,
    top_k: int,
    key_prefix: str,
    feature_index_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    index = st.session_state.get("yolo_feature_index")
    current_dir = st.session_state.get("yolo_feature_index_dir")
    if index is None or (feature_index_dir and current_dir != feature_index_dir):
        if not ensure_yolo_feature_index_loaded(feature_index_dir=feature_index_dir, device=device):
            return
        index = st.session_state.get("yolo_feature_index")
    clear_db_neighbor_state()
    start = time.time()
    with st.spinner("Searching YOLO feature index..."):
        results = index.search_image_bbox(image, bbox_xyxy, top_k=top_k)
    st.session_state["last_results"] = results
    st.session_state["last_results_key_prefix"] = key_prefix
    st.session_state["last_results_context"] = "video"
    st.session_state["last_query_image"] = image.crop(tuple(int(v) for v in bbox_xyxy))
    st.caption(f"Search elapsed: {format_duration(time.time() - start)}")
    show_results(results, key_prefix=key_prefix)


def search_yolo_feature_crop(
    image: Image.Image,
    top_k: int,
    key_prefix: str,
    feature_index_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    index = st.session_state.get("yolo_feature_index")
    current_dir = st.session_state.get("yolo_feature_index_dir")
    if index is None or (feature_index_dir and current_dir != feature_index_dir):
        if not ensure_yolo_feature_index_loaded(feature_index_dir=feature_index_dir, device=device):
            return
        index = st.session_state.get("yolo_feature_index")
    clear_db_neighbor_state()
    start = time.time()
    with st.spinner("Searching YOLO feature index..."):
        results = index.search_crop(image, top_k=top_k)
    st.session_state["last_results"] = results
    st.session_state["last_results_key_prefix"] = key_prefix
    st.session_state["last_results_context"] = "crop"
    st.session_state["last_query_image"] = image
    st.caption(f"Search elapsed: {format_duration(time.time() - start)}")
    show_results(results, key_prefix=key_prefix)


def search_yolo_feature_record(
    record: CropRecord,
    top_k: int,
    key_prefix: str,
    feature_index_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    index = st.session_state.get("yolo_feature_index")
    current_dir = st.session_state.get("yolo_feature_index_dir")
    if index is None or (feature_index_dir and current_dir != feature_index_dir):
        if not ensure_yolo_feature_index_loaded(feature_index_dir=feature_index_dir, device=device):
            return
        index = st.session_state.get("yolo_feature_index")

    start = time.time()
    with st.spinner("Searching YOLO feature index from selected DB sample..."):
        results = index.search_record(record, top_k=top_k, exclude_self=True)
    st.session_state["last_results"] = results
    st.session_state["last_results_key_prefix"] = key_prefix
    st.session_state["last_results_context"] = "record"
    try:
        st.session_state["last_query_image"] = crop_from_record(record)
    except Exception:
        st.session_state["last_query_image"] = None
    st.caption(f"DB re-search elapsed: {format_duration(time.time() - start)}")
    show_results(results, key_prefix=key_prefix)


def run_pending_db_neighbor_search(project: Dict, config: Dict) -> None:
    payload = st.session_state.get("pending_db_neighbor_record")
    if not payload:
        return

    st.session_state["pending_db_neighbor_record"] = None
    record = record_from_state(payload)
    top_k = int(st.session_state.get("pending_db_neighbor_top_k", 20) or 20)
    feature_index_dir = str(project.get("feature_index_dir", ""))

    if not ensure_yolo_feature_index_loaded(feature_index_dir=feature_index_dir, device=config["device"]):
        st.session_state["db_neighbor_error"] = f"Cannot load YOLO feature index: {feature_index_dir}"
        return

    index = st.session_state.get("yolo_feature_index")
    start = time.time()
    try:
        with st.spinner("Searching neighbors from selected DB sample..."):
            results = index.search_record(record, top_k=top_k, exclude_self=True)
        st.session_state["db_neighbor_results"] = results
        st.session_state["db_neighbor_query_record"] = payload
        st.session_state["db_neighbor_error"] = ""
        st.session_state["db_neighbor_elapsed"] = format_duration(time.time() - start)
    except Exception as exc:
        st.session_state["db_neighbor_results"] = []
        st.session_state["db_neighbor_query_record"] = payload
        st.session_state["db_neighbor_error"] = str(exc)


def render_db_neighbor_results(render_key_prefix: str = "db_neighbor") -> None:
    payload = st.session_state.get("db_neighbor_query_record")
    results = st.session_state.get("db_neighbor_results", [])
    error = st.session_state.get("db_neighbor_error", "")
    if not payload and not error:
        return

    st.divider()
    st.subheader("DB Neighbor Results")
    if payload:
        record = record_from_state(payload)
        st.caption(
            f"Query DB sample: {Path(record.image_path).name} | "
            f"{record.class_name} | bbox={list(record.bbox_xyxy)} | "
            f"elapsed={st.session_state.get('db_neighbor_elapsed', '-')}"
        )
    if error:
        st.error(f"DB neighbor search failed: {error}")
        return
    show_results(
        results,
        key_prefix=f"{render_key_prefix}_db_neighbor_results_{payload.get('record_id', 'unknown')}",
    )


@st.cache_data(show_spinner=False)
def cached_cluster_metadata(index_dir: str) -> Dict:
    return load_cluster_metadata(index_dir)


@st.cache_resource(show_spinner=False)
def cached_index_records(index_dir: str):
    return open_record_store(index_dir)


@st.cache_data(show_spinner=False)
def cached_feature_clusters(
    index_dir: str,
    max_points: int,
    n_clusters: int,
    seed: int,
    class_filter: str,
    size_bucket: str,
    clustering_scope: str,
) -> Dict:
    return build_feature_clusters(
        index_dir=index_dir,
        max_points=max_points,
        n_clusters=n_clusters,
        seed=seed,
        class_filter=class_filter or None,
        size_bucket=size_bucket or None,
        clustering_scope=clustering_scope,
    )


@st.cache_data(show_spinner=False)
def cached_similarity_calibration(
    index_dir: str,
    sample_size: int,
    top_k: int,
    seed: int,
    class_filter: str,
    bin_width: float,
) -> Dict:
    from collections import Counter

    import faiss

    root = Path(index_dir)
    index_path = root / "index.faiss"
    features_path = root / "features.npy"
    if not index_path.exists() or not features_path.exists() or not index_records_ready(root):
        raise FileNotFoundError(f"Missing index.faiss/features.npy/records metadata in {root}")

    records = open_record_store(root)
    features = np.load(str(features_path), mmap_mode="r")
    total = min(len(records), int(features.shape[0]))
    if total <= 1:
        return {"detail": pd.DataFrame(), "bins": pd.DataFrame(), "thresholds": pd.DataFrame(), "classes": pd.DataFrame()}

    class_filter = str(class_filter or "").strip()
    candidates = []
    for idx in range(total):
        record = records[idx]
        class_key = f"{record.class_id}: {record.class_name}"
        if class_filter and class_filter != "All" and class_filter not in {str(record.class_name), class_key, str(record.class_id)}:
            continue
        candidates.append(idx)
    if not candidates:
        return {"detail": pd.DataFrame(), "bins": pd.DataFrame(), "thresholds": pd.DataFrame(), "classes": pd.DataFrame()}

    rng = np.random.default_rng(int(seed))
    candidates_arr = np.asarray(candidates, dtype=np.int64)
    actual_sample = min(int(sample_size), int(candidates_arr.size))
    if actual_sample < candidates_arr.size:
        sample_indices = np.sort(rng.choice(candidates_arr, size=actual_sample, replace=False))
    else:
        sample_indices = candidates_arr

    index = faiss.read_index(str(index_path))
    search_k = min(int(top_k) + 1, total)
    rows = []
    batch_size = 512
    for start_idx in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[start_idx : start_idx + batch_size]
        query = np.asarray(features[batch_indices], dtype=np.float32)
        scores, indices = index.search(query, search_k)
        for local_pos, record_idx in enumerate(batch_indices):
            query_record = records[int(record_idx)]
            neighbors = []
            for score, neighbor_idx in zip(scores[local_pos], indices[local_pos]):
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx < 0 or neighbor_idx >= total or neighbor_idx == int(record_idx):
                    continue
                neighbor_record = records[neighbor_idx]
                neighbors.append((float(score), neighbor_idx, neighbor_record))
                if len(neighbors) >= int(top_k):
                    break
            if not neighbors:
                continue

            top1_score, top1_idx, top1_record = neighbors[0]
            topk_same = [int(item[2].class_id) == int(query_record.class_id) for item in neighbors]
            neighbor_classes = [str(item[2].class_name) for item in neighbors]
            class_counts = Counter(neighbor_classes)
            majority_class, majority_count = class_counts.most_common(1)[0] if class_counts else ("", 0)
            majority_ratio = float(majority_count / len(neighbor_classes)) if len(neighbor_classes) else 0.0
            rows.append(
                {
                    "record_idx": int(record_idx),
                    "record_id": int(query_record.record_id),
                    "class_id": int(query_record.class_id),
                    "class_name": str(query_record.class_name),
                    "file_name": Path(query_record.image_path).name,
                    "top1_similarity": top1_score,
                    "top1_record_idx": int(top1_idx),
                    "top1_record_id": int(top1_record.record_id),
                    "top1_class_id": int(top1_record.class_id),
                    "top1_class_name": str(top1_record.class_name),
                    "top1_file_name": Path(top1_record.image_path).name,
                    "top1_same_class": bool(int(top1_record.class_id) == int(query_record.class_id)),
                    "topk_same_class_ratio": float(np.mean(topk_same)),
                    "topk_majority_class": majority_class,
                    "topk_majority_ratio": majority_ratio,
                    "topk": int(len(neighbors)),
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"detail": detail, "bins": pd.DataFrame(), "thresholds": pd.DataFrame(), "classes": pd.DataFrame()}

    safe_width = max(0.001, float(bin_width))
    detail["similarity_bin_start"] = np.floor(detail["top1_similarity"] / safe_width) * safe_width
    detail["similarity_bin_end"] = detail["similarity_bin_start"] + safe_width
    detail["similarity_bin"] = detail.apply(
        lambda row: f"{row.similarity_bin_start:.3f}-{row.similarity_bin_end:.3f}",
        axis=1,
    )

    bins = (
        detail.groupby(["similarity_bin_start", "similarity_bin"], as_index=False)
        .agg(
            count=("top1_similarity", "size"),
            top1_same_class_rate=("top1_same_class", "mean"),
            mean_top1_similarity=("top1_similarity", "mean"),
            mean_topk_same_class_ratio=("topk_same_class_ratio", "mean"),
            mean_topk_majority_ratio=("topk_majority_ratio", "mean"),
        )
        .sort_values("similarity_bin_start", ascending=False)
    )
    for column in [
        "top1_same_class_rate",
        "mean_top1_similarity",
        "mean_topk_same_class_ratio",
        "mean_topk_majority_ratio",
    ]:
        bins[column] = (bins[column] * 100.0).round(2) if column != "mean_top1_similarity" else bins[column].round(4)

    thresholds_rows = []
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.98]
    for threshold in thresholds:
        subset = detail[detail["top1_similarity"] >= threshold]
        if subset.empty:
            thresholds_rows.append(
                {
                    "similarity_threshold": threshold,
                    "count": 0,
                    "coverage_pct": 0.0,
                    "top1_same_class_rate": None,
                    "mean_topk_same_class_ratio": None,
                }
            )
            continue
        thresholds_rows.append(
            {
                "similarity_threshold": threshold,
                "count": int(len(subset)),
                "coverage_pct": round(float(len(subset) / len(detail) * 100.0), 2),
                "top1_same_class_rate": round(float(subset["top1_same_class"].mean() * 100.0), 2),
                "mean_topk_same_class_ratio": round(float(subset["topk_same_class_ratio"].mean() * 100.0), 2),
            }
        )
    thresholds_df = pd.DataFrame(thresholds_rows)

    classes = (
        detail.groupby(["class_id", "class_name"], as_index=False)
        .agg(
            count=("top1_similarity", "size"),
            mean_top1_similarity=("top1_similarity", "mean"),
            top1_same_class_rate=("top1_same_class", "mean"),
            mean_topk_same_class_ratio=("topk_same_class_ratio", "mean"),
        )
        .sort_values("count", ascending=False)
    )
    classes["mean_top1_similarity"] = classes["mean_top1_similarity"].round(4)
    classes["top1_same_class_rate"] = (classes["top1_same_class_rate"] * 100.0).round(2)
    classes["mean_topk_same_class_ratio"] = (classes["mean_topk_same_class_ratio"] * 100.0).round(2)

    return {
        "detail": detail.sort_values("top1_similarity", ascending=False),
        "bins": bins,
        "thresholds": thresholds_df,
        "classes": classes,
        "sample_size": int(len(detail)),
        "total_candidates": int(candidates_arr.size),
    }


CLUSTER_CUSTOM_COLUMNS = [
    "record_id",
    "record_idx",
    "image_path",
    "label_path",
    "class_id",
    "class_name",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "image_width",
    "image_height",
    "annotation_line",
    "cluster_label",
    "size_bucket",
    "area_pct",
    "file_name",
]

CLICK_MAP_PALETTE = [
    "#38bdf8",
    "#f97316",
    "#10b981",
    "#a855f7",
    "#f43f5e",
    "#eab308",
    "#14b8a6",
    "#fb7185",
    "#818cf8",
    "#f59e0b",
    "#22c55e",
    "#06b6d4",
    "#c084fc",
    "#facc15",
    "#60a5fa",
    "#34d399",
    "#f472b6",
    "#fb923c",
]


def cluster_custom_data_from_row(row: pd.Series) -> List:
    return [row[column] for column in CLUSTER_CUSTOM_COLUMNS]


def cluster_custom_data_from_record_idx(df: pd.DataFrame, record_idx: int) -> Optional[List]:
    matches = df[df["record_idx"].astype(int) == int(record_idx)]
    if matches.empty:
        return None
    return cluster_custom_data_from_row(matches.iloc[0])


def crop_record_from_cluster_custom(selected: List) -> CropRecord:
    return CropRecord(
        record_id=int(selected[0]),
        image_path=str(selected[2]),
        label_path=str(selected[3]),
        class_id=int(selected[4]),
        class_name=str(selected[5]),
        bbox_xyxy=(int(selected[6]), int(selected[7]), int(selected[8]), int(selected[9])),
        image_width=int(selected[10]),
        image_height=int(selected[11]),
        annotation_line=int(selected[12]),
    )


def cluster_compare_key(custom: List) -> int:
    return int(custom[1])


def add_cluster_compare_point(custom: List) -> None:
    points = list(st.session_state.get("cluster_compare_points", []) or [])
    key = cluster_compare_key(custom)
    points = [point for point in points if cluster_compare_key(point) != key]
    points.append(list(custom))
    st.session_state["cluster_compare_points"] = points[-2:]


def cluster_row_by_custom(df: pd.DataFrame, custom: List) -> Optional[pd.Series]:
    matches = df[df["record_idx"].astype(int) == cluster_compare_key(custom)]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_cluster_record_panel(
    record: CropRecord,
    title: str,
    meta: str,
    key_prefix: str,
    show_image: bool,
) -> None:
    st.caption(title)
    if meta:
        st.caption(meta)
    if show_image:
        thumb_ok = render_record_thumb(record, badge=f"{record.class_id} {record.class_name}")
        if not thumb_ok:
            st.caption(f"record {record.record_id} image load failed")
    else:
        st.caption("image hidden")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("View", key=f"{key_prefix}_view", use_container_width=True, disabled=not show_image):
            set_preview_image(crop_from_record(record), f"{title} | record={record.record_id}")
            st.rerun()
    with action_col2:
        if st.button("Data", key=f"{key_prefix}_data", use_container_width=True):
            open_data_location(record.image_path)
    render_path_selector(record.image_path, record, key=f"{key_prefix}_path")
    if st.button("Neighbors", key=f"{key_prefix}_neighbors", use_container_width=True):
        request_db_neighbor_search(record, 20)
        st.rerun()


def cluster_hover_text(row: pd.Series) -> str:
    return (
        f"record={int(row.record_id)}<br>"
        f"class={int(row.class_id)} {html.escape(str(row.class_name))}<br>"
        f"cluster={html.escape(str(row.cluster_label))}<br>"
        f"size={html.escape(str(row.size_bucket))} area={float(row.area_pct):.2f}%<br>"
        f"{html.escape(str(row.file_name))}"
    )


def build_cluster_3d_hover_figure(df: pd.DataFrame, color_by: str, seed: int, max_points: int = 5000) -> go.Figure:
    if len(df) > max_points:
        plot_df = df.sample(n=max_points, random_state=int(seed)).copy()
    else:
        plot_df = df.copy()

    fig = go.Figure()
    color_by = str(color_by or "cluster_label")
    if color_by == "area_ratio":
        fig.add_trace(
            go.Scatter3d(
                x=[float(value) for value in plot_df["x"].tolist()],
                y=[float(value) for value in plot_df["y"].tolist()],
                z=[float(value) for value in plot_df["z"].tolist()],
                mode="markers",
                name="area_ratio",
                customdata=[int(value) for value in plot_df["record_idx"].tolist()],
                text=[cluster_hover_text(row) for _, row in plot_df.iterrows()],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=9,
                    opacity=0.82,
                    color=[float(value) for value in plot_df["area_ratio"].tolist()],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="area"),
                ),
            )
        )
    else:
        plot_df["_cluster_color"] = plot_df[color_by].astype(str)
        groups = sorted(plot_df["_cluster_color"].unique().tolist())
        for group_index, group_name in enumerate(groups):
            group = plot_df[plot_df["_cluster_color"] == group_name]
            fig.add_trace(
                go.Scatter3d(
                    x=[float(value) for value in group["x"].tolist()],
                    y=[float(value) for value in group["y"].tolist()],
                    z=[float(value) for value in group["z"].tolist()],
                    mode="markers",
                    name=str(group_name),
                    customdata=[int(value) for value in group["record_idx"].tolist()],
                    text=[cluster_hover_text(row) for _, row in group.iterrows()],
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(
                        size=7,
                        opacity=0.84,
                        color=CLICK_MAP_PALETTE[group_index % len(CLICK_MAP_PALETTE)],
                    ),
                )
            )

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        clickmode="event+select",
        dragmode="turntable",
        height=720,
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        font=dict(color="#dbeafe"),
        legend=dict(bgcolor="rgba(7, 17, 31, 0.78)", font=dict(color="#dbeafe")),
        scene=dict(
            bgcolor="#07111f",
            xaxis=dict(backgroundcolor="#07111f", gridcolor="#1d3a57", zerolinecolor="#2b5c86", color="#dbeafe"),
            yaxis=dict(backgroundcolor="#07111f", gridcolor="#1d3a57", zerolinecolor="#2b5c86", color="#dbeafe"),
            zaxis=dict(backgroundcolor="#07111f", gridcolor="#1d3a57", zerolinecolor="#2b5c86", color="#dbeafe"),
        ),
    )
    return fig


def cross_class_overlap_filter(
    df: pd.DataFrame,
    percentile: float,
    max_points: int,
    neighbor_depth: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if df.empty or df["class_name"].nunique() < 2:
        return df.iloc[0:0].copy(), pd.DataFrame(), 0.0

    from sklearn.neighbors import NearestNeighbors

    work_df = df.reset_index(drop=True).copy()
    coords = work_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    class_values = work_df["class_name"].astype(str).to_numpy()
    n_neighbors = min(len(work_df), max(2, int(neighbor_depth)))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    nearest_dist = np.full(len(work_df), np.inf, dtype=np.float32)
    nearest_pos = np.full(len(work_df), -1, dtype=np.int32)
    for pos in range(len(work_df)):
        for dist, candidate in zip(distances[pos][1:], indices[pos][1:]):
            if class_values[candidate] != class_values[pos]:
                nearest_dist[pos] = float(dist)
                nearest_pos[pos] = int(candidate)
                break

    finite_mask = np.isfinite(nearest_dist)
    if not finite_mask.any():
        return work_df.iloc[0:0].copy(), pd.DataFrame(), 0.0

    threshold = float(np.percentile(nearest_dist[finite_mask], float(percentile)))
    seed_positions = np.where(nearest_dist <= threshold)[0].tolist()
    selected_positions = set(seed_positions)
    for pos in seed_positions:
        partner = int(nearest_pos[pos])
        if partner >= 0:
            selected_positions.add(partner)

    ordered_positions = sorted(
        selected_positions,
        key=lambda pos: float(nearest_dist[pos]) if np.isfinite(nearest_dist[pos]) else float("inf"),
    )
    if max_points > 0:
        ordered_positions = ordered_positions[: int(max_points)]

    overlap_df = work_df.iloc[ordered_positions].copy()
    overlap_df["nearest_other_distance"] = [float(nearest_dist[pos]) for pos in ordered_positions]
    overlap_df["nearest_other_record_id"] = [
        int(work_df.iloc[int(nearest_pos[pos])].record_id) if nearest_pos[pos] >= 0 else -1
        for pos in ordered_positions
    ]
    overlap_df["nearest_other_class"] = [
        str(work_df.iloc[int(nearest_pos[pos])].class_name) if nearest_pos[pos] >= 0 else ""
        for pos in ordered_positions
    ]
    overlap_df["nearest_other_file"] = [
        str(work_df.iloc[int(nearest_pos[pos])].file_name) if nearest_pos[pos] >= 0 else ""
        for pos in ordered_positions
    ]

    pair_rows = []
    seen_pairs = set()
    for pos in seed_positions:
        partner = int(nearest_pos[pos])
        if partner < 0:
            continue
        left = work_df.iloc[pos]
        right = work_df.iloc[partner]
        pair_key = tuple(sorted((int(left.record_idx), int(right.record_idx))))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        pair_rows.append(
            {
                "distance": round(float(nearest_dist[pos]), 5),
                "record_id": int(left.record_id),
                "class": f"{int(left.class_id)} {left.class_name}",
                "file_name": str(left.file_name),
                "nearest_record_id": int(right.record_id),
                "nearest_class": f"{int(right.class_id)} {right.class_name}",
                "nearest_file_name": str(right.file_name),
                "cluster": str(left.cluster_label),
                "nearest_cluster": str(right.cluster_label),
            }
        )
    pair_df = pd.DataFrame(pair_rows).sort_values("distance").head(300) if pair_rows else pd.DataFrame()
    return overlap_df, pair_df, threshold


def plotly_state_selected_points(plot_state: Any) -> List[Dict]:
    if not plot_state:
        return []

    selection = plot_state.get("selection") if isinstance(plot_state, dict) else getattr(plot_state, "selection", None)
    if not selection:
        return []

    points = selection.get("points", []) if isinstance(selection, dict) else getattr(selection, "points", [])
    normalized = []
    for point in points or []:
        if isinstance(point, dict):
            normalized.append(point)
            continue
        try:
            normalized.append(dict(point))
            continue
        except Exception:
            pass
        item = {}
        for key in ("curve_number", "curveNumber", "point_number", "pointNumber", "point_index", "pointIndex", "customdata"):
            if hasattr(point, key):
                item[key] = getattr(point, key)
        if item:
            normalized.append(item)
    return normalized


def event_custom_data_from_plotly_event(event: Dict, fig) -> Optional[List]:
    if not event:
        return None

    for key in ("customdata", "customData", "custom_data"):
        value = event.get(key)
        if value is not None:
            return list(value) if isinstance(value, (list, tuple)) else [value]

    curve_number = event.get("curveNumber", event.get("curve_number"))
    point_number = event.get(
        "pointNumber",
        event.get("pointIndex", event.get("point_number", event.get("point_index"))),
    )
    if curve_number is None or point_number is None:
        return None

    try:
        custom = fig.data[int(curve_number)].customdata[int(point_number)]
        return list(custom) if isinstance(custom, (list, tuple, np.ndarray)) else [custom]
    except Exception:
        return None


def feature_cluster_tab(project: Dict, config: Dict) -> None:
    st.subheader("Feature Clustering")
    feature_index_dir = str(project.get("feature_index_dir", ""))
    if not Path(feature_index_dir, "features.npy").exists():
        st.warning(f"features.npy not found: {feature_index_dir}")
        return

    metadata = cached_cluster_metadata(feature_index_dir)
    class_options = ["All"] + sorted(metadata.get("class_counts", {}).keys())
    size_options = ["All"] + list(SIZE_BUCKET_ORDER)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_points = st.number_input(
            "Sample points",
            min_value=500,
            max_value=50000,
            value=5000,
            step=500,
            key="cluster_max_points",
        )
    with col2:
        n_clusters = st.number_input(
            "Clusters per group",
            min_value=1,
            max_value=100,
            value=24,
            step=1,
            key="cluster_n_clusters",
        )
    with col3:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="cluster_seed")
    with col4:
        clustering_scope = st.selectbox(
            "Cluster scope",
            ["global", "per_class", "class_size"],
            format_func=lambda value: {
                "global": "Global",
                "per_class": "Per class",
                "class_size": "Class + size",
            }[value],
            key="cluster_scope",
        )

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        class_filter = st.selectbox("Class filter", class_options, index=0, key="cluster_class_filter")
    with filter_col2:
        size_filter = st.selectbox(
            "Size filter",
            size_options,
            index=0,
            key="cluster_size_filter",
            format_func=lambda value: "All" if value == "All" else SIZE_BUCKET_LABELS.get(value, value),
        )
    with filter_col3:
        color_by = st.selectbox(
            "3D color",
            ["cluster_label", "class_name", "size_bucket", "area_ratio"],
            format_func=lambda value: {
                "cluster_label": "Cluster",
                "class_name": "Class",
                "size_bucket": "Size bucket",
                "area_ratio": "BBox area",
            }[value],
            key="cluster_color_by",
        )

    if st.button("Run clustering", type="primary", key="btn_run_feature_clustering"):
        next_request = {
            "index_dir": feature_index_dir,
            "max_points": int(max_points),
            "n_clusters": int(n_clusters),
            "seed": int(seed),
            "class_filter": "" if class_filter == "All" else class_filter,
            "size_bucket": "" if size_filter == "All" else size_filter,
            "clustering_scope": clustering_scope,
            "color_by": color_by,
        }
        st.session_state["cluster_request"] = next_request
        st.session_state["cluster_result"] = None
        st.session_state["cluster_result_request"] = None
        st.session_state["cluster_graph_selected"] = None
        st.session_state["cluster_compare_points"] = []
        st.session_state["cluster_graph_event_status"] = ""

    request = st.session_state.get("cluster_request")
    if not request:
        st.caption(
            f"Index records={metadata.get('total_records', 0):,}. "
            "Run clustering to check class/size mixing in YOLO feature space."
        )
        class_count_df = pd.DataFrame(
            [{"class": key, "count": value} for key, value in metadata.get("class_counts", {}).items()]
        ).sort_values("count", ascending=False)
        size_count_df = pd.DataFrame(
            [
                {"size_bucket": SIZE_BUCKET_LABELS.get(key, key), "count": metadata.get("size_counts", {}).get(key, 0)}
                for key in SIZE_BUCKET_ORDER
            ]
        )
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(class_count_df, use_container_width=True, hide_index=True)
        with c2:
            st.dataframe(size_count_df, use_container_width=True, hide_index=True)
        return

    result = st.session_state.get("cluster_result")
    result_request = st.session_state.get("cluster_result_request")
    if result is None or result_request != request:
        start = time.time()
        with st.spinner("Building 3D feature clusters..."):
            result = cached_feature_clusters(
                request["index_dir"],
                int(request["max_points"]),
                int(request["n_clusters"]),
                int(request["seed"]),
                str(request.get("class_filter", "")),
                str(request.get("size_bucket", "")),
                str(request.get("clustering_scope", "global")),
            )
        st.session_state["cluster_result"] = result
        st.session_state["cluster_result_request"] = dict(request)
        st.session_state["cluster_result_elapsed"] = time.time() - start
    result_elapsed = float(st.session_state.get("cluster_result_elapsed", 0.0) or 0.0)
    df = result["df"]
    if df.empty:
        st.warning(result.get("message", "No clustering records found."))
        return

    df = df.copy()
    df["area_pct"] = df["area_ratio"] * 100.0
    df["file_name"] = df["image_path"].map(lambda value: Path(str(value)).name)
    bbox_parts = pd.DataFrame(df["bbox_xyxy"].tolist(), columns=["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
    for column in bbox_parts.columns:
        df[column] = bbox_parts[column].astype(int)
    df["bbox_text"] = df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].apply(
        lambda row: f"[{row.bbox_x1}, {row.bbox_y1}, {row.bbox_x2}, {row.bbox_y2}]",
        axis=1,
    )
    st.caption(
        f"sample={result['sample_size']:,}/{result['total_records']:,} | "
        f"clusters={result['n_clusters']} | "
        f"scope={request.get('clustering_scope', 'global')} | "
        f"PCA explained={sum(result['explained_variance_ratio']) * 100:.1f}% | "
        f"elapsed={format_duration(result_elapsed)}"
    )

    st.subheader("Cluster Display")
    display_col1, display_col2, display_col3, display_col4 = st.columns(4)
    display_class_options = ["All"] + sorted(df["class_name"].astype(str).unique().tolist())
    display_group_options = ["All"] + sorted(df["cluster_label"].astype(str).unique().tolist())
    with display_col1:
        display_class = st.selectbox(
            "Display class",
            display_class_options,
            index=0,
            key="cluster_display_class",
        )
    with display_col2:
        close_overlap_only = st.checkbox(
            "Close overlaps only",
            value=False,
            key="cluster_close_overlap_only",
        )
    with display_col3:
        overlap_percentile = st.slider(
            "Overlap closest %",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="cluster_overlap_percentile",
            disabled=not close_overlap_only,
        )
    with display_col4:
        display_group = st.selectbox(
            "Display group",
            display_group_options,
            index=0,
            key="cluster_display_group",
            disabled=close_overlap_only,
        )

    overlap_pairs = pd.DataFrame()
    if close_overlap_only:
        overlap_df, overlap_pairs, overlap_threshold = cross_class_overlap_filter(
            df,
            percentile=float(overlap_percentile),
            max_points=3000,
        )
        if display_class != "All" and not overlap_df.empty:
            focus_rows = overlap_df[overlap_df["class_name"].astype(str) == display_class]
            reverse_rows = overlap_df[overlap_df["nearest_other_class"].astype(str) == display_class]
            focus_record_ids = set(focus_rows["record_id"].astype(int).tolist())
            focus_record_ids.update(int(value) for value in focus_rows["nearest_other_record_id"].tolist() if int(value) >= 0)
            focus_record_ids.update(reverse_rows["record_id"].astype(int).tolist())
            focus_record_ids.update(int(value) for value in reverse_rows["nearest_other_record_id"].tolist() if int(value) >= 0)
            display_df = overlap_df[overlap_df["record_id"].astype(int).isin(focus_record_ids)].copy()
        else:
            display_df = overlap_df.copy()
        if display_df.empty:
            st.warning("No cross-class overlap candidates found in the current class filter.")
        else:
            st.caption(
                f"close overlap candidates={len(display_df):,} | "
                f"threshold={overlap_threshold:.5f} in 3D PCA space | "
                "colored/grouped by class"
            )
            if not overlap_pairs.empty:
                with st.expander("Closest Cross-Class Pairs", expanded=False):
                    st.dataframe(overlap_pairs, use_container_width=True, hide_index=True)
    else:
        display_df = df.copy()
        if display_class != "All":
            display_df = display_df[display_df["class_name"].astype(str) == display_class]
        if display_group != "All":
            display_df = display_df[display_df["cluster_label"].astype(str) == display_group]
        st.caption(f"displayed points={len(display_df):,}/{len(df):,}")

    if display_df.empty:
        return

    graph_col, preview_col = st.columns([3, 1])
    with graph_col:
        st.subheader("3D Cluster View")
        enable_hover_preview = st.checkbox(
            "Hover preview",
            value=False,
            key="cluster_enable_hover_preview",
        )
        st.caption("Click a point to preview/compare. Enable hover preview only when needed.")
        selected_points_3d = []
        plot_color_by = "class_name" if close_overlap_only else str(request.get("color_by", "cluster_label"))
        fig = build_cluster_3d_hover_figure(
            display_df,
            color_by=plot_color_by,
            seed=int(request["seed"]),
        )
        if plotly_events is not None:
            selected_points_3d = plotly_events(
                fig,
                click_event=True,
                select_event=False,
                hover_event=bool(enable_hover_preview),
                override_height=720,
                override_width="100%",
                key=f"cluster_3d_events_{'hover' if enable_hover_preview else 'click'}",
            )
        else:
            plot_state_3d = st.plotly_chart(
                fig,
                use_container_width=True,
                key="cluster_3d_plot",
                on_select="rerun",
                selection_mode="points",
                theme=None,
            )

    selected_event = None
    selected_fig = None
    selected_source = ""
    if selected_points_3d:
        selected_event = selected_points_3d[0]
        selected_fig = fig
        selected_source = "3D"

    if selected_event is not None and selected_fig is not None:
        raw_custom = event_custom_data_from_plotly_event(selected_event, selected_fig)
        custom = raw_custom
        if raw_custom:
            custom = cluster_custom_data_from_record_idx(df, int(raw_custom[0]))
        if custom:
            st.session_state["cluster_graph_selected"] = custom
            st.session_state["cluster_graph_event_status"] = (
                f"selected record={custom[0]} | {custom[5]} | {custom[13]} | {selected_source}"
            )
            if not enable_hover_preview:
                add_cluster_compare_point(custom)
        else:
            st.session_state["cluster_graph_event_status"] = "point event received, but custom data was empty"

    with preview_col:
        st.subheader("Point Preview")
        show_preview_image = st.checkbox(
            "Show image",
            value=True,
            key="cluster_preview_show_image",
        )
        status = st.session_state.get("cluster_graph_event_status", "")
        if status:
            st.caption(status)
        lookup = st.text_input(
            "Find sampled point",
            value="",
            placeholder="record id / file name / class / cluster",
            key="cluster_point_lookup",
        )
        lookup_df = display_df
        query = lookup.strip()
        if query:
            query_lower = query.lower()
            mask = (
                display_df["record_id"].astype(str).str.contains(query, case=False, regex=False, na=False)
                | display_df["file_name"].astype(str).str.lower().str.contains(query_lower, regex=False, na=False)
                | display_df["class_name"].astype(str).str.lower().str.contains(query_lower, regex=False, na=False)
                | display_df["cluster_label"].astype(str).str.lower().str.contains(query_lower, regex=False, na=False)
            )
            lookup_df = display_df[mask]
        if lookup_df.empty:
            st.caption("No matching sampled points.")
        else:
            point_options = (
                lookup_df.sort_values(["class_name", "cluster_label", "record_id"])["record_idx"]
                .astype(int)
                .head(300)
                .tolist()
            )

            def format_point_option(record_idx: int) -> str:
                match = df[df["record_idx"].astype(int) == int(record_idx)]
                if match.empty:
                    return str(record_idx)
                row = match.iloc[0]
                return (
                    f"{int(row.record_id)} | {int(row.class_id)} {row.class_name} | "
                    f"{row.cluster_label} | {row.file_name}"
                )

            selected_record_idx = st.selectbox(
                "Manual preview",
                point_options,
                format_func=format_point_option,
                key="cluster_manual_preview_row",
            )
            if st.button("Show Point", key="cluster_manual_show_point", use_container_width=True):
                row = df[df["record_idx"].astype(int) == int(selected_record_idx)].iloc[0]
                custom = cluster_custom_data_from_row(row)
                st.session_state["cluster_graph_selected"] = custom
                st.session_state["cluster_graph_event_status"] = (
                    f"manual record={custom[0]} | {custom[5]} | {custom[13]}"
                )
                add_cluster_compare_point(custom)
                st.rerun()
        selected = st.session_state.get("cluster_graph_selected")
        if not selected:
            st.caption("Click a point in the 3D graph, or use Manual preview.")
        else:
            try:
                record = crop_record_from_cluster_custom(selected)
                render_cluster_record_panel(
                    record,
                    title=f"Selected | {record.class_id} {record.class_name}",
                    meta=f"{selected[13]} | {selected[14]} | area={float(selected[15]):.2f}%",
                    key_prefix=f"cluster_graph_preview_{record.record_id}",
                    show_image=bool(show_preview_image),
                )
                st.caption(str(selected[16]))
                if st.button("Add to Compare", key="cluster_graph_add_compare", use_container_width=True):
                    add_cluster_compare_point(selected)
                    st.rerun()
            except Exception as exc:
                st.warning(f"Preview load failed: {exc}")

    compare_points = list(st.session_state.get("cluster_compare_points", []) or [])
    if compare_points:
        st.subheader("Point Compare")
        if len(compare_points) == 1:
            st.caption("One point selected. Click one more point to compare.")
        if len(compare_points) >= 2:
            left_row = cluster_row_by_custom(df, compare_points[0])
            right_row = cluster_row_by_custom(df, compare_points[1])
            if left_row is not None and right_row is not None:
                left_xyz = left_row[["x", "y", "z"]].to_numpy(dtype=np.float32)
                right_xyz = right_row[["x", "y", "z"]].to_numpy(dtype=np.float32)
                distance_3d = float(np.linalg.norm(left_xyz - right_xyz))
                same_class = str(left_row.class_name) == str(right_row.class_name)
                st.caption(
                    f"same_class={same_class} | 3D PCA distance={distance_3d:.5f} | "
                    f"left={int(left_row.class_id)} {left_row.class_name} | "
                    f"right={int(right_row.class_id)} {right_row.class_name}"
                )
        compare_cols = st.columns(2)
        for compare_idx, custom in enumerate(compare_points[-2:]):
            try:
                record = crop_record_from_cluster_custom(custom)
                row = cluster_row_by_custom(df, custom)
                meta = f"{custom[13]} | {custom[14]} | area={float(custom[15]):.2f}%"
                if row is not None:
                    meta = f"x={float(row.x):.3f}, y={float(row.y):.3f}, z={float(row.z):.3f} | {meta}"
                with compare_cols[compare_idx % 2]:
                    render_cluster_record_panel(
                        record,
                        title=f"Compare {compare_idx + 1} | {record.class_id} {record.class_name}",
                        meta=meta,
                        key_prefix=f"cluster_compare_{compare_idx}_{record.record_id}",
                        show_image=bool(show_preview_image),
                    )
            except Exception as exc:
                with compare_cols[compare_idx % 2]:
                    st.warning(f"Compare point load failed: {exc}")
        clear_col1, clear_col2 = st.columns([1, 3])
        with clear_col1:
            if st.button("Clear Compare", key="cluster_compare_clear", use_container_width=True):
                st.session_state["cluster_compare_points"] = []
                st.rerun()

    summary = result.get("summary", pd.DataFrame())
    if close_overlap_only:
        st.subheader("Close Overlap Class Summary")
        class_counts = (
            display_df.groupby(["class_id", "class_name"], as_index=False)
            .agg(
                count=("record_id", "size"),
                mean_nearest_other_distance=("nearest_other_distance", "mean"),
                min_nearest_other_distance=("nearest_other_distance", "min"),
            )
            .sort_values(["count", "min_nearest_other_distance"], ascending=[False, True])
        )
        class_counts["mean_nearest_other_distance"] = class_counts["mean_nearest_other_distance"].round(5)
        class_counts["min_nearest_other_distance"] = class_counts["min_nearest_other_distance"].round(5)
        pair_counts = (
            display_df.groupby(["class_name", "nearest_other_class"], as_index=False)
            .agg(
                count=("record_id", "size"),
                mean_distance=("nearest_other_distance", "mean"),
                min_distance=("nearest_other_distance", "min"),
            )
            .sort_values(["count", "min_distance"], ascending=[False, True])
        )
        pair_counts["mean_distance"] = pair_counts["mean_distance"].round(5)
        pair_counts["min_distance"] = pair_counts["min_distance"].round(5)
        counts_col1, counts_col2 = st.columns(2)
        with counts_col1:
            st.dataframe(class_counts, use_container_width=True, hide_index=True)
        with counts_col2:
            st.dataframe(pair_counts, use_container_width=True, hide_index=True)
    else:
        st.subheader("Mixed Cluster Summary")
        st.caption("Low class_purity means multiple classes are close in feature space. Low size_purity means different bbox scales are mixed.")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        counts_col1, counts_col2 = st.columns(2)
        with counts_col1:
            class_counts = (
                df.groupby(["cluster_label", "class_name"])
                .size()
                .reset_index(name="count")
                .sort_values(["cluster_label", "count"], ascending=[True, False])
            )
            st.dataframe(class_counts, use_container_width=True, hide_index=True)
        with counts_col2:
            size_counts = (
                df.groupby(["cluster_label", "size_bucket"])
                .size()
                .reset_index(name="count")
                .sort_values(["cluster_label", "count"], ascending=[True, False])
            )
            st.dataframe(size_counts, use_container_width=True, hide_index=True)

    sample_source_df = display_df
    if close_overlap_only:
        sample_group_column = "class_name"
        sample_options = sorted(sample_source_df["class_name"].astype(str).unique().tolist())
        sample_select_label = "Class samples"
    else:
        sample_group_column = "cluster_label"
        sample_options = sorted(sample_source_df["cluster_label"].astype(str).unique().tolist())
        if not summary.empty:
            summary_clusters = [cluster for cluster in summary["cluster_label"].tolist() if cluster in sample_options]
            sample_options = summary_clusters or sample_options
        sample_select_label = "Cluster samples"
    selected_sample_group = st.selectbox(
        sample_select_label,
        sample_options,
        key="cluster_sample_select",
    )
    selected_sample_key = slugify(str(selected_sample_group))
    show_sample_images = st.checkbox(
        "Show sample images",
        value=True,
        key="cluster_sample_show_images",
    )
    cluster_df = (
        sample_source_df[sample_source_df[sample_group_column].astype(str) == str(selected_sample_group)]
        .sort_values(["class_name", "size_bucket", "record_id"])
        .head(48)
    )
    cols = st.columns(5)
    for idx, row in enumerate(cluster_df.itertuples(index=False)):
        record = CropRecord(
            record_id=int(row.record_id),
            image_path=str(row.image_path),
            label_path=str(row.label_path),
            class_id=int(row.class_id),
            class_name=str(row.class_name),
            bbox_xyxy=tuple(row.bbox_xyxy),
            image_width=int(row.image_width),
            image_height=int(row.image_height),
            annotation_line=int(row.annotation_line),
        )
        with cols[idx % 5]:
            st.caption(f"{int(row.class_id)} {row.class_name} | {row.size_bucket} | {float(row.area_pct):.2f}%")
            try:
                thumb_ok = True
                if show_sample_images:
                    thumb_ok = render_record_thumb(record, wrapper_class="cluster-sample-image")
                    if not thumb_ok:
                        st.caption(f"record {row.record_id} image load failed")
                else:
                    st.caption("image hidden")
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button(
                        "View",
                        key=f"cluster_preview_{selected_sample_key}_{row.record_id}",
                        use_container_width=True,
                        disabled=not show_sample_images,
                    ):
                        crop = crop_from_record(record)
                        set_preview_image(
                            crop,
                            f"{sample_select_label}={selected_sample_group} | {row.class_name} | record={row.record_id}",
                        )
                        st.rerun()
                with action_col2:
                    if st.button("Data", key=f"cluster_open_data_{selected_sample_key}_{row.record_id}", use_container_width=True):
                        open_data_location(record.image_path)
                render_path_selector(
                    record.image_path,
                    record,
                    key=f"cluster_path_{selected_sample_key}_{row.record_id}",
                )
                if st.button("Neighbors", key=f"cluster_neighbors_{selected_sample_key}_{row.record_id}", use_container_width=True):
                    request_db_neighbor_search(record, 20)
                    st.rerun()
            except Exception:
                st.caption(f"record {row.record_id} load failed")

    render_selected_paths_panel(key_prefix="cluster_selected_paths")
    run_pending_db_neighbor_search(project, config)
    render_db_neighbor_results("cluster")
    render_preview_image()


def crop_search_tab(project: Dict, config: Dict) -> None:
    st.subheader("Crop Image Search")
    st.caption(
        "Uploaded crop search uses the crop as a full image. "
        "For the most faithful YOLO feature match, use Video Detection and search the selected bbox on the original frame."
    )
    feature_index_dir = str(project.get("feature_index_dir", ""))
    if st.session_state.get("yolo_feature_index") is None:
        st.caption(f"YOLO feature index will auto-load on first search: {feature_index_dir}")
    else:
        loaded_dir = st.session_state.get("yolo_feature_index_dir")
        index = st.session_state.get("yolo_feature_index")
        st.caption(f"YOLO feature index ready: {loaded_dir} | records={len(index.records):,}")
    top_k = st.slider("Top-k", min_value=5, max_value=100, value=20, step=5, key="crop_topk")
    uploaded = st.file_uploader(
        "오감지 crop 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="crop_query_uploader",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 3])
        rendered_search_results = False
        with left:
            st.markdown('<div class="query-preview-title">Query crop</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="query-preview-meta">{html.escape(Path(uploaded.name).name)}</div>',
                unsafe_allow_html=True,
            )
            render_thumb_image(image, uploaded.name, cache_key=f"query:{uploaded.name}:{image.size}")
            run_yolo = st.button(
                "Search by YOLO Feature",
                type="primary",
                use_container_width=True,
                key="btn_search_crop_yolo_feature",
            )
            if st.button("View", key="btn_preview_query_crop", use_container_width=True):
                set_preview_image(image, f"Query crop | {uploaded.name}")
                st.rerun()
        if run_yolo:
            with right:
                search_yolo_feature_crop(
                    image,
                    top_k=top_k,
                    key_prefix="crop_yolo_feature_results",
                    feature_index_dir=feature_index_dir,
                    device=config["device"],
                )
                rendered_search_results = True
        if (
            not rendered_search_results
            and st.session_state.get("last_results")
            and st.session_state.get("last_results_context") == "crop"
        ):
            with right:
                st.caption("Last crop search results")
                show_results(
                    st.session_state.get("last_results", []),
                    key_prefix=st.session_state.get("last_results_key_prefix", "crop_yolo_feature_results"),
                )


def is_fire_w122_model(weights_path: str) -> bool:
    name = Path(weights_path).name.upper()
    return (
        "PJ_FIRE" in name
        or "FIRE" in name
        or "W122" in name
        or "화재" in Path(weights_path).name
    )


def infer_repo_path_for_model(weights_path: str) -> str:
    name = Path(weights_path).name.upper()
    if (
        "YOLOV7" in name
        or is_fire_w122_model(weights_path)
    ) and Path("external/yolov7").exists():
        return "external/yolov7"
    if "YOLOV9" in name and Path("external/yolov9").exists():
        return "external/yolov9"
    if Path("external/yolov9").exists():
        return "external/yolov9"
    if Path("external/yolov7").exists():
        return "external/yolov7"
    return ""


def default_yolo_feature_index_dir(weights_path: str) -> str:
    if not weights_path:
        return "artifacts/yolo_feature_index"
    if is_fire_w122_model(weights_path):
        return FIREDB_YOLO_FEATURE_INDEX_DIR
    stem = Path(weights_path).stem
    return f"artifacts/yolo_feature_index_{stem}"


def maybe_auto_load_yolo_feature_index(feature_index_dir: str, device: str) -> None:
    current_dir = st.session_state.get("yolo_feature_index_dir")
    if st.session_state.get("yolo_feature_index") is not None and current_dir == feature_index_dir:
        return
    if st.session_state.get("yolo_feature_index") is not None and current_dir != feature_index_dir:
        st.session_state["yolo_feature_index"] = None
        st.session_state["yolo_feature_index_dir"] = None

    root = Path(feature_index_dir)
    if not (root / "index.faiss").exists() or not (root / "config.json").exists() or not index_records_ready(root):
        return

    try:
        with st.spinner("Loading YOLO feature index..."):
            st.session_state["yolo_feature_index"] = YoloFeatureIndex.load(
                feature_index_dir,
                device=device,
            )
            st.session_state["yolo_feature_index_dir"] = feature_index_dir
        st.success("YOLO feature index auto-loaded")
    except Exception as exc:
        st.warning(f"YOLO feature index auto-load failed: {exc}")


def ensure_yolo_feature_index_loaded(feature_index_dir: Optional[str] = None, device: Optional[str] = None) -> bool:
    target_dir = feature_index_dir or st.session_state.get("yolo_feature_index_dir") or FIREDB_YOLO_FEATURE_INDEX_DIR
    target_device = device or st.session_state.get("cfg_device", "cpu")
    current_dir = st.session_state.get("yolo_feature_index_dir")
    if st.session_state.get("yolo_feature_index") is not None and current_dir == target_dir:
        return True

    root = Path(target_dir)
    if not (root / "index.faiss").exists() or not (root / "config.json").exists() or not index_records_ready(root):
        st.warning(f"YOLO feature index files not found: {target_dir}")
        return False

    start = time.time()
    status = st.empty()
    try:
        with st.spinner(f"Loading YOLO feature index: {target_dir}"):
            st.session_state["yolo_feature_index"] = YoloFeatureIndex.load(
                target_dir,
                device=target_device,
            )
            st.session_state["yolo_feature_index_dir"] = target_dir
        status.success(f"YOLO feature index loaded in {format_duration(time.time() - start)}")
        return True
    except Exception as exc:
        st.session_state["yolo_feature_index"] = None
        st.session_state["yolo_feature_index_dir"] = None
        status.error(f"YOLO feature index load failed: {exc}")
        return False


def show_yolo_feature_index_status(weights_path: str) -> None:
    index = st.session_state.get("yolo_feature_index")
    loaded_dir = st.session_state.get("yolo_feature_index_dir")
    if index is None:
        st.caption("YOLO feature index: not loaded")
        return

    config = getattr(index, "config", {}) or {}
    index_weight = config.get("weights_path", "")
    model_name = Path(weights_path).name if weights_path else ""
    index_model_name = Path(index_weight).name if index_weight else ""
    st.caption(
        f"Loaded YOLO feature index: {loaded_dir} | records={len(index.records):,} | dim={config.get('dim', '-')}"
    )
    if model_name and index_model_name and model_name != index_model_name:
        st.warning(
            f"Model/index mismatch: selected={model_name}, index_built_with={index_model_name}"
        )
    elif model_name and index_model_name:
        st.success(f"Model/index matched: {model_name}")


def yolo_detector_key(
    repo_path: str,
    weights_path: str,
    device: str,
    img_size: int,
    class_names: Dict[int, str],
) -> tuple:
    class_key = tuple(sorted((int(key), str(value)) for key, value in class_names.items()))
    return (
        str(Path(repo_path).resolve()) if repo_path else "",
        str(Path(weights_path).resolve()) if weights_path else "",
        str(device),
        int(img_size),
        class_key,
    )


def get_yolo_detector(
    repo_path: str,
    weights_path: str,
    device: str,
    img_size: int,
    conf_thres: float,
    class_names: Dict[int, str],
) -> YoloV7Detector:
    key = yolo_detector_key(repo_path, weights_path, device, img_size, class_names)
    detector = st.session_state.get("yolo_detector")
    if detector is not None and st.session_state.get("yolo_detector_key") == key:
        detector.conf_thres = float(conf_thres)
        st.caption("YOLO detector cache reused")
        return detector

    st.session_state["yolo_detector"] = None
    st.session_state["yolo_detector_key"] = None
    with st.spinner("Loading YOLO detector..."):
        detector = YoloV7Detector(
            repo_path=repo_path or None,
            weights_path=weights_path,
            device=device,
            img_size=int(img_size),
            conf_thres=float(conf_thres),
            class_names=class_names,
        )
    st.session_state["yolo_detector"] = detector
    st.session_state["yolo_detector_key"] = key
    st.success("YOLO detector loaded")
    return detector


def model_selector(key: str, default_path: str = "", label: str = "YOLO model") -> str:
    model_options = [str(path) for path in model_files()]
    options = model_options + ["Custom path..."]
    default_index = 0
    for idx, option in enumerate(options):
        if default_path and option == default_path:
            default_index = idx
            break
    else:
        for idx, option in enumerate(model_options):
            if is_fire_w122_model(option):
                default_index = idx
                break
        else:
            for idx, option in enumerate(model_options):
                if "YOLOV7" in Path(option).name.upper():
                    default_index = idx
                    break

    selected = st.selectbox(
        label,
        options,
        index=default_index if options else 0,
        format_func=lambda value: Path(value).name if value != "Custom path..." else value,
        key=key,
    )
    if selected == "Custom path...":
        return st.text_input("YOLO weights path", value=default_path, key=f"{key}_custom")

    st.caption(selected)
    return selected


def project_form_defaults() -> Dict:
    project = active_project() or default_fire_project() or {}
    return dict(project)


def normalize_project(project: Dict) -> Dict:
    name = str(project.get("name", "")).strip() or slugify(Path(str(project.get("weights_path", "model"))).stem)
    normalized = dict(project)
    normalized["name"] = name
    layout = str(project.get("dataset_layout", DATASET_LAYOUT_SINGLE) or DATASET_LAYOUT_SINGLE).strip()
    if layout == DATASET_LAYOUT_NESTED_JPEGIMAGES_LABELS:
        layout = DATASET_LAYOUT_NESTED_IMAGE_LABELS
    if layout not in {DATASET_LAYOUT_SINGLE, DATASET_LAYOUT_NESTED_IMAGE_LABELS}:
        layout = DATASET_LAYOUT_SINGLE
    normalized["dataset_layout"] = layout
    normalized["images_dir"] = str(project.get("images_dir", "")).strip()
    normalized["labels_dir"] = str(project.get("labels_dir", "")).strip()
    normalized["data_yaml"] = str(project.get("data_yaml", "")).strip()
    normalized["weights_path"] = str(project.get("weights_path", "")).strip()
    normalized["repo_path"] = str(project.get("repo_path", "")).strip()
    normalized["feature_index_dir"] = str(project.get("feature_index_dir", "")).strip()
    normalized["img_size"] = int(project.get("img_size", 640) or 640)
    normalized["expand"] = float(project.get("expand", 0.08) or 0.08)
    normalized["class_ids"] = str(project.get("class_ids", "")).strip()
    normalized["max_records"] = int(project.get("max_records", 0) or 0)
    normalized["build_max_workers"] = int(project.get("build_max_workers", 0) or 0)
    normalized["feature_batch_size"] = int(project.get("feature_batch_size", 0) or 0)
    normalized["image_size_cache"] = str(project.get("image_size_cache", "")).strip()
    faiss_type = str(project.get("faiss_type", "ivfpq") or "ivfpq").strip().lower()
    normalized["faiss_type"] = faiss_type if faiss_type in {"flat", "ivfpq"} else "ivfpq"
    normalized["faiss_nlist"] = int(project.get("faiss_nlist", 4096) or 4096)
    normalized["faiss_nprobe"] = int(project.get("faiss_nprobe", 32) or 32)
    normalized["faiss_train_size"] = int(project.get("faiss_train_size", 200000) or 200000)
    normalized["faiss_gpu"] = config_bool(project.get("faiss_gpu", False))
    normalized["script_text"] = str(project.get("script_text", ""))
    normalized["script_path"] = str(project.get("script_path", "")).strip()
    return normalized


def default_project_script(project: Dict) -> str:
    name = str(project.get("name", "project"))
    return f"""# Project: {name}
# Purpose: custom notes or commands for this feature project.
# This script is saved with the project but is not executed automatically by the app.

$ProjectName = "{name}"
$FeatureIndexDir = "{project.get('feature_index_dir', '')}"
$WeightsPath = "{project.get('weights_path', '')}"
$DatasetLayout = "{project.get('dataset_layout', DATASET_LAYOUT_SINGLE)}"
$ImagesDir = "{project.get('images_dir', '')}"
$LabelsDir = "{project.get('labels_dir', '')}"
$ClassIds = "{project.get('class_ids', '')}"
$MaxRecords = "{project.get('max_records', 0)}"
$FeatureBatchSize = "{project.get('feature_batch_size', 0)}"
$ImageSizeCache = "{project.get('image_size_cache', '')}"
$FaissType = "{project.get('faiss_type', 'ivfpq')}"
$FaissGpu = "{project.get('faiss_gpu', False)}"

Write-Host "Project: $ProjectName"
Write-Host "Dataset layout: $DatasetLayout"
Write-Host "Feature index: $FeatureIndexDir"
"""


def default_project_script_path(project: Dict) -> str:
    return str(Path("artifacts") / "project_scripts" / f"{slugify(str(project.get('name', 'project')))}.ps1")


def write_project_script(project: Dict) -> Dict:
    project = dict(project)
    script_text = str(project.get("script_text", ""))
    script_path = str(project.get("script_path", "")).strip() or default_project_script_path(project)
    if script_text.strip():
        path = Path(script_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script_text, encoding="utf-8")
        project["script_path"] = str(path)
    return project


def project_required_errors(project: Dict) -> List[str]:
    errors = []
    layout = str(project.get("dataset_layout", DATASET_LAYOUT_SINGLE) or DATASET_LAYOUT_SINGLE)
    required = ["images_dir", "data_yaml", "weights_path", "repo_path", "feature_index_dir"]
    if layout == DATASET_LAYOUT_SINGLE:
        required.insert(1, "labels_dir")
    for key in required:
        if not str(project.get(key, "")).strip():
            errors.append(f"{key} is required")
    return errors


def path_check_row(item: str, path_text: str, required: bool = True, kind: str = "path") -> Dict:
    path = Path(str(path_text or ""))
    exists = path.exists()
    if exists:
        status = "OK"
        detail = str(path)
    elif required:
        status = "FAIL"
        detail = f"not found: {path}"
    else:
        status = "WARN"
        detail = f"will be created or optional: {path}"
    return {"item": item, "status": status, "kind": kind, "detail": detail}


def project_preflight_rows(project: Dict) -> List[Dict]:
    project = normalize_project(project)
    rows: List[Dict] = []
    layout = str(project.get("dataset_layout", DATASET_LAYOUT_SINGLE))
    rows.append({"item": "dataset_layout", "status": "OK", "kind": "config", "detail": layout})
    rows.append(path_check_row("images_dir / dataset_root", project.get("images_dir", ""), required=True, kind="input"))
    if layout == DATASET_LAYOUT_SINGLE:
        rows.append(path_check_row("labels_dir", project.get("labels_dir", ""), required=True, kind="input"))
    else:
        pairs = discover_nested_image_label_pairs(project.get("images_dir", ""))
        rows.append(
            {
                "item": "nested image/label pairs",
                "status": "OK" if pairs else "FAIL",
                "kind": "input",
                "detail": f"{len(pairs):,} pairs found (*.JPEGImages|images + labels)",
            }
        )
    rows.append(path_check_row("data_yaml", project.get("data_yaml", ""), required=True, kind="input"))
    rows.append(path_check_row("weights_path", project.get("weights_path", ""), required=True, kind="model"))
    rows.append(path_check_row("repo_path", project.get("repo_path", ""), required=True, kind="model"))

    output_dir = Path(str(project.get("feature_index_dir", "")))
    output_parent = output_dir.parent if output_dir.name else Path(".")
    rows.append(
        {
            "item": "feature_index_dir",
            "status": "OK" if output_parent.exists() else "WARN",
            "kind": "output",
            "detail": f"{output_dir} | parent {'exists' if output_parent.exists() else 'will be created'}",
        }
    )
    records_json = project_records_json(project)
    rows.append(
        {
            "item": "records_json",
            "status": "WARN" if records_json.exists() else "OK",
            "kind": "cache",
            "detail": f"{records_json} {'exists and will be reused unless rebuild is checked' if records_json.exists() else 'will be created'}",
        }
    )
    return rows


def project_preflight_errors(project: Dict) -> List[str]:
    rows = project_preflight_rows(project)
    return [f"{row['item']}: {row['detail']}" for row in rows if row.get("status") == "FAIL"]


def tail_text(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(data[-lines:])
    except Exception as exc:
        return f"Cannot read {path}: {exc}"


def parse_duration_seconds(value: str) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    total = 0
    matched = False
    for amount, unit in re.findall(r"(\d+)\s*(h|m|s)", text):
        matched = True
        amount_i = int(amount)
        if unit == "h":
            total += amount_i * 3600
        elif unit == "m":
            total += amount_i * 60
        else:
            total += amount_i
    return total if matched else None


def parse_progress_line(line: str) -> Optional[Dict]:
    match = re.search(
        r"(?P<pct>\d+(?:\.\d+)?)%\s+\|\s+(?P<message>.*?)\s+\|\s+elapsed=(?P<elapsed>.*?)(?:\s+eta=(?P<eta>.*))?$",
        str(line).strip(),
    )
    if not match:
        return None
    message = match.group("message")
    done_total = re.search(r"(\d+)\s*/\s*(\d+)", message)
    return {
        "pct": float(match.group("pct")),
        "message": message,
        "done": int(done_total.group(1)) if done_total else None,
        "total": int(done_total.group(2)) if done_total else None,
        "elapsed": (match.group("elapsed") or "").strip(),
        "eta": (match.group("eta") or "").strip(),
        "eta_seconds": parse_duration_seconds(match.group("eta") or ""),
    }


def last_progress_from_log(path: Path) -> Optional[Dict]:
    text = tail_text(path, lines=300)
    if not text:
        return None
    progress = None
    for line in text.splitlines():
        parsed = parse_progress_line(line)
        if parsed:
            progress = parsed
    return progress


def regex_last_int(path: Path, pattern: str) -> Optional[int]:
    text = tail_text(path, lines=300)
    value = None
    for match in re.finditer(pattern, text):
        value = int(str(match.group(1)).replace(",", ""))
    return value


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        try:
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
            if not handle:
                return False
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return bool(ok) and int(code.value) == still_active
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="ascii", errors="ignore").strip())
    except Exception:
        return None


def build_progress_summary(project: Dict) -> Dict:
    project = normalize_project(project)
    log_root = project_log_root(project)
    records_json = project_records_json(project)
    feature_root = Path(project.get("feature_index_dir", ""))
    ready = (
        (feature_root / "index.faiss").exists()
        and (feature_root / "features.npy").exists()
        and (feature_root / "config.json").exists()
        and index_records_ready(feature_root)
    )

    launcher_pid = read_pid(log_root / "launcher.pid")
    running = bool(launcher_pid and pid_is_running(launcher_pid))
    launcher_stderr = tail_text(log_root / "launcher.stderr.log", lines=80)
    launcher_log = tail_text(log_root / "launcher.log", lines=200)
    prepare_progress = last_progress_from_log(log_root / "prepare_records.out.log")
    prepare_records = regex_last_int(log_root / "prepare_records.out.log", r"records=([\d,]+)")
    merge_progress = last_progress_from_log(log_root / "merge.out.log")

    shard_rows = []
    total_done = 0
    total_records = 0
    eta_values = []
    for shard_log in sorted(log_root.glob("shard_*.out.log")):
        idx_match = re.search(r"shard_(\d+)\.out\.log", shard_log.name)
        shard_idx = int(idx_match.group(1)) if idx_match else len(shard_rows)
        progress = last_progress_from_log(shard_log)
        records = regex_last_int(shard_log, r"records=([\d,]+)")
        pid = read_pid(log_root / f"shard_{shard_idx}.pid")
        done = progress.get("done") if progress else None
        total = progress.get("total") if progress else records
        if done is not None and total:
            total_done += int(done)
            total_records += int(total)
        elif records:
            total_records += int(records)
        if progress and progress.get("eta_seconds") is not None:
            eta_values.append(int(progress["eta_seconds"]))
        exit_match = re.search(rf"shard {shard_idx} finished exit_code=(\d+)", launcher_log)
        shard_rows.append(
            {
                "shard": shard_idx,
                "pid": pid or "",
                "running": bool(pid and pid_is_running(pid)),
                "records": records or total or "",
                "done": done if done is not None else "",
                "pct": round(float(progress["pct"]), 2) if progress else "",
                "eta": progress.get("eta", "") if progress else "",
                "message": progress.get("message", "") if progress else "",
                "exit_code": exit_match.group(1) if exit_match else "",
                "log": shard_log.name,
            }
        )

    merge_started = "starting merge" in launcher_log
    merge_done_match = re.search(r"merge finished exit_code=(\d+)", launcher_log)
    failure_text = ""
    if launcher_stderr.strip():
        failure_text = launcher_stderr.strip()
    failed_shards = [row for row in shard_rows if str(row.get("exit_code")) not in ("", "0")]
    if failed_shards:
        failure_text = failure_text or f"Shard failures: {failed_shards}"

    if ready:
        stage = "Completed"
        pct = 1.0
        eta_seconds = 0
    elif failure_text and not running:
        stage = "Failed"
        pct = 0.0
        eta_seconds = None
    elif merge_started and not merge_done_match:
        stage = "Merging"
        pct = 0.95 + (float(merge_progress["pct"]) / 100.0 * 0.05) if merge_progress else 0.95
        eta_seconds = merge_progress.get("eta_seconds") if merge_progress else None
    elif shard_rows:
        stage = "Encoding features"
        pct = float(total_done / total_records) if total_done and total_records else 0.0
        eta_seconds = max(eta_values) if eta_values else None
    elif prepare_progress:
        stage = "Preparing records"
        pct = float(prepare_progress["pct"]) / 100.0
        eta_seconds = prepare_progress.get("eta_seconds")
    elif launcher_pid:
        stage = "Launching" if running else "Stopped"
        pct = 0.0
        eta_seconds = None
    else:
        stage = "Not started"
        pct = 0.0
        eta_seconds = None

    finish_time = ""
    if eta_seconds and eta_seconds > 0:
        finish_time = (datetime.now() + timedelta(seconds=int(eta_seconds))).strftime("%H:%M:%S")

    return {
        "log_root": log_root,
        "stage": stage,
        "pct": max(0.0, min(1.0, pct)),
        "eta_seconds": eta_seconds,
        "finish_time": finish_time,
        "launcher_pid": launcher_pid,
        "launcher_running": running,
        "records_json": records_json,
        "prepare_records": prepare_records,
        "shard_rows": shard_rows,
        "failure_text": failure_text,
        "launcher_log": launcher_log,
        "merge_progress": merge_progress,
        "ready": ready,
    }


def start_project_feature_build(
    project: Dict,
    device_arg: str,
    num_shards: int,
    max_workers: int,
    force_prepare: bool = False,
) -> int:
    project = normalize_project(project)
    log_root = project_log_root(project)
    log_root.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.log", "*.pid"):
        for old_file in log_root.glob(pattern):
            try:
                old_file.unlink()
            except OSError:
                pass

    cmd = [
        sys.executable,
        "-u",
        str(Path("scripts") / "run_yolo_feature_project_build.py"),
        "--project-name",
        project["name"],
        "--images-dir",
        project["images_dir"],
        "--labels-dir",
        project["labels_dir"],
        "--data-yaml",
        project["data_yaml"],
        "--repo-path",
        project["repo_path"],
        "--weights-path",
        project["weights_path"],
        "--index-dir",
        project["feature_index_dir"],
        "--device",
        device_arg,
        "--img-size",
        str(project["img_size"]),
        "--expand",
        str(project["expand"]),
        "--dataset-layout",
        str(project.get("dataset_layout", DATASET_LAYOUT_SINGLE)),
        "--num-shards",
        str(max(1, int(num_shards))),
        "--max-workers",
        str(max(0, int(max_workers))),
        "--max-records",
        str(max(0, int(project.get("max_records", 0) or 0))),
        "--class-ids",
        str(project.get("class_ids", "")),
        "--feature-batch-size",
        str(max(0, int(project.get("feature_batch_size", 0) or 0))),
        "--image-size-cache",
        str(project.get("image_size_cache", "")),
        "--faiss-type",
        str(project.get("faiss_type", "ivfpq")),
        "--nlist",
        str(project.get("faiss_nlist", 4096)),
        "--nprobe",
        str(project.get("faiss_nprobe", 32)),
        "--train-size",
        str(project.get("faiss_train_size", 200000)),
        "--records-json",
        str(project_records_json(project)),
        "--shard-root",
        str(project_shard_root(project)),
        "--log-root",
        str(log_root),
    ]
    if config_bool(project.get("faiss_gpu", False)):
        cmd.append("--faiss-gpu")
    if force_prepare:
        cmd.append("--force-prepare")
    stdout = (log_root / "launcher.stdout.log").open("w", encoding="utf-8")
    stderr = (log_root / "launcher.stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(Path.cwd()), stdout=stdout, stderr=stderr)
    stdout.close()
    stderr.close()
    (log_root / "launcher.pid").write_text(str(proc.pid), encoding="ascii")
    return int(proc.pid)


def project_build_status(project: Dict) -> None:
    summary = build_progress_summary(project)
    log_root = summary["log_root"]
    st.caption(f"Build log: {log_root}")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Stage", summary["stage"])
    with metric_col2:
        st.metric("Progress", f"{summary['pct'] * 100:.1f}%")
    with metric_col3:
        eta_seconds = summary.get("eta_seconds")
        st.metric("Remaining", format_duration(int(eta_seconds)) if eta_seconds is not None else "-")
    with metric_col4:
        st.metric("ETA finish", summary.get("finish_time") or "-")
    st.progress(float(summary["pct"]))

    pid = summary.get("launcher_pid")
    if pid:
        st.caption(f"Launcher PID: {pid} | running={summary.get('launcher_running')}")
    if summary.get("prepare_records"):
        st.caption(f"Prepared records: {int(summary['prepare_records']):,}")
    if summary.get("records_json"):
        st.caption(f"Records JSON: {summary['records_json']}")
    if summary.get("merge_progress"):
        progress = summary["merge_progress"]
        st.caption(f"Merge: {progress.get('message', '')} | eta={progress.get('eta', '-')}")

    shard_rows = summary.get("shard_rows", [])
    if shard_rows:
        st.subheader("Shard Progress")
        st.dataframe(pd.DataFrame(shard_rows), use_container_width=True, hide_index=True)

    if st.button("Refresh Build Status", key=f"refresh_build_status_{slugify(str(project.get('name', 'project')))}"):
        st.rerun()

    auto_refresh = st.checkbox(
        "Auto refresh build status every 10s",
        value=False,
        key=f"auto_refresh_build_status_{slugify(str(project.get('name', 'project')))}",
    )

    launcher = summary.get("launcher_log", "")
    if launcher:
        with st.expander("Launcher log", expanded=False):
            st.code("\n".join(launcher.splitlines()[-40:]), language="text")

    shard_logs = sorted(log_root.glob("shard_*.out.log"))
    if shard_logs:
        latest = shard_logs[-1]
        with st.expander(f"{latest.name} tail", expanded=False):
            st.code(tail_text(latest, lines=40), language="text")

    prepare_err = tail_text(log_root / "prepare_records.err.log", lines=40)
    shard_errs = "\n".join(tail_text(path, lines=40) for path in sorted(log_root.glob("shard_*.err.log")) if path.exists())
    merge_err = tail_text(log_root / "merge.err.log", lines=40)
    launcher_err = tail_text(log_root / "launcher.stderr.log", lines=80)
    errors = "\n".join(part for part in (launcher_err, prepare_err, shard_errs, merge_err, summary.get("failure_text", "")) if str(part).strip())
    if errors.strip():
        with st.expander("Build errors", expanded=True):
            st.code(errors, language="text")

    if auto_refresh and summary.get("stage") not in {"Completed", "Failed", "Not started", "Stopped"}:
        time.sleep(10)
        st.rerun()


def project_manager_tab(config: Dict) -> None:
    st.subheader("Feature Project Builder")
    st.caption("A project binds one model, one YOLO txt DB, and one YOLO feature index.")

    projects = load_projects()
    if projects:
        st.dataframe(pd.DataFrame([project_to_row(project) for project in projects]), use_container_width=True, hide_index=True)
    else:
        st.info(f"No projects registered yet. Registry: {PROJECTS_PATH}")

    names = [project.get("name", "") for project in projects]
    selected_name = st.selectbox(
        "Project to edit",
        ["New project"] + names,
        index=1 if names else 0,
        key="project_manager_select",
    )
    if selected_name != "New project":
        selected_project = get_project(selected_name) or {}
        set_active_project(selected_project)
        defaults = dict(selected_project)
    else:
        defaults = project_form_defaults()

    with st.form("project_form"):
        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("Project name", value=str(defaults.get("name", "")) or "new_project")
            description = st.text_input("Description", value=str(defaults.get("description", "")))
            weights_path = model_selector(
                key="project_form_model_select",
                default_path=str(defaults.get("weights_path", "")),
                label="Model weights",
            )
            repo_path_default = str(defaults.get("repo_path", "")) or infer_repo_path_for_model(weights_path)
            repo_path = st.text_input("YOLO repo path", value=repo_path_default)
            feature_default = str(defaults.get("feature_index_dir", "")) or default_yolo_feature_index_dir(weights_path)
            feature_index_dir = st.text_input("YOLO feature index dir", value=feature_default)
        with col2:
            dataset_layout = st.selectbox(
                "Dataset layout",
                [DATASET_LAYOUT_SINGLE, DATASET_LAYOUT_NESTED_IMAGE_LABELS],
                index=1
                if str(defaults.get("dataset_layout", DATASET_LAYOUT_SINGLE))
                in {DATASET_LAYOUT_NESTED_JPEGIMAGES_LABELS, DATASET_LAYOUT_NESTED_IMAGE_LABELS}
                else 0,
                format_func=lambda value: {
                    DATASET_LAYOUT_SINGLE: "Single images dir + labels dir",
                    DATASET_LAYOUT_NESTED_IMAGE_LABELS: "Nested */JPEGImages or */images + */labels",
                }[value],
            )
            images_dir = st.text_input(
                "Images dir / Dataset root",
                value=str(defaults.get("images_dir", FIREDB_IMAGES_DIR)),
                help=(
                    "Parent folder that contains multiple subfolders, each with JPEGImages/images and labels."
                    if dataset_layout == DATASET_LAYOUT_NESTED_IMAGE_LABELS
                    else "Folder containing images. Relative paths are mapped to Labels dir."
                ),
            )
            labels_dir = st.text_input(
                "Labels dir (ignored for nested layout)",
                value=str(defaults.get("labels_dir", FIREDB_LABELS_DIR)),
                help=(
                    "Ignored for nested layout. Labels are resolved as sibling labels folders next to each JPEGImages/images folder."
                    if dataset_layout == DATASET_LAYOUT_NESTED_IMAGE_LABELS
                    else "Folder containing YOLO txt labels matching Images dir relative paths."
                ),
            )
            st.caption(
                "Nested layout: put the parent root in Images dir / Dataset root. "
                "The app finds every */JPEGImages or */images with sibling */labels."
            )
            data_yaml = st.text_input("data.yaml", value=str(defaults.get("data_yaml", FIREDB_DATA_YAML)))
            img_size = st.number_input("Image size", min_value=320, max_value=1536, value=int(defaults.get("img_size", 640) or 640), step=32)
            expand = st.number_input("BBox expand", min_value=0.0, max_value=0.5, value=float(defaults.get("expand", 0.08) or 0.08), step=0.01)
            class_ids = st.text_input(
                "Build class ids",
                value=str(defaults.get("class_ids", "")),
                help="Optional comma-separated class ids. Empty means all classes.",
            )
            max_records = st.number_input(
                "Max records (0 = all)",
                min_value=0,
                max_value=100_000_000,
                value=int(defaults.get("max_records", 0) or 0),
                step=10000,
                help="Use 0 for full DB. Use a positive value only for smoke tests.",
            )
            default_cache_path = str(defaults.get("image_size_cache", "")) or str(
                Path("artifacts") / "image_size_cache" / f"{slugify(str(project_name))}.sqlite"
            )
            image_size_cache = st.text_input(
                "Image size cache",
                value=default_cache_path,
                help="SQLite cache for image width/height. Reuses size if image path, mtime, and file size match.",
            )

        provisional = {
            "name": project_name,
            "dataset_layout": dataset_layout,
            "images_dir": images_dir,
            "labels_dir": labels_dir,
            "data_yaml": data_yaml,
            "weights_path": weights_path,
            "repo_path": repo_path,
            "feature_index_dir": feature_index_dir,
        }
        script_path = st.text_input(
            "Project script path",
            value=str(defaults.get("script_path", "")) or default_project_script_path(provisional),
        )
        script_text = st.text_area(
            "Project script",
            value=str(defaults.get("script_text", "")) or default_project_script(provisional),
            height=260,
            help="Saved with the project and written to the script path. The app does not execute this automatically.",
        )

        save_project = st.form_submit_button("Save Project", type="primary")

    project = normalize_project(
        {
            "name": project_name,
            "description": description,
            "dataset_layout": dataset_layout,
            "images_dir": images_dir,
            "labels_dir": labels_dir,
            "data_yaml": data_yaml,
            "weights_path": weights_path,
            "repo_path": repo_path,
            "feature_index_dir": feature_index_dir,
            "img_size": img_size,
            "expand": expand,
            "class_ids": class_ids,
            "max_records": max_records,
            "image_size_cache": image_size_cache,
            "build_max_workers": int(defaults.get("build_max_workers", 0) or 0),
            "feature_batch_size": int(defaults.get("feature_batch_size", 0) or 0),
            "faiss_type": str(defaults.get("faiss_type", "ivfpq") or "ivfpq"),
            "faiss_gpu": config_bool(defaults.get("faiss_gpu", False)),
            "script_path": script_path,
            "script_text": script_text,
        }
    )

    if project.get("dataset_layout") == DATASET_LAYOUT_NESTED_IMAGE_LABELS:
        pairs = discover_nested_image_label_pairs(project["images_dir"])
        if pairs:
            st.caption(f"Nested layout detected pairs: {len(pairs):,}")
            with st.expander("Detected image/label pairs", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"image_dir": str(image_root), "labels": str(label_root)}
                            for image_root, label_root in pairs[:500]
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                if len(pairs) > 500:
                    st.caption(f"Showing first 500 of {len(pairs):,} pairs.")
        else:
            st.warning("Nested layout selected, but no */JPEGImages or */images + sibling */labels pairs were found.")

    if project.get("script_text"):
        st.download_button(
            "Download Project Script",
            data=project["script_text"].encode("utf-8-sig"),
            file_name=Path(project.get("script_path") or default_project_script_path(project)).name,
            mime="text/plain",
            key="download_project_script",
        )

    if save_project:
        errors = project_required_errors(project)
        if errors:
            st.error("; ".join(errors))
        else:
            project = write_project_script(project)
            saved = upsert_project(project)
            set_active_project(saved)
            st.success(f"Project saved: {saved['name']}")
            st.rerun()

    st.divider()
    st.subheader("Build Feature Index")
    st.caption("This starts a background process. You can leave the page open and watch the logs below.")
    st.subheader("Build Readiness")
    preflight_rows = project_preflight_rows(project)
    preflight_df = pd.DataFrame(preflight_rows)
    st.dataframe(preflight_df, use_container_width=True, hide_index=True)
    blocking_preflight = [row for row in preflight_rows if row.get("status") == "FAIL"]
    if blocking_preflight:
        st.error("Build cannot start until FAIL items are fixed.")

    build_col1, build_col2, build_col3, build_col4, build_col5 = st.columns([1, 1, 1, 1, 2])
    with build_col1:
        default_build_device = "0" if config["device"] == "cuda" else "cpu"
        build_device = st.text_input("Build device", value=default_build_device, key="project_build_device")
    with build_col2:
        default_shards = 32 if int(project.get("max_records", 0) or 0) == 0 else 1
        num_shards = st.number_input(
            "Total shards",
            min_value=1,
            max_value=256,
            value=int(defaults.get("num_shards", default_shards) or default_shards),
            step=1,
            key="project_build_shards",
        )
    with build_col3:
        max_workers = st.number_input(
            "Parallel workers",
            min_value=0,
            max_value=64,
            value=int(project.get("build_max_workers", 0) or 0),
            step=1,
            key="project_build_max_workers",
            help="0 uses the number of selected devices. Local 1 GPU: 1. 8 GPU server: 8.",
        )
    with build_col4:
        feature_batch_size = st.number_input(
            "Feature batch",
            min_value=0,
            max_value=64,
            value=int(project.get("feature_batch_size", 0) or 0),
            step=1,
            key="project_feature_batch_size",
            help="0=auto. 1=single-image legacy mode. Larger values batch images per YOLO forward.",
        )
    with build_col5:
        st.caption(f"Output: {project.get('feature_index_dir', '')}")
        st.caption("Use comma-separated devices for multi-GPU, e.g. 0,1,2,3.")
        faiss_type = st.selectbox(
            "Final FAISS index",
            ["ivfpq", "flat"],
            index=0 if str(project.get("faiss_type", "ivfpq")) == "ivfpq" else 1,
            key="project_build_faiss_type",
            help="ivfpq is compressed and recommended for full large DB. flat is exact but needs much more RAM.",
        )
        faiss_gpu = st.checkbox(
            "Use FAISS GPU if available",
            value=config_bool(project.get("faiss_gpu", False)),
            key="project_build_faiss_gpu",
            help="Uses faiss-gpu during final index train/add when available. Falls back to CPU on this Windows faiss-cpu environment.",
        )
        force_prepare = st.checkbox(
            "Rebuild records metadata",
            value=False,
            key="project_force_prepare_records",
            help="Re-scan image/label folders instead of reusing existing records metadata.",
        )
    project["build_max_workers"] = int(max_workers)
    project["feature_batch_size"] = int(feature_batch_size)
    project["faiss_type"] = faiss_type
    project["faiss_gpu"] = bool(faiss_gpu)

    start_build = st.button("Start Feature Build", type="primary", key="btn_start_project_feature_build")
    if start_build:
        errors = project_required_errors(project)
        errors.extend(project_preflight_errors(project))
        if errors:
            st.error("; ".join(errors))
        else:
            project = write_project_script(project)
            saved = upsert_project(project)
            set_active_project(saved)
            pid = start_project_feature_build(
                saved,
                build_device,
                int(num_shards),
                int(max_workers),
                force_prepare=bool(force_prepare),
            )
            st.success(f"Feature build started: PID {pid}")

    if st.button("Set As Active Project", key="btn_set_active_project"):
        project = write_project_script(project)
        saved = upsert_project(project)
        set_active_project(saved)
        st.success(f"Active project: {saved['name']}")

    if selected_name != "New project" and st.button("Delete Project Entry", key="btn_delete_project_entry"):
        delete_project(selected_name)
        if st.session_state.get("active_project_name") == selected_name:
            st.session_state["active_project_name"] = None
            st.session_state["active_project"] = None
        st.warning(f"Deleted project entry: {selected_name}")
        st.rerun()

    project_build_status(project)


def video_detection_tab(project: Dict, config: Dict) -> None:
    st.subheader("Video Detection -> Select Crop -> Search")
    with st.expander("YOLO detector settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            weights_path = str(project.get("weights_path", ""))
            repo_path = str(project.get("repo_path", ""))
            feature_index_dir = str(project.get("feature_index_dir", ""))
            st.text_input("YOLO model", value=weights_path, disabled=True, key="project_video_weights_path")
            st.text_input("YOLO repo", value=repo_path, disabled=True, key="project_video_repo_path")
            feature_index_dir = st.text_input(
                "YOLO feature index",
                value=feature_index_dir,
                disabled=True,
                key="project_video_feature_index_dir",
            )
            loaded_feature_dir = st.session_state.get("yolo_feature_index_dir")
            if st.session_state.get("yolo_feature_index") is None or loaded_feature_dir != feature_index_dir:
                st.caption("Feature index loads when you press Load YOLO Feature Index or start a search.")
            if st.button("Load YOLO Feature Index", key="btn_load_yolo_feature_index"):
                try:
                    st.session_state["yolo_feature_index"] = YoloFeatureIndex.load(
                        feature_index_dir,
                        device=config["device"],
                    )
                    st.session_state["yolo_feature_index_dir"] = feature_index_dir
                    st.success("YOLO feature index loaded")
                except Exception as exc:
                    st.error(f"YOLO feature index load failed: {exc}")
            loaded_feature_dir = st.session_state.get("yolo_feature_index_dir")
            if loaded_feature_dir:
                show_yolo_feature_index_status(weights_path)
            img_size = st.number_input(
                "Image size",
                min_value=320,
                max_value=1536,
                value=int(project.get("img_size", 640) or 640),
                step=32,
                key="video_img_size",
            )
        with col2:
            conf_thres = st.slider(
                "Detection confidence",
                min_value=0.01,
                max_value=0.95,
                value=0.25,
                step=0.01,
                key="video_conf_thres",
            )
            frame_stride = st.number_input(
                "Frame stride",
                min_value=1,
                max_value=300,
                value=15,
                step=1,
                key="video_frame_stride",
            )
            max_frames = st.number_input(
                "Max processed frames",
                min_value=1,
                max_value=10000,
                value=300,
                step=10,
                key="video_max_frames",
            )
            max_detections = st.number_input(
                "Max detection crops",
                min_value=1,
                max_value=2000,
                value=200,
                step=10,
                key="video_max_detections",
            )

    video_file = st.file_uploader(
        "동영상 업로드",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )
    default_video = ""
    video_files = sorted(Path("video").glob("*.mp4")) if Path("video").exists() else []
    if video_files:
        default_video = str(video_files[0])
    local_video_path = st.text_input(
        "또는 로컬 동영상 경로",
        value=default_video,
        key="video_local_path",
    )

    if st.button("Run YOLO Detection", type="primary", key="btn_run_yolo_detection"):
        if not weights_path:
            st.error("YOLO weights path가 필요합니다.")
            return
        if not video_file and not local_video_path:
            st.error("동영상 업로드 또는 로컬 경로가 필요합니다.")
            return

        class_names = load_class_names(str(project.get("data_yaml", ""))) or {0: "person"}
        progress_bar = st.progress(0)
        status = st.empty()
        detect_start = time.time()

        def progress(done: int, total: int, message: str) -> None:
            progress_bar.progress(min(1.0, done / max(1, total)))
            status.caption(progress_with_eta(done, total, message, detect_start))

        try:
            detector = get_yolo_detector(
                repo_path=repo_path,
                weights_path=weights_path,
                device=config["device"],
                img_size=int(img_size),
                conf_thres=float(conf_thres),
                class_names=class_names,
            )

            if video_file:
                suffix = Path(video_file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_file.read())
                    video_path = tmp.name
            else:
                video_path = local_video_path

            st.session_state["last_video_path"] = video_path
            detections = collect_video_detections(
                video_path=video_path,
                detector=detector,
                frame_stride=int(frame_stride),
                max_frames=int(max_frames),
                max_detections=int(max_detections),
                progress=progress,
            )
            st.session_state["video_detections"] = detections
            progress_bar.progress(1.0)
            status.caption(
                f"Done. {len(detections)} crops collected. "
                f"Elapsed {format_duration(time.time() - detect_start)}"
            )
        except Exception as exc:
            st.error(f"Detection failed: {exc}")

    detections: List[Detection] = st.session_state.get("video_detections", [])
    if detections:
        st.markdown(f"Detected crops: **{len(detections)}**")
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)
        with control_col1:
            top_k = st.slider(
                "Video crop top-k",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key="video_topk",
            )
        with control_col2:
            max_display = st.slider(
                "Displayed crops",
                min_value=10,
                max_value=200,
                value=min(50, max(10, len(detections))),
                step=10,
                key="video_displayed_crops",
            )
        class_options = ["All"] + sorted({str(det.class_name) for det in detections})
        with control_col3:
            video_class_filter = st.selectbox("Crop class", class_options, key="video_crop_class_filter")
        with control_col4:
            video_size_filter = st.selectbox(
                "Crop size",
                ["All"] + list(SIZE_BUCKET_ORDER),
                key="video_crop_size_filter",
                format_func=lambda value: "All" if value == "All" else SIZE_BUCKET_LABELS.get(value, value),
            )

        group_mode = st.selectbox(
            "Group crops",
            ["Class + Size", "Class", "Size", "None"],
            key="video_crop_group_mode",
        )

        filtered_detections = []
        for det in detections:
            info = detection_size_info(det)
            if video_class_filter != "All" and str(det.class_name) != video_class_filter:
                continue
            if video_size_filter != "All" and info["size_bucket"] != video_size_filter:
                continue
            filtered_detections.append(det)

        if not filtered_detections:
            st.warning("No detected crops match the selected class/size filter.")
            run_pending_db_neighbor_search(project, config)
            render_db_neighbor_results("video_empty")
            return

        summary_rows = []
        for det in filtered_detections:
            info = detection_size_info(det)
            summary_rows.append(
                {
                    "class_name": det.class_name,
                    "size_bucket": SIZE_BUCKET_LABELS.get(info["size_bucket"], info["size_bucket"]),
                    "count": 1,
                }
            )
        video_summary = (
            pd.DataFrame(summary_rows)
            .groupby(["class_name", "size_bucket"], as_index=False)["count"]
            .sum()
            .sort_values(["class_name", "size_bucket"])
        )
        st.dataframe(video_summary, use_container_width=True, hide_index=True)

        selected_for_search = None
        rendered_video_search_results = False
        display_detections = filtered_detections[: int(max_display)]
        grouped: Dict[str, List[Detection]] = {}
        for det in display_detections:
            grouped.setdefault(detection_group_name(det, group_mode), []).append(det)

        for group_name, group_detections in grouped.items():
            if group_mode != "None":
                st.markdown(
                    f'<div class="group-band">{html.escape(group_name)} ({len(group_detections)})</div>',
                    unsafe_allow_html=True,
                )
            gallery_cols = st.columns(5)
            for idx, det in enumerate(group_detections):
                info = detection_size_info(det)
                with gallery_cols[idx % 5]:
                    st.caption(
                        f"ID {det.det_id} | {det.class_name} {det.confidence:.2f} | "
                        f"{SIZE_BUCKET_LABELS.get(info['size_bucket'], info['size_bucket'])}"
                    )
                    render_thumb_image(
                        det.crop,
                        f"{det.class_name} {det.det_id}",
                        cache_key=f"video:{det.det_id}:{det.frame_index}:{det.bbox_xyxy}",
                    )
                    if st.button(
                        "View",
                        key=f"btn_preview_video_crop_{det.det_id}",
                        use_container_width=True,
                    ):
                        set_preview_image(
                            det.crop,
                            (
                                f"ID {det.det_id} | frame={det.frame_index} | "
                                f"{det.class_name} {det.confidence:.2f} | "
                                f"{SIZE_BUCKET_LABELS.get(info['size_bucket'], info['size_bucket'])}"
                            ),
                        )
                        st.rerun()
                    if st.button(
                        "Search",
                        key=f"btn_search_video_crop_{det.det_id}",
                        use_container_width=True,
                    ):
                        selected_for_search = det

        if selected_for_search is not None:
            st.divider()
            st.subheader("Video Crop Search Results")
            render_thumb_image(
                selected_for_search.crop,
                "Selected false-positive candidate",
                cache_key=f"video:selected:{selected_for_search.det_id}:{selected_for_search.frame_index}:{selected_for_search.bbox_xyxy}",
            )
            video_path = st.session_state.get("last_video_path")
            if not video_path:
                st.warning("동영상 경로가 없어 원본 frame을 읽을 수 없습니다.")
            else:
                try:
                    frame_image = read_video_frame(video_path, selected_for_search.frame_index)
                    search_yolo_feature_bbox(
                        frame_image,
                        selected_for_search.bbox_xyxy,
                        top_k=top_k,
                        key_prefix=f"video_yolo_feature_results_{selected_for_search.det_id}",
                        feature_index_dir=feature_index_dir,
                        device=config["device"],
                    )
                    rendered_video_search_results = True
                except Exception as exc:
                    st.error(f"YOLO feature search failed: {exc}")

        if (
            not rendered_video_search_results
            and st.session_state.get("last_results")
            and st.session_state.get("last_results_context") == "video"
        ):
            st.divider()
            st.subheader("Video Crop Search Results")
            show_results(
                st.session_state.get("last_results", []),
                key_prefix=st.session_state.get("last_results_key_prefix", "video_yolo_feature_results"),
            )

        run_pending_db_neighbor_search(project, config)
        render_db_neighbor_results("video")


def render_calibration_examples(feature_index_dir: str, detail: pd.DataFrame, key_prefix: str) -> None:
    st.subheader("Evidence Samples")
    st.caption("Each pair shows a sampled DB bbox and its nearest non-self DB neighbor from the same feature index.")

    if detail.empty:
        st.info("No detail rows available.")
        return

    records = cached_index_records(feature_index_dir)
    if not records:
        st.warning("records metadata is empty or unavailable.")
        return

    work = detail.copy()
    if "top1_record_idx" not in work.columns:
        id_to_idx = {int(record.record_id): idx for idx, record in enumerate(records)}
        work["top1_record_idx"] = work["top1_record_id"].map(lambda value: id_to_idx.get(int(value), -1))

    bins = (
        work[["similarity_bin_start", "similarity_bin"]]
        .drop_duplicates()
        .sort_values("similarity_bin_start", ascending=False)["similarity_bin"]
        .astype(str)
        .tolist()
        if "similarity_bin" in work.columns
        else []
    )

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
    with ctrl1:
        match_filter = st.selectbox(
            "Match",
            ["All", "Same class", "Different class"],
            index=0,
            key=f"{key_prefix}_match_filter",
        )
    with ctrl2:
        min_similarity = st.slider(
            "Min similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key=f"{key_prefix}_min_similarity",
        )
    with ctrl3:
        selected_bin = st.selectbox(
            "Similarity bin",
            ["All"] + bins,
            index=0,
            key=f"{key_prefix}_bin_filter",
        )
    with ctrl4:
        sort_mode = st.selectbox(
            "Sort",
            ["Different first", "Highest similarity", "Lowest similarity", "Random"],
            index=0,
            key=f"{key_prefix}_sort",
        )
    with ctrl5:
        max_pairs = st.number_input(
            "Pairs",
            min_value=2,
            max_value=60,
            value=12,
            step=2,
            key=f"{key_prefix}_max_pairs",
        )

    examples = work[work["top1_similarity"].astype(float) >= float(min_similarity)].copy()
    if selected_bin != "All" and "similarity_bin" in examples.columns:
        examples = examples[examples["similarity_bin"].astype(str) == str(selected_bin)]
    if match_filter == "Same class":
        examples = examples[examples["top1_same_class"].astype(bool)]
    elif match_filter == "Different class":
        examples = examples[~examples["top1_same_class"].astype(bool)]

    if examples.empty:
        st.warning("No sample pairs match the current visual filters.")
        return

    examples["_different"] = ~examples["top1_same_class"].astype(bool)
    if sort_mode == "Different first":
        examples = examples.sort_values(["_different", "top1_similarity"], ascending=[False, False])
    elif sort_mode == "Highest similarity":
        examples = examples.sort_values("top1_similarity", ascending=False)
    elif sort_mode == "Lowest similarity":
        examples = examples.sort_values("top1_similarity", ascending=True)
    else:
        examples = examples.sample(frac=1.0, random_state=42)
    examples = examples.head(int(max_pairs))

    st.caption(f"showing {len(examples):,} / {len(work):,} sampled pairs")
    pair_columns = st.columns(2)
    for pos, row in enumerate(examples.itertuples(index=False)):
        record_idx = int(getattr(row, "record_idx"))
        top1_record_idx = int(getattr(row, "top1_record_idx"))
        if record_idx < 0 or record_idx >= len(records) or top1_record_idx < 0 or top1_record_idx >= len(records):
            continue

        query_record = records[record_idx]
        neighbor_record = records[top1_record_idx]
        same_class = bool(getattr(row, "top1_same_class"))
        status = "SAME" if same_class else "DIFF"
        status_color = "#22c55e" if same_class else "#f97316"
        card_key = f"{key_prefix}_{pos}_{query_record.record_id}_{neighbor_record.record_id}"

        with pair_columns[pos % 2]:
            st.markdown(
                f"""
                <div class="group-band">
                  <span style="color:{status_color};font-weight:800;">{status}</span>
                  &nbsp; sim={float(getattr(row, "top1_similarity")):.4f}
                  &nbsp; topk_same={float(getattr(row, "topk_same_class_ratio")) * 100.0:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )
            q_col, n_col = st.columns(2)
            with q_col:
                st.caption(f"Query | {query_record.class_id} {query_record.class_name}")
                if render_record_thumb(query_record, badge=f"Q {query_record.class_id} {query_record.class_name}"):
                    if st.button("View Q", key=f"{card_key}_view_q", use_container_width=True):
                        set_preview_image(
                            crop_from_record(query_record),
                            f"Calibration query | record={query_record.record_id} | {query_record.class_name}",
                        )
                        st.rerun()
                st.caption(Path(query_record.image_path).name)
                if st.button("Data Q", key=f"{card_key}_data_q", use_container_width=True):
                    open_data_location(query_record.image_path)
                render_path_selector(query_record.image_path, query_record, key=f"{card_key}_select_q")
            with n_col:
                st.caption(f"Top1 | {neighbor_record.class_id} {neighbor_record.class_name}")
                if render_record_thumb(neighbor_record, badge=f"N {neighbor_record.class_id} {neighbor_record.class_name}"):
                    if st.button("View N", key=f"{card_key}_view_n", use_container_width=True):
                        set_preview_image(
                            crop_from_record(neighbor_record),
                            f"Calibration top1 | record={neighbor_record.record_id} | {neighbor_record.class_name}",
                        )
                        st.rerun()
                st.caption(Path(neighbor_record.image_path).name)
                if st.button("Data N", key=f"{card_key}_data_n", use_container_width=True):
                    open_data_location(neighbor_record.image_path)
                render_path_selector(neighbor_record.image_path, neighbor_record, key=f"{card_key}_select_n")

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Query Neighbors", key=f"{card_key}_neighbors_q", use_container_width=True):
                    request_db_neighbor_search(query_record, 20)
                    st.rerun()
            with btn_col2:
                if st.button("Top1 Neighbors", key=f"{card_key}_neighbors_n", use_container_width=True):
                    request_db_neighbor_search(neighbor_record, 20)
                    st.rerun()


def calibration_tab(project: Dict, config: Dict) -> None:
    st.subheader("Similarity Calibration")
    feature_index_dir = str(project.get("feature_index_dir", ""))
    root = Path(feature_index_dir)
    if not (root / "index.faiss").exists() or not (root / "features.npy").exists() or not index_records_ready(root):
        st.warning(f"FAISS index/features/records not found: {feature_index_dir}")
        return

    metadata = cached_cluster_metadata(feature_index_dir)
    total_records = int(metadata.get("total_records", 0) or 0)
    class_options = ["All"] + sorted(metadata.get("class_counts", {}).keys())
    st.caption(
        "DB leave-one-out evidence. Each sampled DB bbox searches the same DB while excluding itself. "
        "The rates below are empirical same-class support, not direct YOLO detection probability."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        sample_size = st.number_input(
            "Samples",
            min_value=1,
            max_value=max(1, total_records),
            value=min(5000, max(1, total_records)),
            step=500,
            key="calibration_sample_size",
        )
    with col2:
        top_k = st.number_input("Top-k", min_value=1, max_value=100, value=20, step=1, key="calibration_top_k")
    with col3:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="calibration_seed")
    with col4:
        bin_width = st.selectbox(
            "Bin width",
            [0.01, 0.02, 0.05],
            index=1,
            format_func=lambda value: f"{value:.2f}",
            key="calibration_bin_width",
        )
    with col5:
        class_filter = st.selectbox("Class", class_options, index=0, key="calibration_class_filter")

    if st.button("Run Calibration", type="primary", key="btn_run_similarity_calibration"):
        next_request = {
            "index_dir": feature_index_dir,
            "sample_size": int(sample_size),
            "top_k": int(top_k),
            "seed": int(seed),
            "class_filter": str(class_filter),
            "bin_width": float(bin_width),
        }
        st.session_state["calibration_request"] = next_request
        st.session_state["calibration_result"] = None
        st.session_state["calibration_result_request"] = None

    request = st.session_state.get("calibration_request")
    if not request:
        st.info(f"Ready. DB records={total_records:,}. Run calibration to create similarity evidence tables.")
        return

    result = st.session_state.get("calibration_result")
    result_request = st.session_state.get("calibration_result_request")
    if result is None or result_request != request:
        start = time.time()
        with st.spinner("Running similarity calibration..."):
            result = cached_similarity_calibration(
                request["index_dir"],
                int(request["sample_size"]),
                int(request["top_k"]),
                int(request["seed"]),
                str(request.get("class_filter", "")),
                float(request.get("bin_width", 0.02)),
            )
        st.session_state["calibration_result"] = result
        st.session_state["calibration_result_request"] = dict(request)
        st.session_state["calibration_result_elapsed"] = time.time() - start

    detail = result.get("detail", pd.DataFrame())
    if detail.empty:
        st.warning("No calibration records found for the selected class/filter.")
        return

    elapsed = float(st.session_state.get("calibration_result_elapsed", 0.0) or 0.0)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Samples used", f"{int(result.get('sample_size', len(detail))):,}")
    with metric_col2:
        st.metric("Candidates", f"{int(result.get('total_candidates', 0)):,}")
    with metric_col3:
        st.metric("Top-k", str(int(request["top_k"])))
    with metric_col4:
        st.metric("Elapsed", format_duration(elapsed))

    render_calibration_examples(feature_index_dir, detail, key_prefix="calibration_examples")

    st.subheader("Threshold Evidence")
    st.caption(
        "Interpretation: for DB samples whose nearest non-self neighbor similarity is at least the threshold, "
        "this table shows how often that nearest neighbor has the same label class."
    )
    thresholds_df = result.get("thresholds", pd.DataFrame())
    st.dataframe(thresholds_df, use_container_width=True, hide_index=True, key="calibration_threshold_table")

    st.subheader("Similarity Bins")
    bins_df = result.get("bins", pd.DataFrame())
    st.dataframe(bins_df, use_container_width=True, hide_index=True, key="calibration_bins_table")

    class_col1, class_col2 = st.columns(2)
    with class_col1:
        st.subheader("Per-Class Evidence")
        st.dataframe(result.get("classes", pd.DataFrame()), use_container_width=True, hide_index=True, key="calibration_class_table")
    with class_col2:
        st.subheader("Top Detail")
        detail_display = detail[
            [
                "record_id",
                "class_id",
                "class_name",
                "top1_similarity",
                "top1_same_class",
                "top1_record_id",
                "top1_class_id",
                "top1_class_name",
                "topk_same_class_ratio",
                "file_name",
                "top1_file_name",
            ]
        ].head(500)
        st.dataframe(detail_display, use_container_width=True, hide_index=True, key="calibration_detail_table")

    down_col1, down_col2, down_col3 = st.columns(3)
    with down_col1:
        st.download_button(
            "Download Thresholds CSV",
            thresholds_df.to_csv(index=False).encode("utf-8-sig"),
            "similarity_threshold_evidence.csv",
            "text/csv",
            key="calibration_thresholds_download_csv",
            use_container_width=True,
        )
    with down_col2:
        st.download_button(
            "Download Bins CSV",
            bins_df.to_csv(index=False).encode("utf-8-sig"),
            "similarity_bins_evidence.csv",
            "text/csv",
            key="calibration_bins_download_csv",
            use_container_width=True,
        )
    with down_col3:
        st.download_button(
            "Download Detail CSV",
            detail.to_csv(index=False).encode("utf-8-sig"),
            "similarity_calibration_detail.csv",
            "text/csv",
            key="calibration_detail_download_csv",
            use_container_width=True,
        )

    render_selected_paths_panel(key_prefix="calibration_selected_paths")
    run_pending_db_neighbor_search(project, config)
    render_db_neighbor_results("calibration")
    render_preview_image()


def status_panel(project: Optional[Dict]) -> None:
    if not project:
        st.warning("No active project selected.")
        return
    st.caption(
        f"Active project: {project.get('name')} | "
        f"model={Path(str(project.get('weights_path', ''))).name} | "
        f"index={project.get('feature_index_dir', '')}"
    )
    yolo_index = st.session_state.get("yolo_feature_index")
    yolo_dir = str(project.get("feature_index_dir", "")) or FIREDB_YOLO_FEATURE_INDEX_DIR
    if yolo_index is None:
        if Path(yolo_dir, "index.faiss").exists():
            st.info(f"YOLO feature index is available and will auto-load on first YOLO search: {yolo_dir}")
        else:
            st.warning(f"YOLO feature index not found: {yolo_dir}")
    else:
        st.success(f"YOLO feature index ready: {len(yolo_index.records):,} training boxes | {yolo_dir}")


def search_page(config: Dict) -> None:
    st.subheader("DB Search")
    projects = load_projects()
    if not projects:
        st.warning("No projects registered. Create or save a project in the Feature Projects tab first.")
        return

    names = [str(project.get("name", "")) for project in projects]
    current_name = st.session_state.get("active_project_name") or names[0]
    default_index = names.index(current_name) if current_name in names else 0
    selected_name = st.selectbox("Project", names, index=default_index, key="search_project_select")
    project = get_project(selected_name) or projects[default_index]
    set_active_project(project)

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.text_input("Dataset layout", value=str(project.get("dataset_layout", DATASET_LAYOUT_SINGLE)), disabled=True, key="search_project_layout")
        st.text_input("Model", value=str(project.get("weights_path", "")), disabled=True, key="search_project_model")
        st.text_input("Feature index", value=str(project.get("feature_index_dir", "")), disabled=True, key="search_project_index")
    with detail_col2:
        st.text_input("Images", value=str(project.get("images_dir", "")), disabled=True, key="search_project_images")
        st.text_input("Labels", value=str(project.get("labels_dir", "")), disabled=True, key="search_project_labels")

    if st.button("Load Project Index", type="primary", key="btn_load_search_project_index"):
        if ensure_yolo_feature_index_loaded(
            feature_index_dir=str(project.get("feature_index_dir", "")),
            device=config["device"],
        ):
            st.success(f"Loaded project index: {project.get('name')}")

    status_panel(project)

    tab_crop, tab_video, tab_cluster, tab_calibration, tab_last = st.tabs(
        ["Crop Image Search", "Video Detection Search", "Feature Clustering", "Calibration", "Last Results"]
    )
    with tab_crop:
        crop_search_tab(project, config)
        run_pending_db_neighbor_search(project, config)
        render_db_neighbor_results("crop")
        render_preview_image()
    with tab_video:
        video_detection_tab(project, config)
        render_preview_image()
    with tab_cluster:
        feature_cluster_tab(project, config)
    with tab_calibration:
        calibration_tab(project, config)
    with tab_last:
        query = st.session_state.get("last_query_image")
        if query is not None:
            st.image(query, caption="Last query", width=260)
        show_results(st.session_state.get("last_results", []), key_prefix="last_results")
        run_pending_db_neighbor_search(project, config)
        render_db_neighbor_results("last")
        render_preview_image()


def main() -> None:
    init_state()
    ensure_default_projects()
    apply_theme()
    st.title("YOLOv7 False Positive Sample Finder")
    st.caption("Project-based YOLO feature search for false-positive analysis.")

    config = sidebar_config()
    tab_projects, tab_search = st.tabs(["Feature Projects", "DB Search"])
    with tab_projects:
        project_manager_tab(config)
    with tab_search:
        search_page(config)


if __name__ == "__main__":
    main()
