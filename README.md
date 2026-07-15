# YOLO False Positive Feature Search

Streamlit tool for investigating YOLO object-detection false positives by searching similar training samples from an indexed feature database.

The main workflow is:

1. Build a feature project from YOLO-format labels and training images.
2. Extract ROI-pooled YOLO feature vectors from detector feature maps.
3. Build a FAISS index over the feature vectors.
4. Search by uploaded crop image, selected video detection crop, or an existing DB record.
5. Inspect nearest neighbors, DB-neighbors, and 3D clustering views.

## Features

- YOLO txt dataset support.
- Single-folder and nested dataset layouts:
  - `images` + `labels`
  - `JPEGImages` + `labels`
  - root folders containing repeated image/label pairs
- YOLOv7 local repo support, with YOLOv9 repo paths left configurable.
- Feature extraction from YOLO internal P3/P4/P5 detection feature maps.
- Batch feature extraction with automatic GPU batch sizing.
- SQLite image-size cache for faster repeated metadata preparation.
- Sharded feature build for single-GPU and multi-GPU servers.
- FAISS `IndexFlatIP` or compressed `IndexIVFPQ` final index.
- Optional FAISS GPU train/add path when `faiss-gpu` is available.
- Exact reranking from saved `features.npy` after IVFPQ candidate search.
- Streamlit UI for project setup, crop search, video detection search, DB-neighbor search, and clustering.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU model inference, install the PyTorch build that matches your CUDA runtime.

FAISS GPU is optional. The default dependency is `faiss-cpu` because Windows environments commonly do not provide a stable `faiss-gpu` wheel. On a Linux CUDA server, install a compatible `faiss-gpu` package and run the merge step with `--faiss-gpu`.

## Run The UI

```powershell
streamlit run app.py
```

The default local URL is:

```text
http://localhost:8501
```

## Dataset Format

YOLO labels must use normalized xywh format:

```text
class_id x_center y_center width height
```

Typical layout:

```text
dataset/
  images/
    train/
      sample.jpg
  labels/
    train/
      sample.txt
  data.yaml
```

Nested layout is also supported:

```text
dataset_root/
  subset_a/
    JPEGImages/
    labels/
  subset_b/
    images/
    labels/
```

## Build A YOLO Feature Index

Single process example:

```powershell
python -u scripts/build_yolo_feature_index.py `
  --images-dir db/data/JPEGImages `
  --labels-dir db/data/labels `
  --data-yaml db/data/data.yaml `
  --repo-path external/yolov7 `
  --weights-path model/model.pt `
  --index-dir artifacts/yolo_feature_index `
  --device 0 `
  --feature-batch-size 0
```

Project/sharded build example:

```powershell
python -u scripts/run_yolo_feature_project_build.py `
  --project-name safety_env `
  --images-dir V:\dataset\images `
  --labels-dir V:\dataset\labels `
  --data-yaml data.yaml `
  --repo-path external/yolov7 `
  --weights-path model/model.pt `
  --index-dir artifacts/yolo_feature_index_safety_env `
  --device 0 `
  --num-shards 64 `
  --max-workers 1 `
  --feature-batch-size 0 `
  --faiss-type ivfpq
```

Multi-GPU server example:

```powershell
python -u scripts/run_yolo_feature_project_build.py `
  --project-name safety_env `
  --images-dir /data/safety/images `
  --labels-dir /data/safety/labels `
  --data-yaml /data/safety/data.yaml `
  --repo-path external/yolov7 `
  --weights-path model/model.pt `
  --index-dir artifacts/yolo_feature_index_safety_env `
  --device 0,1,2,3,4,5,6,7 `
  --num-shards 64 `
  --max-workers 8 `
  --feature-batch-size 0 `
  --faiss-type ivfpq `
  --faiss-gpu
```

## FAISS Strategy

For small indexes, `flat` gives exact inner-product search.

For large indexes, `ivfpq` is recommended:

1. FAISS IVFPQ searches a larger candidate set.
2. The app reloads candidate vectors from `features.npy`.
3. Candidates are reranked by exact cosine / inner-product score.

This keeps the interactive index small while preserving better final ranking quality.

## Generated Files

Feature index output contains:

```text
features.npy          float32 feature matrix
records.jsonl         crop metadata
record_offsets.npy    random-access offsets for records.jsonl
config.json           model/index metadata
index.faiss           FAISS index
```

These generated files can be large and are intentionally excluded from git.

## Notes

- Do not commit model weights, video files, datasets, or generated feature indexes.
- Keep YOLO repositories configurable through the UI/project settings.
- FAISS GPU use is recorded in `config.json` as `faiss_gpu_requested`, `faiss_gpu_used`, and `faiss_gpu_reason`.

## Data Curation Roadmap

For the research-backed plan on false-positive cause search, clustering, coreset selection, and training DB reduction, see:

```text
data_curation_research_and_clustering_plan.md
```

## Build A Curation Report

The curation report uses the saved YOLO feature index directly. It does not re-extract model features.

```powershell
python -u scripts/build_curation_report.py `
  --index-dir artifacts/yolo_feature_index_safety_env `
  --output-dir artifacts/curation_reports/safety_env/report_10k `
  --max-query-records 10000 `
  --top-k 50 `
  --rerank-k 200 `
  --duplicate-threshold 0.98 `
  --cross-class-threshold 0.90
```

Key outputs:

```text
curation_recommendations.csv
image_recommendations.csv
near_duplicates.csv
duplicate_groups.csv
cross_class_overlap.csv
representatives.csv
boundary_samples.csv
rare_samples.csv
summary.json
```

To create a reduced dataset manifest or copied dataset:

```powershell
python -u scripts/export_reduced_dataset.py `
  --report-dir artifacts/curation_reports/safety_env/report_10k `
  --output-dir artifacts/reduced_datasets/safety_env/report_10k `
  --images-root V:\dataset\images `
  --labels-root V:\dataset\labels `
  --data-yaml data.yaml `
  --mode manifest
```
