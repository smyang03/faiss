# Data Curation And Clustering Plan For YOLO False Positive Analysis

## Goal

This project is not only a nearest-neighbor viewer. The practical goal is a data engine for object detection:

1. Find training DB samples that are visually/model-feature similar to a false positive.
2. Group the existing training DB by feature space to find redundancy, mixed classes, label noise, and hard negatives.
3. Produce actionable data curation outputs for retraining: keep, review, add hard-negative, relabel, or drop candidates.
4. Validate every recommendation through retraining or targeted evaluation on fixed videos/test sets.

The current index already has the core asset needed for this:

- `features.npy`: YOLO ROI-pooled P3/P4/P5 features, normalized, shape `(3,867,553, 1792)`
- `records.jsonl`: crop metadata
- `index.faiss`: approximate search index with exact reranking from `features.npy`

This means most curation and clustering additions can run without re-extracting YOLO features.

## Relevant Recent Research

### 1. Automatic data engines for object detection

**AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving, CVPR 2024**

Paper: https://arxiv.org/abs/2403.17373  
CVF: https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_AIDE_An_Automatic_Data_Engine_for_Object_Detection_in_Autonomous_CVPR_2024_paper.pdf

Useful idea for this project:

- Treat failure cases as triggers for data search, curation, labeling, retraining, and verification.
- The project already implements the "failure case -> retrieve related data" part.
- The missing piece is a structured curation report and retraining verification loop.

Recommended addition:

- Add `Failure Case Bundle`: query crop, top-k similar DB samples, cross-class overlaps, suspected missing hard negatives, and fixed evaluation video references.

### 2. Coreset selection for object detection

**Coreset Selection for Object Detection, CVPRW 2024**

Paper: https://arxiv.org/abs/2404.09161  
CVF: https://openaccess.thecvf.com/content/CVPR2024W/DDCV/papers/Lee_Coreset_Selection_for_Object_Detection_CVPRW_2024_paper.pdf

Useful idea for this project:

- Object detection coreset selection is harder than classification because one image can contain multiple object instances.
- The paper uses object/class-aware representative features and balances representativeness and diversity.

Recommended addition:

- Do not reduce data only at image level.
- Use object-level crop features first, then aggregate to image-level recommendations.
- For each class and size bucket, select medoids/representatives and keep boundary or mixed samples.

### 3. Dataset pruning adapted to object detection

**Extending Dataset Pruning to Object Detection: A Variance-based Approach, 2025**

Paper: https://arxiv.org/abs/2505.17245

Useful idea for this project:

- Detection pruning needs object-level attribution, scoring, and image-level aggregation.
- Confidence and IoU variance can identify informative samples better than size/balance alone.

Recommended addition:

- Add optional detector-pass metrics later: prediction confidence variance, IoU stability, missed/extra detection flags.
- Combine these with feature clustering instead of using clustering alone.

### 4. Online data curation for object detection

**DetGain: Online Data Curation for Object Detection via Marginal Contributions to Dataset-level AP, 2025**

Paper: https://arxiv.org/abs/2511.14197

Useful idea for this project:

- Data curation can be dynamic during training, selecting samples that help AP most at the current model state.
- This is heavier than the current offline feature index, but it gives a later direction.

Recommended addition:

- Keep the current project as offline curation.
- Later add training-log integration: per-image loss, prediction quality, and teacher-student gap.

### 5. Redundancy-aware pruning at scale

**InfoMax: Data Pruning by Information Maximization, 2025**

Paper: https://arxiv.org/abs/2506.01701

Useful idea for this project:

- Useful pruning should maximize sample importance while minimizing redundancy from similar samples.
- Similarity matrix sparsification and dataset partitioning make million-scale pruning feasible.

Recommended addition:

- Build a sparse kNN graph from FAISS top-k neighbors.
- Score each sample by `importance - redundancy`.
- At first, use simple importance proxies: class rarity, cluster boundary, cross-class overlap, FP-neighbor hit count, and manual review flag.

### 6. Zero-shot/unlabeled coreset selection

**ZCore: Zero-Shot Coreset Selection, 2024**

Paper: https://arxiv.org/abs/2411.15349

Useful idea for this project:

- Coreset selection can use embedding coverage and redundancy without labels.
- Our project already has model-specific YOLO features, so the same principle can be applied without CLIP/DINO at first.

Recommended addition:

- Add coverage/redundancy scores from current YOLO feature space.
- Optionally compare with CLIP/DINO later as a second-view embedding.

### 7. Group data attribution

**Generalized Group Data Attribution, 2024**

Paper: https://arxiv.org/abs/2410.09940

Useful idea for this project:

- Attribution to individual samples is expensive.
- Attribution to groups can speed up influence-style analysis by treating clusters as units.

Recommended addition:

- Use clusters as attribution groups.
- If later adding TRAK/TracIn/dattri, run it on cluster representatives or suspicious cluster groups first.

### 8. TRAK and data attribution reliability

**Imperfect Influence, Preserved Rankings: A Theory of TRAK for Data Attribution, 2026**

Paper: https://arxiv.org/abs/2602.01312

Useful idea for this project:

- Data attribution may be approximate, but ranking preservation is often the practical target.
- This matches the project goal: not proving a single exact cause, but ranking likely candidate training samples/groups.

Recommended addition:

- Present influence results as ranked evidence, not as absolute cause.
- Cross-check similarity rank, cluster membership, and attribution rank.

## Current Clustering Coverage

Implemented now:

- PCA coordinates: 3D and 2D visualization from sampled features.
- MiniBatchKMeans:
  - global
  - per class
  - class + size bucket
- Cluster summary:
  - class purity
  - size purity
  - dominant class
  - dominant size
- Cross-class overlap view:
  - currently based on PCA 3D nearest-neighbor distance
- Point preview and two-point comparison.
- DB-neighbor search from selected records.

Important limitation:

- PCA 2D/3D is for visualization. Final curation decisions should use original 1792-d normalized feature distances.

## Additional Clustering Methods To Add

### P0: Original-feature nearest-neighbor graph

Purpose:

- Near duplicate grouping
- Cross-class overlap in real feature space
- Reduction candidate generation

Method:

- Use FAISS to compute top-k neighbors for sampled or full records.
- Build sparse neighbor graph with cosine similarity.
- Output:
  - duplicate groups
  - cross-class nearest pairs
  - same-class redundancy clusters
  - suspicious mixed-class pairs

Why first:

- It directly uses the existing FAISS index.
- It avoids the distortion of PCA space.
- It is the most aligned with false-positive search.

### P1: Medoid / representative selection

Purpose:

- Choose representative samples per cluster for dataset reduction.

Method:

- For each class + size + cluster group:
  - compute centroid
  - select nearest sample to centroid as medoid
  - also keep a few farthest/boundary samples

Output:

- `keep_representative.csv`
- `keep_boundary.csv`
- `drop_near_duplicate_candidates.csv`

### P1: BisectingKMeans

Purpose:

- More hierarchical grouping than flat KMeans.
- Useful for large classes with broad visual modes.

Availability:

- Available in current scikit-learn environment.

Recommended use:

- Sample or per-class/class-size feature groups.
- Good alternative UI option next to MiniBatchKMeans.

### P1: BIRCH

Purpose:

- Incremental clustering for large feature sets.
- Useful for coarse compression before more expensive clustering.

Availability:

- Available in current scikit-learn environment.

Recommended use:

- Per-class or class-size subsets.
- Good for building coarse groups, then selecting representatives.

### P2: HDBSCAN

Purpose:

- Find variable-density groups and noise/outliers without predefining cluster count.
- Useful for label-noise and rare-pattern discovery.

Availability:

- Available in current scikit-learn environment.

Recommended use:

- Sampled features only, or per-class subsets.
- Full 3.86M x 1792 direct HDBSCAN is not practical.

Output:

- dense groups
- noise samples
- small rare clusters

### P2: DBSCAN / OPTICS

Purpose:

- Density-based near-duplicate and outlier detection.

Availability:

- Available in current scikit-learn environment.

Recommended use:

- Not for full high-dimensional DB directly.
- Use on PCA-50, per-class sample, or FAISS-neighborhood subgraphs.

### P2: K-center greedy / farthest-point sampling

Purpose:

- Diversity-preserving subset selection.
- Good for "reduce DB while preserving coverage".

Method:

- Start from medoids or high-importance samples.
- Iteratively keep samples farthest from the current kept set.
- Use FAISS for scalable approximate distance.

Output:

- `coreset_keep_10pct.csv`
- `coreset_keep_20pct.csv`
- coverage statistics

### P2: Facility-location / submodular coreset

Purpose:

- Select samples that maximize coverage and minimize redundancy.

Method:

- Approximate with sparse kNN graph.
- Objective: high coverage of all samples with low duplicate selection.

Recommended use:

- Best long-term method for training DB reduction.
- More implementation work than KMeans.

### P3: Graph community clustering

Purpose:

- Detect communities from kNN graph.
- Strong for mixed-class region analysis.

Method:

- FAISS kNN graph -> community detection such as Leiden/Louvain.

Dependency:

- Requires extra packages such as `igraph`/`leidenalg` or `networkx`.

Recommended use:

- Add only after P0/P1 reports are stable.

## Recommended Curation Report

Add a new report tab or CLI command:

```text
scripts/build_curation_report.py
```

Inputs:

- feature index dir
- class filter
- size bucket filter
- top-k neighbor count
- duplicate threshold
- representative budget

Outputs:

- `cluster_summary.csv`
- `near_duplicates.csv`
- `cross_class_overlap.csv`
- `representatives.csv`
- `boundary_samples.csv`
- `drop_candidates.csv`
- `review_candidates.csv`

Recommended decision labels:

- `KEEP_REPRESENTATIVE`
- `KEEP_BOUNDARY`
- `REVIEW_LABEL`
- `REVIEW_CROSS_CLASS_OVERLAP`
- `ADD_HARD_NEGATIVE`
- `DROP_NEAR_DUPLICATE`
- `DO_NOT_DROP_RARE_CLUSTER`

## Practical Priority

1. Move cross-class overlap from PCA 3D distance to original 1792-d feature cosine distance.
2. Add near-duplicate grouping using FAISS top-k neighbors.
3. Add representative/medoid selection per class + size + cluster.
4. Add curation CSV export with keep/review/drop recommendations.
5. Add clustering method selector:
   - MiniBatchKMeans
   - BisectingKMeans
   - BIRCH
   - HDBSCAN for sampled/per-class mode
6. Add optional coreset builder:
   - k-center greedy
   - sparse facility-location approximation
7. Later add training-aware scoring:
   - detector loss
   - prediction confidence variance
   - teacher-student gap
   - AP contribution proxy

## Recommended Interpretation Rules

- Similarity search finds candidate causes, not guaranteed causes.
- Mixed-class clusters are review targets, not automatic deletion targets.
- Near duplicates are safe reduction candidates only when class, size, source, and visual content agree.
- Rare clusters should usually be kept, even if small.
- Data reduction must be validated by retraining/evaluation:
  - mAP
  - class AP
  - FP count on fixed videos
  - confidence of known false positives
  - missed true positives

