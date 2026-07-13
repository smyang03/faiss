# 오감지 원인 추적 설계 메모

## 배경

커스터마이징한 객체 감지 모델을 런타임에 적용한 뒤 오감지가 발생했다. 목표는 오감지된 detection이 기존 학습 DB의 어떤 샘플과 관련이 있는지, 또는 어떤 학습 요인 때문에 발생했는지 추정하는 것이다.

핵심 질문은 두 가지로 나뉜다.

1. **샘플을 찾을 것인가**
   - 오감지 crop과 유사한 학습 이미지 또는 bbox crop을 찾는다.
   - 사람이 라벨 오류, class 혼동, annotation 누락, 배경 편향을 검수하기 쉽다.
   - 실무 PoC로 가장 빠르고 설명 가능하다.

2. **학습 웨이트/영향을 찾을 것인가**
   - 특정 오감지 score를 키운 학습 샘플이 무엇인지 gradient attribution으로 추정한다.
   - 단순 유사도보다 원인성에 가까운 분석이 가능하다.
   - detector loss, target function, checkpoint 관리가 필요해 구현 난도가 높다.

## 최종 방향

최종 설계는 **샘플 검색을 1차 목표로 두고, attribution 계열 분석을 섞는 하이브리드 방식**으로 간다.

즉, 먼저 “이 오감지와 비슷한 학습 샘플”을 찾고, 이후 “이 오감지 score를 실제로 밀어올린 학습 샘플”을 dattri/TRAK 계열로 재랭킹한다.

## 현재 전제

- 모델: YOLOv7
- 학습 annotation: YOLO txt format
- 1차 목표: 오감지와 유사하거나 오감지 score를 지지한 학습 샘플 찾기
- 2차 목표: dattri/TRAK으로 influence 기반 재랭킹

YOLO txt annotation은 일반적으로 다음 형식이다.

```text
class_id x_center y_center width height
```

좌표는 이미지 크기 기준 normalized 값이므로 crop 추출 시 pixel 좌표로 변환한다.

```text
x1 = (x_center - width / 2) * image_width
y1 = (y_center - height / 2) * image_height
x2 = (x_center + width / 2) * image_width
y2 = (y_center + height / 2) * image_height
```

YOLOv7 기준으로는 1차 샘플 검색 MVP와 2차 attribution 모두 진행 가능하다. 다만 attribution에서는 NMS 이후 bbox만으로는 부족하고, 가능하면 NMS 전 prediction tensor에서 해당 오감지 bbox와 IoU가 가장 높은 candidate를 찾아 raw score를 target으로 잡는 것이 좋다.

## 권장 파이프라인

### 1. 오감지 query 정의

런타임에서 발생한 false positive detection을 query로 정의한다.

필요 정보:

- 원본 이미지 경로
- 오감지 bbox
- predicted class
- confidence score
- 가능하면 pre-NMS prediction index
- 모델 출력의 raw logit 또는 objectness/class logit

YOLO 계열 예시:

```text
target_score = objectness_logit + class_logit[pred_class]
```

Faster R-CNN 계열 예시:

```text
target_score = roi_class_logit[pred_class]
```

### 2. 학습 DB object patch 인덱싱

학습 DB의 GT bbox 또는 학습에 사용된 object region을 crop/patch 단위로 관리한다.

샘플별 메타데이터:

- image path
- bbox
- label
- annotation id
- split
- source dataset/version
- augmentation 여부가 추적 가능하면 augmentation metadata

YOLO txt 기준 권장 입력 구조:

```text
dataset/
  images/
    train/
      xxx.jpg
    val/
      yyy.jpg
  labels/
    train/
      xxx.txt
    val/
      yyy.txt
  data.yaml
```

이미지와 라벨은 stem 기준으로 매칭한다.

```text
images/train/abc.jpg <-> labels/train/abc.txt
```

`data.yaml`의 `names` 또는 별도 class names 파일을 읽어 `class_id -> class_name`을 매핑한다.

### 3. 유사 샘플 검색

먼저 embedding 기반으로 유사한 학습 샘플을 찾는다.

사용 후보:

- 모델 backbone feature
- detector neck/head 직전 feature
- DINOv2
- CLIP/OpenCLIP
- 필요 시 FAISS/Qdrant로 vector index 구성

이 단계의 목적은 전체 학습 DB를 바로 attribution하지 않고 후보를 줄이는 것이다.

예시:

```text
FP crop -> embedding -> top 500~5000 similar train patches
```

### 4. attribution 분석

후보군 또는 전체 학습셋에 대해 training data attribution을 수행한다.

우선순위:

1. **dattri + TracIn**
   - checkpoint가 여러 개 있을 때 1순위
   - MIT 라이센스
   - PyTorch model, loss_func, target_func를 직접 정의 가능

2. **MadryLab/TRAK**
   - 대규모 attribution에 강함
   - custom `AbstractModelOutput`으로 detector target을 정의할 수 있음

3. **Captum TracInCP**
   - PyTorch 생태계 안정성 좋음
   - top-k proponents/opponents API 제공

4. **Kronfluence**
   - final checkpoint만 있을 때 검토
   - EKFAC/KFAC 기반 influence function

attribution에서 찾고 싶은 것은 다음이다.

- positive influence: 오감지 score를 키운 것으로 추정되는 학습 샘플
- negative influence: 오감지 score를 억제한 것으로 추정되는 학습 샘플

### 5. 결과 병합

최종 결과는 유사도 순위와 influence 순위를 함께 보여준다.

출력 예시:

| rank | train image | train label | bbox | similarity | influence | note |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | path/to/image.jpg | class_a | xyxy | 0.91 | 12.4 | label 확인 필요 |

우선 검수 대상:

- 유사도와 influence가 모두 높은 샘플
- influence는 높은데 시각적으로 덜 유사한 샘플
- 같은 predicted class로 몰리는 샘플
- 특정 배경/촬영 조건에 치우친 샘플

### 6. 검증

원인 후보를 찾은 뒤에는 반드시 ablation으로 검증한다.

검증 방법:

- top-k 영향 샘플 라벨 수정
- annotation 누락 보완
- 오염 샘플 제거
- hard negative 추가
- 소규모 fine-tune 후 동일 런타임 이미지에서 FP score 변화 확인

성공 기준:

- 해당 false positive confidence 감소
- 같은 패턴의 오감지 재발률 감소
- 정상 true positive 성능이 크게 떨어지지 않음

## 샘플 찾기 vs 웨이트 찾기

이번 목적에서는 **웨이트 자체를 찾는 것보다 샘플을 찾는 것이 우선**이다.

이유:

- 웨이트 단위 원인 분석은 detector 구조상 해석이 어렵다.
- 특정 weight/channel이 문제라고 해도 데이터 수정 액션으로 바로 이어지기 어렵다.
- 현업에서 수정 가능한 대상은 대개 학습 샘플, 라벨, negative data, augmentation 정책이다.

따라서 최종 판단은 다음과 같다.

```text
1차 목표: 오감지를 유발하거나 지지한 학습 샘플 찾기
2차 보조: gradient attribution으로 유사 샘플 후보를 재랭킹/검증
비목표: 특정 weight 하나를 직접 원인으로 단정하기
```

## 최종 결론

라이센스 제약이 없다면 **FiftyOne 또는 유사한 데이터 브라우저를 샘플 분석 워크벤치로 쓰고, dattri/TRAK을 attribution 엔진으로 붙이는 방식**이 가장 실용적이다.

구현 순서는 다음이 적절하다.

1. 오감지 bbox crop 수집
2. 학습 DB bbox crop/metadata 정리
3. embedding similarity top-k 검색
4. dattri/TRAK으로 오감지 target score attribution 수행
5. similarity rank와 influence rank 비교
6. top-k 샘플 검수
7. 샘플 수정/제거/추가 후 재학습 또는 fine-tune으로 검증

## 현재 데모 구현 방향

현재 PoC는 **샘플 검색을 1차 목표**로 두고, YOLO 내부 feature map 기반 검색을 추가했다.

- 입력 DB: YOLO txt annotation + 원본 이미지
- 모델: `model/화재_train__pj_fire_8class_v2__mn_w122.pt`
- YOLO repo: `external/yolov7`
- feature: YOLO Detect head 직전 입력 feature map(P3/P4/P5)을 bbox ROI mean pooling 후 concat/normalize
- index: `features.npy`, `records.json`, `config.json`, `index.faiss`
- 최종 fireDB index 경로: `artifacts/yolo_feature_index_fire_8class_w122`

런타임 흐름:

```text
동영상 입력 -> YOLO 검출 -> crop 후보 선택 -> YOLO feature query -> FAISS top-k 학습 bbox 검색
crop 이미지 입력 -> crop 전체 bbox로 YOLO feature query -> FAISS top-k 학습 bbox 검색
```

로컬 RTX 4090 실행:

```powershell
conda run -n yolov7 python -u scripts/run_firedb_yolov7_feature_build.py
```

8 GPU 서버 실행:

```powershell
$env:YOLO_FEATURE_DEVICES="0,1,2,3,4,5,6,7"
conda run -n yolov7 python -u scripts/run_firedb_yolov7_feature_build.py
```

서버에서 shard 수를 GPU 수와 다르게 잡고 싶으면 다음처럼 지정한다.

```powershell
$env:YOLO_FEATURE_DEVICES="0,1,2,3,4,5,6,7"
$env:YOLO_FEATURE_NUM_SHARDS="16"
conda run -n yolov7 python -u scripts/run_firedb_yolov7_feature_build.py
```

진행 로그:

```text
artifacts/firedb_firemodel_logs/launcher.log
artifacts/firedb_firemodel_logs/prepare_records.out.log
artifacts/firedb_firemodel_logs/shard_0.out.log
artifacts/firedb_firemodel_logs/merge.out.log
```

## 리서치 검토: 현재 방식의 타당성

결론: 현재 방식은 실무적으로 타당하다. 다만 표현은 **오감지 원인 확정**이 아니라 **YOLO 내부 feature 공간에서 유사하게 반응한 학습 bbox 후보 검색**이 맞다.

근거:

- Fast R-CNN은 전체 이미지에서 convolution feature map을 만든 뒤, 각 RoI에 대해 RoI pooling으로 fixed-length feature vector를 추출한다. 즉 bbox/region 단위로 feature map을 pooling해 객체 표현을 만드는 방식은 object detection 계열의 표준 아이디어다.
  - https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
- FPN은 여러 scale의 feature map(P2/P3/P4/P5)을 사용해 multi-scale object representation을 구성한다. YOLO 계열의 P3/P4/P5 feature를 함께 쓰는 것은 이 multi-scale 표현을 활용하는 방향과 맞다.
  - https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf
- FiftyOne Brain은 object patch 단위 similarity index를 공식 기능으로 제공한다. 즉 detection bbox/object patch를 embedding화해서 유사 객체를 찾는 워크플로우는 실제 데이터 검수 툴에서도 쓰인다.
  - https://docs.voxel51.com/brain.html#object-similarity
- Deep k-Nearest Neighbors 계열 연구는 neural network의 layer representation에서 test input과 가까운 training points를 찾고, 이를 interpretability/robustness/OOD 판단에 활용한다. 현재 방식은 이를 object detection bbox 단위로 적용한 형태에 가깝다.
  - https://arxiv.org/abs/1803.04765
  - https://proceedings.mlr.press/v162/sun22d.html
- FAISS 공식 문서 기준으로 L2 normalize된 vector에 inner product search를 적용하면 cosine similarity 검색으로 해석할 수 있다. 현재 구현은 이 조건을 따른다.
  - https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances

현재 구현의 해석:

```text
YOLO feature map + bbox ROI pooling + L2 normalize + FAISS cosine top-k
= 모델이 내부 feature 공간에서 비슷하게 본 학습 bbox 후보 검색
```

주의점:

- YOLOv7은 Faster R-CNN처럼 명시적인 RoI head가 있는 모델이 아니므로, 이 방식은 detector head 입력 feature map에 RoI pooling을 얹은 분석용 adaptation이다.
- crop 이미지만 넣는 검색은 원본 frame context와 scale 정보가 사라져 결과가 흔들릴 수 있다.
- 가장 신뢰할 수 있는 query는 **원본 frame + detection bbox** 방식이다.
- 유사도가 높은 샘플이 곧 “학습에 영향을 준 샘플”이라는 뜻은 아니다. 영향도까지 보려면 TracIn/TRAK/dattri 같은 training data attribution이 필요하다.

운영 검증 기준:

1. 학습 bbox를 query로 넣었을 때 자기 자신 또는 거의 같은 frame이 top-k에 나오는지 확인한다.
2. 같은 class/비슷한 배경/비슷한 bbox scale이 top-k에 몰리는지 확인한다.
3. 오감지 query에서 top-k 샘플을 라벨 오류, annotation 누락, hard negative 부족 관점으로 검수한다.
4. top-k 후보를 수정/추가/제거 후 재학습하여 동일 오감지 score가 낮아지는지 ablation으로 확인한다.

다음 개선 후보:

- query mode를 기본적으로 `원본 frame + bbox`로 유도한다.
- P3/P4/P5 각각의 유사도와 concat 유사도를 함께 표시한다.
- 같은 predicted class만 검색 / 전체 class 검색을 선택 가능하게 한다.
- top-k 결과의 class 분포, source 영상 분포, bbox 크기 분포를 같이 요약한다.
- 이후 TracIn/TRAK/dattri로 top-k 후보를 재랭킹한다.
