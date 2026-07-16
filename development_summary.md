# YOLO Feature Search / Curation 개발 요약

작성일: 2026-07-16  
현재 커밋: `d060443`  
실행 URL: `http://localhost:8501`  
주요 프로젝트: `fire_8class_w122`, `safety_env`

## 1. 개발 목표

커스텀 YOLO 계열 객체 감지 모델에서 오검출이 발생했을 때, 기존 학습 DB 중 어떤 샘플과 유사한지 찾고, feature 기반으로 중복/준중복 데이터를 시각적으로 검토한 뒤 실제 경량화 DB를 추출할 수 있게 만드는 것이 목표였다.

핵심 질문은 두 가지다.

- 오검출 crop이 기존 학습 DB의 어떤 bbox crop과 가까운가?
- 학습 DB 안에서 너무 유사한 데이터 그룹을 찾아 대표 샘플만 남기고 줄일 수 있는가?

## 2. 전체 구조

Streamlit UI 중심으로 구성했다.

- `app.py`: 프로젝트 관리, crop 검색, 영상 검출 검색, clustering, curation/reduction UI
- `fp_finder/yolo_feature_index.py`: YOLO feature index 로딩/검색
- `fp_finder/feature_clustering.py`: feature 기반 2D/3D clustering, class/size metadata
- `fp_finder/curation.py`: curation report, similarity reduction plan, reduced dataset export
- `scripts/build_similarity_reduction_plan.py`: feature 기반 경량화 plan CLI
- `scripts/export_similarity_reduction_plan.py`: 경량화 dataset export CLI
- `scripts/run_health_check.py`: 주요 기능 smoke/health check
- `scripts/run_ui_regression.py`: 브라우저 UI regression

## 3. Feature 검색 방식

현재 핵심 feature는 YOLOv7 feature map을 사용한다.

- P3/P4/P5 feature map에서 bbox ROI pooling
- 각 scale feature를 평균 pooling
- P3, P4, P5 vector를 concat
- 현재 FireDB index dimension: `1792`

의미:

- P3: 세밀한 edge/texture/local pattern
- P4: 중간 형태
- P5: 큰 구조/semantic pattern

검색은 FAISS 기반 nearest neighbor search로 동작한다. crop 이미지 또는 영상에서 검출된 crop을 같은 방식으로 feature화하고, DB feature index에서 top-k 유사 bbox crop을 찾는다.

## 4. UI 주요 기능

### 4.1 Feature Projects

프로젝트 단위로 모델, DB, feature index를 묶는다.

- 모델 weight 경로
- YOLO repo 경로
- image root / label root
- dataset layout
- feature index 경로
- batch size / shard / FAISS 설정
- 프로젝트별 script 저장
- class/size metadata cache 생성

### 4.2 Crop Image Search

crop 이미지를 직접 넣고 DB에서 유사 bbox crop을 검색한다.

- crop 업로드
- feature 추출
- FAISS top-k 검색
- 결과 crop card 표시
- 유사 DB 결과에서 다시 neighbor search 가능
- 선택한 이미지 경로 export 가능

### 4.3 Video Detection Search

동영상을 넣으면 detector로 객체 bbox를 검출하고 crop별로 DB 검색을 수행한다.

- 동영상 frame sampling
- YOLO detector inference
- 검출 crop 카드 표시
- crop별 유사 DB 검색
- preview / data location / neighbor search 제공

### 4.4 Feature Clustering

저장된 feature index를 이용해 데이터 분포를 시각화한다.

- 3D/2D projection 선택
- class별 보기
- cluster별 보기
- close-overlap only 모드
- point click preview
- 두 점 선택 compare
- hover preview는 메모리 이슈 방지를 위해 선택 사용

## 5. Curation / Reduction 핵심 기능

### 5.1 Curation Report

feature kNN 기반으로 다음 항목을 생성한다.

- near duplicate
- duplicate group
- cross-class overlap
- representative sample
- boundary sample
- rare sample
- image-level recommendation

이 기능은 오검출 원인 후보, 클래스 혼동 후보, 라벨 검토 후보를 찾기 위한 분석 기능이다.

### 5.2 Similarity Reduction Planner

목표 감소율을 정하지 않고, 실제 feature space에서 매우 가까운 same-class/same-size group만 묶는다.

기본 원칙:

- 같은 class끼리만 reduction group 생성
- 같은 size bucket끼리만 묶기
- group마다 대표 sample 유지
- cross-class와 가까운 sample은 자동 drop하지 않고 보호
- 결과적으로 자연스럽게 줄일 수 있는 양을 산출

주요 출력:

- `reduction_summary.json`
- `reduction_groups.csv`
- `reduction_group_members.csv`
- `reduction_drop_records.csv`
- `reduction_keep_records.csv`
- `reduction_image_plan.csv`
- `reduction_tight_edges.csv`

## 6. 시각화 개선 사항

사용자가 표와 숫자만 보고는 "왜 줄여도 되는지" 납득하기 어렵기 때문에 시각 검증 화면을 강화했다.

### 6.1 Overview

`Visual -> Overview`에 다음을 배치했다.

- record flow Sankey
- class/action stacked chart
- image keep/drop pie chart
- group size vs mean similarity scatter
- class별 summary table
- at-a-glance sample board

### 6.2 At-a-glance Sample Board

이번에 추가한 핵심 시각화다.

한 화면에서 여러 유사 그룹의 실제 crop을 바로 볼 수 있다.

- 한 줄 = 하나의 tight feature group
- 왼쪽 = 유지할 representative crop
- 오른쪽 = drop/protected 후보 crop
- 선 = representative와 후보 간 similarity 관계
- 초록 = 대표 keep
- 빨강 = drop candidate
- 노랑 = protected keep

기본 board mode:

- `One Top Group Per Class`: 클래스별 대표 중복 그룹을 하나씩 보여줌
- `Largest Drop Groups`: drop 후보가 가장 많은 그룹 위주
- `Largest Group`: group size가 큰 순서
- `Highest Similarity`: 평균 similarity가 높은 순서

성능상 네트워크 드라이브 이미지를 자동으로 수십 장 열면 앱이 느려질 수 있어서, `Generate Sample Board` 버튼으로 한 번 생성하고 PNG로 캐시한다.

현재 생성된 기본 board:

```text
artifacts/reduction_plans/fire_8class_w122/full_t0995/reduction_sample_board_All_One_Top_Group_Per_Class_8x4.png
```

### 6.3 Evidence Map

선택한 reduction group 하나를 상세하게 보여준다.

- 대표 crop 1장
- drop/protected 후보 여러 장
- similarity 선 연결
- PNG 다운로드 가능
- evidence rows table 표시

용도:

- sample board로 전체 패턴 확인
- Evidence Map으로 특정 group 상세 확인
- Group Compare로 개별 crop 카드 확인

## 7. 실제 FireDB 분석 결과

기준 plan:

```text
artifacts/reduction_plans/fire_8class_w122/full_t0995
```

조건:

- full plan
- threshold: `0.995`
- same class only: `true`
- same size only: `true`
- top-k: `30`
- rerank-k: `200`
- feature dim: `1792`

요약:

| 항목 | 값 |
|---|---:|
| 전체 record | 102,087 |
| planned record | 102,087 |
| tight edges | 690,423 |
| tight groups | 9,002 |
| grouped records | 63,013 |
| representative records | 9,002 |
| protected records | 3,472 |
| drop record candidates | 52,791 |
| record reduction | 51.71% |
| image drop candidates | 36,569 |
| safe image drop candidates | 36,569 |

이미지 기준:

| 항목 | 값 |
|---|---:|
| 전체 image | 78,275 |
| keep image | 41,706 |
| drop image candidate | 36,569 |
| image reduction | 46.72% |

해석:

- 보수적인 `0.995`에서도 bbox/record 기준 약 절반이 매우 유사한 group으로 묶였다.
- 실제 데이터 경량화 가능성은 충분히 있다.
- 다만 cross-class 근접 sample은 보호되므로 자동 삭제하지 않는 것이 맞다.

클래스별 경향:

| class | drop 비율 해석 |
|---|---|
| `other_smoke` | 매우 높은 중복성 |
| `ir_fire` | 매우 높은 중복성 |
| `other_fire` | 매우 높은 중복성 |
| `ramp` | 높은 중복성 |
| `ir_ramp` | 높은 중복성 |
| `fire` | 상대적으로 다양한 sample 포함 |
| `smoke` | 상대적으로 다양한 sample 포함 |
| `ir_smoke` | 가장 적게 줄어듦 |

상위 중복 group 예:

| group | class | group size | drop | mean sim |
|---:|---|---:|---:|---:|
| G8845 | other_fire | 358 | 350 | 0.999667 |
| G8846 | other_fire | 285 | 255 | 0.999678 |
| G8847 | other_fire | 219 | 215 | 0.999632 |

## 8. Reduced Dataset Export

Similarity Reduction Planner의 Export 탭 또는 CLI로 실제 경량화 dataset을 생성할 수 있다.

지원 mode:

- `manifest`: 목록만 생성
- `copy`: 실제 이미지/라벨 복사
- `hardlink`: 가능하면 hardlink 생성, 실패 시 copy

label policy:

- `filtered`: YOLO txt label을 kept bbox annotation만 남기도록 재작성
- `original`: 원본 label 그대로 복사

운영 추천:

1. 먼저 `manifest`로 결과 확인
2. Visual tab에서 sample board / evidence map 확인
3. 문제 없으면 `copy + filtered`로 실제 학습용 reduced dataset 생성

CLI 예:

```powershell
python -u scripts/export_similarity_reduction_plan.py `
  --plan-dir artifacts/reduction_plans/fire_8class_w122/full_t0995 `
  --output-dir artifacts/reduced_datasets/fire_8class_w122/full_t0995_filtered `
  --images-root V:\dataset\images `
  --labels-root V:\dataset\labels `
  --data-yaml firedb_v1_data.yaml `
  --mode copy `
  --label-policy filtered
```

## 9. 검증 결과

실행한 검증:

- `python -m py_compile app.py fp_finder/curation.py scripts/export_similarity_reduction_plan.py`
- Streamlit AppTest: 예외 0개
- sample index health check: `10/10 PASS`
- fire index health check: `10/10 PASS`
- browser regression: console error 0개, page error 0개, warning 0개
- Streamlit stderr: 0 byte

최근 UI regression 결과:

```text
artifacts/ui_regression/sample_board_latest/ui_regression_summary.json
```

최근 Fire health check 결과:

```text
artifacts/health_checks/sample_board_fire/health_check_summary.json
```

## 10. 현재 운영 상태

Streamlit 서버:

```text
http://localhost:8501
```

최근 확인:

- HTTP status: `200`
- PID: `950384`

GitHub push 완료:

```text
https://github.com/smyang03/faiss.git
branch: main
latest commit: d060443
```

## 11. 운영 가이드

일반 분석 순서:

1. `Feature Projects`에서 프로젝트 선택/검증
2. `Crop Image Search` 또는 `Video Detection Search`에서 오검출 crop 검색
3. 검색된 DB neighbor를 다시 neighbor search
4. `Feature Clustering`에서 class/cluster overlap 확인
5. `Curation Report -> Similarity Reduction Planner`에서 reduction plan 선택
6. `Visual -> Overview`의 sample board로 전체 중복 패턴 확인
7. `Evidence Map`에서 특정 group 상세 검토
8. `Export`에서 `manifest` 또는 `copy + filtered` 실행
9. reduced dataset으로 재학습 후 기존 full dataset과 성능 비교

주의:

- sample plan은 검토용이다. 실제 삭제/추출 판단은 `partial_plan=false`인 full plan 기준으로 해야 한다.
- `0.985`는 더 많이 줄지만 공격적이다.
- 현재 운영 1차 기준은 `0.995`가 더 보수적이고 적합하다.
- cross-class protected sample은 모델이 헷갈리는 경계 sample일 수 있으므로 자동 삭제하면 안 된다.

## 12. 다음 개선 후보

- sample board에 group click -> Evidence Map 자동 이동
- class별 reduction board를 한 번에 batch 생성
- reduced dataset 생성 후 원본/축소본 class 분포 비교 리포트
- 학습 결과 mAP/오검출률 비교 리포트 연결
- protected cross-class sample 전용 검수 화면 강화
- label 품질 오류 후보 자동 리포트
