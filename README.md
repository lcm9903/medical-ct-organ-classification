# 복부 CT 장기 분류 (OrganAMNIST) — 머신러닝 · 전이학습 · 이미지 증강 비교

64×64 흑백 복부 CT axial 패치 이미지를 **11개 장기 클래스**(좌/우 구분 포함)로 분류하는 프로젝트입니다.
전통적 머신러닝(RandomForest)과 ResNet50 전이학습을 비교하고, 소량 데이터에서 **이미지 증강(Data Augmentation)** 이 성능과 과적합에 어떤 영향을 주는지 실험했습니다.

> AI 헬스케어 7기 「의료 이미지 분석을 위한 컴퓨터 비전(CV)」 개별 미션

---

## 📁 노트북 구성

| 노트북 | 내용 |
|---|---|
| `[AI 헬스케어 7기] 이채민_개별 미션 2차.ipynb` | 전체 Train(클래스당 500장) 사용. **HOG 특성공학 + RandomForest(3-Fold CV)** vs **ResNet50 (Feature Extraction → Fine-tuning)** 비교 |
| `이미지 증강.ipynb` | 클래스당 200장으로 **RandomForest(GridSearchCV) vs ResNet50** 비교 후, 클래스당 50장(총 550장)으로 **CNN 증강 유/무 성능 비교** |

---

## 🗂 데이터

- **데이터셋**: OrganAMNIST 형태의 복부 CT 패치 (64×64, 1채널, PNG)
- **클래스(11)**: `bladder`, `femur-left`, `femur-right`, `heart`, `kidney-left`, `kidney-right`, `liver`, `lung-left`, `lung-right`, `pancreas`, `spleen`
- **분포**: 클래스당 Train 500장 / Test 200장 (총 5,500 / 2,200) → 클래스 불균형 없음
- 폴더명이 곧 라벨(Y)입니다.

```
03_OrganAMNIST/
 ├─ train/
 │   ├─ bladder/*.png
 │   ├─ femur-left/*.png
 │   └─ ... (11개 클래스)
 └─ test/
     ├─ bladder/*.png
     └─ ...
```

---

## 🔧 전처리

1. 폴더 순회 → 이미지를 배열로 변환 (`(n, 64, 64, 1)`)
2. 픽셀 값 0~1 정규화 (`/255`)
3. 폴더명 → 정수 라벨 인코딩 (`LabelEncoder` 또는 `enumerate`)
4. ResNet50 입력용: 1채널 → 3채널 복제(`np.repeat`), 필요 시 96×96 리사이즈, `preprocess_input` 적용

---

## 🧪 실험 1. RandomForest vs ResNet50 (전체 데이터, 클래스당 500장)

**RandomForest**
- 특성공학: 원본 픽셀 대신 **HOG** 특성 추출 (orientations=9, 8×8 cell, 2×2 block → 1,764차원)
- `n_estimators=300`, `class_weight='balanced'`, StratifiedKFold 3-Fold CV
- CV 정확도 **0.8455 (±0.0092)**

**ResNet50 전이학습**
- ImageNet 가중치, 96×96×3 입력, `GAP → Dense(256) → Dropout(0.4) → Softmax`
- 1단계 Feature Extraction: base 동결, Adam(1e-4), EarlyStopping
- 2단계 Fine-tuning: 마지막 30개 레이어 unfreeze, Adam(1e-6)

| 모델 | Test Accuracy | Macro F1 |
|---|:---:|:---:|
| RandomForest (HOG + 3-Fold CV) | **0.736** | 0.734 |
| ResNet50 (Feature Extraction) | 0.656 | 0.605 |
| ResNet50 (Fine-tuning) | 0.704 | 0.651 |

**RandomForest 주요 오분류 쌍** (혼동행렬 기준)
- kidney-right → bladder (51건), femur-left ↔ kidney-left / femur-right ↔ kidney-right (각 30건), kidney-left → bladder (30건)
- 좌/우 신장, 대퇴골처럼 **형태가 유사한 좌우 쌍과 인접 장기** 간 혼동이 집중됨

> ⚠️ **ResNet50 결과 해석 시 주의**
> 이 노트북의 ResNet50은 `pancreas`, `spleen` 두 클래스의 precision/recall이 **0.000**이고 학습 중 `val_accuracy`가 약 0.09에 머물렀습니다.
> 데이터를 클래스 순서대로 로드한 뒤 **셔플 없이 `validation_split=0.2`** 를 사용해, 마지막 20%(= pancreas·spleen 전체)가 검증셋으로만 빠지고 학습에는 한 번도 쓰이지 않았기 때문입니다.
> 따라서 표의 ResNet50 수치는 모델 자체의 한계가 아니라 데이터 분할 문제로 낮아진 값이며, 셔플 후 재학습하면 실험 2처럼 크게 개선될 것으로 예상됩니다.

---

## 🧪 실험 2. RandomForest vs ResNet50 (클래스당 200장, 셔플 적용)

- 클래스당 200장(총 2,200장)만 사용, `sklearn.utils.shuffle`로 섞어서 구성
- **RandomForest**: 픽셀 평탄화(4,096차원) + `MinMaxScaler`, GridSearchCV(`max_depth` 3~5, cv=3)
- **ResNet50**: 64×64×3 그대로 입력, base 동결, `GAP → Dense(128) → BN → ReLU → Dropout(0.5) → Softmax`, 10 epochs

| 모델 | Train Acc | Test Accuracy | Macro F1 |
|---|:---:|:---:|:---:|
| RandomForest (raw pixel, depth≤5) | 0.73 | 0.60 | 0.60 |
| ResNet50 (Feature Extraction) | val 0.94 | **0.86** | 0.86 |

- RandomForest는 고차원(2,200 × 4,096) 픽셀 입력과 얕은 트리로 인해 `kidney-left`(F1 0.20), `pancreas`, `spleen` 분류가 특히 약함
- ResNet50은 셔플을 적용하자 모든 클래스를 학습해 정확도 0.86 달성. 여전히 **좌/우 신장(F1 0.61 / 0.71)** 이 가장 어려운 클래스

---

## 🧪 실험 3. 소량 데이터에서 이미지 증강 효과 (클래스당 50장, 총 550장)

**CNN 구조**: `(증강 레이어) → Rescaling(1/255) → Conv(32)-Pool → Conv(32)-Pool → Dense(128) → Dropout(0.5) → Softmax`
40 epochs, `validation_split=0.2`

**증강 레이어 (Online / 실시간 증강)**

| 기법 | 설정 | 비고 |
|---|---|---|
| RandomRotation | 0.03 (약 ±10.8°) | 과도한 회전 지양 |
| RandomTranslation | 가로/세로 6% | 빈 공간은 0으로 채움 |
| RandomZoom | ±10% | |
| RandomContrast | 10% | 강도 변환은 약하게 |
| RandomBrightness | 0.05 (약 ±12.75) | |
| ~~RandomFlip~~ | 미사용 | **좌/우 장기 라벨이 뒤바뀌므로 제외** |

**결과**

| 모델 | 최종 Train Acc | 최종 Val Acc | Test Accuracy | Macro F1 |
|---|:---:|:---:|:---:|:---:|
| CNN (증강 없음) | 0.950 | 0.809 | 0.67 | 0.67 |
| CNN (증강 적용) | 0.882 | 0.864 | **0.73** | **0.73** |

- 증강 적용 시 Train–Val 간격이 좁아져 **과적합이 억제**되고, Test 정확도가 **+6%p** 향상
- 증강 모델은 매 epoch 다른 이미지를 보므로 수렴이 조금 더 느림
- 데이터가 적을수록 증강의 이득이 크며, 증강은 정보를 새로 만드는 것이 아니라 **부족한 데이터의 정보 손실을 완화하는 도구**임

---

## 💡 의료 영상 증강 시 고려사항

- **좌우 반전 금지**: 좌/우 장기가 별도 클래스이므로 flip 시 라벨이 틀어짐
- **과도한 회전 지양**: 해부학적 방향성이 깨짐
- **강도 변환**: CT 밝기는 HU(Hounsfield Unit)라는 물리 단위이므로, 단순 밝기 조정보다 **Window Jittering**(window center/width 변경)이 더 적절
- **Offline vs Online**: 사전 증강은 학습이 빠르지만 같은 증강본을 반복, 실시간 증강(모델 앞단 레이어)은 매 epoch 새로운 변형을 학습
- 증강은 **학습 데이터에만** 적용하고 검증/평가 데이터에는 적용하지 않음

---

## 📌 결론

1. 전처리·특성공학만 잘 해도 RandomForest(HOG)가 0.74 수준의 baseline을 제공하지만, 픽셀을 그대로 쓰면 0.60으로 떨어짐 → **특성 표현이 고전 ML 성능을 좌우**
2. 적절히 셔플된 데이터에서 ResNet50 전이학습은 **0.86**으로 가장 높은 성능을 보임
3. 모든 모델에서 공통적으로 **좌/우 신장**과 bladder·pancreas·spleen 등 인접 연부조직 장기 구분이 가장 어려움
4. 소량 데이터(550장)에서는 도메인을 고려한 증강만으로 과적합이 줄고 Test 정확도가 0.67 → 0.73으로 향상

## 🔜 개선 방향

- 실험 1 ResNet50을 **셔플(또는 stratified split) 후 재학습**하여 RandomForest와 공정하게 비교
- Fine-tuning 학습률(1e-6)을 1e-5 수준으로 조정해 비교
- 전체 데이터(500장/클래스)에서 증강 효과 검증, CT 특화 증강(Window Jittering, Elastic) 추가
- 좌/우 신장 오분류 개선을 위해 위치 정보를 보존하는 입력 설계 검토

---

## ⚙️ 실행 환경

- Google Colab (GPU), Google Drive 마운트
- Python 3, TensorFlow / Keras, scikit-learn, scikit-image, NumPy, pandas, Pillow
- 시각화: matplotlib, seaborn, plotly, koreanize-matplotlib

```bash
pip install tensorflow scikit-learn scikit-image pillow pandas numpy matplotlib seaborn plotly koreanize-matplotlib
```

노트북 상단의 데이터 경로(`DATA_DIR`, `train_path`, `test_path`)를 실행 환경에 맞게 수정한 뒤 순서대로 실행하면 됩니다.
