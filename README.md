# K의료영상기관 복부 CT 장기(Organ) 분류 프로젝트

AI 헬스케어 7기 개별 미션 2차 — 이채민

## 프로젝트 개요
OrganAMNIST 형태의 64×64 흑백 복부 CT axial 패치 이미지를 이용해 11개 장기 클래스를 분류하는 모델을 구성한다. 전통적 머신러닝(RandomForest + 특성공학 + 교차검증)과 ResNet50 전이학습 모델의 성능을 비교한다.

## 데이터 폴더 구조
```
DATA_DIR/
 ├─ Train/
 │   ├─ <organ_class_1>/*.png
 │   ├─ <organ_class_2>/*.png
 │   └─ ... (11개 장기 클래스 폴더)
 └─ Test/
     ├─ <organ_class_1>/*.png
     ├─ <organ_class_2>/*.png
     └─ ...
```
폴더명이 곧 라벨(Y)이 된다. (실행 환경에 맞게 `DATA_DIR` 경로만 수정)

## 실행 환경
- Google Colab 기준
- 주요 라이브러리: `pandas`, `numpy`, `matplotlib`, `seaborn`, `koreanize-matplotlib`, `PIL`, `scikit-learn`, `scikit-image`(HOG), `tensorflow`(Keras, ResNet50)

## 분석/모델링 단계
1. **문제 1 — 클래스 확인 및 분포 시각화**: Train 폴더의 장기(클래스) 목록 확인, Train/Test 클래스별 이미지 수 막대그래프로 불균형 여부 점검
2. **문제 2 — 데이터 로드/전처리**: 이미지를 배열로 변환, 0~1 정규화, 폴더명을 라벨로 인코딩(`LabelEncoder`), `X_train/X_test/Y_train/Y_test` 구성
3. **문제 3 — RandomForest 분류 모델**:
   - 특성공학: 원본 픽셀 대신 **HOG(Histogram of Oriented Gradients)** 특성 추출
   - 3-Fold 교차검증(`StratifiedKFold`) 적용
   - 클래스별 정밀도/재현율/F1-score, 전체 정확도 평가
4. **문제 4 — 혼동행렬 분석**: Test Set 혼동행렬 히트맵 시각화 및 오분류가 집중되는 클래스 쌍 확인
5. **문제 5 — ResNet50 전이학습**:
   - 흑백 1채널 → 3채널 변환, 64×64 → 96×96로 리사이즈
   - **1단계(Feature Extraction)**: ImageNet 사전학습 가중치 고정, 새 분류 헤드(GlobalAveragePooling+Dense)만 학습
   - 클래스 불균형 보정을 위한 `class_weight` 계산 및 적용
   - **2단계(Fine-tuning)**: base_model 뒤쪽 30개 레이어 unfreeze 후 낮은 학습률로 추가 학습
   - 두 단계 각각 학습곡선, 정확도/분류리포트, 혼동행렬 확인 → `organ_resnet50_finetuned_model.h5` 저장
6. **최종 비교**: RandomForest vs ResNet50(Feature Extraction) vs ResNet50(Fine-tuning) Test Accuracy 비교 막대그래프

## 산출물
- `organ_resnet50_finetuned_model.h5` : Fine-tuning 완료된 ResNet50 장기 분류 모델
- 클래스별 이미지 수 비교표, 혼동행렬 히트맵(RandomForest/ResNet50 FE/ResNet50 FT), 성능 비교 막대그래프

## 결론 요약
- 11개 장기 클래스 간 Train/Test 데이터 불균형 여부 확인
- HOG + 3-Fold CV 기반 RandomForest로 형태가 유사한 인접 장기(예: 좌/우 신장, 좌/우 폐 등) 간 오분류 경향 파악
- ResNet50은 class_weight로 불균형을 보정했으며, Fine-tuning 이후 복부 CT 장기 패치에 특화된 특징 학습으로 Feature Extraction 단계 대비 성능 향상 경향 확인
