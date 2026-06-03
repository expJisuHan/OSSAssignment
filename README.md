# OSSAssignment - Breast Cancer Classification

## 프로젝트 소개

본 프로젝트는 scikit-learn에서 제공하는 Breast Cancer Wisconsin Dataset을 활용하여 유방암 종양의 악성(Malignant) 여부를 예측하는 머신러닝 분류 모델을 구현한 과제이다.

데이터 전처리, 모델 학습, 성능 평가 및 모델 비교 과정을 수행하였으며 GitHub Commit을 통해 단계별 개발 과정을 관리하였다.

---

# 사용 데이터셋

## Breast Cancer Wisconsin Dataset

scikit-learn에서 제공하는 대표적인 분류 데이터셋으로 종양의 다양한 특성 정보를 이용하여 악성 종양(Malignant)과 양성 종양(Benign)을 분류한다.

### 데이터 정보

| 항목    | 내용   |
| ----- | ---- |
| 데이터 수 | 569개 |
| 특성 수  | 30개  |
| 클래스 수 | 2개   |

### 클래스 정보

| 값 | 의미                |
| - | ----------------- |
| 0 | Malignant (악성 종양) |
| 1 | Benign (양성 종양)    |

---

# 개발 환경

* Python 3.11
* scikit-learn
* NumPy

---

# 데이터 분할

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

* 학습 데이터 : 80%
* 테스트 데이터 : 20%

---

# 사용 모델

## Logistic Regression

초기 분류 모델로 Logistic Regression을 사용하였다.

```python
LogisticRegression(max_iter=5000)
```

데이터 정규화를 위해 StandardScaler를 적용하였다.

### 정규화

```python
scaler = StandardScaler()
```

---

## Random Forest Classifier

모델 성능 비교를 위해 Random Forest Classifier를 적용하였다.

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

---

# 모델 성능 평가

사용한 평가 지표

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-Score

---

# 실험 결과

## Logistic Regression

### Accuracy

```text
0.9736842105263158
```

약 97.37%의 정확도를 기록하였다.

### Confusion Matrix

```text
[[41  2]
 [ 1 70]]
```

### Classification Report

```text
Precision : 0.98 / 0.97
Recall    : 0.95 / 0.99
F1-score  : 0.96 / 0.98
```

---

## Random Forest

### Accuracy

```text
0.9649122807017544
```

약 96.49%의 정확도를 기록하였다.

### Confusion Matrix

```text
[[40  3]
 [ 1 70]]
```

### Classification Report

```text
Precision : 0.98 / 0.96
Recall    : 0.93 / 0.99
F1-score  : 0.95 / 0.97
```

---

# 모델 성능 비교

| 모델                  | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 97.37%   |
| Random Forest       | 96.49%   |

### 결과

본 실험에서는 Logistic Regression 모델이 Random Forest 모델보다 높은 정확도를 보였다.

```text
Logistic Regression Accuracy : 0.9737
Random Forest Accuracy       : 0.9649
```

따라서 Breast Cancer Dataset에서는 Logistic Regression 모델이 더 우수한 분류 성능을 보이는 것으로 확인되었다.

---

# GitHub Commit 내역

## Commit 1

### Initial upload : 유방암 데이터셋 분류 모델 구현

구현 내용

* Breast Cancer Dataset 로드
* Train/Test Split
* Logistic Regression 모델 학습
* 예측 수행
* Accuracy 출력

---

## Commit 2

### Train model : 데이터 정규화 적용 및 모델 학습 성능 개선

구현 내용

* StandardScaler 추가
* 데이터 정규화 수행
* 모델 학습 안정성 향상

결과

* Accuracy : 97.37%

---

## Commit 3

### Evaluation : 혼동 행렬 및 분류 성능 평가 지표 추가

구현 내용

* Confusion Matrix 추가
* Classification Report 추가
* Precision, Recall, F1-score 분석 가능

결과

* Accuracy : 97.37%
* Confusion Matrix 출력
* Classification Report 출력

---

## Commit 4

### Parameter 변경 실험 : Random Forest 모델 적용 및 성능 비교

구현 내용

* Random Forest Classifier 추가
* Logistic Regression과 성능 비교
* Accuracy 기반 모델 분석

결과

* Logistic Regression : 97.37%
* Random Forest : 96.49%

---

# 결론

Breast Cancer Wisconsin Dataset을 이용하여 유방암 종양 분류 모델을 구현하였다. Logistic Regression 모델을 통해 97.37%의 높은 정확도를 달성하였으며, 데이터 정규화와 성능 평가 지표를 추가하여 모델을 분석하였다.

또한 Random Forest 모델을 적용하여 성능을 비교한 결과 Logistic Regression이 더 우수한 성능을 보였다. 이를 통해 데이터셋의 특성에 따라 단순한 선형 모델이 복잡한 앙상블 모델보다 더 좋은 결과를 낼 수 있음을 확인하였다.
