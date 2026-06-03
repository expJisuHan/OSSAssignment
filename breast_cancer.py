from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

data = load_breast_cancer()

X = data.data
y = data.target

print("입력 데이터 형태:", X.shape)
print("타겟 데이터 형태:", y.shape)
print("특성 이름 개수:", len(data.feature_names))
print("클래스 이름:", data.target_names)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n[분류 모델 평가 결과]")
print("Accuracy:", accuracy)

print("\n실제값:")
print(y_test)

print("\n예측값:")
print(y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\n[Confusion Matrix]")
print(cm)

print("\n[Classification Report]")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)