from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=5000)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n==============================")
print("Logistic Regression 결과")
print("==============================")

print("Accuracy:", lr_accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")
print(
    classification_report(
        y_test,
        lr_pred,
        target_names=data.target_names
    )
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n==============================")
print("Random Forest 결과")
print("==============================")

print("Accuracy:", rf_accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=data.target_names
    )
)

print("\n==============================")
print("모델 성능 비교")
print("==============================")

print(f"Logistic Regression Accuracy : {lr_accuracy:.4f}")
print(f"Random Forest Accuracy       : {rf_accuracy:.4f}")

if rf_accuracy > lr_accuracy:
    print("→ Random Forest 모델의 성능이 더 우수합니다.")
elif rf_accuracy < lr_accuracy:
    print("→ Logistic Regression 모델의 성능이 더 우수합니다.")
else:
    print("→ 두 모델의 성능이 동일합니다.")