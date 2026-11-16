

# CELL 1: Import Libraries
# PR7: Classification using Logistic Regression and KNN



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



# CELL 2: Load Dataset

data = load_iris()
X, y = data.data, data.target

print("Dataset shape:", X.shape, "| Classes:", data.target_names)


# CELL 3: Preprocessing - Train-Test Split + Feature Scaling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# CELL 4: Train Models

log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_pred_knn = knn.predict(X_test)



# CELL 5: Evaluate Models


def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average='weighted'), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, average='weighted'), 4))
    print("F1-Score :", round(f1_score(y_true, y_pred, average='weighted'), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


# Evaluate Logistic Regression
evaluate("Logistic Regression", y_test, y_pred_lr)

# Evaluate KNN
evaluate("KNN (k=5)", y_test, y_pred_knn)
