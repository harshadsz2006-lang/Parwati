"""Classification using Logistic Regression and KNN (Iris dataset)"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# Load data
data = load_iris()
X, y = data.data, data.target
print("Dataset shape:", X.shape, "| Classes:", data.target_names)

# Train-test split and feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate both models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}
for name, model in models.items():
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    for label, metric in [("Precision", precision_score), ("Recall", recall_score), ("F1-Score", f1_score)]:
        print(f"{label:<9}:", round(metric(y_test, y_pred, average="weighted"), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
