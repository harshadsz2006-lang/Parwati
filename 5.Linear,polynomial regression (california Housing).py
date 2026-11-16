import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load data: median income -> median house value
df = fetch_california_housing(as_frame=True).frame
print("Dataset shape:", df.shape)
print(df.head())

X, y = df[["MedInc"]], df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Define, train, and evaluate both models
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": make_pipeline(
        PolynomialFeatures(degree=3), StandardScaler(), LinearRegression()
    ),
}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "MSE": mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R2 Score": r2_score(y_test, pred),
    })

print("\nComparison Table:")
print(pd.DataFrame(rows).round(4).to_string(index=False))

# Plot regression curves over the full income range
X_range = pd.DataFrame({"MedInc": np.linspace(X["MedInc"].min(), X["MedInc"].max(), 300)})

plt.figure(figsize=(8, 5))
plt.scatter(X_test["MedInc"], y_test, s=8, alpha=0.3, label="Actual data")
for (name, model), color in zip(models.items(), ["red", "green"]):
    plt.plot(X_range["MedInc"], model.predict(X_range), color=color, label=f"{name} fit")

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Linear vs Polynomial Regression - California Housing")
plt.legend()
plt.tight_layout()
plt.savefig("regression_comparison.png", dpi=150)
plt.show()
print("Graph saved as regression_comparison.png")
