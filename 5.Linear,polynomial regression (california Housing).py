# ==================== CELL 1: Import Libraries ====================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==================== CELL 2: Load California Housing Dataset ====================

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape:", df.shape)
print(df.head())


# ==================== CELL 3: Select Features and Target ====================

# Select median income as the feature
X = df[['MedInc']]

# Select median house value as the target
y = df['MedHouseVal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==================== CELL 4: Train Linear Regression Model ====================

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred_lin = lin_model.predict(X_test)


# ==================== CELL 5: Train Polynomial Regression Model ====================

# Polynomial Regression (degree 3)
poly_model = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearRegression()
)

poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

print("Both models trained successfully.")


# ==================== CELL 6: Evaluate Model Performance ====================

def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Performance:")
    print(f"MSE      : {mse:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"R2 Score : {r2:.4f}")

    return mse, mae, r2


# Evaluate both models
lin_scores = evaluate(y_test, y_pred_lin, "Linear Regression")

poly_scores = evaluate(
    y_test, y_pred_poly, "Polynomial Regression (Degree 3)"
)


# ==================== CELL 7: Create Comparison Table ====================

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression"],
    "MSE": [lin_scores[0], poly_scores[0]],
    "MAE": [lin_scores[1], poly_scores[1]],
    "R2 Score": [lin_scores[2], poly_scores[2]]
})

print("\nComparison Table:")
print(comparison.round(4).to_string(index=False))


# ==================== CELL 8: Generate Values for Plotting ====================

X_range = np.linspace(
    X['MedInc'].min(),
    X['MedInc'].max(),
    300
).reshape(-1, 1)


# ==================== CELL 9: Predict Values Using Both Models ====================

y_range_lin = lin_model.predict(
    pd.DataFrame(X_range, columns=['MedInc'])
)

y_range_poly = poly_model.predict(
    pd.DataFrame(X_range, columns=['MedInc'])
)


# ==================== CELL 10: Plot Regression Curves ====================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test['MedInc'],
    y_test,
    s=8,
    alpha=0.3,
    label="Actual data"
)

plt.plot(
    X_range.ravel(),
    y_range_lin,
    color="red",
    label="Linear fit"
)

plt.plot(
    X_range.ravel(),
    y_range_poly,
    color="green",
    label="Polynomial fit (degree 3)"
)

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Linear vs Polynomial Regression - California Housing")
plt.legend()
plt.tight_layout()

plt.savefig("regression_comparison.png", dpi=150)
plt.show()

print("Graph saved as regression_comparison.png")
