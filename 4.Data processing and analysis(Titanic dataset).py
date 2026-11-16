"""Data Preprocessing and EDA - Titanic Dataset"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = sns.load_dataset("titanic")
print(df.shape, "\n", df.head())

# 2. Cleaning and missing values
df = df.drop_duplicates().drop(columns=["deck", "embark_town", "alive", "class"])
df = df.fillna({"age": df["age"].median(), "embarked": df["embarked"].mode()[0]}).dropna()

# 3. Encoding
df["sex"] = (df["sex"] == "male").astype(int)
df = pd.get_dummies(df, columns=["embarked", "who"], drop_first=True, dtype=int)

# 4. Statistical analysis
print(df.describe())
print(df.corr(numeric_only=True)["survived"].sort_values(ascending=False))

# 5. Histograms
df[["age", "fare", "pclass"]].hist(figsize=(10, 6), bins=20)
plt.suptitle("Histograms")
plt.savefig("histograms.png")
plt.show()

# 6. Boxplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, col in zip(axes, ["age", "fare"]):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col.title())
plt.tight_layout()
plt.savefig("boxplots.png")
plt.show()

# 7. Feature scaling
num_cols = ["age", "fare", "sibsp", "parch"]
df[num_cols] = StandardScaler().fit_transform(df[num_cols])

# 8. Train-test split
X, y = df.drop(columns=["survived"]), df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)
