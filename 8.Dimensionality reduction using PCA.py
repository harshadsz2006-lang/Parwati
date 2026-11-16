
# ================= CELL 1: Import Libraries =================

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ================= CELL 2: Load and Preprocess Data =================

X, y = load_digits(return_X_y=True)

print("Original shape:", X.shape)
print("Number of features:", X.shape[1])
print("Number of classes:", len(np.unique(y)))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

print("Training samples:", len(X_tr_s))
print("Testing samples:", len(X_te_s))


# ================= CELL 3: Evaluation Function =================

def evaluate(Xa, Xb, label):
    clf = LogisticRegression(max_iter=2000)
    t = time.time()
    clf.fit(Xa, y_tr)
    training_time = time.time() - t
    acc = accuracy_score(y_te, clf.predict(Xb))

    print(f"{label:<25} features={Xa.shape[1]:<3} "
          f"accuracy={acc:.4f} train_time={training_time:.4f}s")
    return acc


# ================= CELL 4: PCA and Accuracy Comparison =================

acc_full = evaluate(X_tr_s, X_te_s, "Without PCA")

pca95 = PCA(n_components=0.95)
X95 = pca95.fit_transform(X_tr_s)
acc_95 = evaluate(X95, pca95.transform(X_te_s), "PCA (95% variance)")

print("\nComponents kept for 95% variance:", pca95.n_components_)

ks = [2, 5, 10, 20, 30, 40, 64]
accs = []

print("\nComponents vs accuracy")
for k in ks:
    p = PCA(n_components=k)
    accs.append(evaluate(
        p.fit_transform(X_tr_s), p.transform(X_te_s),
        f"PCA ({k} components)"
    ))


# ================= CELL 5: PCA Visualization =================

pca2 = PCA(n_components=2).fit(X_tr_s)
Z = pca2.transform(sc.transform(X))

print("Variance explained by PC1 and PC2:",
      np.round(pca2.explained_variance_ratio_, 4))
print("Total variance explained:",
      round(pca2.explained_variance_ratio_.sum(), 4))

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

s = ax[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=10)
ax[0].set_title("Digits: 2 Principal Components")
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
fig.colorbar(s, ax=ax[0], label="Digit")

cum = np.cumsum(PCA().fit(X_tr_s).explained_variance_ratio_)
ax[1].plot(range(1, len(cum)+1), cum, marker=".")
ax[1].axhline(0.95, color="red", linestyle="--", label="95% variance")
ax[1].set_title("Cumulative Explained Variance")
ax[1].set_xlabel("Number of Components")
ax[1].set_ylabel("Variance Explained")
ax[1].legend()

ax[2].plot(ks, accs, marker="o")
ax[2].axhline(acc_full, color="green", linestyle="--", label="Without PCA")
ax[2].set_title("Accuracy vs Components")
ax[2].set_xlabel("Number of Components")
ax[2].set_ylabel("Accuracy")
ax[2].legend()

plt.tight_layout()
plt.savefig("pr8_pca_plots.png", dpi=120)
plt.show()
print("Saved plots to pr8_pca_plots.png")


# ================= CELL 6: Summary and Conclusion =================

print("\n========== SUMMARY ==========")
print("Original number of features:", X.shape[1])
print("Features retained at 95% variance:", pca95.n_components_)

reduction = 100 * (1 - pca95.n_components_ / X.shape[1])
print(f"Feature reduction: {reduction:.1f}%")
print(f"Accuracy without PCA: {acc_full:.4f}")
print(f"Accuracy with PCA (95% variance): {acc_95:.4f}")

print("\nConclusion:")
print("PCA reduces the number of features while preserving")
print("most of the information in the dataset.")
print("Classification accuracy may increase or decrease")
print("depending on the number of components retained.")
