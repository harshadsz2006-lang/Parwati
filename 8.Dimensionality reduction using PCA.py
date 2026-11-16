import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load, split, and scale the data
X, y = load_digits(return_X_y=True)
print("Original shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
print("Training samples:", len(X_tr_s), "| Testing samples:", len(X_te_s))


def evaluate(Xa, Xb, label):
    """Fit logistic regression on Xa, report test accuracy on Xb."""
    clf = LogisticRegression(max_iter=2000)
    t = time.time()
    clf.fit(Xa, y_tr)
    train_time = time.time() - t
    acc = accuracy_score(y_te, clf.predict(Xb))
    print(f"{label:<25} features={Xa.shape[1]:<3} accuracy={acc:.4f} train_time={train_time:.4f}s")
    return acc


# PCA vs no PCA
acc_full = evaluate(X_tr_s, X_te_s, "Without PCA")

pca95 = PCA(n_components=0.95)
X95 = pca95.fit_transform(X_tr_s)
acc_95 = evaluate(X95, pca95.transform(X_te_s), "PCA (95% variance)")
print("\nComponents kept for 95% variance:", pca95.n_components_)

print("\nComponents vs accuracy")
ks = [2, 5, 10, 20, 30, 40, 64]
accs = []
for k in ks:
    p = PCA(n_components=k)
    accs.append(evaluate(p.fit_transform(X_tr_s), p.transform(X_te_s), f"PCA ({k} components)"))

# Visualization
pca_all = PCA().fit(X_tr_s)
Z = PCA(n_components=2).fit(X_tr_s).transform(sc.transform(X))
ratio = pca_all.explained_variance_ratio_
print("Variance explained by PC1 and PC2:", np.round(ratio[:2], 4))
print("Total variance explained:", round(ratio[:2].sum(), 4))

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

s = ax[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=10)
ax[0].set(title="Digits: 2 Principal Components", xlabel="PC1", ylabel="PC2")
fig.colorbar(s, ax=ax[0], label="Digit")

cum = np.cumsum(ratio)
ax[1].plot(range(1, len(cum) + 1), cum, marker=".")
ax[1].axhline(0.95, color="red", linestyle="--", label="95% variance")
ax[1].set(title="Cumulative Explained Variance", xlabel="Number of Components",
          ylabel="Variance Explained")
ax[1].legend()

ax[2].plot(ks, accs, marker="o")
ax[2].axhline(acc_full, color="green", linestyle="--", label="Without PCA")
ax[2].set(title="Accuracy vs Components", xlabel="Number of Components", ylabel="Accuracy")
ax[2].legend()

plt.tight_layout()
plt.savefig("pr8_pca_plots.png", dpi=120)
plt.show()
print("Saved plots to pr8_pca_plots.png")

# Summary
n_feat = X.shape[1]
print("\n========== SUMMARY ==========")
print("Original number of features:", n_feat)
print("Features retained at 95% variance:", pca95.n_components_)
print(f"Feature reduction: {100 * (1 - pca95.n_components_ / n_feat):.1f}%")
print(f"Accuracy without PCA: {acc_full:.4f}")
print(f"Accuracy with PCA (95% variance): {acc_95:.4f}")
print("""
Conclusion:
PCA reduces the number of features while preserving
most of the information in the dataset.
Classification accuracy may increase or decrease
depending on the number of components retained.""")
