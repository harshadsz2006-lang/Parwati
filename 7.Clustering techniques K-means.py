
# ================= CELL 7: Import Libraries =================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# ================= CELL 8: Load Dataset =================

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# ================= CELL 9: Elbow Method =================

wcss = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)
    wcss.append(model.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method_short.png")
plt.show()


# ================= CELL 10: K-Means Clustering =================

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(X)

print("K-Means Silhouette Score:", round(silhouette_score(X, df["kmeans_cluster"]), 3))
print("\nMean feature values by K-Means cluster:")
print(df.groupby("kmeans_cluster")[iris.feature_names].mean().round(2))

print("\nSpecies vs K-Means clusters:")
print(pd.crosstab(df["species"], df["kmeans_cluster"]))


# ================= CELL 11: Hierarchical Clustering =================

linked = linkage(X, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(linked, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("dendrogram_short.png")
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=3, linkage="ward")
df["hier_cluster"] = hierarchical.fit_predict(X)

print("Hierarchical Silhouette Score:", round(silhouette_score(X, df["hier_cluster"]), 3))


# ================= CELL 12: Comparison and Conclusion =================

print("Species vs K-Means clusters:")
print(pd.crosstab(df["species"], df["kmeans_cluster"]))

print("\nSpecies vs Hierarchical clusters:")
print(pd.crosstab(df["species"], df["hier_cluster"]))

print("\nCluster assignments:")
print(df[["species", "kmeans_cluster", "hier_cluster"]].head(10))

print("\nConclusion:")
print("Both K-Means and Hierarchical Clustering group similar Iris samples.")
print("The Iris dataset contains three actual species.")
print("Setosa is generally separated clearly, while Versicolor and")
print("Virginica may overlap.")
print("Silhouette scores help evaluate cluster separation.")
