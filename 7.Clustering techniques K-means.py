df["species"] = iris.target_names[iris.target]

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Elbow method
ks = range(1, 11)
wcss = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]

plt.figure(figsize=(7, 4))
plt.plot(ks, wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.xticks(ks)
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method_short.png")
plt.show()

# K-Means clustering
df["kmeans_cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)

print("K-Means Silhouette Score:", round(silhouette_score(X, df["kmeans_cluster"]), 3))
print("\nMean feature values by K-Means cluster:")
print(df.groupby("kmeans_cluster")[iris.feature_names].mean().round(2))

# Hierarchical clustering
plt.figure(figsize=(10, 5))
dendrogram(linkage(X, method="ward"), no_labels=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("dendrogram_short.png")
plt.show()

df["hier_cluster"] = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
print("\nHierarchical Silhouette Score:", round(silhouette_score(X, df["hier_cluster"]), 3))

# Comparison with the true species
for label, col in [("K-Means", "kmeans_cluster"), ("Hierarchical", "hier_cluster")]:
    print(f"\nSpecies vs {label} clusters:")
    print(pd.crosstab(df["species"], df[col]))

print("\nCluster assignments:")
print(df[["species", "kmeans_cluster", "hier_cluster"]].head(10))

print("""
Conclusion:
Both K-Means and Hierarchical Clustering group similar Iris samples.
The Iris dataset contains three actual species.
Setosa is generally separated clearly, while Versicolor and
Virginica may overlap. Silhouette scores help evaluate cluster separation.""")
