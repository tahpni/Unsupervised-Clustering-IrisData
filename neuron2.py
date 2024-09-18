import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
import timeit as timer

# Start the timer
start = timer.default_timer()

# Load only necessary columns
df = pd.read_csv(
    'iris copy.csv',
    usecols=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
)

# Map species to numerical values
species_map = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
df['species_num'] = df['variety'].map(species_map).values

# Extract features as a NumPy array
features = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Dimensionality Reduction using PCA
pca = PCA(n_components=2, random_state=42)
features_pca = pca.fit_transform(features_scaled)

# Function to map clusters to species using NumPy
def map_clusters(clusters, species):
    cluster_labels = np.zeros_like(clusters)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        cluster_labels[mask] = np.bincount(species[mask]).argmax()
    return cluster_labels

# KMeans Clustering
kmeans = KMeans(n_clusters=3, n_init=50, random_state=42)
kmeans_clusters = kmeans.fit_predict(features_pca)
kmeans_predicted_species = map_clusters(kmeans_clusters, df['species_num'].values)

# Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=3, n_init=50, random_state=42)
gmm_clusters = gmm.fit_predict(features_pca)
gmm_predicted_species = map_clusters(gmm_clusters, df['species_num'].values)

# Evaluation Metrics Function
def evaluate_clustering(true_labels, predicted_labels, features):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    silhouette = silhouette_score(features, predicted_labels)
    return ari, nmi, silhouette

# Evaluate KMeans
kmeans_ari, kmeans_nmi, kmeans_silhouette = evaluate_clustering(
    df['species_num'].values, kmeans_clusters, features_pca
)

# Evaluate GMM
gmm_ari, gmm_nmi, gmm_silhouette = evaluate_clustering(
    df['species_num'].values, gmm_clusters, features_pca
)

# Print Evaluation Results
print("KMeans Clustering Evaluation:")
print(f"Adjusted Rand Index (ARI): {kmeans_ari * 100:.2f}%")
print(f"Normalized Mutual Information (NMI): {kmeans_nmi * 100:.2f}%")
print(f"Silhouette Score: {kmeans_silhouette:.4f}\n")

print("Gaussian Mixture Model Clustering Evaluation:")
print(f"Adjusted Rand Index (ARI): {gmm_ari * 100:.2f}%")
print(f"Normalized Mutual Information (NMI): {gmm_nmi * 100:.2f}%")
print(f"Silhouette Score: {gmm_silhouette:.4f}\n")

# Choose the better model based on ARI
if kmeans_ari > gmm_ari:
    best_clusters = kmeans_clusters
    best_predicted_species = kmeans_predicted_species
    best_model = 'KMeans'
else:
    best_clusters = gmm_clusters
    best_predicted_species = gmm_predicted_species
    best_model = 'Gaussian Mixture Model'

print(f"Selected Best Model: {best_model}")

# Add clustering results to the DataFrame
df['cluster'] = best_clusters
df['predicted_species'] = best_predicted_species

# Plotting in subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# Original Data with True Labels
scatter0 = axes[0].scatter(
    features_pca[:, 0],
    features_pca[:, 1],
    c=df['species_num'],
    cmap='viridis',
    alpha=0.6
)
axes[0].set_xlabel('PCA Component 1')
axes[0].set_ylabel('PCA Component 2')
axes[0].set_title('Actual Species')
cbar0 = plt.colorbar(scatter0, ax=axes[0], label='Species')
cbar0.solids.set_edgecolor("face")

# KMeans Clusters
scatter1 = axes[1].scatter(
    features_pca[:, 0],
    features_pca[:, 1],
    c=kmeans_clusters,
    cmap='viridis',
    alpha=0.6
)
axes[1].set_xlabel('PCA Component 1')
axes[1].set_ylabel('PCA Component 2')
axes[1].set_title('KMeans Clustering')
cbar1 = plt.colorbar(scatter1, ax=axes[1], label='KMeans Cluster')
cbar1.solids.set_edgecolor("face")

# GMM Clusters
scatter2 = axes[2].scatter(
    features_pca[:, 0],
    features_pca[:, 1],
    c=gmm_clusters,
    cmap='viridis',
    alpha=0.6
)
axes[2].set_xlabel('PCA Component 1')
axes[2].set_ylabel('PCA Component 2')
axes[2].set_title('Gaussian Mixture Model Clustering')
cbar2 = plt.colorbar(scatter2, ax=axes[2], label='GMM Cluster')
cbar2.solids.set_edgecolor("face")

# Adjust layout
plt.tight_layout()

# Stop the timer and print execution time
stop = timer.default_timer()
print(f"Execution Time: {(stop - start):.2f} seconds")

accuracy = accuracy_score(df['species_num'], df['predicted_species'])
print(f'Clustering accuracy: {accuracy * 100:.1f}%')


# Display plots
plt.show()