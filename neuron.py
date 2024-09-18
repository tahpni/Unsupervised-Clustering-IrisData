import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import timeit as timer

start = timer.default_timer()

df = pd.read_csv('iris copy.csv')
dataf = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
scaler = StandardScaler()
dataf_scaled = scaler.fit_transform(dataf)

kmeans = KMeans(n_clusters=3, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(dataf_scaled)

df['species_num'] = df['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

def map_clusters_to_species(clusters, species):
    labels = {}
    for cluster in set(clusters):
        mask = clusters == cluster
        most_common_species = mode(species[mask])[0]
        labels[cluster] = most_common_species
    return labels

cluster_to_species = map_clusters_to_species(df['kmeans_cluster'].values, df['species_num'].values)
df['predicted_species'] = df['kmeans_cluster'].map(cluster_to_species)

accuracy = accuracy_score(df['species_num'], df['predicted_species'])
print(f'Clustering accuracy: {accuracy * 100:.1f}%')

# K Cluster Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['petal.length'], df['petal.width'], c=df['kmeans_cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('KMeans Clustering on Iris Data')
plt.colorbar(label='Predicted Cluster').solids.set_edgecolor("face")

# Real Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['petal.length'], df['petal.width'], c=df['species_num'], cmap='viridis', alpha=0.6)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Actual Species on Iris Data')
plt.colorbar(label='Actual Species').solids.set_edgecolor("face")
stop = timer.default_timer()
print(f'Time: {(stop - start):.1f} seconds')
plt.show()