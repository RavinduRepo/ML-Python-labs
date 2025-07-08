from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


iris = load_iris()
iris_X = iris.data  # features only, unlabeled dataset

# plotting
wcss_iris = [] # Within Cluster Sum of Squares for iris dataset
# Using the Elbow Method to find the optimal number of clusters for the Iris dataset
for k in range(1, 11):
    # kmeans with k clusters
    kmeans_iris = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=0)
    # fit the model (train the model)
    kmeans_iris.fit(iris_X)
    # append the inertia (WCSS) to the list (Inertia is the sum of squared distances of samples to their closest cluster center)
    wcss_iris.append(kmeans_iris.inertia_)

plt.plot(range(1, 11), wcss_iris)
plt.title("Elbow Method for Iris Dataset")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

#find k
# Fit KMeans with the optimal k (3) found from the Elbow method for the iris dataset
kmeans_iris = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=0)
iris_clusters = kmeans_iris.fit_predict(iris_X)

# Explain output
# The output of `kmeans.cluster_centers_` is a NumPy array 
# containing the coordinates of the centroids (centers) of each 
# cluster found by the K-Means algorithm. Each row in this array
# represents the center of a cluster in the feature space, and 
# each column corresponds to a feature. These centroids are the 
# mean positions of all the points assigned to each cluster and 
# are used to define the clusters in the data.

# visualize the clusters
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot data points, colored by cluster assignment
ax.scatter(
    iris_X[:, 0], iris_X[:, 1], iris_X[:, 2],
    c=iris_clusters, cmap='viridis', s=40, alpha=0.7
)

# Plot cluster centers
centers = kmeans_iris.cluster_centers_
ax.scatter(
    centers[:, 0], centers[:, 1], centers[:, 2],
    c='red', s=200, marker='*', label='Centers'
)

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_zlabel('Petal length (cm)')
ax.set_title('Iris Clusters (3D)')
ax.legend()
plt.show()