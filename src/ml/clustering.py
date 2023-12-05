import pandas as pd
import wandb
import src.utils.utilities as utils
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score



class Clustering():
    def __init__(self, args):
        self.save_dir = os.path.join(os.getcwd(), "output", "images", "Clustering")
        self.train = pd.read_csv(args.data_train)
        self.test = pd.read_csv(args.data_test)
        #self.perform_kmeans()
        self.perform_dbscan()

    def perform_kmeans(self):
        k = self.compute_elbow()

        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(self.train)

        # Determine the cluster labels of new_points: labels
        labels = model.predict(self.train)

        # Add the cluster labels to your DataFrame
        self.train['Cluster'] = labels

        # Get top 2 principal components
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.train)

        # Visualize the clusters
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.8)

        # Save
        kmeans_output = os.path.join(self.save_dir, "kmeans", "kmeans.png")
        os.makedirs(os.path.dirname(kmeans_output), exist_ok=True)
        plt.savefig(kmeans_output)
        plt.clf()
        plt.close()


    def evaluate_clustering(self, dataset):
        """ Evaluate the clustering using mutual information score """

        # get the labels
        labels = dataset["account_type"]

        # get the cluster labels
        cluster_labels = dataset["Cluster"]

        # calculate the mutual information score
        mutual_info = mutual_info_score(labels, cluster_labels)

        # log to wandb
        print(f"Mutual information score: {mutual_info}")


    def perform_dbscan(self):
        # TODO: balance human/bot labels
        numerical_columns = utils.get_numeric_columns()
        dataset = self.train[numerical_columns]

        # Apply DBSCAN
        self.determine_dbscan_eps(dataset)
        ms = self.get_optimal_minsamples_dbscan(dataset)
        dbscan = DBSCAN(eps=0.25, min_samples=ms)
        clusters = dbscan.fit_predict(dataset)

        # Add the cluster labels to your DataFrame
        dataset['Cluster'] = clusters

        self.visualize(clusters, dataset)

        # change clusters -1 to 1
        clusters[clusters == 0] = 1
        clusters[clusters == -1] = 0

        # plot confusion matrix between clusters and labels
        sns.heatmap(pd.crosstab(clusters, self.train['account_type']), annot=True, fmt='d')
        plt.xlabel('Bot/Human')
        plt.ylabel('Cluster')
        plt.title('Confusion Matrix')
        plt.show()
        plt.savefig(os.path.join(self.save_dir, "DBSCAN", "confusion_matrix.png"))

        # add clusters into self.train
        self.train['Cluster'] = clusters
        self.evaluate_clustering(self.train)

        return dataset

    def determine_dbscan_eps(self, dataset):
        """ Use K-Nearest Neighbors to determine the optimal eps value for DBSCAN
         
            https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf 
            https://www.khoury.northeastern.edu/home/vip/teach/DMcourse/2_cluster_EM_mixt/notes_slides/revisitofrevisitDBSCAN.pdf
        """
        k_neighbours = self.get_optimal_minsamples_dbscan(dataset) # same as minsamples
        neighbors = NearestNeighbors(n_neighbors=k_neighbours)
        neighbors_fit = neighbors.fit(dataset)
        distances, indices = neighbors_fit.kneighbors(dataset)

        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        
        # retrieve the best eps
        knee_point = self.find_knee_point(distances)
        best_eps = distances[knee_point]

        # plot distances with marked knee point
        plt.plot(distances)
        plt.plot(knee_point, best_eps, 'ro')
        plt.annotate('knee point', xy=(knee_point, best_eps), xytext=(knee_point, best_eps + 0.1))
        plt.xlabel('Data point index')
        plt.ylabel('Distance')

        # save
        output_dir = os.path.join(self.save_dir, "DBSCAN")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "distances.png"))


        return best_eps

    def find_knee_point(self, distances):
        """Find the knee point in the plot of sorted distances.
        As per: https://raghavan.usc.edu//papers/kneedle-simplex11.pdf
        TODO Confirm this algorithm with the paper!
        """
        n_points = len(distances)
        x = np.arange(n_points)
        y = distances

        # Calculate the first derivative (slope)
        dy = np.diff(y) / np.diff(x)

        # Use the second derivative to find the knee point
        ddy = np.diff(dy) / np.diff(x[:-1])

        # Find the index of the knee point
        knee_point_index = np.argmax(dy)

        return knee_point_index
        

    def get_optimal_minsamples_dbscan(self, dataset):
        """ Return twice the number of features in the dataset
        As per https://link.springer.com/article/10.1023/A:1009745219419 
        """
        return dataset.shape[1] * 2



    def visualize(self, clusters, dataset):
        # extract t-SNE 1 and 2
        tsne = TSNE(n_components=3)
        tsne_components = tsne.fit_transform(dataset)

        # Visualize the clusters
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(tsne_components[:, 0], tsne_components[:, 1], tsne_components[:, 2], c=clusters, cmap='viridis', marker='o', s=50, alpha=0.8)
        ax.set_title('DBSCAN Clustering')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
        plt.show()

        # save to output/images/Clustering/DBSCAN.png
        output_dir = os.path.join(self.save_dir, "DBSCAN")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "DBSCAN.png"))

        # With PCA plotting: X-Y axis is the top two principal components

        # Take clusters and project them using either:
            # U-Map
            # T-SNE

        # This gives X-clusters
        # For each datapoint assign the Bot/Human label
            # Ideally looks seperated
            # If there's many, then there's different types of humans/bots

        # Additionally:
            # Mutual label scores
            # Mutual information score (for evaluating clustering (train/val splits))
            

    def compute_elbow(self):
        """ Make an elbow plot """

        inertias = []
        for i in range(1, 10):
            # Create a KMeans instance with k clusters: model
            model = KMeans(n_clusters=i)

            # Fit model to samples
            model.fit(self.train)

            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)

        # scale inertias
        inertias = [i / inertias[0] for i in inertias]

        # Get the best k
        best_k = 0
        for i in range(1, len(inertias)):
            if abs(inertias[i] - inertias[i-1]) < 0.1:
                best_k = i
                break

        # Plot ks vs inertias
        plt.plot(range(1, 10), inertias, '-o')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(range(1, 10))

        # Save
        kmeans_output = os.path.join(self.save_dir, "kmeans", "elbow.png")
        os.makedirs(os.path.dirname(kmeans_output), exist_ok=True)
        plt.savefig(kmeans_output)
        plt.clf()
        plt.close()

        return best_k