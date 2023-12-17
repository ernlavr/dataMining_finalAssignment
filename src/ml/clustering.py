import pandas as pd
import wandb
import src.utils.utilities as utils
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.cluster import DBSCAN, KMeans, OPTICS, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import sklearn.metrics as metrics



class Clustering():
    def __init__(self, args):
        self.save_dir = os.path.join(os.getcwd(), "output", "images", "Clustering")
        self.train = pd.read_csv(args.data_parsed)

        # self.reduced_train = self.tsne_dimensionality_reduction(self.train, 3)
        self.reduced_train = self.pcaDimensionalityReduction(self.train, 3, make_plots=True)
        
        self.perform_kmeans(self.reduced_train)
        self.perform_dbscan(self.reduced_train)

    def perform_kmeans(self, dataset):
        # self.compute_elbow()

        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=8, random_state=1)

        # Fit model to samples
        model.fit(dataset)

        # Determine the cluster labels of new_points: labels
        labels = model.predict(dataset)
        self.train['Cluster'] = labels

        dataset = self.pcaDimensionalityReduction(dataset, 2)
        dataset['Cluster'] = labels

        # Add the cluster labels to your DataFrame
        self.visualize(labels, dataset, name="kmeans")
        self.evaluate_clustering(self.train)
        self.plot_confusion(labels, self.train, "kmeans")


    def evaluate_clustering(self, dataset):
        """ Evaluate the clustering using mutual information score """
        # get the labels
        labels = dataset["account_type"]

        # get the cluster labels
        cluster_labels = dataset["Cluster"]
        

        # calculate the mutual information score
        mutual_info = metrics.adjusted_mutual_info_score(labels, cluster_labels)
        adjusted_rand = metrics.adjusted_rand_score(labels, cluster_labels)
        homogeneity_score = metrics.homogeneity_score(labels, cluster_labels)
        completeness_score = metrics.completeness_score(labels, cluster_labels)
        silhouette_score = metrics.silhouette_score(dataset, dataset["Cluster"])
        pair_confusion_matrix = metrics.cluster.pair_confusion_matrix(labels, cluster_labels)

        # log to wandb
        print(f"Mutual information score: {mutual_info}")
        print(f"Adjusted rand score: {adjusted_rand}")
        print(f"Homogeneity score: {homogeneity_score}")
        print(f"Completeness score: {completeness_score}")
        print(f"Silhouette score: {silhouette_score}")


    def collapse_to_two_clusters(self, labels):
        num_clusters = len(set(labels))

        middle = (num_clusters - 1) // 2
        mapping = {label: 0 if label < middle else 1 for label in set(labels)}

        collapsed_labels = [mapping[label] for label in labels]
        return collapsed_labels

    def perform_dbscan(self, dataset):
        # TODO: balance human/bot labels
        # numerical_columns = utils.get_numeric_columns()
        # dataset = dataset[numerical_columns]

        # Apply DBSCAN
        # remove 'account_type'
        self.determine_dbscan_eps(dataset)
        ms = self.get_optimal_minsamples_dbscan(dataset)
        dbscan = DBSCAN(eps=0.035, min_samples=ms)
        clusters = dbscan.fit_predict(dataset)

        dataset['Cluster'] = clusters
        self.visualize(clusters, dataset, 'DBSCAN')

        # plot confusion matrix between clusters and labels
        sns.heatmap(pd.crosstab(clusters, self.train['account_type']), annot=True, fmt='d')
        plt.xlabel('Bot/Human')
        plt.ylabel('Cluster')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_dir, "DBSCAN", "confusion_matrix.png"))
        plt.clf()
        plt.close()

        # add clusters into self.train
        self.train['Cluster'] = clusters
        self.evaluate_clustering(self.train)

        self.pcaDimensionalityReduction(dataset, 3)

        return dataset
    
    def plot_confusion(self, clusters, dataset, name):
        # plot confusion matrix between clusters and labels
        sns.heatmap(pd.crosstab(clusters, dataset['account_type']), annot=True, fmt='d')
        plt.xlabel('Bot/Human')
        plt.ylabel('Cluster')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_dir, name, f"{name}_conf_matrix.png"))
        plt.clf()
        plt.close()

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
        plt.show()
        plt.savefig(os.path.join(output_dir, "distances.png"))
        plt.clf()
        plt.close()


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
    
    def make_pair_plot(self, pca_results):
        """ Make a pair plot of the PCA results """
        # Plot the pairwise relationships between the principal components colored by the labels
        sns.pairplot(pca_results, hue='account_type')
        plt.savefig(os.path.join(os.getcwd(), "output", "images", 'dimensionality', "pairplot.png"))
        plt.clf()
        plt.close()
        
    def pcaDimensionalityReduction(self, dataset, components, make_plots=False):
        """ Reduce the dimensionality of the dataset using PCA """
        # remove "account_type" column
        if "account_type" in dataset.columns.values:
            dataset = dataset.drop(columns=["account_type"])

        pca = PCA(n_components=components)
        pca_components = pca.fit_transform(dataset)
        
        cols = [f'Dim_{i}' for i in range(components)]
        output = pd.DataFrame(data=pca_components, columns=cols)
        output['account_type'] = self.train['account_type']


        if make_plots:
            self.make_pair_plot(output)
            # Plot the principal components
            fig = plt.figure()
            if components == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(output['Dim_0'].values, output['Dim_1'].values, output['Dim_2'].values, alpha=0.25, c=output['account_type'].values)
            else:
                ax = fig.add_subplot(111)
                ax.scatter(output['Dim_0'].values, output['Dim_1'].values, alpha=0.25)
            plt.xlabel('PC_1')
            plt.ylabel('PC_2')
            ax.set_zlabel('PC_3')
            plt.title('Data after PCA Transformation')
            plt.show()
            # plt.savefig(os.path.join(self.save_dir, "dimensionality", "PCA.png"))
            plt.clf()
            plt.close()

        return output

    def tsne_dimensionality_reduction(self, dataset, components):
        """ Reduce the dimensionality of the dataset using t-SNE """
        # remove "account_type" column
        if "account_type" in dataset.columns.values:
            dataset = dataset.drop(columns=["account_type"])

        for perplexity in [80]:    
            tsne = TSNE(n_components=components, perplexity=perplexity, n_iter=5000)
            tsne_components = tsne.fit_transform(dataset)
            
            cols = [f'Dim_{i}' for i in range(components)]
            output = pd.DataFrame(data=tsne_components, columns=cols)
            output['account_type'] = self.train['account_type'] # add labels back in

            self.make_pair_plot(output)

            # Plot the principal components
            fig = plt.figure()
            if components == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(output['Dim_0'].values, output['Dim_1'].values, output['Dim_2'].values, alpha=0.25)
            else:
                ax = fig.add_subplot(111)
                ax.scatter(output['Dim_0'].values, output['Dim_1'].values, alpha=0.25)
            plt.xlabel('PC_1')
            plt.ylabel('PC_2')
            plt.title('Data after t-SNE Transformation')
            plt.show()
            plt.clf()
            plt.close()

        return output

    def visualize(self, clusters, dataset, name, eps=None):
        # Visualize the clusters
        fig = plt.figure()
        ax = fig.add_subplot()
        scatter = ax.scatter(dataset['Dim_0'].values, dataset['Dim_1'].values, c=clusters, cmap='viridis', marker='o', s=50, alpha=0.8)
        legend = ax.legend(*scatter.legend_elements(), title='Clusters')
        ax.add_artist(legend)
        ax.set_title(f'{name} Clustering; {eps}')
        ax.set_xlabel('PC_1')
        ax.set_ylabel('PC_2')

        # save to output/images/Clustering/DBSCAN.png
        output_dir = os.path.join(self.save_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        plt.show()
        plt.savefig(os.path.join(output_dir, f"{name}.png"))
        plt.clf()
        plt.close()

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
        for i in range(1, 20):
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
        plt.plot(range(1, 20), inertias, '-o')
        plt.xlabel('number of clusters, k')
        plt.ylabel('Inter-cluster variance')
        plt.xticks(range(1, 20))

        # Save
        kmeans_output = os.path.join(self.save_dir, "kmeans", "elbow.png")
        os.makedirs(os.path.dirname(kmeans_output), exist_ok=True)
        plt.savefig(kmeans_output)
        plt.clf()
        plt.close()

        return best_k