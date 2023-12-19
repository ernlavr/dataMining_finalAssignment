import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class Clustering:
    def __init__(self, args):
        self.save_dir = os.path.join(os.getcwd(), "output", "images", "Clustering")
        self.train = pd.read_csv(args.data_parsed)
        self.random_state = 42
        self.reduced_train: pd.DataFrame

    def __call__(self) -> None:
        self.reduced_train = self.pca_dimensionality_reduction(self.train, 3)
        self.perform_kmeans(self.reduced_train)
        self.perform_dbscan(self.reduced_train)

    def pca_dimensionality_reduction(self, dataset, components):
        """Reduce the dimensionality of the dataset using PCA"""
        # remove "account_type" column
        if "account_type" in dataset.columns.values:
            dataset = dataset.drop(columns=["account_type"])

        pca = PCA(n_components=components)
        pca_components = pca.fit_transform(dataset)
        cols = [f"PC_{i}" for i in range(components)]
        output = pd.DataFrame(data=pca_components, columns=cols)
        output["account_type"] = self.train["account_type"]

        self.make_pair_plot(output)
        # Plot the principal components
        fig = plt.figure()
        if components == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                output["PC_0"].values,
                output["PC_1"].values,
                output["PC_2"].values,
                alpha=0.25,
                c=output["account_type"].values,
            )
        else:
            ax = fig.add_subplot(111)
            ax.scatter(output["PC_0"].values, output["PC_1"].values, alpha=0.25)
        plt.xlabel("PC_1")
        plt.ylabel("PC_2")
        ax.set_zlabel("PC_3")
        plt.title("Data after PCA Transformation")
        plt.show()
        # plt.savefig(os.path.join(self.save_dir, "dimensionality", "PCA.png"))
        plt.clf()
        plt.close()

        return output

    def perform_kmeans(self, dataset):
        # drop account_type
        dataset = dataset.drop(columns=["account_type"])
        self._compute_elbow()

        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=8, random_state=self.random_state)

        # Determine the cluster labels of new_points: labels
        labels = model.fit_predict(dataset)
        self.train["Cluster"] = labels

        dataset["Cluster"] = labels

        # Add the cluster labels to your DataFrame
        self._visualize(labels, dataset, name="kmeans")
        self._evaluate_clustering(self.train)
        self._plot_confusion(labels, self.train, "kmeans")
    
    def perform_dbscan(self, dataset):
        # Drop target to prevent cheating
        dataset = dataset.drop(columns=["account_type"])

        # Determine hyperparameters
        self._determine_dbscan_eps(dataset)
        ms = self._get_optimal_minsamples_dbscan(dataset)

        # Cluster
        dbscan = DBSCAN(eps=0.02, min_samples=ms)
        clusters = dbscan.fit_predict(dataset)

        # Visualize
        dataset["Cluster"] = clusters
        self._visualize(clusters, dataset, "DBSCAN")
        sns.heatmap(
            pd.crosstab(clusters, self.train["account_type"]), annot=True, fmt="d"
        )
        plt.xlabel("Labels")
        plt.ylabel("Cluster")
        plt.show()
        plt.savefig(os.path.join(self.save_dir, "DBSCAN", "confusion_matrix.png"))
        plt.clf()
        plt.close()

        # add clusters into self.train
        self.train["Cluster"] = clusters
        self._evaluate_clustering(self.train)

        return dataset

    def make_pair_plot(self, pca_results):
        """Make a pair plot of the PCA results"""
        # Plot the pairwise relationships between the principal components colored by the labels
        sns.pairplot(pca_results, hue="account_type")
        sns.pairplot(pca_results, hue="account_type")
        save_dir = os.path.join(os.getcwd(), "output", "images", "dimensionality")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "pairplot.png"))
        plt.clf()
        plt.close()

    def _evaluate_clustering(self, dataset):
        """Evaluate the clustering using mutual information score"""
        # get the labels
        labels = self.train["account_type"]

        # get the cluster labels
        cluster_labels = dataset["Cluster"]

        # calculate the mutual information score
        mutual_info = metrics.adjusted_mutual_info_score(labels, cluster_labels)
        homogeneity_score = metrics.homogeneity_score(labels, cluster_labels)
        completeness_score = metrics.completeness_score(labels, cluster_labels)
        silhouette_score = metrics.silhouette_score(dataset, dataset["Cluster"])
        pair_confusion_matrix = metrics.cluster.pair_confusion_matrix(
            labels, cluster_labels
        )  # TODO: pair_confusion_matrix unused variable

        # log to wandb
        print(f"Adj. Mutual information score: {round(mutual_info, 3)}")
        print(f"Homogeneity score: {round(homogeneity_score, 3)}")
        print(f"Completeness score: {round(completeness_score, 3)}")
        print(f"Silhouette score: {round(silhouette_score, 3)}")

    def _collapse_to_two_clusters(self, labels):
        num_clusters = len(set(labels))

        middle = (num_clusters - 1) // 2
        mapping = {label: 0 if label < middle else 1 for label in set(labels)}

        collapsed_labels = [mapping[label] for label in labels]
        return collapsed_labels

    def _plot_confusion(self, clusters, dataset, name):
        # plot confusion matrix between clusters and labels

        sns.heatmap(pd.crosstab(clusters, dataset["account_type"]), annot=True, fmt="d")
        plt.xlabel("Labels")
        plt.ylabel("Cluster")
        plt.title("Confusion Matrix")
        plt.show()
        plt.savefig(os.path.join(self.save_dir, name, f"{name}_conf_matrix.png"))
        plt.clf()
        plt.close()

    def _determine_dbscan_eps(self, dataset):
        """Use K-Nearest Neighbors to determine the optimal eps value for DBSCAN
        As per:
        https://www.khoury.northeastern.edu/home/vip/teach/DMcourse/2_cluster_EM_mixt/notes_slides/revisitofrevisitDBSCAN.pdf
        """
        k_neighbours = self._get_optimal_minsamples_dbscan(dataset)  # same as minsamples
        neighbors = NearestNeighbors(n_neighbors=k_neighbours)
        neighbors_fit = neighbors.fit(dataset)
        distances, _ = neighbors_fit.kneighbors(dataset)

        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # retrieve the best eps
        knee_point = self._find_knee_point(distances)
        best_eps = distances[knee_point]

        # plot distances with marked knee point
        plt.plot(distances)
        plt.plot(1980, 0.02, "ro", alpha=0.33)  # hard-coded to match final plot
        plt.annotate("knee point", xy=(1980, 0.02), xytext=(1990, 0.02 + 0.01))
        plt.xlabel("Data point index")
        plt.ylabel("Distance")

        # save
        output_dir = os.path.join(self.save_dir, "DBSCAN")
        os.makedirs(output_dir, exist_ok=True)
        plt.show()
        plt.savefig(os.path.join(output_dir, "distances.png"))
        plt.clf()
        plt.close()

        return best_eps

    def _find_knee_point(self, distances):
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
        ddy = np.diff(dy) / np.diff(x[:-1])  # TODO: ddy unused variable

        # Find the index of the knee point
        knee_point_index = np.argmax(dy)

        return knee_point_index

    def _get_optimal_minsamples_dbscan(self, dataset):
        """Return twice the number of features in the dataset"""
        beta = 10  # empirical value
        return dataset.shape[1] * 2 + beta

    def _visualize(self, clusters, dataset, name, eps=None):
        # Visualize the clusters
        fig = plt.figure()
        ax = fig.add_subplot()
        scatter = ax.scatter(
            dataset["PC_0"].values,
            dataset["PC_1"].values,
            c=clusters,
            cmap="viridis",
            marker="o",
            s=50,
            alpha=0.8,
        )
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        ax.set_title(f"{name} Clustering; {eps}")
        ax.set_xlabel("PC_1")
        ax.set_ylabel("PC_2")

        # save to output/images/Clustering/DBSCAN.png
        output_dir = os.path.join(self.save_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        plt.show()
        plt.savefig(os.path.join(output_dir, f"{name}.png"))
        plt.clf()
        plt.close()

    def _compute_elbow(self):
        """Make an elbow plot"""
        inertias = []
        for i in range(1, 20):
            # Create a KMeans instance with k clusters: model
            model = KMeans(n_clusters=i, random_state=self.random_state)

            # Fit model to samples
            model.fit(self.train)

            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)

        # scale inertias
        inertias = [i / inertias[0] for i in inertias]

        # Get the best k
        best_k = 0
        for i in range(1, len(inertias)):
            if abs(inertias[i] - inertias[i - 1]) < 0.1:
                best_k = i
                break

        # Plot ks vs inertias
        plt.plot(range(1, 20), inertias, "-o")
        plt.xlabel("number of clusters, k")
        plt.ylabel("Inter-cluster variance")
        plt.xticks(range(1, 20))

        # Save
        kmeans_output = os.path.join(self.save_dir, "kmeans", "elbow.png")
        os.makedirs(os.path.dirname(kmeans_output), exist_ok=True)
        plt.savefig(kmeans_output)
        plt.clf()
        plt.close()

        return best_k
