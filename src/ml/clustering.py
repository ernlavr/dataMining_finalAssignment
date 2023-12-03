import pycaret.clustering as clust
import wandb
import src.utils.utilities as utils
import seaborn as sns
import matplotlib.pyplot as plt

class Clustering():
    def __init__(self, dataset):
        self.dataset = dataset
        self.experiment = self.setup_experiment()

    def setup_experiment(self):
        # parse 2/3 of the data for training
        corr = self.dataset.corr()
        f, ax = plt.subplots(figsize=(14, 10))
        hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
        t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)
        plt.savefig("correlation.png")

        data = self.dataset.sample(frac=0.05, random_state=42)
        """ Setup the experiment """
        exp = clust.ClusteringExperiment()
        experiment = exp.setup(
            data=data,

            # Feature definitions
            ordinal_features=None,
            numeric_features=utils.get_numeric_columns(),
            #categorical_features=utils.get_categorical_columns(),
            date_features=utils.get_date_feature_columns(),
            #text_features=utils.get_text_feature_columns(),
            ignore_features=utils.ignore_features_columns(),

            # Preprocessing
            preprocess=True,
            polynomial_features=False, # TODO: sweep with True
            polynomial_degree=2,
            remove_outliers=False, #TODO Change to True
            outliers_method='iforest',

            # Transformation
            transformation=False,
            transformation_method='yeo-johnson',

            # Normalization
            normalize=True,
            normalize_method='zscore', # TODO: sweep with minmax

            # PCA, dimensionality reduction
            pca=True,
            pca_method='linear', # TODO: sweep with kernel
            pca_components=0.99,

            # # Reproducibility and logging
            session_id=42,
            log_experiment=True,
            experiment_name='clustering',
            # log_data=True,
            log_plots=True,

            # # Other
            verbose=True,
            profile=True
        )

        kmeans = experiment.create_model('kmeans')
        kmeans_model = experiment.assign_model(kmeans)

        experiment.plot_model(kmeans_model, plot='elbow')
        experiment.plot_model(kmeans_model, plot='silhouette')
        experiment.plot_model(kmeans_model, plot='cluster')
        kmeans_pred = experiment.predict_model(kmeans, data=data)

        return experiment