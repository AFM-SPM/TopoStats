"Functions for clustering the grainstats data."

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

from topostats.plottingfuncs import Images

from topostats.logs.logs import setup_logger, LOGGER_NAME

LOGGER = setup_logger(LOGGER_NAME)

class ClusterData():
    """Clusters grainstats data"""
    def __init__(self, grainstats: pd.DataFrame, labeled_mask: np.ndarray, image: np.ndarray, pca_cols="all"):
        """Initialises the class
        
        Parameters
        ----------
        grainstats: pd.DataFrame
            A dataframe containing the image's grain statistics.
        """
        self.grainstats = grainstats
        self.labeled_mask = labeled_mask
        self.image = image
        self.pca_cols = pca_cols
        self.mol_nums = None
        self.normalised_df = None
        self.pca = None
        self.components = None
        self.pca_importances = None
        self.dbscan_labels = None
        self.cluster_mask = None
        #assert len(grainstats) == labeled_mask.max()
        
    def cluster_data(self, eps1=0.1, eps2=0.4, min_samples=2):
        self.refine_dataframe()
        self.pca, self.components = self.pca_analysis(self.normalised_df)
        self.pca_importances = self.get_pca_importances(self.pca, self.normalised_df.columns)
        LOGGER.info(f"PCA Importances:\n{self.pca_importances}")
        self.dbscan_labels = self.recursive_dbscan(data=self.components, eps1=eps1, eps2=eps2, min_samples=min_samples)
        self.cluster_mask = self.get_cluster_mask(self.labeled_mask, self.dbscan_labels, self.mol_nums)
        return pd.DataFrame(np.stack([self.mol_nums, self.dbscan_labels], axis=1), columns=["img_grain_no", "cluster_label"])

    def refine_dataframe(self):
        # remove non-important / string features
        numeric_grainstats = self.grainstats.copy().drop(columns=[
        "centre_x", # position dependent
        "centre_y", # position dependent
        "radius_min", # corrolate with radius_mean
        "radius_max", # corrolate with radius_mean
        "radius_median", # corrolate with radius_mean
        "height_min", # corrolate with height_mean
        "height_max", # corrolate with height_mean
        "height_median", # corrolate with height_mean
        "smallest_bounding_width", # corrolate with min_feret & aspect ratio
        "smallest_bounding_length", # corrolate with max_feret & aspect ratio
        "volume",
        "area_cartesian_bbox",
        "smallest_bounding_area",
        "threshold", # str
        "circular", # str
        "image", # str
        "basename"]) # str

        if self.pca_cols != "all":
            if "img_grain_no" not in self.pca_cols:
                self.pca_cols.append("img_grain_no")
            numeric_grainstats = numeric_grainstats[self.pca_cols]
        
        LOGGER.info(f"Performing PCA analysis on following columns:\n{numeric_grainstats.columns}")

        for col_name in numeric_grainstats.columns:
            pd.to_numeric(numeric_grainstats[col_name])
        filtered_grainstats = numeric_grainstats.dropna()
        self.mol_nums = np.array(filtered_grainstats.pop("img_grain_no"))
        LOGGER.info(f"{len(self.mol_nums)} of {self.labeled_mask.max()} grainstats fully calculated")
        self.normalised_df = (filtered_grainstats - filtered_grainstats.min()) / (filtered_grainstats.max() - filtered_grainstats.min())

    @staticmethod
    def pca_analysis(normalised_data):
        n_components = 0
        pca_var_sum = 0
        while pca_var_sum < 80 or n_components <= 1:
            n_components += 1
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(normalised_data)
            pca_var_sum = sum(pca.explained_variance_ratio_) * 100
        LOGGER.info(f"PCA Halted at {n_components} components as varience ratio > 80% ({pca_var_sum} from {pca.explained_variance_ratio_}).")

        return pca, components
    
    @staticmethod
    def get_pca_importances(pca, column_names):
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        col_names = [f"PC{i+1}" for i in range(pca.n_components_)]
        loading_matrix = pd.DataFrame(loadings, columns=col_names, index=column_names)
        return loading_matrix.sort_values(by=['PC1'])

    @staticmethod
    def recursive_dbscan(data, eps1=0.1, eps2=0.4, min_samples=2):
        norm_data = (data - data.min()) / (data.max() - data.min())
        clustering = DBSCAN(eps=eps1, min_samples=min_samples).fit(norm_data)
        labels = clustering.labels_
        unique_labels = set(labels)
        LOGGER.info(f"{len(unique_labels)} found on first pass.")
        """
        if -1 in unique_labels:
            # Get heights and areas for -1 labels
            indexes = np.where(labels==-1)[0]
            pca_sub = []
            for i in range(data.shape[0]):
                pca_comp_sub = []
                for j in indexes:
                    pca_comp_sub.append(data[i][j])
                pca_sub.append(pca_comp_sub)

            # Run DBSCAN on new labels
            clustering2 = DBSCAN(eps=eps2, min_samples=min_samples).fit(np.array(pca_sub).T)
            labels2 = clustering2.labels_
            # Update overall labels with new labels
            labels2[labels2!=-1] = labels2[labels2!=-1] + len(unique_labels) - 1
            for i in range(len(indexes)):
                idx = indexes[i]
                labels[idx] = labels2[i]
            unique_labels = set(labels2)
            LOGGER.info(f"{len(unique_labels)} found on second pass.")
        """

        return labels
    
    @staticmethod
    def get_cluster_mask(labeled_grain_mask, dbscan_labels, mol_mappings):
        grain_shape = labeled_grain_mask.shape
        unique_labels = np.unique(dbscan_labels)
        mask_tensor = np.zeros((grain_shape[0], grain_shape[1], len(unique_labels)))
        for i, k in enumerate(unique_labels):
            LOGGER.info(f"Cluster {k} has {(dbscan_labels==k).sum()} items")
            labeled_mask_cp = labeled_grain_mask.copy()
            grain_nums = mol_mappings[dbscan_labels==k] + 1
            for grain_no in grain_nums:
                labeled_mask_cp[labeled_mask_cp==grain_no] = 10000 # grab all pixel grains in a cluster and set to large value
            labeled_mask_cp[labeled_mask_cp!=10000] = 0
            labeled_mask_cp[labeled_mask_cp==10000] = 1
            mask_tensor[:,:,i] += labeled_mask_cp
        # compile tensor into single mask (could be done above)
        clustered_mask = np.zeros_like(mask_tensor[:,:,0])
        for cluster_no in range(mask_tensor.shape[2]):
            clustered_mask[mask_tensor[:,:,cluster_no]==1] = cluster_no + 1
        return clustered_mask
