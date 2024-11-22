import numpy as np
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.stats import gamma
from kneed import KneeLocator
from typing import List
from scanpy import AnnData
import scanpy.external as sce
from numpy.typing import ArrayLike


class scPSS:
    def __init__(self, adata: AnnData, sample_key: str, reference_samples: List[str], query_samples: List[str]):
        self.adata = adata
        self.sample_key = sample_key
        self.reference_samples = reference_samples
        self.query_samples = query_samples


    def harmony_integrate(self, max_iter_harmony=10):
        self.max_iter_harmony = max_iter_harmony
        # Check if 'X_pca' already exists in .obsm
        if 'X_pca' not in self.adata.obsm:
            print("X_pca not found. Performing PCA...")
            # Perform PCA using Scanpy
            sc.tl.pca(self.adata)
        else:
            print("X_pca already exists in the AnnData object.")

        # Perform integration using Harmony
        print(f"Performing Harmony integration using key: {self.sample_key}...")
        sce.pp.harmony_integrate(self.adata, key=self.sample_key, max_iter_harmony=max_iter_harmony)


    def __get_kth_nn_distance__(self, distances: np.array, k: int) -> np.array:
        """Returns the k-th nearest neighbor distances from a distance matrix."""
        return distances[:, k]


    def find_optimal_k(self, reference_distances: np.array, query_distances: np.array, ks: ArrayLike[int], initial_p_vals: ArrayLike[float], store=False) -> int:
        """
        Finds the optimal k by analyzing outlier ratios for different k values.
        """
        outlier_ratios_for_k = []

        for k in ks:
            reference_kth_distances = reference_distances[:, k+1]
            query_kth_distances = query_distances[:, k]

            a_fit, loc_fit, scale_fit = gamma.fit(reference_kth_distances)
            q = 1 - np.array(initial_p_vals)
            thresholds = gamma.ppf(q, a=a_fit, loc=loc_fit, scale=scale_fit)
            outlier_ratios = np.mean(query_kth_distances[:, None] > thresholds, axis=0)
            outlier_ratios_for_k.append(outlier_ratios)

        outlier_ratios_for_k = np.array(outlier_ratios_for_k)
        mean_outlier_ratio_for_k = np.mean(outlier_ratios_for_k, axis=1)
        index_of_optimal = np.argmax(mean_outlier_ratio_for_k)
        optimal_k = ks[index_of_optimal]

        if store:
            self.outlier_ratio_for_k = outlier_ratios_for_k

        return optimal_k


    def find_optimal_p_val(self, reference_distances: np.array, query_distances: np.array, k: int, store=False) -> float:
        """
        Finds the optimal p-value for a specific k.
        """
        reference_kth_distances = reference_distances[:, k+1]
        query_kth_distances = query_distances[:, k]

        a_fit, loc_fit, scale_fit = gamma.fit(reference_kth_distances)
        qs = np.arange(850, 1001, 5) * 0.001
        thresholds = gamma.ppf(qs, a=a_fit, loc=loc_fit, scale=scale_fit)
        outlier_ratios = np.mean(query_kth_distances[:, None] > thresholds, axis=0)

        ps = 1 - qs
        kneedle = KneeLocator(ps, outlier_ratios, curve='concave', direction='increasing')
        optimal_p = kneedle.knee + .005

        if store:
            self.outlier_ratios_for_p_val = outlier_ratios

        return optimal_p


    def find_optimal_parameters(self, n_comps: ArrayLike[int], ks: ArrayLike[int], initial_p_vals: ArrayLike[float]):
        """
        Finds the optimal parameters for different numbers of components.
        """
        self.n_comps = n_comps
        self.ks = ks
        self.initial_p_vals = initial_p_vals

        reference_adata = self.adata[self.obs[self.sample_key].isin(self.reference_samples)]
        query_adata = self.adata[self.obs[self.sample_key].isin(self.query_samples)]

        optimal_parameters_for_n_comp = {}
        max_outlier_ratio = 0
        optimal_n_comp = None

        obsm_str = 'X_pca_harmony'
        for n_comp in n_comps:
            X_reference = reference_adata.obsm[obsm_str][:, :n_comp]
            X_query = query_adata.obsm[obsm_str][:, :n_comp]

            reference_distances = cdist(X_reference, X_reference)
            reference_distances = np.sort(reference_distances, axis=1)

            query_distances = cdist(X_query, X_reference)
            query_distances = np.sort(query_distances, axis=1)

            optimal_k = self.find_optimal_k(reference_distances, query_distances, ks, initial_p_vals)
            optimal_p = self.find_optimal_p_val(reference_distances, query_distances, optimal_k)

            reference_kth_distances = reference_distances[:, optimal_k+1]
            query_kth_distances = query_distances[:, optimal_k]

            a_fit, loc_fit, scale_fit = gamma.fit(reference_kth_distances)
            q = 1 - optimal_p
            threshold = gamma.ppf(q, a=a_fit, loc=loc_fit, scale=scale_fit)
            outlier_ratio = np.mean(query_kth_distances > threshold)

            optimal_parameters_for_n_comp[n_comp] = {
                'optimal_k': optimal_k,
                'optimal_p': optimal_p,
                'threshold': threshold,
                'outlier_ratio': outlier_ratio
            }

            if outlier_ratio > max_outlier_ratio:
                max_outlier_ratio = outlier_ratio
                optimal_n_comp = n_comp

            print(n_comp, optimal_parameters_for_n_comp[n_comp])

        self.optimal_parameters_for_n_comp = optimal_parameters_for_n_comp
        self.optimal_n_comp = optimal_n_comp
        self.optimal_k = optimal_parameters_for_n_comp[n_comp]['optimal_k']
        self.optimal_p = optimal_parameters_for_n_comp[n_comp]['optimal_p']
        self.threshold = optimal_parameters_for_n_comp[n_comp]['threshold']
        self.outlier_ratio = optimal_parameters_for_n_comp[n_comp]['outlier_ratio']
