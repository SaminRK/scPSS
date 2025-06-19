import numpy as np
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.stats import gamma, lognorm
from kneed import KneeLocator
from typing import List
from scanpy import AnnData
import scanpy.external as sce
from typing import Optional, List, Dict
from numpy.typing import ArrayLike

from .distance_functions import custom_metrics, CDIST_METRICS
from .qvalue import storey_qvalue


class scPSS:
    def __init__(self, adata: AnnData, sample_key: str, reference_samples: List[str], query_samples: List[str]):
        """
        Initialize the scPSS framework with reference and query sample groups in an AnnData object.

        Args:
            adata (AnnData): Annotated data object containing gene expression and metadata.
            sample_key (str): Key in `adata.obs` that identifies sample groups.
            reference_samples (List[str]): Sample identifiers considered as reference (typically healthy).
            query_samples (List[str]): Sample identifiers considered as query (to be classified).

        Attributes:
            adata (AnnData): A copy of the input AnnData object.
            sample_key (str): Key used to distinguish samples in `adata.obs`.
            reference_samples (List[str]): Sample identifiers considered as reference (typically healthy).
            query_samples (List[str]): Sample identifiers considered as query (to be classified).
            reference_mask (np.ndarray): Boolean mask for selecting reference cells.
            query_mask (np.ndarray): Boolean mask for selecting query cells.
            best_params (dict or None): Stores the best set of parameters after optimization.
        """
        self.adata = adata.copy()
        self.sample_key = sample_key
        self.reference_samples = reference_samples
        self.query_samples = query_samples
        self.reference_mask = self.adata.obs[self.sample_key].isin(self.reference_samples)
        self.query_mask = self.adata.obs[self.sample_key].isin(self.query_samples)
        self.best_params = None
        self.__obsm_str__ = "X_pca"
        self._reference_label = "reference"
        self._healthy_label = "healthy"
        self._pathological_label = "pathological"

    def harmony_integrate(self, max_iter_harmony: int = 10, random_state: int = 100):
        """
        It performs PCA first, then performs Harmony batch correction to correct for batch
        effects using `self.sample_key` as batch key.

        Args:
            max_iter_harmony (int, optional): Maximum number of Harmony iterations.
                Defaults to 10.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to 100.
        """
        sc.pp.pca(self.adata)
        print("✅ PCA Complete.")
        sce.pp.harmony_integrate(
            self.adata, key=self.sample_key, max_iter_harmony=max_iter_harmony, random_state=random_state
        )
        self.__obsm_str__ = "X_pca_harmony"
        print("✅ Harmony Integration Complete.")

    def __get_dist_threshold__(self, reference_dists, q):

        if self.fn_to_fit == "lognormal":
            shape_fit, loc_fit, scale_fit = lognorm.fit(reference_dists, floc=min(0, np.min(reference_dists) - 1e-20))
            threshold = lognorm.ppf(q, s=shape_fit, loc=loc_fit, scale=scale_fit)
            return threshold

        a_fit, loc_fit, scale_fit = gamma.fit(reference_dists)
        threshold = gamma.ppf(q, a=a_fit, loc=loc_fit, scale=scale_fit)
        return threshold

    def __get_p_values__(self, reference_dists, dists):

        if self.fn_to_fit == "lognormal":
            shape_fit, loc_fit, scale_fit = lognorm.fit(reference_dists, floc=min(0, np.min(reference_dists) - 1e-20))
            p_values = 1 - lognorm.cdf(dists, s=shape_fit, loc=loc_fit, scale=scale_fit)
            return p_values

        a_fit, loc_fit, scale_fit = gamma.fit(reference_dists)
        p_values = 1 - gamma.cdf(dists, a=a_fit, loc=loc_fit, scale=scale_fit)
        return p_values

    def __find_optimal_k__(self, dists_ref_ref, dists_que_ref, ks, initial_p_vals, return_outlier_ratios=False):
        outlier_ratios_for_k = []

        q = 1 - np.array(initial_p_vals)
        for k in ks:
            reference_kth_distances = dists_ref_ref[:, k + 1]
            query_kth_distances = dists_que_ref[:, k]
            thresholds = self.__get_dist_threshold__(reference_kth_distances, q)
            outlier_ratios = np.mean(query_kth_distances[:, None] > thresholds, axis=0)
            outlier_ratios_for_k.append(outlier_ratios)

        outlier_ratios_for_k = np.array(outlier_ratios_for_k)
        mean_outlier_ratio_for_k = np.mean(outlier_ratios_for_k, axis=1)
        index_of_optimal = np.argmax(mean_outlier_ratio_for_k)
        optimal_k = ks[index_of_optimal]

        if return_outlier_ratios:
            return optimal_k, outlier_ratios_for_k
        return optimal_k

    def __find_optimal_p_val__(self, dists_ref_ref, dists_que_ref, optimal_k, max_p, return_outlier_ratios=False):
        k = optimal_k
        reference_kth_distances = dists_ref_ref[:, k + 1]
        query_kth_distances = dists_que_ref[:, k]

        qs = np.arange(850, 1001, 5) * 0.001
        thresholds = self.__get_dist_threshold__(reference_kth_distances, qs)

        outlier_ratios = np.mean(query_kth_distances[:, None] > thresholds, axis=0)

        ps = 1 - qs
        kneedle = KneeLocator(ps, outlier_ratios, curve="concave", direction="increasing")
        optimal_p = min(kneedle.knee + 0.005, max_p) if kneedle.knee else None

        if return_outlier_ratios:
            return optimal_p, outlier_ratios

        return optimal_p
    
    def __get_dist_fn__(self, distance_metric="euclidean"):
        if distance_metric in CDIST_METRICS:
            return lambda A, B: cdist(A, B, metric=distance_metric)
        
        if distance_metric in custom_metrics:
            return lambda A, B: custom_metrics[distance_metric](A, B)

        raise ValueError(f"Unsupported distance metric: {distance_metric}")


    def find_optimal_parameters(
        self,
        distance_metric="euclidean",
        search_n_comps: Optional[ArrayLike] = None,
        search_ks: Optional[ArrayLike] = None,
        maximum_p_val: int = 0.1,
        initial_p_vals: Optional[List[float]] = None,
        fn_to_fit: Optional[str] = None,
        verbose: bool = False,
    ) -> List[Dict[str, any]]:
        """
        Optimizes parameters for detecting pathological shifts in query cells based on distance metrics in PC space.

        This method identifies the optimal parameters (`n_comps`, `k`, and `p`) for distinguishing diseased cells (query cells) from healthy cells (reference cells). The optimization process selects parameters that maximize the outlier ratio (the proportion of query cells classified as pathological).

        The parameters selected are:
        - `n_comps`: The number of top principal components used for distance calculations.
        - `k`: The number of nearest neighbors used to calculate the pathological shift score.
        - `p`: The significance threshold below which cells are classified as pathological.

        The function iterates over different values of `n_comps`, `k`, and `p` to find the combination that provides the highest outlier ratio, which is indicative of the best separation between query (diseased) and reference (healthy) cells.

        Args:
            search_n_comps (array-like, optional): A range or list containing the search space for `n_comps`. Default is values from 2 to 25.
            search_ks (array-like, optional): A range or list containing the search space for `k`. Default is values from 5 to 50.
            maximum_p_val (int, optional): Maximum allowed p-value `p` cutoff. Default is 0.1.
            initial_p_vals (list, optional): A list of initial p-values to test (thresholds for labeling cells as pathological). Default is [0.1, 0.05, 0.01].
            fn_to_fit (str, optional): Specifies the function to use for fitting the distance distribution ('lognormal' or 'gamma'). Default is 'lognormal'.
            verbose (bool, optional): If True, prints out the results for each parameter combination. Default is False.

        Returns:
            list of dicts: A list of dictionaries with results of each combination of parameters tested, where each dictionary contains the following keys:
                - `n_comps`: The number of components used.
                - `optimal_k`: The optimal k-value (number of neighbors).
                - `optimal_p`: The optimal p-value (threshold for pathology).
                - `threshold`: The distance threshold used to classify outliers.
                - `outlier_ratio`: The proportion of query cells classified as outliers (pathological).
                - `ks`: The k-values tested.
                - `initial_p_vals`: The p-values tested.
                - `ps`: The p-value thresholds used for outlier classification.

        Notes:

            - The best parameter set is stored in `self.best_params`.

        Example:
            To optimize parameters with default values, call:
                params = model.find_optimal_parameters()
        """
        ad_ref = self.adata[self.adata.obs[self.sample_key].isin(self.reference_samples)]
        ad_que = self.adata[self.adata.obs[self.sample_key].isin(self.query_samples)]

        self.distance_metric = distance_metric

        if search_n_comps is None:
            search_n_comps = np.arange(2, 26)
        if search_ks is None:
            search_ks = np.arange(5, min(sum(self.reference_mask) - 1, 51))
        if initial_p_vals is None:
            initial_p_vals = [0.1, 0.05, 0.01]

        self.fn_to_fit = "lognormal" if fn_to_fit != "gamma" else "gamma"

        best_outlier_ratio = 0
        params = []
        dist_fn = self.__get_dist_fn__(self.distance_metric)
        for n_comps in search_n_comps:
            X_ref = ad_ref.obsm[self.__obsm_str__][:, :n_comps]
            X_que = ad_que.obsm[self.__obsm_str__][:, :n_comps]

            dists_ref_ref = dist_fn(X_ref, X_ref)
            dists_ref_ref = np.sort(dists_ref_ref, axis=1)

            dists_que_ref = dist_fn(X_que, X_ref)
            dists_que_ref = np.sort(dists_que_ref, axis=1)

            optimal_k = self.__find_optimal_k__(dists_ref_ref, dists_que_ref, search_ks, initial_p_vals)
            optimal_p = self.__find_optimal_p_val__(dists_ref_ref, dists_que_ref, optimal_k, maximum_p_val)

            if optimal_p is None:
                continue

            dist_ref_ref = dists_ref_ref[:, optimal_k + 1]
            dist_que_ref = dists_que_ref[:, optimal_k]

            thres = self.__get_dist_threshold__(dist_ref_ref, 1 - optimal_p)
            outlier_ratio = np.mean(dist_que_ref > thres)

            param = {
                "n_comps": n_comps,
                "optimal_k": optimal_k,
                "optimal_p": optimal_p,
                "threshold": thres,
                "outlier_ratio": outlier_ratio,
                "ks": search_ks,
                "max_p_val": maximum_p_val,
                "initial_p_vals": initial_p_vals,
                "ps": 1 - np.arange(850, 1001, 5) * 0.001,
            }
            params.append(param)
            if verbose:
                print(param)

            if outlier_ratio > best_outlier_ratio:
                best_outlier_ratio = outlier_ratio
                self.best_params = param
        
        print("✅ Found Optimal Parameters.")
        return params

    def set_distance_and_condition(self, bh_fdr=0.10):
        """
        Assigns distances and conditions (diseased or healthy) to the query and reference cells in the dataset
        based on the optimal parameters for pathological shift detection.

        This method uses the optimal parameters (`n_comps`, `k`, and `p`) obtained from `self.find_optimal_parameters()`
        to pathological distances of query cells and reference cells and stores the values in `self.adata.obs['scpss_distances']`).
        The function then assigns a "pathological condition" to each cell, classifying query cells as either
        "diseased" or "healthy", based on whether their distance exceeds the calculated threshold. These labels are stored in
        in `self.adata.obs['scpss_condition']`).

        Args:
            None (this function uses previously optimized parameters stored in `self.best_params`)

        Returns:
            None: This function adds the following two objects in `self.adata.obs`:
                - `scpss_distances`: The calculated pathological distances.
                - `scpss_condition`: A categorical variable indicating the condition of each cell ("reference", "healthy", or "diseased").

        Notes:
            - The optimal number of principal components (`n_comps`), nearest neighbors (`k`), and significance threshold (`p`)
            are retrieved from `self.best_params`, which should be set prior by calling the `find_optimal_parameters` function.
            - The distances between query and reference cells are computed using pairwise distance metrics in principal component space.
            - The threshold for classifying a query cell as "diseased" or "healthy" is determined based on the optimal `k` and `p` values.
            - This function assigns the labels to the cells and updates the `adata.obs` DataFrame in place.

        Example:
            After calling `find_optimal_parameters`, you can invoke this function to label and assign conditions to cells as follows:
                model.set_distance_and_condition()

        """
        n_comps = self.best_params["n_comps"]
        ad_ref = self.adata[self.reference_mask]
        ad_que = self.adata[self.query_mask]

        X_ref = ad_ref.obsm[self.__obsm_str__][:, :n_comps]
        X_que = ad_que.obsm[self.__obsm_str__][:, :n_comps]

        dist_fn = self.__get_dist_fn__(self.distance_metric)

        dists_ref_ref = dist_fn(X_ref, X_ref)
        dists_ref_ref = np.sort(dists_ref_ref, axis=1)

        dists_que_ref = dist_fn(X_que, X_ref)
        dists_que_ref = np.sort(dists_que_ref, axis=1)

        optimal_k, outlier_ratios_for_k = self.__find_optimal_k__(
            dists_ref_ref,
            dists_que_ref,
            self.best_params["ks"],
            self.best_params["initial_p_vals"],
            return_outlier_ratios=True,
        )
        self.best_params["outlier_ratios_for_k"] = outlier_ratios_for_k
        optimal_p, outlier_ratios_for_p = self.__find_optimal_p_val__(
            dists_ref_ref, dists_que_ref, optimal_k, self.best_params["max_p_val"], return_outlier_ratios=True
        )
        self.best_params["outlier_ratios_for_p"] = outlier_ratios_for_p

        k = optimal_k
        dist_ref_ref = dists_ref_ref[:, k + 1]
        dist_que_ref = dists_que_ref[:, k]

        thres = self.__get_dist_threshold__(dist_ref_ref, 1 - optimal_p)
        is_pathological = dist_que_ref > thres

        self.adata.obs.loc[self.reference_mask, "scpss_condition"] = self._reference_label
        self.adata.obs.loc[self.query_mask, "scpss_condition"] = [
            self._pathological_label if d else self._healthy_label for d in is_pathological
        ]
        
        self.adata.obs.loc[self.reference_mask, "scpss_outlier"] = dist_ref_ref > thres
        self.adata.obs.loc[self.query_mask, "scpss_outlier"] = dist_que_ref > thres
        self.adata.obs["scpss_outlier"] = np.where(
           self.adata.obs["scpss_outlier"], "Outlier", "Inlier"
        )

        self.adata.obs.loc[self.reference_mask, "scpss_distances"] = dist_ref_ref
        self.adata.obs.loc[self.query_mask, "scpss_distances"] = dist_que_ref

        p_values_ref = self.__get_p_values__(dist_ref_ref, dist_ref_ref)
        p_values_que = self.__get_p_values__(dist_ref_ref, dist_que_ref)

        self.adata.obs.loc[self.reference_mask, "scpss_p_values"] = p_values_ref
        self.adata.obs.loc[self.query_mask, "scpss_p_values"] = p_values_que

        all_p_values = np.concatenate([p_values_ref, p_values_que])

        from statsmodels.stats.multitest import multipletests

        rej_all, pvals_all_corrected, _, _ = multipletests(all_p_values, alpha=bh_fdr, method='fdr_bh')
        
        self.adata.obs.loc[self.reference_mask, "scpss_p_values_bh"] = pvals_all_corrected[:len(p_values_ref)]
        self.adata.obs.loc[self.query_mask, "scpss_p_values_bh"] = pvals_all_corrected[len(p_values_ref):]
        self.adata.obs.loc[self.reference_mask, "scpss_outlier_bh"] = rej_all[:len(p_values_ref)]
        self.adata.obs.loc[self.query_mask, "scpss_outlier_bh"] = rej_all[len(p_values_ref):]
        self.adata.obs["scpss_outlier_bh"] = np.where(
           self.adata.obs["scpss_outlier_bh"], "Outlier", "Inlier"
        )

        labels = np.where(rej_all, self._pathological_label, self._healthy_label)
        self.adata.obs.loc[self.reference_mask, "scpss_condition_bh"] = self._reference_label
        self.adata.obs.loc[self.query_mask, "scpss_condition_bh"] = labels[len(p_values_ref):]

        qvalues = storey_qvalue(self.adata.obs["scpss_p_values"])
        self.adata.obs["scpss_q_values"] = qvalues

        print("✅ Stored distances and conditions in Anndata object.")
        