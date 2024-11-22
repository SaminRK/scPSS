import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gamma
from kneed import KneeLocator


def get_kth_nn_distance(distances: np.array, k: int) -> np.array:
    return distances[:, k]



def find_optimal_k(reference_distances: np.array, query_distances: np.array, lower_limit: int, upper_limit: int, p_vals: List[int]) -> int:

    outlier_ratios_for_k = []

    for k in range(lower_limit, upper_limit + 1):
        reference_kth_distances = reference_distances[:, k+1]
        query_kth_distances = query_distances[:, k]

        a_fit, loc_fit, scale_fit = gamma.fit(reference_kth_distances)
        q = 1 - np.array(p_vals)
        thresholds = gamma.ppf(q, a=a_fit, loc=loc_fit, scale=scale_fit)
        outlier_ratios = np.mean(query_kth_distances[:, None] > thresholds, axis=0)
        outlier_ratios_for_k.append(outlier_ratios)
    
    outlier_ratios_for_k = np.array(outlier_ratios_for_k)
    mean_outlier_ratio_for_k = np.mean(outlier_ratios_for_k, axis=1)
    index_of_optimal = np.argmax(mean_outlier_ratio_for_k)
    optimal_k = index_of_optimal + upper_limit

    return optimal_k


def find_optimal_p_val(reference_distances, query_distances, k: int):

    reference_kth_distances = reference_distances[:, k+1]
    query_kth_distances = query_distances[:, k]
    
    qs = np.arange(850, 1000, 5) * 0.001
    a_fit, loc_fit, scale_fit = gamma.fit(reference_kth_distances)
    thresholds = gamma.ppf(qs, a=a_fit, loc=loc_fit, scale=scale_fit)
    outlier_ratios = np.mean(query_kth_distances[:, None] > thresholds, axis=0)

    kneedle = KneeLocator(qs, outlier_ratios, curve='concave', direction='increasing')
    optimal_p = 1 - (kneedle.knee + .005)

    return optimal_p



def find_optimal_parameters(reference_adata, query_adata, n_comps_upper_limit: int):

    optimal_parameters_for_n_comps = {}
    
    obsm_str = 'X_pca_harmony'
    for n_comps in range(2, n_comps_upper_limit+1):
        X_reference = reference_adata.obsm[obsm_str][:, :n_comps]
        X_query = query_adata.obsm[obsm_str][:, :n_comps]

        reference_distances = cdist(X_reference, X_reference)
        reference_distances = np.sort(reference_distances, axis=1)

        query_distances = cdist(X_query, X_reference)
        query_distances = np.sort(query_distances, axis=1)

        optimal_k = find_optimal_k(reference_distances, query_distances, 5, 50, [.1, .05, .01])
        optimal_p = find_optimal_p_val(reference_distances, query_distances, optimal_k)

        reference_kth_distances = reference_distances[:, optimal_k+1]
        query_kth_distances = query_distances[:, optimal_k]

        a_fit, loc_fit, scale_fit = gamma.fit(reference_kth_distances)
        q = 1 - optimal_p
        threshold = gamma.ppf(q, a=a_fit, loc=loc_fit, scale=scale_fit)
        outlier_ratio = np.mean(query_kth_distances > threshold)

        optimal_parameters_for_n_comps[n_comps] = {
            'optimal_k': optimal_k,
            'optimal_p': optimal_p,
            'threshold': threshold,
            'outlier_ratio': outlier_ratio
        }

