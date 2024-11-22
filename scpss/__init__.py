import numpy as np
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.stats import gamma
from kneed import KneeLocator
from typing import List
from scanpy import AnnData
import scanpy.external as sce
from numpy.typing import ArrayLike

def find_optimal_k(dists_ref_ref, dists_que_ref):
    outlier_ratios_for_k = []

    ks = np.arange(5, 51)
    initial_p_vals = [0.1, 0.05, 0.01]

    for k in ks:
        reference_kth_distances = dists_ref_ref[:, k + 1]
        query_kth_distances = dists_que_ref[:, k]

        a_fit, loc_fit, scale_fit = gamma.fit(reference_kth_distances)
        q = 1 - np.array(initial_p_vals)
        thresholds = gamma.ppf(q, a=a_fit, loc=loc_fit, scale=scale_fit)
        outlier_ratios = np.mean(query_kth_distances[:, None] > thresholds, axis=0)
        outlier_ratios_for_k.append(outlier_ratios)

    outlier_ratios_for_k = np.array(outlier_ratios_for_k)
    mean_outlier_ratio_for_k = np.mean(outlier_ratios_for_k, axis=1)
    index_of_optimal = np.argmax(mean_outlier_ratio_for_k)
    optimal_k = ks[index_of_optimal]

    return optimal_k

def find_optimal_p_val(dists_ref_ref, dists_que_ref, optimal_k):
    k = optimal_k
    reference_kth_distances = dists_ref_ref[:, k + 1]
    query_kth_distances = dists_que_ref[:, k]

    a_fit, loc_fit, scale_fit = gamma.fit(reference_kth_distances)
    qs = np.arange(850, 1001, 5) * 0.001
    thresholds = gamma.ppf(qs, a=a_fit, loc=loc_fit, scale=scale_fit)
    outlier_ratios = np.mean(query_kth_distances[:, None] > thresholds, axis=0)

    ps = 1 - qs
    kneedle = KneeLocator(
        ps, outlier_ratios, curve="concave", direction="increasing"
    )
    optimal_p = kneedle.knee + 0.005 if kneedle.knee else None

    return optimal_p

obsm_str = 'X_pca_harmony'

def find_optimal_parameters(ad_ref, ad_que, n_comps):
    params = []
    for n_comps in range(2, n_comps+1):
        X_ref = ad_ref.obsm[obsm_str][:, :n_comps]
        X_que = ad_que.obsm[obsm_str][:, :n_comps]

        dists_ref_ref = cdist(X_ref, X_ref)
        dists_ref_ref = np.sort(dists_ref_ref, axis=1)

        dists_que_ref = cdist(X_que, X_ref)
        dists_que_ref = np.sort(dists_que_ref, axis=1)

        optimal_k = find_optimal_k(dists_ref_ref, dists_que_ref)
        optimal_p = find_optimal_p_val(dists_ref_ref, dists_que_ref, optimal_k)

        dist_ref_ref = dists_ref_ref[:, optimal_k+1]
        dist_que_ref = dists_que_ref[:, optimal_k]

        a_fit, loc_fit, scale_fit = gamma.fit(dist_ref_ref)
        pct = 1 - optimal_p
        thres = gamma.ppf(pct, a=a_fit, loc=loc_fit, scale=scale_fit)
        outlier_ratio = np.mean(dist_que_ref > thres)

        param = {
            'n_comps': n_comps,
            'optimal_k': optimal_k,
            'optimal_p': optimal_p,
            'threshold': thres,
            'outlier_ratio': outlier_ratio
        }
        params.append(param)
        print(param)


    return params

