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
        self.ad = adata.copy()
        self.sample_key = sample_key
        self.reference_samples = reference_samples
        self.query_samples = query_samples
        self.reference_mask = self.ad.obs[self.sample_key].isin(self.reference_samples)
        self.query_mask = self.ad.obs[self.sample_key].isin(self.query_samples)
        self.best_params = None
        self.obsm_str = 'X_pca_harmony'
        
    
    def harmony_integrate(self, max_harmony_iter = 10, random_state: int = 100):
        if 'X_pca_harmony' in self.ad.obsm:
            print("Harmony integration already done. Skipping")
            return
        if 'X_pca' not in self.ad.obsm:
            print("PCA not done. Doing it now...")
            sc.pp.pca(self.ad)
        sce.pp.harmony_integrate(self.ad, key=self.sample_key, max_harmony_iter=max_harmony_iter, random_state=random_state)
    
    def get_dist_threshold(self, reference_dists, p_val):
        a_fit, loc_fit, scale_fit = gamma.fit(reference_dists)
        threshold = gamma.ppf(1 - p_val, a=a_fit, loc=loc_fit, scale=scale_fit)
        return threshold

    def find_optimal_k(self, dists_ref_ref, dists_que_ref, ks, initial_p_vals, return_outlier_ratios=False):
        outlier_ratios_for_k = []

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

        if return_outlier_ratios:
            return optimal_k, outlier_ratios_for_k
        return optimal_k

    def find_optimal_p_val(self, dists_ref_ref, dists_que_ref, optimal_k, return_outlier_ratios=False):
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

        if return_outlier_ratios:
            return optimal_p, outlier_ratios

        return optimal_p


    def find_optimal_parameters(self, max_n_comps, ks=None, initial_p_vals=None, verbose=False):
        ad_ref = self.ad[self.ad.obs[self.sample_key].isin(self.reference_samples)]
        ad_que = self.ad[self.ad.obs[self.sample_key].isin(self.query_samples)]

        if ks is None:
            ks = np.arange(5, 51)
        if initial_p_vals is None:
            initial_p_vals = [0.1, 0.05, 0.01]


        best_outlier_ratio = 0
        params = []
        for n_comps in range(2, max_n_comps+1):
            X_ref = ad_ref.obsm[self.obsm_str][:, :n_comps]
            X_que = ad_que.obsm[self.obsm_str][:, :n_comps]

            dists_ref_ref = cdist(X_ref, X_ref)
            dists_ref_ref = np.sort(dists_ref_ref, axis=1)

            dists_que_ref = cdist(X_que, X_ref)
            dists_que_ref = np.sort(dists_que_ref, axis=1)

            optimal_k = self.find_optimal_k(dists_ref_ref, dists_que_ref, ks, initial_p_vals)
            optimal_p = self.find_optimal_p_val(dists_ref_ref, dists_que_ref, optimal_k)

            dist_ref_ref = dists_ref_ref[:, optimal_k+1]
            dist_que_ref = dists_que_ref[:, optimal_k]

            thres = self.get_dist_threshold(dist_ref_ref, optimal_p)
            outlier_ratio = np.mean(dist_que_ref > thres)

            param = {
                'n_comps': n_comps,
                'optimal_k': optimal_k,
                'optimal_p': optimal_p,
                'threshold': thres,
                'outlier_ratio': outlier_ratio,
                'ks': ks,
                'initial_p_vals': initial_p_vals,
                'ps': 1 - np.arange(850, 1001, 5) * 0.001,
            }
            params.append(param)
            if verbose:
                print(param)

            if outlier_ratio > best_outlier_ratio:
                best_outlier_ratio = outlier_ratio
                self.best_params = param

        return params
    

    def set_distance_and_condition(self):
        n_comps = self.best_params['n_comps']
        ad_ref = self.ad[self.ad.obs[self.sample_key].isin(self.reference_samples)]
        ad_que = self.ad[self.ad.obs[self.sample_key].isin(self.query_samples)]

        X_ref = ad_ref.obsm[self.obsm_str][:, :n_comps]
        X_que = ad_que.obsm[self.obsm_str][:, :n_comps]

        dists_ref_ref = cdist(X_ref, X_ref)
        dists_ref_ref = np.sort(dists_ref_ref, axis=1)

        dists_que_ref = cdist(X_que, X_ref)
        dists_que_ref = np.sort(dists_que_ref, axis=1)

        optimal_k, outlier_ratios_for_k = self.find_optimal_k(dists_ref_ref, dists_que_ref,
                                                              self.best_params['ks'], self.best_params['initial_p_vals'], return_outlier_ratios=True)
        self.best_params['outlier_ratios_for_k'] = outlier_ratios_for_k
        optimal_p, outlier_ratios_for_p = self.find_optimal_p_val(dists_ref_ref, dists_que_ref, optimal_k, return_outlier_ratios=True)
        self.best_params['outlier_ratios_for_p'] = outlier_ratios_for_p

        k = optimal_k
        dist_ref_ref = dists_ref_ref[:, k+1]
        dist_que_ref = dists_que_ref[:, k]

        thres = self.get_dist_threshold(dist_ref_ref, optimal_p)
        predicted_diseased = dist_que_ref > thres
        
        self.ad.obs.loc[self.reference_mask, 'scpss_condition'] = 'reference'
        self.ad.obs.loc[self.query_mask, 'scpss_condition'] = [
            'diseased' if d else 'healthy' for d in predicted_diseased
        ]

        self.ad.obs.loc[self.reference_mask, 'scpss_distances'] = dist_ref_ref
        self.ad.obs.loc[self.query_mask, 'scpss_distances'] = dist_que_ref
