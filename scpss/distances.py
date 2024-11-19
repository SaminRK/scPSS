import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gamma

def find_pk(dists_ref_ref, dists_que_ref, k, q=0.9):
    dists_ref_ref_ = dists_ref_ref[:, k+1]
    dists_que_ref_ = dists_que_ref[:, k]

    a_fit, loc_fit, scale_fit = gamma.fist(dists_ref_ref_)
    thres = gamma.ppf(q, a=a_fit, loc=loc_fit, scale=scale_fit)

    true_positive_ratio = np.mean(dists_ref_ref_ < thres)
    outlier_ratio = np.mean(dists_que_ref_ > thres)

    return true_positive_ratio, outlier_ratio

start_k = 5
end_k = 50

def get_k_outlier_ratio_for_q(dists_ref_ref, dists_que_ref, q):

    K_outlier_ratio = []

    for k in range(start_k, end_k):
        tp, o = find_pk(dists_ref_ref, dists_que_ref, k, q)
        K_outlier_ratio.append((k, o))

    return K_outlier_ratio

def get_q_outlier_ratio_for_k(k, dists_ref_ref, dists_que_ref):
    q_outlier_ratio = []

    for iq in range(850, 1001, 5):
        q = 0.001 * iq
        tp, o = find_pk(dists_ref_ref, dists_que_ref, k, q)
        q_outlier_ratio.append((q, o))

    return q_outlier_ratio

from kneed import KneeLocator

obsm_str = 'X_pca_harmony'

def process_with_pc_comp(ad_ref, ad_que, n_comps):
    X_ref = ad_ref.obsm[obsm_str][:, :n_comps]
    X_que = ad_que.obsm[obsm_str][:, :n_comps]

    dists_ref_ref = cdist(X_ref, X_ref)
    dists_ref_ref = np.sort(dists_ref_ref, axis=1)

    dists_que_ref = cdist(X_que, X_ref)
    dists_que_ref = np.sort(dists_que_ref, axis=1)

    K_outlier_ratio_1 = get_k_outlier_ratio_for_q(dists_ref_ref, dists_que_ref, 0.9)
    K_outlier_ratio_2 = get_k_outlier_ratio_for_q(dists_ref_ref, dists_que_ref, 0.95)
    K_outlier_ratio_3 = get_k_outlier_ratio_for_q(dists_ref_ref, dists_que_ref, 0.99)

    average_outlier_ratios = [(a[1]+b[1]+c[1])/3 for a, b, c in zip(K_outlier_ratio_1, K_outlier_ratio_2, K_outlier_ratio_3)]
    max_index = np.argmax(average_outlier_ratios)
    optimal_k = max_index + start_k

    q_outlier_ratio = get_q_outlier_ratio_for_k(optimal_k, dists_ref_ref, dists_que_ref)

    x_values, y_values = zip(*q_outlier_ratio)
    x_values = [1 - x for x in x_values]

    kneedle = KneeLocator(x_values, y_values, curve='concave', direction='increasing')
    knee = kneedle.knee
    optimal_p = knee + .005

    dist_ref_ref = dists_ref_ref[:, optimal_k+1]
    dist_que_ref = dists_que_ref[:, optimal_k]

    a_fit, loc_fit, scale_fit = gamma.fit(dist_ref_ref)
    pct = 1 - optimal_p
    thres = gamma.ppf(pct, a=a_fit, loc=loc_fit, scale=scale_fit)
    predicted_diseased = np.mean(dist_que_ref > thres)

    return optimal_k, optimal_p, thres, predicted_diseased
