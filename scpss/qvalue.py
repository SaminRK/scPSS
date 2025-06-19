import numpy as np

def storey_qvalue(pvals, lambda_=0.5):
    pvals = np.asarray(pvals)
    m = float(len(pvals))
    pi0 = np.mean(pvals > lambda_) / (1 - lambda_)
    pi0 = min(pi0, 1.0)

    # Sort p-values
    order = np.argsort(pvals)
    ordered_pvals = pvals[order]

    # Compute q-values
    qvals = pi0 * m * ordered_pvals / np.arange(1, m+1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]  # Ensure monotonicity

    # Reorder to original
    qval_final = np.empty_like(qvals)
    qval_final[order] = qvals
    return qval_final
