from sklearn.metrics import silhouette_score, calinski_harabasz_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from scipy.stats import mode

def compute_basic_metrics(features, labels):
    sil = silhouette_score(features, labels)
    ch = calinski_harabasz_score(features, labels)
    return sil, ch

def compute_multi_modal_metrics(y_true, y_pred):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return nmi, ari

def cluster_purity(y_true, y_pred):
    clusters = np.unique(y_pred)
    total = 0
    for c in clusters:
        if c == -1:
            continue
        idx = np.where(y_pred == c)
        total += np.sum(y_true[idx] == mode(y_true[idx])[0])
    return total / len(y_true)
