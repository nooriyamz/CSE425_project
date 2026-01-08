from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def kmeans_cluster(features, n_clusters=3):
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(features)
    sil = silhouette_score(features, labels)
    ch = calinski_harabasz_score(features, labels)
    return labels, sil, ch

def agglomerative_cluster(features, n_clusters=3):
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)
    sil = silhouette_score(features, labels)
    db = davies_bouldin_score(features, labels)
    return labels, sil, db

def dbscan_cluster(features, eps=0.8, min_samples=2):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
    mask = labels != -1
    if mask.sum() > 1:
        sil = silhouette_score(features[mask], labels[mask])
        db = davies_bouldin_score(features[mask], labels[mask])
        return labels, sil, db
    return labels, None, None
