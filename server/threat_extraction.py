import numpy as np
from sklearn.cluster import DBSCAN

def extract_anomalies_from_scores(features, scores, percentile_threshold=95):
    cutoff = np.percentile(scores, percentile_threshold)
    idx = np.where(scores >= cutoff)[0]
    if len(idx) == 0:
        return {}
    feats = features[idx]
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(feats)
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(int(idx[i]))
    return clusters
