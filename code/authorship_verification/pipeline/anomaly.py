"""
Anomaly Detection Module
========================
Isolation Forest + threshold clustering for the final verdict.
"""
import numpy as np
from typing import Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import cfg


class AuthorshipAnomalyDetector:
    """Final decision: Isolation Forest -> threshold clustering -> verdict."""

    def __init__(self, config=None):
        c = config or cfg.pipeline
        self.isolation_forest = IsolationForest(
            contamination=c.isolation_forest_contamination,
            n_estimators=c.isolation_forest_n_estimators,
            random_state=c.isolation_forest_random_state,
            max_features=1.0,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, genuine_features: np.ndarray):
        """Fit on GENUINE pairs only (learn what 'normal' looks like)."""
        features_scaled = self.scaler.fit_transform(genuine_features)
        self.isolation_forest.fit(features_scaled)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        features_scaled = self.scaler.transform(features)
        predictions = self.isolation_forest.predict(features_scaled)
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        return {
            "predictions": predictions,
            "anomaly_scores": anomaly_scores,
            "is_genuine": predictions == 1,
        }

    def predict_single(self, features: np.ndarray) -> Dict[str, float]:
        result = self.predict(features.reshape(1, -1))
        score = result["anomaly_scores"][0]
        is_genuine = result["is_genuine"][0]
        confidence = 1.0 / (1.0 + np.exp(-score * 2))
        return {
            "verdict": "genuine" if is_genuine else "impostor",
            "anomaly_score": float(score),
            "confidence": float(confidence if is_genuine else 1 - confidence),
        }


def extract_pair_features(dtw_distance, similarity_signal_a,
                           similarity_signal_b=None,
                           mean_embedding_distance=None) -> np.ndarray:
    """Build a 16-dim feature vector for one text pair."""
    features = [dtw_distance]

    if len(similarity_signal_a) > 0:
        features.extend([
            np.mean(similarity_signal_a), np.std(similarity_signal_a),
            np.min(similarity_signal_a),  np.max(similarity_signal_a),
            np.median(similarity_signal_a), np.var(similarity_signal_a),
        ])
    else:
        features.extend([0.0] * 6)

    if similarity_signal_b is not None and len(similarity_signal_b) > 0:
        features.extend([
            np.mean(similarity_signal_b), np.std(similarity_signal_b),
            np.min(similarity_signal_b),  np.max(similarity_signal_b),
        ])
    else:
        features.extend([0.0] * 4)

    emb = mean_embedding_distance if mean_embedding_distance is not None else 0.0
    features.append(emb)

    if similarity_signal_b is not None and len(similarity_signal_b) > 0 and len(similarity_signal_a) > 0:
        min_len = min(len(similarity_signal_a), len(similarity_signal_b))
        corr = np.corrcoef(similarity_signal_a[:min_len], similarity_signal_b[:min_len])[0, 1]
        features.append(0.0 if np.isnan(corr) else corr)
        features.append(np.max(similarity_signal_a) - np.min(similarity_signal_a))
        features.append(np.max(similarity_signal_b) - np.min(similarity_signal_b))
    else:
        features.extend([0.0, 0.0, 0.0])

    features.append(dtw_distance / (emb + 1e-8))

    return np.array(features, dtype=np.float64)


def kmedoids_cluster(anomaly_scores: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """
    Split anomaly scores into genuine/impostor via a median threshold.
    Label 1 = genuine (higher anomaly score = more "normal").
    """
    threshold = np.median(anomaly_scores)
    labels = (anomaly_scores >= threshold).astype(int)
    means = [anomaly_scores[labels == 0].mean(), anomaly_scores[labels == 1].mean()]
    if means[0] > means[1]:
        labels = 1 - labels
    return labels
