"""
Dynamic Time Warping (DTW) Module
==================================
Aligns two style signals of possibly different lengths.
"""
import numpy as np
from typing import Tuple

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    _FASTDTW = True
except ImportError:
    _FASTDTW = False


def compute_dtw_distance(signal_a: np.ndarray, signal_b: np.ndarray, window: int = None) -> float:
    """
    DTW distance between two 1D signals. Signals are z-score normalized
    first so comparisons are fair across different scales. Uses fastdtw
    when available (much faster than the pure-Python fallback).
    """
    if len(signal_a) == 0 or len(signal_b) == 0:
        return float("inf")

    def normalize(s):
        std = np.std(s)
        return (s - np.mean(s)) / (std if std > 1e-8 else 1.0)

    a = normalize(signal_a)
    b = normalize(signal_b)

    if _FASTDTW:
        dist, _ = fastdtw(a.reshape(-1, 1), b.reshape(-1, 1), dist=euclidean)
        return float(dist)

    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window) if window else 1
        j_end   = min(m + 1, i + window + 1) if window else m + 1
        for j in range(j_start, j_end):
            cost = abs(a[i-1] - b[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return float(dtw[n, m])


def compute_dtw_with_path(signal_a, signal_b, window=None):
    """DTW distance AND the optimal alignment path (for plotting)."""
    n, m = len(signal_a), len(signal_b)
    if n == 0 or m == 0:
        return float("inf"), []
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window) if window else 1
        j_end   = min(m + 1, i + window + 1) if window else m + 1
        for j in range(j_start, j_end):
            cost = abs(signal_a[i-1] - signal_b[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    path, i, j = [], n, m
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            argmin = np.argmin([dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1]])
            if argmin == 0: i -= 1; j -= 1
            elif argmin == 1: i -= 1
            else: j -= 1
    path.reverse()
    return float(dtw[n, m]), path
