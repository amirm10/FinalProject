"""
Signal Construction Module
==========================
Converts chunk-level similarity scores into a 1D style signal.
"""
from typing import List
import numpy as np
import torch


def build_similarity_signal(similarity_matrix, aggregation: str = "combined") -> np.ndarray:
    """
    Aggregation options: diagonal, row_max, row_mean, col_max, combined.
    "combined" blends row_max + row_mean + col_max + col_mean for a
    richer signal than any single method alone.
    """
    mat = similarity_matrix.numpy() if isinstance(similarity_matrix, torch.Tensor) else similarity_matrix
    n_a, n_b = mat.shape

    if aggregation == "diagonal":
        diag_len = min(n_a, n_b)
        return np.array([mat[i, i] for i in range(diag_len)])
    elif aggregation == "row_max":
        return np.max(mat, axis=1)
    elif aggregation == "row_mean":
        return np.mean(mat, axis=1)
    elif aggregation == "col_max":
        return np.max(mat, axis=0)
    elif aggregation == "combined":
        row_max = np.max(mat, axis=1)
        row_mean = np.mean(mat, axis=1)
        col_max = np.max(mat, axis=0)
        col_mean = np.mean(mat, axis=0)
        min_len = min(len(row_max), len(col_max))
        return (row_max[:min_len]  * 0.4 +
                row_mean[:min_len] * 0.3 +
                col_max[:min_len]  * 0.2 +
                col_mean[:min_len] * 0.1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def build_style_signal(chunk_embeddings, chunk_mask, reference_embedding=None) -> np.ndarray:
    """Per-chunk cosine similarity to the text's mean (or given) embedding."""
    valid = chunk_mask.bool()
    embeddings = chunk_embeddings[valid].detach().cpu()
    if len(embeddings) == 0:
        return np.array([])
    if reference_embedding is None:
        reference = embeddings.mean(dim=0, keepdim=True)
    else:
        reference = reference_embedding.unsqueeze(0).detach().cpu()
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, reference.expand_as(embeddings), dim=1
    )
    return similarities.numpy()


def aggregate_batch_signals(similarity_matrices: List, aggregation: str = "combined") -> List[np.ndarray]:
    return [build_similarity_signal(mat, aggregation) for mat in similarity_matrices]
