"""
Evaluation Module
=================
Metrics: Accuracy, F1-Score, Robustness Score.
Visualization: loss curves, confusion matrix, signal plots.

Corresponds to: Section 7 (Success Metrics).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all success metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
    }


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """Plot train/val loss curves (TR-01 convergence check)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(history["val_loss"], label="Val Loss", color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves (TR-01: Should decrease steadily)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["learning_rates"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Impostor", "Genuine"])
    ax.set_yticklabels(["Impostor", "Genuine"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_dtw_alignment(signal_a, signal_b, path=None, save_path=None):
    """Visualize DTW alignment between two style signals."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    axes[0].plot(signal_a, color="blue", label="Text A Signal")
    axes[0].set_ylabel("Similarity")
    axes[0].set_title("Style Signal A")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(signal_b, color="orange", label="Text B Signal")
    axes[1].set_ylabel("Similarity")
    axes[1].set_xlabel("Chunk Index")
    axes[1].set_title("Style Signal B")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
