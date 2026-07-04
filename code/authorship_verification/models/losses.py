"""
Loss Functions
==============
Contrastive loss for Siamese network training.

Teaches the network to minimize distance for same-author pairs
and maximize distance for different-author pairs (up to a margin).

Creates the "Separating Manifold" shown in slide 7.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (Chopra et al., 2005).
    Labels: 1 = same author, 0 = different author.
    """

    def __init__(self, margin: Optional[float] = None):
        super().__init__()
        self.margin = margin if margin is not None else cfg.training.contrastive_margin

    def forward(self, embedding_a, embedding_b, labels):
        distances = F.pairwise_distance(embedding_a, embedding_b, p=2)
        same_loss = labels * distances.pow(2)
        diff_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        return 0.5 * (same_loss + diff_loss).mean()


class ContrastiveLossWithMining(ContrastiveLoss):
    """
    Extension with hard example mining.
    Upweights hard positives (same author but far) and
    hard negatives (different author but close).
    """

    def __init__(self, margin: Optional[float] = None, hard_weight: float = 2.0):
        super().__init__(margin)
        self.hard_weight = hard_weight

    def forward(self, embedding_a, embedding_b, labels):
        distances = F.pairwise_distance(embedding_a, embedding_b, p=2)

        with torch.no_grad():
            hard_pos = (labels == 1) & (distances > self.margin * 0.5)
            hard_neg = (labels == 0) & (distances < self.margin * 0.5)
            weights = 1.0 + (hard_pos | hard_neg).float() * (self.hard_weight - 1.0)

        same_loss = labels * distances.pow(2)
        diff_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        return 0.5 * (weights * (same_loss + diff_loss)).mean()
