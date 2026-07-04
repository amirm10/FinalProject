"""
Siamese Network (Corrected Architecture)
=========================================
Two identical towers sharing weights.

Pipeline per tower:
  1. Each chunk -> BERT -> mean pool -> 768-dim vector
  2. Stack chunks -> (N, 768)
  3. CNN-BiLSTM on chunk sequence -> text embedding (256-dim)

Then: Euclidean distance between text embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from models.encoder import BertChunkEncoder, StyleSequenceEncoder


class SiameseNetwork(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.bert_encoder = BertChunkEncoder(config)
        self.style_encoder = StyleSequenceEncoder(config)

    def _encode_chunks(self, input_ids, attention_mask, token_type_ids, chunk_mask):
        """
        Step 1: BERT + mean pool each chunk -> (batch, num_chunks, 768)
        Step 2: CNN-BiLSTM on chunk sequence -> (batch, embedding_dim)

        Returns: (text_embedding, chunk_vectors)
        """
        batch_size, num_chunks, seq_len = input_ids.shape

        # Flatten all chunks into one big batch for BERT
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)
        flat_types = token_type_ids.view(-1, seq_len)

        # BERT + mean pool -> (batch*num_chunks, 768)
        flat_vectors = self.bert_encoder(flat_ids, flat_mask, flat_types)

        # Reshape -> (batch, num_chunks, 768)
        chunk_vectors = flat_vectors.view(batch_size, num_chunks, -1)
        chunk_vectors = chunk_vectors * chunk_mask.unsqueeze(-1)

        # CNN-BiLSTM on chunk sequence -> text embedding
        text_embedding = self.style_encoder(chunk_vectors, chunk_mask)

        return text_embedding, chunk_vectors

    def forward(self, batch: dict) -> dict:
        # Tower A
        emb_a, chunk_vec_a = self._encode_chunks(
            batch["input_ids_a"], batch["attention_mask_a"],
            batch["token_type_ids_a"], batch["chunk_mask_a"],
        )
        # Tower B (same weights)
        emb_b, chunk_vec_b = self._encode_chunks(
            batch["input_ids_b"], batch["attention_mask_b"],
            batch["token_type_ids_b"], batch["chunk_mask_b"],
        )

        distances = F.pairwise_distance(emb_a, emb_b, p=2)

        return {
            "embedding_a": emb_a,
            "embedding_b": emb_b,
            "chunk_embeddings_a": chunk_vec_a,
            "chunk_embeddings_b": chunk_vec_b,
            "distances": distances,
            "labels": batch["labels"],
        }

    def get_similarity_scores(self, chunk_emb_a, chunk_emb_b, chunk_mask_a, chunk_mask_b):
        """Per-chunk cosine similarity matrices for signal construction."""
        batch_size = chunk_emb_a.shape[0]
        results = []
        for i in range(batch_size):
            valid_a = chunk_mask_a[i].bool()
            valid_b = chunk_mask_b[i].bool()
            ea = chunk_emb_a[i][valid_a]
            eb = chunk_emb_b[i][valid_b]
            sim = F.cosine_similarity(ea.unsqueeze(1), eb.unsqueeze(0), dim=2)
            results.append(sim.detach().cpu())
        return results
