"""
Encoder Module (Corrected Architecture)
========================================
Matches presentation slide 6:
  Input -> BERT -> Mean Pooling -> CNN-BiLSTM -> FC+ReLU -> Embedding

Two components:
  1. BertChunkEncoder: Per-chunk BERT + mean pooling -> 768-dim vector
  2. StyleSequenceEncoder: CNN-BiLSTM over the sequence of chunk vectors

The CNN-BiLSTM now sees patterns ACROSS chunks (inter-chunk style flow),
not patterns across tokens within a single chunk.
"""

import torch
import torch.nn as nn
from transformers import BertModel

from config import cfg


class BertChunkEncoder(nn.Module):
    """
    Encodes a single text chunk: BERT -> Mean Pooling -> 768-dim vector.
    This is the first stage of slide 6.
    """

    def __init__(self, config=None):
        super().__init__()
        c = config or cfg.model
        self.bert = BertModel.from_pretrained(c.bert_model_name)
        self._freeze_bert_layers(c.freeze_bert_layers)

    def _freeze_bert_layers(self, num_layers: int):
        if num_layers <= 0:
            return
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(min(num_layers, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args: input_ids, attention_mask: (batch, seq_len)
        Returns: (batch, 768) mean-pooled BERT representation
        """
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = bert_out.last_hidden_state  # (batch, seq_len, 768)

        # Mean pooling over non-padded tokens
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (hidden * mask).sum(dim=1)          # (batch, 768)
        counts = mask.sum(dim=1).clamp(min=1)        # (batch, 1)
        pooled = summed / counts                     # (batch, 768)

        return pooled


class StyleSequenceEncoder(nn.Module):
    """
    Processes the SEQUENCE of chunk vectors through CNN -> BiLSTM -> FC.
    This is the second stage of slide 6.

    Input:  (batch, num_chunks, 768) -- sequence of BERT chunk vectors
    Output: (batch, embedding_dim)   -- style embedding for the full text
    """

    def __init__(self, config=None):
        super().__init__()
        c = config or cfg.model

        # CNN: captures local patterns across neighboring chunks
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=c.bert_hidden_size,
                out_channels=c.cnn_num_filters,
                kernel_size=ks,
                padding=ks // 2,
            )
            for ks in c.cnn_kernel_sizes
        ])
        self.cnn_activation = nn.ReLU()
        self.cnn_dropout = nn.Dropout(c.cnn_dropout)

        cnn_total_filters = c.cnn_num_filters * len(c.cnn_kernel_sizes)

        # BiLSTM: captures sequential style flow across chunks
        self.bilstm = nn.LSTM(
            input_size=cnn_total_filters,
            hidden_size=c.bilstm_hidden_size,
            num_layers=c.bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=c.bilstm_dropout if c.bilstm_num_layers > 1 else 0.0,
        )
        bilstm_output_size = c.bilstm_hidden_size * 2

        # FC head
        self.fc = nn.Sequential(
            nn.Linear(bilstm_output_size, c.fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(c.fc_dropout),
            nn.Linear(c.fc_hidden_dim, c.embedding_dim),
        )

    def forward(self, chunk_vectors, chunk_mask):
        """
        Args:
            chunk_vectors: (batch, num_chunks, 768)
            chunk_mask:    (batch, num_chunks) -- 1 for real, 0 for padding
        Returns:
            embedding: (batch, embedding_dim)
        """
        num_chunks = chunk_vectors.size(1)

        # CNN expects (batch, channels, seq_len)
        x = chunk_vectors.permute(0, 2, 1)  # (batch, 768, num_chunks)

        conv_outputs = []
        for conv in self.convs:
            out = self.cnn_activation(conv(x))
            out = out[:, :, :num_chunks]  # trim for even kernels
            conv_outputs.append(out)

        x = self.cnn_dropout(torch.cat(conv_outputs, dim=1))
        x = x.permute(0, 2, 1)  # (batch, num_chunks, total_filters)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)  # (batch, num_chunks, hidden*2)

        # Masked mean pooling over valid chunks
        mask = chunk_mask.unsqueeze(-1)  # (batch, num_chunks, 1)
        summed = (lstm_out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts  # (batch, hidden*2)

        # FC -> embedding
        embedding = self.fc(pooled)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)

        return embedding
