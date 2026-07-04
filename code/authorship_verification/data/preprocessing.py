"""
Text Preprocessing Module
=========================
Cleans raw text, tokenizes with BERT, and splits into fixed-size chunks.
"""
import re
from typing import Dict, List
import torch
from transformers import BertTokenizer
from config import cfg


class TextPreprocessor:
    """Cleans raw text and prepares it for BERT tokenization."""

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(cfg.data.bert_model_name)

    def clean(self, text: str) -> str:
        """Minimal cleaning that preserves stylistic features."""
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+\.\S+", "", text)
        text = re.sub(r"[^\x20-\x7E\n\t]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text(self, text: str, max_chunks: int = 6) -> List[Dict]:
        """
        Split text into chunks of at most max_chunk_tokens tokens.
        max_chunks caps the number of chunks per text (OOM fix): without
        it, long texts produce many chunks, and batch_size * num_chunks
        * 2 towers can exceed a 16GB GPU's memory during the BERT forward pass.
        """
        max_len = cfg.data.max_chunk_tokens
        all_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(all_tokens) == 0:
            return []

        window_size = max_len - 2
        max_total_tokens = window_size * max_chunks
        all_tokens = all_tokens[:max_total_tokens]

        chunks = []
        for start in range(0, len(all_tokens), window_size):
            window = all_tokens[start : start + window_size]
            input_ids = (
                [self.tokenizer.cls_token_id]
                + window
                + [self.tokenizer.sep_token_id]
            )
            attention_mask = [1] * len(input_ids)
            padding_length = max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

            chunks.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.zeros(max_len, dtype=torch.long),
            })

            if len(chunks) >= max_chunks:
                break

        return chunks

    def preprocess(self, text: str) -> List[Dict]:
        """Full pipeline: clean -> chunk."""
        cleaned = self.clean(text)
        if not cleaned:
            return []
        return self.chunk_text(cleaned)


def validate_chunks(chunks: List[Dict]) -> bool:
    """Validation helper for test TP-02."""
    max_len = cfg.data.max_chunk_tokens
    tokenizer = BertTokenizer.from_pretrained(cfg.data.bert_model_name)
    for i, chunk in enumerate(chunks):
        ids = chunk["input_ids"]
        if len(ids) != max_len:
            print(f"Chunk {i}: expected length {max_len}, got {len(ids)}")
            return False
        if ids[0].item() != tokenizer.cls_token_id:
            print(f"Chunk {i}: missing [CLS] token at position 0")
            return False
    return True
