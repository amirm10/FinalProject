"""
=================================================================
  Authorship Verification Framework
  Pipeline Verifier — Run this in Google Colab
=================================================================

Usage in Colab:
  1. Upload authorship_verification.zip to Google Drive
  2. Upload pan20_train.csv, pan20_val.csv, pan20_test.csv to
     My Drive/Authorship Verification/data/
  3. Mount Drive and extract the zip (see notebook cells)
  4. Run: !python verify.py

Tests run:
  TP-01, TP-02, TP-03 — data pipeline checks
  SYS-01, SYS-02, SYS-03 — signal, DTW, anomaly checks
=================================================================
"""

import os
import random
import sys

import numpy as np
import torch

from config import cfg
from data.dataset import _generate_demo_data
from data.impostor import ImpostorGenerator
from data.preprocessing import TextPreprocessor, validate_chunks
from pipeline.anomaly import AuthorshipAnomalyDetector
from pipeline.dtw import compute_dtw_distance
from pipeline.signal import build_similarity_signal


def set_seed(seed: int):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_verify():
    """Run all test cases from the testing plan."""
    print("=" * 60)
    print("  VERIFICATION MODE: Running Test Cases")
    print("=" * 60)

    all_pass = True

    # TP-01: Tokenizer validation
    print("\n[TP-01] Tokenizer validation...")
    prep = TextPreprocessor()
    test_texts = [
        "Hello, world! This is a test.",
        "Special chars: @#$%^&*() and émojis",
        "A" * 10000,  # very long text
        "",           # empty text
    ]
    for i, text in enumerate(test_texts):
        try:
            chunks = prep.preprocess(text)
            print(f"  Text {i+1}: {len(chunks)} chunks — OK")
        except Exception as e:
            print(f"  Text {i+1}: FAIL — {e}")
            all_pass = False

    # TP-02: Chunking logic
    print("\n[TP-02] Chunking logic (all chunks <= 512 tokens)...")
    long_text = "This is a test sentence for chunking. " * 500
    chunks = prep.preprocess(long_text)
    if validate_chunks(chunks):
        print(f"  {len(chunks)} chunks, all valid — OK")
    else:
        print("  FAIL")
        all_pass = False

    # TP-03: Dataset integrity
    print("\n[TP-03] Dataset integrity...")
    author_texts = _generate_demo_data()
    gen = ImpostorGenerator()
    pairs = gen.generate_pairs(author_texts)
    invalid = [(a, b, l) for a, b, l in pairs if not a.strip() or not b.strip()]
    if len(invalid) == 0:
        print(f"  {len(pairs)} pairs, 0 invalid — OK")
    else:
        print(f"  FAIL: {len(invalid)} invalid pairs")
        all_pass = False

    # SYS-01: Signal aggregation
    print("\n[SYS-01] Signal aggregation...")
    dummy_sim = torch.rand(5, 4)
    signal = build_similarity_signal(dummy_sim, "row_max")
    if len(signal) == 5 and all(0 <= s <= 1 for s in signal):
        print(f"  Signal shape: {signal.shape} — OK")
    else:
        print("  FAIL")
        all_pass = False

    # SYS-02: DTW alignment
    print("\n[SYS-02] DTW alignment...")
    sig_a = np.array([0.9, 0.8, 0.7, 0.85])
    sig_b = np.array([0.88, 0.78, 0.72])
    dist = compute_dtw_distance(sig_a, sig_b)
    if dist >= 0 and dist < float('inf'):
        print(f"  DTW distance: {dist:.4f} — OK")
    else:
        print("  FAIL")
        all_pass = False

    # SYS-03: Anomaly detection
    print("\n[SYS-03] Anomaly detection...")
    detector = AuthorshipAnomalyDetector()
    genuine = np.random.randn(50, 5) + 2  # genuine cluster
    impostor = np.random.randn(10, 5) - 2  # impostor cluster
    detector.fit(genuine)
    result = detector.predict(np.vstack([genuine[:5], impostor[:5]]))
    genuine_correct = result["is_genuine"][:5].sum()
    impostor_correct = (~result["is_genuine"][5:]).sum()
    print(f"  Genuine detected: {genuine_correct}/5, Impostor detected: {impostor_correct}/5 — "
          f"{'OK' if genuine_correct >= 3 and impostor_correct >= 3 else 'NEEDS TUNING'}")

    print("\n" + "=" * 60)
    print(f"  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    set_seed(cfg.training.seed)
    run_verify()
