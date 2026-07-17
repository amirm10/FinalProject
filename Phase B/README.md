# Authorship Verification Using Impostor Projections and Siamese Networks

**Capstone Project — Phase B**  
Team Code: 26-1-R-27  
Student: Ameer Masoud  
Supervisor: Prof. Zeev Volkovich  
Braude College of Engineering — Department of Software Engineering

---

## Overview

This project determines whether two arbitrary text samples were written by the **same author**. It combines a BERT-based Siamese network with a CNN–BiLSTM stylistic head, trained under the **Impostor Projection** methodology, and converts the resulting embeddings into a final verdict through a signal-based inference pipeline: Dynamic Time Warping (DTW) alignment followed by Isolation Forest anomaly detection.

Unlike a standard binary classifier trained on both genuine and impostor pairs, the Impostor Projection approach trains the anomaly detector **exclusively on genuine (same-author) pairs**. The system learns what consistent single-author writing looks like; impostor pairs are then flagged as statistical deviations from that learned profile, rather than being explicitly labeled during the detector's own training.

The system is evaluated on **PAN20**, a standard, publicly available authorship-verification benchmark.

## Pipeline at a Glance

```
Text A, Text B
     │
     ▼
Preprocessing & Chunking          (clean text → BERT tokenizer → fixed-size chunks, capped at 6/text)
     │
     ▼
Siamese BERT + CNN–BiLSTM         (shared-weight towers → per-chunk style embeddings)
     │
     ▼
Per-Chunk Similarity Matrix       (cosine similarity, Text A chunks × Text B chunks)
     │
     ▼
Style Signal Construction        (combined aggregation → 1D signal per text)
     │
     ▼
DTW Alignment                    (z-score normalized, length-robust alignment distance)
     │
     ▼
Feature Vector (16-dim)          (DTW distance, signal statistics, embedding distance, derived ratios)
     │
     ▼
Isolation Forest (genuine-only fit)   ← trained only on genuine pairs (Impostor Projection)
     │
     ▼
Threshold tuned on validation set (max F1)
     │
     ▼
Genuine / Impostor Verdict + Confidence Score
```

> **Design note:** the decision threshold is selected exclusively from the validation set, then applied unchanged to the test set. The test set is never used to choose the threshold — only to measure performance afterward.

## Results (PAN20 Test Set — 1,616 pairs: 795 genuine, 821 impostor)

| Metric        | Baseline (mixed-fit, median split) | Improved (genuine-only fit, tuned threshold) |
|---------------|:-----------------------------------:|:----------------------------------------------:|
| Accuracy      | 59.2%                               | **69.1%**                                       |
| F1-Score      | 58.9%                               | **72.3%**                                       |
| Precision     | 58.4%                               | 64.6%                                           |
| Recall        | 59.4%                               | 82.0%                                           |
| Genuine detection  | 59.4%                          | **82.0%**                                       |
| Impostor detection | 59.1%                          | 56.5%                                           |

Exact values are stored in [`results/final_metrics.json`](results/final_metrics.json) and [`results/robustness_analysis.json`](results/robustness_analysis.json). The improvement comes entirely from the decision strategy — **the same trained model** is used for both rows; no retraining was performed between them.

## Folder Structure

```
Authorship Verification/
├── README.md                                        ← this file
├── Authorship_Verification_Final_Submission.ipynb   ← main notebook — run this
│
├── code/
│   └── authorship_verification.zip                  ← full source (config.py, data/, models/, pipeline/, training/, utils/)
│
├── data/                                             ← not included in this archive (see Note below)
│   └── pan20_{train,val,test}.csv
│
├── checkpoints/                                      ← not included in this archive (see Note below)
│   ├── best_model_final.pt
│   └── checkpoint_epoch_{1..N}.pt
│
└── results/
    ├── final_metrics.json              ← baseline + improved accuracy/F1/precision/recall
    ├── final_metrics_improved.json     ← improved metrics + checkpoint used + its val_loss
    ├── robustness_analysis.json        ← baseline + improved genuine/impostor detection rates
    ├── robustness_analysis_final.json  ← improved robustness, with raw pair counts
    ├── test_predictions.npz            ← baseline predictions, scores, and 16-dim features
    ├── test_predictions_improved.npz   ← improved predictions and scores
    ├── loss_curves.png                 ← training/validation loss + learning-rate schedule
    ├── confusion_matrix_baseline.png
    ├── confusion_matrix_improved.png
    ├── dtw_alignment_genuine.png       ← sample DTW alignment, true genuine pair
    └── dtw_alignment_impostor.png      ← sample DTW alignment, true impostor pair
```

> **Note on `data/` and `checkpoints/`:** these folders are omitted from the lightweight code/results archive (this repository) due to file size — the PAN20 CSV splits and the trained model checkpoint (several hundred MB combined) are kept on Google Drive instead. They **must** be present at `MyDrive/Authorship Verification/{data,checkpoints}/` for the notebook to run. See **How to Run** below.

## How to Run

1. Create the folder `MyDrive/Authorship Verification/` in Google Drive.
2. Inside it, place: `code/authorship_verification.zip`, `data/pan20_train.csv`, `data/pan20_val.csv`, `data/pan20_test.csv`, and `checkpoints/best_model_final.pt`.
3. Upload `Authorship_Verification_Final_Submission.ipynb` to Google Colab.
4. Run the notebook top to bottom. **One runtime restart is required partway through** (the notebook prints an explicit instruction when this point is reached — restart, then continue from the next cell without re-running the setup cells above it).
5. To reproduce the reported results **without retraining**, skip the Training section (Section 7) — the notebook loads the existing best checkpoint before that point and the rest of the pipeline runs inference and evaluation only.

Full step-by-step instructions, including expected console output at each stage, are in the **User Guide** (Appendix A of the Project Book).

## Notebook Structure

| Category | Sections | Content |
|---|---|---|
| A. Setup | 1–3 | Mount Drive, install dependencies, restart runtime, apply source-code patches, environment check |
| B. Data | 4 | Load PAN20 train / validation / test splits |
| C. Model & Training | 5–8 | Build the Siamese network, load checkpoint, train (optional), plot loss curves |
| D. Inference Pipeline | 9–10 | Run inference on the test set, baseline (mixed-fit) anomaly detection |
| E. Evaluation | 11–13 | Baseline metrics, robustness, sample DTW alignment visualization |
| F. Accuracy Improvement | 14 | Genuine-only Isolation Forest fit + validation-tuned threshold |
| G. Save Results | 15 | Export all metrics, predictions, and plots |
| Appendix | — | Resume inference from any checkpoint without retraining; verifies the true best checkpoint by its internal `val_loss`, not by filename |

## Key Engineering Decisions

| Area | Decision |
|---|---|
| Batch size | 4, with 8 gradient-accumulation steps (effective batch 32) — fits a 16GB GPU |
| BERT layers frozen | 6 of 12 |
| Contrastive margin | 1.5 |
| Early stopping | patience 2 epochs |
| Anomaly detector | Isolation Forest, 200 trees, fit on genuine pairs only |
| Feature vector | 16-dimensional: DTW distance, signal statistics (×2 texts), embedding distance, signal correlation, signal ranges, DTW/embedding ratio |
| Style signal aggregation | combined: 0.4·row-max + 0.3·row-mean + 0.2·col-max + 0.1·col-mean |
| DTW implementation | `fastdtw`, z-score normalized signals |
| Checkpoint selection | verified by each checkpoint's internal `val_loss` field, never by filename alone |

See the **Maintenance Guide** (Appendix B of the Project Book) for the full rationale, known issues, and instructions for retraining or modifying the system.

## Documentation Set

| Document | Purpose |
|---|---|
| `Project_Book.docx` | Full written report: background, methodology, engineering process, results, conclusions, lessons learned |
| `User_Guide.docx` (also Appendix A of the Project Book) | Step-by-step operating instructions for running a verification check |
| `Maintenance_Guide.docx` (also Appendix B of the Project Book) | Architecture, dependencies, installation from scratch, retraining instructions, known issues |
| `Poster_A0.pdf` | A0 presentation poster: problem, methodology, results, conclusions |

## Citation / Acknowledgment

This project builds on the Impostor Projection methodology introduced by Seidman (2013) and on the PAN20 Authorship Verification shared task dataset.

---

*Capstone Project — Phase B · Team Code 26-1-R-27 · Braude College of Engineering, Department of Software Engineering*
