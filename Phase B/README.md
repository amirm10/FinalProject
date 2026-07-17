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

## Repository Contents

This repository contains everything required to reproduce the project **except** the PAN20 dataset and the trained model checkpoints.

These files are distributed separately through Google Drive because they exceed GitHub's storage limits.

The download link is provided in:s

```
data_set_link.txt
```

After downloading them, follow the instructions in the **How to Run** section.

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
## Repository Structure

```
Phase B/
│
├── code/                                   ← Full project source code (.zip)
│
├── results/                                ← Evaluation results, plots, metrics and predictions
│
├── Authorship_Verification_Final_Submission.ipynb
│                                           ← Main notebook used for the complete project
│
├── Authorship_Verification_Presentation_Safe_Demo.ipynb
│                                           ← Lightweight notebook used for the live demonstration
│
├── Project_Book.docx                       ← Final project report
│
├── User_Guide.docx                         ← User documentation
│
├── Maintenance_Guide.docx                  ← Maintenance documentation
│
├── Poster.pdf                              ← Final project poster
│
├── README.md                               ← This file
│
└── data_set_link.txt                       ← Google Drive link for the dataset and trained checkpoints
```
> > **Important Note**
>
> The **PAN20 dataset** and the trained **model checkpoints** are **not included in this GitHub repository** because their size exceeds GitHub's file size limitations.
>
> Instead, both are provided through the Google Drive link located in:
>
> ```
> data_set_link.txt
> ```
>
> The Google Drive folder contains:
>
> - `data/`
>   - `pan20_train.csv`
>   - `pan20_val.csv`
>   - `pan20_test.csv`
>
> - `checkpoints/`
>   - `best_model_final.pt`
>
> Download these folders and place them inside your Google Drive before running the notebook.

## How to Run

1. Clone or download this repository.

2. Open `data_set_link.txt`.

3. Download the Google Drive archive containing:

   - `data/`
   - `checkpoints/`

4. Copy these folders into:

```
MyDrive/Authorship Verification/
```

so the structure becomes:

```
MyDrive/
└── Authorship Verification/
    ├── data/
    ├── checkpoints/
```

5. Upload the following project files from this repository to the same Google Drive folder:

- `Authorship_Verification_Final_Submission.ipynb`
- `code/`
- `results/`

6. Open `Authorship_Verification_Final_Submission.ipynb` in Google Colab.

7. Execute the notebook from top to bottom.

8. One runtime restart will be requested automatically during setup.

9. To reproduce the published benchmark results, skip the optional training stage and load the supplied trained checkpoint.

10. The notebook will reproduce the reported evaluation metrics and regenerate all result files.

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
