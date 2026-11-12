# Fall Detection Generalization Across Vulnerable Populations

**Master's Thesis — Polish Japanese Academy of Information Technology**  
**Student:** Aksa  
**Supervisor:** TBD

## Research Question

How well do existing IMU-based fall detection models generalize from young-adult data to elderly data, and what signal characteristics drive the gap?

## Datasets

| Dataset | Source | Age Group | Labels | Role |
|---|---|---|---|---|
| **SisFall** | [Sucerquia et al., 2017](http://sistemic.udea.edu.co/en/research/projects/english-falls/) | Young (19–30) + Elderly (60–75) | Falls (F01–F15) + ADLs (D01–D19) | Primary — fall detection |
| **HAR70+** | [Ustad et al., 2023](https://doi.org/10.3390/s23052368) / Kaggle | Elderly (70–95) | ADL activity labels | Elderly ADL distribution |
| **Laavanya** | Kaggle | Elderly | Fall / non-fall | Cross-dataset evaluation |

> Raw data is not committed. See `data/README.md` for download instructions.

## Project Structure

```
fall-detection-generalization/
├── src/
│   ├── data/          # Dataset loaders & harmonization
│   ├── features/      # Time/frequency feature extraction
│   ├── models/        # Threshold, classical ML, deep learning
│   ├── evaluation/    # Protocols (P1/P2/P3) and metrics
│   └── visualization/ # Plots and reports
├── notebooks/         # Exploratory analysis
├── scripts/           # CLI entry points
├── configs/           # Experiment YAML configs
├── data/              # Raw (gitignored) and processed data
└── results/           # Saved model outputs and figures
```

## Evaluation Protocols

- **P1 — Within-dataset, subject-independent:** grouped k-fold CV (baseline)
- **P2 — Cross-population:** train on young adults (SisFall), test on elderly
- **P3 — Cross-dataset:** train on one dataset, evaluate on independent cohort

## Models

1. **Threshold baseline** — SVM magnitude peak + post-impact inactivity (Bourke et al., 2007)
2. **Random Forest** on hand-crafted time/frequency features
3. **SVM + XGBoost** on hand-crafted features with SHAP interpretability
4. **1D-CNN** (and optionally CNN-LSTM) on raw windowed IMU signals

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py --dataset sisfall
python scripts/run_experiments.py --config configs/baseline.yaml
```

## Research Questions

- **RQ1:** Which IMU signal properties differ systematically between elderly and young adults?
- **RQ2:** How much does fall detection performance degrade when tested on elderly vs. young data?
- **RQ3:** Do elderly-trained models transfer across independent elderly cohorts?
- **RQ4:** Which population-informed adjustments (feature selection, retraining, SHAP) reduce the gap?
