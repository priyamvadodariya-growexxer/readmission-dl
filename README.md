# Readmission-DL — City General Hospital 30-day Readmission Prediction

**Student name:** 
Priyam Vadodaria </br>
**Student ID:**
1128 </br>
**Submission date:**
27 March, 2026 

---

## Problem

Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).

---

## My model

**Architecture:**
<!-- Describe your network: layer sizes, activations, regularisation -->
A shallow Multi-Layer Perceptron (MLP) was used with the following structure:
Input → Linear(64) → ReLU → BatchNorm → Dropout(0.3) → Linear(32) → ReLU → Dropout(0.2) → Linear(1). </br>
A shallow Multi-Layer Perceptron (MLP) was used with the following structure:
Input → Linear(64) → ReLU → BatchNorm → Dropout(0.3) → Linear(32) → ReLU → Dropout(0.2) → Linear(1).

**Key preprocessing decisions:**
<!-- Summarise the most important choices — 2–3 sentences -->
Invalid and corrupted values (e.g., age = 999) were cleaned. Categorical values were one hot encoded to avoid introducing artificial ordinal relationships

**How I handled class imbalance:**
<!-- What technique and why -->
Class imbalance (~9% positive class) was handled using the pos_weight parameter in BCEWithLogitsLoss, which increases the penalty for misclassifying minority class samples. This approach was preferred over resampling methods as it maintains the original data distribution and is more stable for deep learning models. The classification threshold was additionally tuned to optimize F1-score, aligning model decisions with the objective of improving minority class detection.

---

## Results on validation set

| Metric | Value |
|--------|-------|
| AUROC | 0.937075|
| F1 (minority class) |0.581395 |
| Precision (minority) | 0.480769|
| Recall (minority) | 0.735294|
| Decision threshold used | 0.210000|

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (optional — pretrained weights included)

```bash
python notebooks/solution.ipynb  # or run cells in order
```

### 3. Run inference on the test set

```bash
python predict.py --input ../data/test.csv --output ../data/predictions.csv
```

The output CSV will contain two columns: `patient_id` and `readmission_probability`.

---

## Repository structure

```
readmission-dl/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── solution.ipynb
├── src/
│   └── predict.py
├── models/
|   └── .wt files
├── DECISIONS.md
├── requirements.txt
└── README.md
```

---

## Limitations and honest assessment

<!-- What would you improve with more time? Where might this model fail in production? -->
In production, the model may fail when encountering data distribution shifts (e.g., different hospitals, coding schemes, or patient demographics), especially since categorical encodings and feature distributions are learned from a single dataset. With more time, I would explore more robust validation (e.g., external validation datasets), advanced feature engineering (including temporal features), and compare against strong tabular baselines such as gradient boosting models to further improve performance and reliability.