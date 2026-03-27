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

**Key preprocessing decisions:**
<!-- Summarise the most important choices — 2–3 sentences -->

**How I handled class imbalance:**
<!-- What technique and why -->

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
