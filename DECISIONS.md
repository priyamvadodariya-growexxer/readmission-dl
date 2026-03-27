# Decision log

This file documents three key decision points from your pipeline.
**Each entry is mandatory. Vague or generic answers will be penalised.**
The question to answer is not "what does this technique do" — it is "why did YOU make THIS choice given YOUR data."

---

## Decision 1: Data cleaning strategy

### What I did:
- Fixed invalid values:
  - Replaced `age = 999` with NaN and imputed with median  
  - Removed physiologically impossible blood pressure values (<50) and imputed  
- Parsed `admission_date` safely and extracted `admission_month`  
- Created a missing indicator for `glucose_level_mgdl` and used median imputation  
- Applied log transforms to skewed features:
  - `length_of_stay_days`, `prior_admissions_1yr`, `n_medications_discharge`  
- Added feature interactions / risk flags:
  - High comorbidity indicator  
  - Interaction between comorbidity and prior admissions  
- Applied one-hot encoding for categorical variables  
- Added a final global NaN safety check  

### Why I did it:
- `age = 999` distorted distribution heavily  
- Blood pressure contained impossible values → data corruption  
- Glucose missingness was informative (not random)  
- Several features were right-skewed  
- Features had low linear correlation → interactions important  
- Encoded categorical values had no ordinal meaning  

### What I considered and rejected:
- Dropping rows → dataset too small  
- Dropping glucose → lost signal  
- Treating encoded categories as ordinal → incorrect assumption  
- Using embeddings → overkill for dataset size  

### What would happen if I was wrong:
- Model learns corrupted patterns  
- Important signals lost  
- Poor generalization and unstable predictions  


## Decision 2: Model architecture and handling class imbalance

### What I did:
- Used shallow MLP (64 → 32 → 1)
- Added ReLU, BatchNorm, Dropout  
- Used BCEWithLogitsLoss with pos_weight  
- Tuned pos_weight  
- Used Adam optimizer with weight decay  
- Applied early stopping and 5-fold stratified CV  

### Why I did it:
- Small dataset → avoid deep networks  
- Tabular data → MLP sufficient  
- Severe imbalance (~9%) → required weighting  
- CV showed variability → needed robustness  

### What I considered and rejected:
- Deep networks → overfitting risk  
- Tree-based models → assignment constraint  
- SMOTE → unstable for DL  
- Ignoring imbalance → trivial model  

### What would happen if I was wrong:
- Overfitting or underfitting  
- Poor minority detection  
- Unstable performance  


## Decision 3: Evaluation metric and threshold selection

### What I did:
- Used AUROC for ranking  
- Used F1-score as main metric  
- Tuned threshold instead of 0.5  
- Used OOF predictions for stable threshold  
- Explored calibration  

### Why I did it:
- Accuracy misleading due to imbalance  
- AUC high but F1 moderate → threshold issue  
- Default threshold suboptimal  
- Needed stable threshold across folds  
- Real-world focus on recall  

### What I considered and rejected:
- Accuracy → misleading  
- Fixed threshold → suboptimal  
- Per-fold thresholds → inconsistent  
- Evaluating on test → data leakage  

### What would happen if I was wrong:
- High AUC but poor real-world performance  
- Missed high-risk patients  
---

*Word count guidance: aim for 80–150 words per decision. More is not better — precision is.*
