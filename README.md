🧠 Methodology (CRISP-DM)

Phase 1 — Business & Data Understanding

Goal: prioritize care by catching high-risk patients (Recall-first).

Metrics: AUPRC (class imbalance), Recall, Precision, F1, AUROC; Brier score for calibration.

EDA: schema, missingness, target balance, distributions, key relationships.

Phase 2 — Data Preparation

Pipeline (ColumnTransformer):

Numerics: median impute → Yeo–Johnson → RobustScaler.

Binary flags: mode impute; keep 0/1.

Leakage control: drop time from features.

Phase 3 — Modeling

Baselines → Logistic Regression (L2, class_weight='balanced'), Decision Tree, Random Forest.

5-fold stratified OOF CV for honest comparison (same preprocessing in all folds).

Phase 4 — Evaluation

Select model by CV AUPRC.

Tune threshold to hit Recall ≈ 0.80 (from OOF PR curve); verify on test.

Report test ROC/PR, confusion matrices (default vs tuned), calibration (Brier + reliability).

Fairness slices: by sex and age bands at the tuned operating point.

Cost curve: explore trade-offs (example cost = 5×FN + 1×FP).

Phase 5 — Deployment

Export single scikit-learn pipeline (.joblib) + threshold in config.json.

FastAPI service: /predict returns probabilities & labels using the tuned threshold.

Model Card for governance & ops.