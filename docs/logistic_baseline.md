# Phase 3: Logistic Regression Baseline

## 1. Objective
The objective of this phase is to establish a genuine, reproducible, and leakage-safe Logistic Regression baseline model. This baseline provides a reference point for evaluating more advanced nonlinear models (like XGBoost) in future phases. Logistic Regression is used for its simplicity, interpretability, and fast training.

## 2. Data
The model was trained and evaluated on the datasets generated in Phase 2B:
- **Training Dataset**: `data/processed/train.csv` (198,608 rows)
- **Validation Dataset**: `data/processed/validation.csv` (42,559 rows)
- **Test Dataset**: `data/processed/test.csv` (42,560 rows) - **Strictly isolated and untouched in this phase**.

## 3. Class Imbalance
The dataset contains an extreme class imbalance, representing approximately ~0.1667% fraud.
- **Training Fraud Count**: 331 frauds (~0.1667%)
- **Validation Fraud Count**: 71 frauds (~0.1668%)

## 4. Preprocessing
We utilized the existing Phase 2B preprocessing pipeline. The CSV files contain engineered but unscaled features. To correctly prepare the data for Logistic Regression without double preprocessing:
- We extracted the `RobustScaler` from `models/preprocessing/logistic_preprocessor.joblib`.
- We applied `.transform()` directly to the unscaled feature matrices, ensuring no data leakage from the validation set and zero redundant feature engineering.

## 5. Model Configuration
The Logistic Regression model was configured as follows:
- Algorithm: `sklearn.linear_model.LogisticRegression`
- `random_state`: 42
- `class_weight`: "balanced"
- `max_iter`: 1000 (to ensure successful convergence)
- No hyperparameter search or tuning was performed.

## 6. Class Weighting
Due to the extreme class imbalance, the `class_weight="balanced"` parameter was utilized. This explicitly instructs the model to heavily penalize errors on the minority class (fraud) proportionally to its scarcity, forcing the model to detect fraud rather than defaulting to predicting everything as legitimate. SMOTE, oversampling, and undersampling were deliberately omitted to maintain a clean baseline without synthetic data risks.

## 7. Threshold
The standard `0.5` classification threshold was used for baseline reporting. 
> [!NOTE]
> Threshold 0.5 is used strictly as an unoptimized baseline reference. Final threshold selection will be performed later using validation-set analysis and business trade-off considerations.

## 8. Validation Results
The baseline model was evaluated strictly on the validation set.
- **Precision**: 0.0495 (4.95%)
- **Recall**: 0.8873 (88.73%)
- **F1-Score**: 0.0938
- **PR-AUC**: 0.6999
- **ROC-AUC**: 0.9658
- **Accuracy**: 0.9714 (Secondary metric)

**Confusion Matrix (Validation Set)**:
- True Negative (TN): 41279
- False Positive (FP): 1209
- False Negative (FN): 8
- True Positive (TP): 63

**Rates**:
- False Positive Rate (FPR): 0.0285 (2.85%)
- False Negative Rate (FNR): 0.1127 (11.27%)

## 9. Business Interpretation
- **False Positive (FP - 1209 cases)**: A legitimate transaction incorrectly flagged as fraud. 
  - *Potential business impact*: Results in a transaction decline, causing customer friction, negative experience, and unnecessary manual investigation efforts for the support team.
- **False Negative (FN - 8 cases)**: A fraudulent transaction incorrectly classified as legitimate.
  - *Potential business impact*: Results in direct financial loss, fraud exposure for the institution, and customer/account security risks.

## 10. Coefficients
The model's strongest absolute coefficients identify which features are most associated with the predictions. Since V1-V28 are anonymized PCA features, these coefficients indicate statistical association, **not causation**.

Top 5 Strongest Coefficients:
1. `V4`: 1.7289
2. `V12`: -1.3460
3. `V14`: -1.2529
4. `V10`: -1.2068
5. `V1`: 0.9788

## 11. Limitations
- **Linear Decision Boundary**: Logistic Regression can only model linear relationships, potentially missing complex interaction effects present in credit card fraud.
- **Extreme Class Imbalance**: Despite class weighting, the precision is very low (~5%), meaning the vast majority of flagged transactions are false alarms.
- **Anonymized Features**: The PCA-derived nature of V1-V28 limits human intuition and domain-specific feature engineering.
- **Unoptimized Threshold**: The 0.5 threshold is almost certainly sub-optimal for our specific, yet-to-be-defined business cost matrix.

## 12. Next Phase
The next phase will involve evaluating **XGBoost** as an advanced nonlinear model. XGBoost is expected to model complex feature interactions natively and significantly improve Precision-Recall performance without the rigid linear constraints of this baseline.
