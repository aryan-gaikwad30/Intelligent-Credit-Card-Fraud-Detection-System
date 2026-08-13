# Phase 4: XGBoost Baseline

## 1. Objective
The objective of this phase is to establish a reproducible XGBoost baseline model to determine whether a nonlinear tree-based model can improve upon the Logistic Regression baseline under the exact same data conditions, with a primary focus on improving the Precision-Recall Area Under Curve (PR-AUC).

## 2. Dataset
The baseline uses the Phase 2B engineered data:
- **Training rows**: 198,608
- **Validation rows**: 42,559
- **Training Legitimate Count**: 198,277
- **Training Fraud Count**: 331 (~0.1667%)

## 3. Train/Validation Split
The identical 70/15/15 deterministic split (random seed 42) from Phase 2B was preserved, ensuring a mathematically fair baseline comparison against Logistic Regression.

## 4. Test-set Isolation
The `test.csv` dataset was strictly isolated and completely unreferenced during this phase. It will be used only for final evaluation when the final model and thresholds are chosen.

## 5. Feature Representation
We utilized the raw engineered features derived from Phase 2B, which replaces raw Time and Amount with cyclic functions (`time_of_day_sin`, `time_of_day_cos`) and `amount_log1p`. 

## 6. Why XGBoost Does Not Use Scaling
Unlike distance-based models (e.g., Logistic Regression), XGBoost uses decision trees which are invariant to monotonic transformations of the independent variables. Feature scaling (like `RobustScaler`) does not affect the tree-building process or split points, so we feed the unscaled engineered features directly to the model.

## 7. Model Configuration
The baseline XGBoost (`xgboost.XGBClassifier`) configuration was set to:
- `n_estimators`: 300
- `max_depth`: 5
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `random_state`: 42
- `n_jobs`: -1
- `importance_type`: "gain"

## 8. Class Imbalance Strategy
Instead of synthetic data resampling (SMOTE) which can distort distributions, we managed the extreme class imbalance natively by weighting positive instances. We configured the `scale_pos_weight` parameter to apply a heavier penalty to false negatives.

## 9. Calculated scale_pos_weight
The parameter was computed strictly from the training partition:
`scale_pos_weight = negative_count (198,277) / positive_count (331) = 599.02`

## 10. Validation Metrics
Evaluated on the standard 0.5 threshold:
- **Precision**: 0.9167 (91.67%)
- **Recall**: 0.7746 (77.46%)
- **F1-Score**: 0.8397
- **PR-AUC**: 0.8437
- **ROC-AUC**: 0.9726
- **Accuracy**: 0.9995

## 11. Confusion Matrix Interpretation
**Validation Confusion Matrix**:
- TN: 42483
- FP: 5
- FN: 16
- TP: 55
XGBoost drastically reduces False Positives (down to 5) while maintaining strong True Positives (55). It missed 16 frauds. This reflects a massive boost in precision, making the system highly reliable when an alert fires.

## 12. PR-AUC Interpretation
The PR-AUC of 0.8437 reflects outstanding performance on the highly imbalanced dataset. It shows that across all probability thresholds, XGBoost maintains an excellent balance of capturing fraud without generating excessive false alarms.

## 13. ROC-AUC Interpretation
ROC-AUC (0.9726) shows the model's high capability to rank random positive instances higher than random negative instances. However, due to extreme class imbalance, the PR-AUC is the far more informative metric for actual business value.

## 14. Feature Importance
Feature importance was extracted using "gain" (the average training loss reduction gained when using a feature for splitting).
**Top 5 Features**:
1. `V14` (Gain: ~0.379)
2. `V10` (Gain: ~0.128)
3. `V4` (Gain: ~0.060)
4. `V12` (Gain: ~0.046)
5. `V19` (Gain: ~0.030)
*Note: Because V1-V28 are PCA-derived, these importances indicate strong statistical predictive power, not necessarily real-world causal factors.*

## 15. Logistic Regression Comparison
| Metric | Logistic Regression | XGBoost |
|---|---|---|
| PR-AUC | 0.7000 | 0.8437 |
| ROC-AUC | 0.9658 | 0.9726 |
| Precision | 0.0495 | 0.9167 |
| Recall | 0.8873 | 0.7746 |
| F1 | 0.0938 | 0.8397 |
| Accuracy | 0.9714 | 0.9995 |
| FP | 1209 | 5 |
| FN | 8 | 16 |
| FPR | 0.0285 | 0.0001 |
| FNR | 0.1127 | 0.2254 |

## 16. Model Improvement/Degradation
- **PR-AUC Absolute Change**: +0.1437
- **PR-AUC Relative Change**: +20.52%
XGBoost demonstrates a profound improvement over the linear baseline. The non-linear trees captured complex interactions that allowed the model to eliminate 99.6% of False Positives (1209 -> 5) while preserving strong recall, leading to the massive precision leap from 4.95% to 91.67%.

## 17. Limitations
- The model's baseline Recall (77.46%) is currently lower than Logistic Regression's (88.73%) at the 0.5 threshold.
- The tree depth and parameters were entirely arbitrary baseline guesses and may be sub-optimal.

## 18. Why Hyperparameter Tuning Was Not Performed Yet
Phase 4 aims strictly to validate whether a tree-based architecture inherently improves upon Logistic Regression under baseline conditions. Hyperparameter tuning is reserved for a future, focused optimization phase.

## 19. Why Threshold Optimization Was Not Performed Yet
The default 0.5 threshold was used strictly for a fair baseline evaluation. Optimizing the threshold requires assessing business cost matrices (cost of a false positive vs false negative) which is reserved for a later phase.

## 20. Next Phase
With XGBoost firmly established as a superior predictive architecture for this dataset, the next phases will involve systematic hyperparameter optimization and business-aligned threshold tuning.
