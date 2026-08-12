# Phase 2A: Data Cleaning & Validation

## 1. Cleaning Objective
Validate the dataset schema, investigate duplicates, zero amounts, and invalid values, and apply data-cleaning policies to produce a clean, reproducible dataset without modifying the raw source.

## 2. Raw Dataset
- Rows: 284807
- Columns: 31
- Fraud transactions: 492 (0.1727%)
- Legitimate transactions: 284315

## 3. Schema Validation
- Validation passed: All expected columns are present.
- Target unique values: [0, 1]
- Unexpected target values: []
- Missing target values: 0

## 4. Duplicate Investigation
- Total rows involved in duplicate groups: 1854
- Exact duplicate rows that can be removed: 1081
- Conflicting target groups (identical features, different targets): 0
- Fraud vs Legitimate among duplicates: 32 fraud, 1822 legitimate.
- **Decision:** REMOVE exact duplicates.
- **Reasoning:** Exact duplicates across all 31 columns provide no new information and can cause train/test data leakage (artificially inflating evaluation metrics if a duplicate is in both sets).

## 5. Zero-Amount Investigation
- Total zero-amount transactions: 1825
- Fraud vs Legitimate (Zero Amount): 27 fraud, 1798 legitimate.
- Fraud vs Legitimate (Non-Zero Amount): 465 fraud, 282517 legitimate.
- **Decision:** RETAIN zero-amount transactions.
- **Reasoning:** Zero-amount transactions are often valid authorization checks (e.g., verifying a card before a recurring charge). There is no evidence they are corrupted records, and they even contain some fraudulent attempts.

## 6. Numerical Validation
- Non-numeric columns: []
- Missing or infinite values found: {}

## 7. Outlier Policy
Outliers (such as the highly skewed Amount) were NOT removed in this phase. Statistical outliers are not automatically invalid or erroneous transactions. Handling of these distributions will be considered during the preprocessing/feature engineering phase.

## 8. Cleaning Rules
1. Drop exact duplicate rows to prevent data leakage.
2. Retain zero-amount transactions.
3. Drop rows with non-finite values (NaN, Inf) or invalid Target classes (not 0 or 1).

## 9. Before vs After

| Metric | Raw | Cleaned |
| --- | ---: | ---: |
| Rows | 284807 | 283726 |
| Columns | 31 | 31 |
| Class 0 | 284315 | 283253 |
| Class 1 | 492 | 473 |
| Fraud % | 0.1727% | 0.1667% |
| Duplicate rows | 1081 | 0 |
| Zero-amount rows | 1825 | 1808 |

## 10. Final Dataset
- Location: `data/processed/creditcard_clean.csv`
- Dimensions: 283726 rows x 31 columns

## 11. Data Leakage Considerations
Removing exact duplicates ensures that identical records do not appear in both the training and test sets. The cleaning process evaluates row validity individually and does not compute any aggregate statistics (like means or scalers) that would leak global target information.

## 12. Deferred Decisions
The following are NOT decided yet:
- feature engineering
- scaling
- train/validation/test splitting
- class imbalance strategy
- SMOTE/resampling
- model selection
- threshold selection
