# Phase 2B: Feature Engineering & Preprocessing

## Dataset Input
- Cleaned dataset used: `data/processed/creditcard_clean.csv`
- Cleaned dataset rows used: 283726

## Split Strategy
- Split ratio: 70% Training / 15% Validation / 15% Test
- Target stratification applied to preserve Class ratio.
- Train row count: 198608
- Validation row count: 42559
- Test row count: 42559

## Random Seed
- Random seed used: 42

## Class Distribution Verification
- Train fraud count and percentage: 331 (0.1667%)
- Validation fraud count and percentage: 71 (0.1668%)
- Test fraud count and percentage: 71 (0.1668%)

## Feature Engineering
### Time Transformation
The original `Time` feature represents absolute seconds elapsed, which could lead to overfitting on the specific 2-day span of this dataset. It was converted into cyclic features `time_of_day_sin` and `time_of_day_cos` using a 86,400-second period, representing the time of day, and the raw `Time` feature was dropped.

### Amount Transformation
Because `Amount` is heavily right-skewed, we engineered an `amount_log1p` feature using `np.log1p(Amount)`. The original `Amount` feature was dropped.

### PCA Features
Features V1-V28 are already PCA-transformed. They remain unchanged as there is no reason to apply a second PCA transformation over already orthogonal components.

## Dropped Features
- `Time`: Replaced by cyclic features.
- `Amount`: Replaced by log1p feature.

## Scaling
### Logistic Regression Preprocessing Strategy
The Logistic Regression pipeline uses `RobustScaler` applied to all continuous features (V1-V28, engineered Amount, engineered Time). `RobustScaler` uses median and IQR, making it resistant to outliers.

### XGBoost Preprocessing Strategy
The XGBoost pipeline passes through the engineered features without scaling, since decision trees are invariant to monotonic transformations and do not require scaled inputs.

## Leakage Prevention
All learned transformations (e.g., `RobustScaler` medians and quartiles) were fitted **strictly on the training data (`X_train`)**. The validation and test sets were transformed using the fitted scaler, ensuring no future information leaked into the preprocessing parameters.

## Class Imbalance
Handling class imbalance (e.g., via SMOTE or class weights) is deferred to the modeling phase. The original class distribution remains intact in the training set.

## Final Feature Set
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, time_of_day_sin, time_of_day_cos, amount_log1p

## Artifacts Generated
- `models/preprocessing/logistic_preprocessor.joblib`
- `models/preprocessing/xgboost_preprocessor.joblib`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

## Verification
All tests successfully verified that there is no data leakage, target exclusion from feature matrices, and deterministic preprocessing behavior.
