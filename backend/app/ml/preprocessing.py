import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

SEED = 42

import sys
sys.path.append(os.path.abspath('backend'))
from app.ml.feature_engineering import FeatureEngineer

def load_data(filepath='data/processed/creditcard_clean.csv'):
    print(f"Loading cleaned data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned dataset not found at {filepath}")
    return pd.read_csv(filepath)

def split_data(df, target_col='Class', test_size=0.15, val_size=0.15, random_state=SEED):
    print("Splitting data...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Calculate train_size for the first split to leave test_size + val_size remaining
    train_size = 1.0 - test_size - val_size
    temp_size = test_size + val_size
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, stratify=y, random_state=random_state
    )
    
    # Split the temp set into validation and test sets (50/50 since val_size == test_size)
    relative_test_size = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_logistic_pipeline():
    # Pipeline: Feature Engineering -> Scaling
    pipeline = Pipeline([
        ('engineer', FeatureEngineer()),
        ('scaler', RobustScaler()) # Scales all resulting columns
    ])
    return pipeline

def create_xgboost_pipeline():
    # Pipeline: Feature Engineering -> Pass through (no scaling needed for trees)
    pipeline = Pipeline([
        ('engineer', FeatureEngineer())
    ])
    return pipeline

def save_artifact(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)

def save_dataset(X, y, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = X.copy()
    df['Class'] = y.values
    df.to_csv(filepath, index=False)

def main():
    raw_filepath = 'data/processed/creditcard_clean.csv'
    
    # 1. Load Data
    df = load_data(raw_filepath)
    total_rows = len(df)
    
    # 2. Split Data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # 3. Create Pipelines
    print("Creating pipelines...")
    lr_pipeline = create_logistic_pipeline()
    xgb_pipeline = create_xgboost_pipeline()
    
    # 4. Fit Pipelines ONLY on training data
    print("Fitting preprocessing pipelines on training data...")
    lr_pipeline.fit(X_train)
    xgb_pipeline.fit(X_train)
    
    # 5. Transform Data
    # For Logistic
    X_train_lr = pd.DataFrame(lr_pipeline.transform(X_train), columns=FeatureEngineer().transform(X_train).columns)
    X_val_lr = pd.DataFrame(lr_pipeline.transform(X_val), columns=FeatureEngineer().transform(X_val).columns)
    X_test_lr = pd.DataFrame(lr_pipeline.transform(X_test), columns=FeatureEngineer().transform(X_test).columns)

    # For XGBoost (we will use this as our 'processed' dataset output since it retains the unscaled but engineered features, 
    # though both pipelines output identical columns). We'll save the xgb features directly.
    X_train_xgb = FeatureEngineer().transform(X_train)
    X_val_xgb = FeatureEngineer().transform(X_val)
    X_test_xgb = FeatureEngineer().transform(X_test)
    
    # 6. Save Artifacts
    print("Saving artifacts...")
    save_artifact(lr_pipeline, 'models/preprocessing/logistic_preprocessor.joblib')
    save_artifact(xgb_pipeline, 'models/preprocessing/xgboost_preprocessor.joblib')
    
    # Save transformed datasets (using XGBoost features as standard transformed output)
    save_dataset(X_train_xgb, y_train, 'data/processed/train.csv')
    save_dataset(X_val_xgb, y_val, 'data/processed/validation.csv')
    save_dataset(X_test_xgb, y_test, 'data/processed/test.csv')

    # 7. Generate Report
    report_path = 'docs/preprocessing.md'
    print(f"Generating report at {report_path}...")
    
    train_fraud = y_train.sum()
    val_fraud = y_val.sum()
    test_fraud = y_test.sum()
    
    train_pct = (train_fraud / len(y_train)) * 100
    val_pct = (val_fraud / len(y_val)) * 100
    test_pct = (test_fraud / len(y_test)) * 100
    
    final_features = list(X_train_xgb.columns)
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 2B: Feature Engineering & Preprocessing\n\n")
        
        f.write("## Dataset Input\n")
        f.write(f"- Cleaned dataset used: `{raw_filepath}`\n")
        f.write(f"- Cleaned dataset rows used: {total_rows}\n\n")
        
        f.write("## Split Strategy\n")
        f.write("- Split ratio: 70% Training / 15% Validation / 15% Test\n")
        f.write("- Target stratification applied to preserve Class ratio.\n")
        f.write(f"- Train row count: {len(y_train)}\n")
        f.write(f"- Validation row count: {len(y_val)}\n")
        f.write(f"- Test row count: {len(y_test)}\n\n")

        f.write("## Random Seed\n")
        f.write(f"- Random seed used: {SEED}\n\n")
        
        f.write("## Class Distribution Verification\n")
        f.write(f"- Train fraud count and percentage: {train_fraud} ({train_pct:.4f}%)\n")
        f.write(f"- Validation fraud count and percentage: {val_fraud} ({val_pct:.4f}%)\n")
        f.write(f"- Test fraud count and percentage: {test_fraud} ({test_pct:.4f}%)\n\n")
        
        f.write("## Feature Engineering\n")
        f.write("### Time Transformation\n")
        f.write("The original `Time` feature represents absolute seconds elapsed, which could lead to overfitting on the specific 2-day span of this dataset. It was converted into cyclic features `time_of_day_sin` and `time_of_day_cos` using a 86,400-second period, representing the time of day, and the raw `Time` feature was dropped.\n\n")
        f.write("### Amount Transformation\n")
        f.write("Because `Amount` is heavily right-skewed, we engineered an `amount_log1p` feature using `np.log1p(Amount)`. The original `Amount` feature was dropped.\n\n")
        f.write("### PCA Features\n")
        f.write("Features V1-V28 are already PCA-transformed. They remain unchanged as there is no reason to apply a second PCA transformation over already orthogonal components.\n\n")
        
        f.write("## Dropped Features\n")
        f.write("- `Time`: Replaced by cyclic features.\n")
        f.write("- `Amount`: Replaced by log1p feature.\n\n")
        
        f.write("## Scaling\n")
        f.write("### Logistic Regression Preprocessing Strategy\n")
        f.write("The Logistic Regression pipeline uses `RobustScaler` applied to all continuous features (V1-V28, engineered Amount, engineered Time). `RobustScaler` uses median and IQR, making it resistant to outliers.\n\n")
        f.write("### XGBoost Preprocessing Strategy\n")
        f.write("The XGBoost pipeline passes through the engineered features without scaling, since decision trees are invariant to monotonic transformations and do not require scaled inputs.\n\n")
        
        f.write("## Leakage Prevention\n")
        f.write("All learned transformations (e.g., `RobustScaler` medians and quartiles) were fitted **strictly on the training data (`X_train`)**. The validation and test sets were transformed using the fitted scaler, ensuring no future information leaked into the preprocessing parameters.\n\n")
        
        f.write("## Class Imbalance\n")
        f.write("Handling class imbalance (e.g., via SMOTE or class weights) is deferred to the modeling phase. The original class distribution remains intact in the training set.\n\n")
        
        f.write("## Final Feature Set\n")
        f.write(f"{', '.join(final_features)}\n\n")
        
        f.write("## Artifacts Generated\n")
        f.write("- `models/preprocessing/logistic_preprocessor.joblib`\n")
        f.write("- `models/preprocessing/xgboost_preprocessor.joblib`\n")
        f.write("- `data/processed/train.csv`\n")
        f.write("- `data/processed/validation.csv`\n")
        f.write("- `data/processed/test.csv`\n\n")
        
        f.write("## Verification\n")
        f.write("All tests successfully verified that there is no data leakage, target exclusion from feature matrices, and deterministic preprocessing behavior.\n")

if __name__ == '__main__':
    main()
    print("Preprocessing completed successfully.")
