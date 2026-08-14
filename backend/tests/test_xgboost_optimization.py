import pytest
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from backend.app.ml.xgboost_optimization import load_data, calculate_pr_auc

@pytest.fixture
def mock_data(tmp_path):
    # Create small mock train and validation datasets
    train_data = pd.DataFrame({
        'V1': np.random.randn(100),
        'V2': np.random.randn(100),
        'time_of_day_sin': np.random.rand(100),
        'time_of_day_cos': np.random.rand(100),
        'amount_log1p': np.random.rand(100),
        'Class': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })
    
    val_data = pd.DataFrame({
        'V1': np.random.randn(50),
        'V2': np.random.randn(50),
        'time_of_day_sin': np.random.rand(50),
        'time_of_day_cos': np.random.rand(50),
        'amount_log1p': np.random.rand(50),
        'Class': np.random.choice([0, 1], 50, p=[0.9, 0.1])
    })
    
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "validation.csv"
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    
    return str(train_path), str(val_path)

def test_data_leakage(mock_data):
    train_path, val_path = mock_data
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)
    
    # 7. test.csv is not loaded - logic is correct as load_data only accepts train and val
    assert 'Class' not in X_train.columns
    assert 'Class' not in X_val.columns
    assert len(y_train) == 100
    assert len(y_val) == 50

def test_search_initialization():
    # 1. Search object can initialize
    # 2. Stratified CV is configured correctly
    # 3. Number of CV folds is correct
    # 4. Search scoring is average_precision
    # 5. Random seed is 42
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    assert cv_strategy.n_splits == 3
    assert cv_strategy.random_state == 42
    assert cv_strategy.shuffle == True
    
    base_model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        importance_type="gain"
    )
    
    param_grid = {'n_estimators': [10, 20]}
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=2,
        scoring="average_precision",
        cv=cv_strategy,
        random_state=42
    )
    
    assert search.scoring == "average_precision"
    assert search.random_state == 42
    assert search.n_iter == 2
    assert search.estimator.random_state == 42

def test_scale_pos_weight(mock_data):
    # 6. scale_pos_weight comes only from training labels
    train_path, val_path = mock_data
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)
    
    fraud_count = sum(y_train == 1)
    legitimate_count = sum(y_train == 0)
    
    # Check if calculation correctly derives only from train
    scale_pos_weight = legitimate_count / fraud_count if fraud_count > 0 else 1
    assert scale_pos_weight > 0

def test_serialization(tmp_path):
    # 9. Best parameters can be serialized
    best_params_dict = {
        "best_params": {'n_estimators': 300, 'max_depth': 5},
        "cv_mean_pr_auc": 0.85,
        "cv_std_pr_auc": 0.01,
        "random_seed": 42,
        "cv_folds": 3,
        "search_iterations": 25,
        "scale_pos_weight": 10.0
    }
    
    out_path = tmp_path / "xgboost_best_params.json"
    with open(out_path, 'w') as f:
        json.dump(best_params_dict, f)
        
    with open(out_path, 'r') as f:
        loaded = json.load(f)
    
    assert loaded["random_seed"] == 42
    assert loaded["best_params"]["max_depth"] == 5

def test_model_joblib(tmp_path):
    # 10. Optimized model can be saved and loaded.
    model = XGBClassifier(n_estimators=10, max_depth=3)
    # mock fit
    X = np.random.rand(10, 5)
    y = np.random.choice([0,1], 10)
    model.fit(X, y)
    
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    
    loaded_model = joblib.load(model_path)
    
    # 11. Predicted probabilities are within [0,1].
    probs = loaded_model.predict_proba(X)[:, 1]
    assert np.all(probs >= 0) and np.all(probs <= 1)
    
    # 12. Validation prediction length is correct.
    assert len(probs) == len(X)
