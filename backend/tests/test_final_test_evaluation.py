import pytest
import os
import json
import pandas as pd
import numpy as np

# We mainly test that the config matches expectations 
# and the basic metric math holds up for arbitrary test arrays

def test_config_stability():
    config_path = 'models/final_model_config.json'
    assert os.path.exists(config_path), "final_model_config.json must exist"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    assert config['model_name'] == 'Phase 4 XGBoost Baseline'
    assert config['threshold'] == 0.31
    assert config['model_artifact'] == 'models/xgboost_baseline.joblib'

def test_test_data_isolation():
    # Verify that test.csv exists and has correct structure
    # but we don't fit anything to it.
    test_path = 'data/processed/test.csv'
    assert os.path.exists(test_path), "test.csv must exist"
    
    df = pd.read_csv(test_path, nrows=5)
    assert 'Class' in df.columns
    
def test_metric_math():
    from backend.app.ml.final_test_evaluation import calculate_pr_auc
    y_true = np.array([0, 1, 0, 1, 0])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    
    auc_val = calculate_pr_auc(y_true, y_probs)
    assert auc_val == 1.0
