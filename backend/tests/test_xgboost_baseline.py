import os
import json
import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch

from app.ml.xgboost_baseline import (
    load_and_verify_data,
    calculate_scale_pos_weight,
    train_xgboost,
    predict_probabilities,
    evaluate_model,
    save_artifacts,
    extract_feature_importance,
    EXPECTED_COLUMNS
)

@pytest.fixture
def mock_data():
    np.random.seed(42)
    # create matching schema
    df_train = pd.DataFrame(np.random.randn(100, len(EXPECTED_COLUMNS)-1), columns=EXPECTED_COLUMNS[:-1])
    df_train['Class'] = np.random.randint(0, 2, 100)
    
    df_val = pd.DataFrame(np.random.randn(20, len(EXPECTED_COLUMNS)-1), columns=EXPECTED_COLUMNS[:-1])
    df_val['Class'] = np.random.randint(0, 2, 20)
    
    return df_train, df_val

def test_load_and_verify_data_success(mock_data):
    df_train, df_val = mock_data
    df_train.to_csv('temp_train_xgb.csv', index=False)
    df_val.to_csv('temp_val_xgb.csv', index=False)
    
    try:
        X_train, y_train, X_val, y_val = load_and_verify_data('temp_train_xgb.csv', 'temp_val_xgb.csv')
        assert 'Class' not in X_train.columns
        assert 'Class' not in X_val.columns
        assert y_train.name == 'Class'
        assert y_val.name == 'Class'
    finally:
        if os.path.exists('temp_train_xgb.csv'): os.remove('temp_train_xgb.csv')
        if os.path.exists('temp_val_xgb.csv'): os.remove('temp_val_xgb.csv')

def test_load_and_verify_data_schema_mismatch(mock_data):
    df_train, df_val = mock_data
    df_train_bad = df_train.drop(columns=['amount_log1p'])
    df_train_bad.to_csv('temp_train_bad.csv', index=False)
    df_val.to_csv('temp_val_good.csv', index=False)
    
    try:
        with pytest.raises(ValueError, match="Schema mismatch"):
            load_and_verify_data('temp_train_bad.csv', 'temp_val_good.csv')
    finally:
        if os.path.exists('temp_train_bad.csv'): os.remove('temp_train_bad.csv')
        if os.path.exists('temp_val_good.csv'): os.remove('temp_val_good.csv')

def test_calculate_scale_pos_weight():
    y_train = pd.Series([0, 0, 0, 1])
    spw, neg, pos = calculate_scale_pos_weight(y_train)
    assert pos == 1
    assert neg == 3
    assert spw == 3.0

def test_test_set_not_loaded_in_xgb_baseline(mock_data):
    df_train, df_val = mock_data
    
    def mock_read_csv(filepath, **kwargs):
        if 'test.csv' in filepath:
            raise ValueError("test.csv should NOT be loaded")
        if 'train.csv' in filepath:
            return df_train.copy()
        if 'validation.csv' in filepath:
            return df_val.copy()
        return pd.DataFrame()

    with patch('pandas.read_csv', side_effect=mock_read_csv):
        X_t, y_t, X_v, y_v = load_and_verify_data('train.csv', 'validation.csv')
        assert len(X_t) == 100

def test_model_training_and_predictions(mock_data):
    df_train, df_val = mock_data
    X_train = df_train.drop(columns=['Class'])
    y_train = df_train['Class']
    X_val = df_val.drop(columns=['Class'])
    y_val = df_val['Class']
    
    # ensure at least one positive class
    if y_train.sum() == 0:
        y_train.iloc[0] = 1
        
    spw, _, _ = calculate_scale_pos_weight(y_train)
    
    model = train_xgboost(X_train, y_train, spw)
    
    # Test probabilities
    y_prob = predict_probabilities(model, X_val)
    assert len(y_prob) == len(y_val)
    assert all((y_prob >= 0.0) & (y_prob <= 1.0))
    
    # Test metrics
    metrics = evaluate_model(y_val, y_prob)
    
    assert np.isfinite(metrics['pr_auc'])
    assert np.isfinite(metrics['roc_auc'])
    assert np.isfinite(metrics['precision'])
    assert np.isfinite(metrics['recall'])
    
    # Confusion matrix sums to validation size
    total_samples = metrics['true_negative'] + metrics['false_positive'] + metrics['false_negative'] + metrics['true_positive']
    assert total_samples == len(y_val)
    
    # Feature importance
    df_fi = extract_feature_importance(model, X_train.columns)
    assert len(df_fi) == len(X_train.columns)
    assert 'importance' in df_fi.columns

def test_save_and_load_model_and_metrics(mock_data, tmp_path):
    df_train, df_val = mock_data
    X_train = df_train.drop(columns=['Class'])
    y_train = df_train['Class']
    X_val = df_val.drop(columns=['Class'])
    y_val = df_val['Class']
    
    if y_train.sum() == 0:
        y_train.iloc[0] = 1
        
    spw, _, _ = calculate_scale_pos_weight(y_train)
    model = train_xgboost(X_train, y_train, spw)
    
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    
    metrics = {'true_negative': 15}
    save_artifacts(model, metrics, str(metrics_path), str(model_path))
    
    assert model_path.exists()
    assert metrics_path.exists()
    
    loaded_model = joblib.load(model_path)
    loaded_model.predict(X_val)
    
    with open(metrics_path, 'r') as f:
        loaded_metrics = json.load(f)
    
    assert loaded_metrics['true_negative'] == metrics['true_negative']
