import os
import json
import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch, ANY
from app.ml.logistic_baseline import (
    load_data,
    load_preprocessor,
    prepare_features,
    train_baseline,
    predict_probabilities,
    evaluate_model,
    save_model,
    save_metrics
)

@pytest.fixture
def mock_data():
    # Create a very small mock dataset
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'V{i}' for i in range(1, 6)])
    y_train = pd.Series(np.random.randint(0, 2, 100), name='Class')
    
    X_val = pd.DataFrame(np.random.randn(20, 5), columns=[f'V{i}' for i in range(1, 6)])
    y_val = pd.Series(np.random.randint(0, 2, 20), name='Class')
    
    return X_train, y_train, X_val, y_val

@pytest.fixture
def mock_scaler():
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    # Fit it on some random data to simulate a pre-fitted scaler
    scaler.fit(np.random.randn(100, 5))
    return scaler

def test_load_data_excludes_target():
    # Create temp csvs
    train_df = pd.DataFrame({'V1': [1, 2], 'Class': [0, 1]})
    val_df = pd.DataFrame({'V1': [3, 4], 'Class': [1, 0]})
    train_df.to_csv('temp_train.csv', index=False)
    val_df.to_csv('temp_val.csv', index=False)
    
    try:
        X_train, y_train, X_val, y_val = load_data('temp_train.csv', 'temp_val.csv')
        assert 'Class' not in X_train.columns
        assert 'Class' not in X_val.columns
        assert y_train.name == 'Class'
        assert y_val.name == 'Class'
    finally:
        os.remove('temp_train.csv')
        os.remove('temp_val.csv')

def test_prepare_features_does_not_fit(mock_data, mock_scaler):
    X_train, _, X_val, _ = mock_data
    
    # We will patch the fit method of RobustScaler to raise an error if called
    with patch.object(mock_scaler, 'fit', side_effect=Exception("fit should not be called")) as mock_fit:
        X_train_scaled, X_val_scaled = prepare_features(X_train, X_val, mock_scaler)
        
        # Verify shape
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        mock_fit.assert_not_called()

def test_test_set_not_loaded_in_baseline(mock_data, mock_scaler):
    X_train, y_train, X_val, y_val = mock_data
    
    # The main flow calls pd.read_csv. We want to ensure it NEVER is called with test.csv
    # We will mock pd.read_csv to return our mock dfs, but assert it's not called with test.csv
    def mock_read_csv(filepath, **kwargs):
        if 'test.csv' in filepath:
            raise ValueError("test.csv should NOT be loaded")
        if 'train.csv' in filepath:
            df = X_train.copy()
            df['Class'] = y_train.values
            return df
        if 'validation.csv' in filepath:
            df = X_val.copy()
            df['Class'] = y_val.values
            return df
        return pd.DataFrame()

    with patch('pandas.read_csv', side_effect=mock_read_csv):
        # execute load_data
        X_t, y_t, X_v, y_v = load_data('train.csv', 'validation.csv')
        assert len(X_t) == 100
        assert len(X_v) == 20

def test_model_training_and_predictions(mock_data):
    X_train, y_train, X_val, y_val = mock_data
    
    model = train_baseline(X_train, y_train)
    
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
    assert np.isfinite(metrics['f1'])
    assert np.isfinite(metrics['accuracy'])
    assert np.isfinite(metrics['false_positive_rate'])
    assert np.isfinite(metrics['false_negative_rate'])
    
    # Confusion matrix sums to validation size
    total_samples = metrics['true_negative'] + metrics['false_positive'] + metrics['false_negative'] + metrics['true_positive']
    assert total_samples == len(y_val)

def test_save_and_load_model_and_metrics(mock_data, tmp_path):
    X_train, y_train, X_val, y_val = mock_data
    
    model = train_baseline(X_train, y_train)
    y_prob = predict_probabilities(model, X_val)
    metrics = evaluate_model(y_val, y_prob)
    
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    
    save_model(model, str(model_path))
    save_metrics(metrics, str(metrics_path))
    
    assert model_path.exists()
    assert metrics_path.exists()
    
    loaded_model = joblib.load(model_path)
    # just checking if predict works
    loaded_model.predict(X_val)
    
    with open(metrics_path, 'r') as f:
        loaded_metrics = json.load(f)
    
    assert loaded_metrics['true_negative'] == metrics['true_negative']

def test_preprocessing_artifact_loads_cleanly():
    import sys
    import subprocess
    # Run a clean python process that only appends 'backend' to sys.path and loads the joblib
    # It must not inject FeatureEngineer into __main__
    script = (
        "import sys, os, joblib\n"
        "sys.path.append(os.path.abspath('backend'))\n"
        "pipeline = joblib.load('models/preprocessing/logistic_preprocessor.joblib')\n"
        "print('Loaded successfully:', type(pipeline))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Clean load failed: {result.stderr}"
    assert "Loaded successfully" in result.stdout

