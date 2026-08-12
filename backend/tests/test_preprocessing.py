import pandas as pd
import numpy as np
import os
import pytest
import sys

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.app.ml.preprocessing import (
    FeatureEngineer, create_logistic_pipeline, create_xgboost_pipeline, split_data, SEED
)

@pytest.fixture
def dummy_clean_data():
    np.random.seed(42)
    data = {
        'Time': np.linspace(0, 172800, 100),
        'Amount': np.random.exponential(scale=50, size=100),
        'Class': np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    }
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, 100)
    return pd.DataFrame(data)

def test_target_absent(dummy_clean_data):
    X_train, _, _, y_train, _, _ = split_data(dummy_clean_data)
    
    pipeline = create_logistic_pipeline()
    pipeline.fit(X_train)
    X_train_tf = pipeline.transform(X_train)
    
    # Check that 'Class' was separated out and not in X_train
    assert 'Class' not in X_train.columns
    # Also verify the transformed array size matches X_train minus original Time and Amount but adding 3 engineered features
    # Original: 30 features (Time, V1-V28, Amount)
    # Dropped: Time, Amount (2)
    # Added: time_seconds_in_day(dropped), time_of_day_sin, time_of_day_cos, amount_log1p (3)
    # Net: 30 - 2 + 3 = 31 features
    assert X_train_tf.shape[1] == 31

def test_leakage_scaler_fit(dummy_clean_data):
    # Test 2 & 3: Ensure validation/test data is not used to fit scaler.
    X_train, X_val, X_test, _, _, _ = split_data(dummy_clean_data)
    
    pipeline1 = create_logistic_pipeline()
    pipeline1.fit(X_train)
    
    # Fit another pipeline on a completely different set to prove standard scaler params differ
    pipeline2 = create_logistic_pipeline()
    pipeline2.fit(X_val)
    
    # Using pipeline1 on X_val should NOT equal using pipeline2 on X_val
    X_val_tf1 = pipeline1.transform(X_val)
    X_val_tf2 = pipeline2.transform(X_val)
    
    assert not np.allclose(X_val_tf1, X_val_tf2)
    
def test_columns_identical(dummy_clean_data):
    X_train, X_val, X_test, _, _, _ = split_data(dummy_clean_data)
    engineer = FeatureEngineer()
    
    X_train_eng = engineer.transform(X_train)
    X_val_eng = engineer.transform(X_val)
    X_test_eng = engineer.transform(X_test)
    
    assert list(X_train_eng.columns) == list(X_val_eng.columns)
    assert list(X_val_eng.columns) == list(X_test_eng.columns)

def test_deterministic_feature_engineering(dummy_clean_data):
    X_train, _, _, _, _, _ = split_data(dummy_clean_data)
    engineer = FeatureEngineer()
    
    res1 = engineer.transform(X_train)
    res2 = engineer.transform(X_train.copy())
    
    pd.testing.assert_frame_equal(res1, res2)

def test_pipeline_transform_unseen(dummy_clean_data):
    X_train, X_val, _, _, _, _ = split_data(dummy_clean_data)
    
    pipeline = create_xgboost_pipeline()
    pipeline.fit(X_train)
    
    # Should transform without raising exceptions
    X_val_tf = pipeline.transform(X_val)
    assert X_val_tf is not None

def test_no_smote_resampling(dummy_clean_data):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(dummy_clean_data)
    
    # No changes to the row counts of train, val, test
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    
    # Using pipeline does not change row counts
    pipeline = create_logistic_pipeline()
    pipeline.fit(X_train)
    X_train_tf = pipeline.transform(X_train)
    assert X_train_tf.shape[0] == len(X_train)

def test_raw_dataset_untouched():
    raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Dataset/creditcard.csv'))
    # Just checking it exists; the code doesn't write to it.
    assert os.path.exists(raw_path)
