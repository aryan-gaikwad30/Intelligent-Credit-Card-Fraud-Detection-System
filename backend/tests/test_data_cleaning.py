import pandas as pd
import numpy as np
import os
import pytest
import sys

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.app.ml.data_cleaning import (
    validate_schema, validate_target, analyze_duplicates, 
    analyze_zero_amounts, analyze_invalid_values, clean_data
)

@pytest.fixture
def dummy_raw_data():
    return pd.DataFrame({
        'Time': [0, 1, 2, 2, 3],
        'V1': [0.1, 0.2, 0.3, 0.3, np.nan],
        'V2': [0.4, 0.5, 0.6, 0.6, 0.7],
        'V3': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V4': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V5': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V6': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V7': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V8': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V9': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V10': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V11': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V12': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V13': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V14': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V15': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V16': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V17': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V18': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V19': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V20': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V21': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V22': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V23': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V24': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V25': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V26': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V27': [0.1, 0.1, 0.1, 0.1, 0.1],
        'V28': [0.1, 0.1, 0.1, 0.1, 0.1],
        'Amount': [100.0, 0.0, 50.0, 50.0, 20.0],
        'Class': [0, 1, 0, 0, 3] # One valid 0, one valid 1, one duplicate, one invalid class
    })

def test_schema_validation(dummy_raw_data):
    valid, expected_cols = validate_schema(dummy_raw_data)
    assert valid
    assert expected_cols[0] == 'Time'
    assert expected_cols[-1] == 'Class'

def test_schema_validation_fails_on_missing_cols():
    df = pd.DataFrame({'Time': [0], 'Amount': [10], 'Class': [0]})
    with pytest.raises(ValueError, match="Schema validation failed. Missing columns"):
        validate_schema(df)

def test_target_validation(dummy_raw_data):
    res = validate_target(dummy_raw_data)
    assert 3 in res['unexpected_values']
    assert res['missing_targets'] == 0

def test_duplicate_analysis(dummy_raw_data):
    res = analyze_duplicates(dummy_raw_data)
    assert res['exact_duplicate_rows_to_remove'] == 1

def test_zero_amount_analysis(dummy_raw_data):
    res = analyze_zero_amounts(dummy_raw_data)
    assert res['zero_count'] == 1
    assert res['zero_fraud'] == 1
    assert res['non_zero_fraud'] == 0

def test_invalid_values_analysis(dummy_raw_data):
    res = analyze_invalid_values(dummy_raw_data)
    assert res['invalid_counts']['V1']['na'] == 1

def test_clean_data(dummy_raw_data):
    clean_df = clean_data(dummy_raw_data)
    assert len(clean_df) == 3  
    # 5 initial rows:
    # row 0: valid, class 0
    # row 1: valid, class 1 (zero amount)
    # row 2: valid, class 0 
    # row 3: duplicate of row 2 (dropped)
    # row 4: nan in V1, invalid class 3 (dropped)
    # Actually row 2 and 3 are exact duplicates. One is kept. So 3 rows kept.
    # Wait, row 0 kept. Row 1 kept. Row 2 kept. Row 3 dropped (duplicate). Row 4 dropped (NaN & invalid class).
    assert len(clean_df) == 3
    assert clean_df.duplicated().sum() == 0
    assert 3 not in clean_df['Class'].unique()
    assert clean_df['V1'].isna().sum() == 0
    
    # ensure zero amount is kept
    assert len(clean_df[clean_df['Amount'] == 0]) == 1
