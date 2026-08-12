import pandas as pd
import numpy as np
import os

def load_raw_data(filepath='Dataset/creditcard.csv'):
    print(f"Loading raw dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    return pd.read_csv(filepath)

def validate_schema(df):
    print("Validating schema...")
    expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    actual_columns = df.columns.tolist()
    
    missing_cols = set(expected_columns) - set(actual_columns)
    extra_cols = set(actual_columns) - set(expected_columns)
    
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing columns: {missing_cols}")
    
    return True, expected_columns

def validate_target(df):
    print("Validating target column...")
    unique_vals = df['Class'].unique()
    invalid_targets = [v for v in unique_vals if v not in [0, 1]]
    target_dtype = df['Class'].dtype
    
    return {
        'unique_values': unique_vals.tolist(),
        'target_dtype': str(target_dtype),
        'missing_targets': df['Class'].isnull().sum(),
        'unexpected_values': invalid_targets
    }

def analyze_duplicates(df):
    print("Analyzing duplicates...")
    is_duplicate = df.duplicated(keep=False)
    dup_rows = df[is_duplicate]
    total_dup_rows = len(dup_rows)
    
    dup_groups = dup_rows.groupby(list(df.columns)).size().reset_index(name='count')
    num_dup_groups = len(dup_groups)
    
    # Check for conflicting targets in rows that are identical except for Class
    features_only = df.drop(columns=['Class'])
    feature_dups = df[features_only.duplicated(keep=False)]
    conflicting_groups = feature_dups.groupby(list(features_only.columns))['Class'].nunique()
    conflicts = (conflicting_groups > 1).sum()

    dup_fraud = dup_rows['Class'].sum()
    dup_legit = total_dup_rows - dup_fraud

    return {
        'total_duplicate_rows': total_dup_rows,
        'exact_duplicate_rows_to_remove': df.duplicated().sum(),
        'num_duplicate_groups': num_dup_groups,
        'conflicting_target_groups': conflicts,
        'duplicate_fraud_count': int(dup_fraud),
        'duplicate_legit_count': int(dup_legit)
    }

def analyze_zero_amounts(df):
    print("Analyzing zero amounts...")
    zero_amts = df[df['Amount'] == 0]
    non_zero_amts = df[df['Amount'] > 0]
    
    return {
        'zero_count': len(zero_amts),
        'zero_fraud': int((zero_amts['Class'] == 1).sum()),
        'zero_legit': int((zero_amts['Class'] == 0).sum()),
        'non_zero_fraud': int((non_zero_amts['Class'] == 1).sum()),
        'non_zero_legit': int((non_zero_amts['Class'] == 0).sum())
    }

def analyze_invalid_values(df):
    print("Analyzing invalid values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = set(df.columns) - set(numeric_cols)
    
    invalid_counts = {}
    for col in numeric_cols:
        is_na = df[col].isna().sum()
        is_inf = np.isinf(df[col]).sum()
        if is_na > 0 or is_inf > 0:
            invalid_counts[col] = {'na': int(is_na), 'inf': int(is_inf)}
            
    return {
        'non_numeric_columns': list(non_numeric_cols),
        'invalid_counts': invalid_counts,
        'total_invalid_rows': sum([v['na'] + v['inf'] for v in invalid_counts.values()])
    }

def clean_data(df):
    print("Cleaning data...")
    # 1. Duplicates Policy
    # We remove exact duplicates to prevent train/test leakage.
    df_clean = df.drop_duplicates()
    
    # 2. Zero-Amount Policy
    # We retain zero-amount transactions because they can be valid authorization checks.
    
    # 3. Invalid Values Policy
    # Drop rows with NaN or Inf (though Phase 1 found none)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols)
    
    # Drop invalid targets
    df_clean = df_clean[df_clean['Class'].isin([0, 1])]
    
    return df_clean

def save_clean_data(df, filepath='data/processed/creditcard_clean.csv'):
    print(f"Saving clean data to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath

def generate_cleaning_report(raw_df, clean_df, schema_res, target_res, dup_res, zero_res, inv_res, report_path='docs/data_cleaning.md'):
    print(f"Generating report at {report_path}...")
    
    raw_rows = len(raw_df)
    clean_rows = len(clean_df)
    raw_cols = raw_df.shape[1]
    clean_cols = clean_df.shape[1]
    
    raw_fraud = raw_df['Class'].sum()
    clean_fraud = clean_df['Class'].sum()
    
    raw_fraud_pct = (raw_fraud / raw_rows) * 100
    clean_fraud_pct = (clean_fraud / clean_rows) * 100
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 2A: Data Cleaning & Validation\n\n")
        
        f.write("## 1. Cleaning Objective\n")
        f.write("Validate the dataset schema, investigate duplicates, zero amounts, and invalid values, and apply data-cleaning policies to produce a clean, reproducible dataset without modifying the raw source.\n\n")
        
        f.write("## 2. Raw Dataset\n")
        f.write(f"- Rows: {raw_rows}\n")
        f.write(f"- Columns: {raw_cols}\n")
        f.write(f"- Fraud transactions: {raw_fraud} ({raw_fraud_pct:.4f}%)\n")
        f.write(f"- Legitimate transactions: {raw_rows - raw_fraud}\n\n")
        
        f.write("## 3. Schema Validation\n")
        f.write(f"- Validation passed: All expected columns are present.\n")
        f.write(f"- Target unique values: {target_res['unique_values']}\n")
        f.write(f"- Unexpected target values: {target_res['unexpected_values']}\n")
        f.write(f"- Missing target values: {target_res['missing_targets']}\n\n")
        
        f.write("## 4. Duplicate Investigation\n")
        f.write(f"- Total rows involved in duplicate groups: {dup_res['total_duplicate_rows']}\n")
        f.write(f"- Exact duplicate rows that can be removed: {dup_res['exact_duplicate_rows_to_remove']}\n")
        f.write(f"- Conflicting target groups (identical features, different targets): {dup_res['conflicting_target_groups']}\n")
        f.write(f"- Fraud vs Legitimate among duplicates: {dup_res['duplicate_fraud_count']} fraud, {dup_res['duplicate_legit_count']} legitimate.\n")
        f.write("- **Decision:** REMOVE exact duplicates.\n")
        f.write("- **Reasoning:** Exact duplicates across all 31 columns provide no new information and can cause train/test data leakage (artificially inflating evaluation metrics if a duplicate is in both sets).\n\n")
        
        f.write("## 5. Zero-Amount Investigation\n")
        f.write(f"- Total zero-amount transactions: {zero_res['zero_count']}\n")
        f.write(f"- Fraud vs Legitimate (Zero Amount): {zero_res['zero_fraud']} fraud, {zero_res['zero_legit']} legitimate.\n")
        f.write(f"- Fraud vs Legitimate (Non-Zero Amount): {zero_res['non_zero_fraud']} fraud, {zero_res['non_zero_legit']} legitimate.\n")
        f.write("- **Decision:** RETAIN zero-amount transactions.\n")
        f.write("- **Reasoning:** Zero-amount transactions are often valid authorization checks (e.g., verifying a card before a recurring charge). There is no evidence they are corrupted records, and they even contain some fraudulent attempts.\n\n")
        
        f.write("## 6. Numerical Validation\n")
        f.write(f"- Non-numeric columns: {inv_res['non_numeric_columns']}\n")
        f.write(f"- Missing or infinite values found: {inv_res['invalid_counts']}\n\n")
        
        f.write("## 7. Outlier Policy\n")
        f.write("Outliers (such as the highly skewed Amount) were NOT removed in this phase. Statistical outliers are not automatically invalid or erroneous transactions. Handling of these distributions will be considered during the preprocessing/feature engineering phase.\n\n")
        
        f.write("## 8. Cleaning Rules\n")
        f.write("1. Drop exact duplicate rows to prevent data leakage.\n")
        f.write("2. Retain zero-amount transactions.\n")
        f.write("3. Drop rows with non-finite values (NaN, Inf) or invalid Target classes (not 0 or 1).\n\n")
        
        f.write("## 9. Before vs After\n\n")
        f.write("| Metric | Raw | Cleaned |\n")
        f.write("| --- | ---: | ---: |\n")
        f.write(f"| Rows | {raw_rows} | {clean_rows} |\n")
        f.write(f"| Columns | {raw_cols} | {clean_cols} |\n")
        f.write(f"| Class 0 | {raw_rows - raw_fraud} | {clean_rows - clean_fraud} |\n")
        f.write(f"| Class 1 | {raw_fraud} | {clean_fraud} |\n")
        f.write(f"| Fraud % | {raw_fraud_pct:.4f}% | {clean_fraud_pct:.4f}% |\n")
        f.write(f"| Duplicate rows | {dup_res['exact_duplicate_rows_to_remove']} | {clean_df.duplicated().sum()} |\n")
        f.write(f"| Zero-amount rows | {zero_res['zero_count']} | {len(clean_df[clean_df['Amount'] == 0])} |\n\n")
        
        f.write("## 10. Final Dataset\n")
        f.write("- Location: `data/processed/creditcard_clean.csv`\n")
        f.write(f"- Dimensions: {clean_rows} rows x {clean_cols} columns\n\n")
        
        f.write("## 11. Data Leakage Considerations\n")
        f.write("Removing exact duplicates ensures that identical records do not appear in both the training and test sets. The cleaning process evaluates row validity individually and does not compute any aggregate statistics (like means or scalers) that would leak global target information.\n\n")
        
        f.write("## 12. Deferred Decisions\n")
        f.write("The following are NOT decided yet:\n")
        f.write("- feature engineering\n")
        f.write("- scaling\n")
        f.write("- train/validation/test splitting\n")
        f.write("- class imbalance strategy\n")
        f.write("- SMOTE/resampling\n")
        f.write("- model selection\n")
        f.write("- threshold selection\n")
        
def main():
    raw_filepath = 'Dataset/creditcard.csv'
    clean_filepath = 'data/processed/creditcard_clean.csv'
    
    # 1. Load Data
    raw_df = load_raw_data(raw_filepath)
    
    # 2. Schema Validation
    _, schema_res = validate_schema(raw_df)
    
    # 3. Target Validation
    target_res = validate_target(raw_df)
    
    # 4. Duplicate Investigation
    dup_res = analyze_duplicates(raw_df)
    
    # 5. Zero-Amount Investigation
    zero_res = analyze_zero_amounts(raw_df)
    
    # 6. Numerical Validation
    inv_res = analyze_invalid_values(raw_df)
    
    # 7. Clean Data
    clean_df = clean_data(raw_df)
    
    # 8. Save Clean Data
    save_clean_data(clean_df, clean_filepath)
    
    # 9. Generate Report
    generate_cleaning_report(raw_df, clean_df, schema_res, target_res, dup_res, zero_res, inv_res)

if __name__ == '__main__':
    main()
    print("Data cleaning completed successfully.")
