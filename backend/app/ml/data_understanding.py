import pandas as pd
import numpy as np
import os

def analyze():
    print("Loading dataset...")
    df = pd.read_csv('Dataset/creditcard.csv')
    print("Dataset loaded successfully.")

    os.makedirs('docs', exist_ok=True)
    with open('docs/data_understanding.md', 'w', encoding='utf-8') as f:
        f.write("# Phase 1: Data Understanding\n\n")

        # 1. Dataset Shape
        f.write("## Dataset Overview\n\n")
        f.write(f"Number of rows: {df.shape[0]}\n")
        f.write(f"Number of columns: {df.shape[1]}\n\n")

        # 2. Schema
        f.write("## Schema\n\n")
        f.write("| Column | Data Type | Non-null Count | Missing Count | Unique Count |\n")
        f.write("| ------ | --------- | -------------: | ------------: | -----------: |\n")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notnull().sum()
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            f.write(f"| {col} | {dtype} | {non_null} | {missing} | {unique} |\n")
        f.write("\n")

        # 3. Target Column
        target_col = 'Class' if 'Class' in df.columns else None
        
        # 4. Class Distribution
        f.write("## Class Distribution\n\n")
        if target_col:
            f.write(f"Target column: {target_col}\n")
            f.write(f"Target data type: {df[target_col].dtype}\n")
            f.write(f"Target unique values: {df[target_col].unique().tolist()}\n\n")
            
            value_counts = df[target_col].value_counts()
            for val, count in value_counts.items():
                pct = (count / df.shape[0]) * 100
                f.write(f"Class {val}:\n")
                f.write(f"Count: {count}\n")
                f.write(f"Percentage: {pct:.4f}%\n\n")
            
            if len(value_counts) >= 2:
                majority_count = value_counts.max()
                minority_count = value_counts.min()
                imbalance_ratio = majority_count / minority_count
                f.write(f"Imbalance ratio (majority_count / minority_count): {imbalance_ratio:.2f}\n\n")

        # 5. Missing Values
        f.write("## Missing Values\n\n")
        total_missing = df.isnull().sum()
        missing_cols = total_missing[total_missing > 0]
        if missing_cols.empty:
            f.write("No missing values detected in the raw dataset.\n\n")
        else:
            for col, count in missing_cols.items():
                pct = (count / df.shape[0]) * 100
                f.write(f"- {col}: {count} missing ({pct:.4f}%)\n")
            f.write("\n")

        # 6. Duplicate Analysis
        f.write("## Duplicate Analysis\n\n")
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / df.shape[0]) * 100
        f.write(f"Total duplicate rows: {duplicate_count}\n")
        f.write(f"Percentage of duplicate rows: {duplicate_pct:.4f}%\n\n")

        # 7. Numerical Summary
        f.write("## Numerical Summary\n\n")
        f.write("Summary statistics for all numerical features:\n\n")
        num_summary = df.describe().T
        f.write(num_summary.to_markdown())
        f.write("\n\nImportant observations:\n")
        f.write("- Features V1 to V28 appear to be standardized (mean near 0).\n")
        f.write("- Time and Amount features are on completely different scales compared to V1-V28.\n\n")

        # 9. Time Analysis
        f.write("## Time Analysis\n\n")
        if 'Time' in df.columns:
            time_min = df['Time'].min()
            time_max = df['Time'].max()
            time_median = df['Time'].median()
            time_mean = df['Time'].mean()
            duration_hours = time_max / 3600
            duration_days = duration_hours / 24
            
            f.write(f"- Minimum: {time_min}\n")
            f.write(f"- Maximum: {time_max}\n")
            f.write(f"- Median: {time_median}\n")
            f.write(f"- Mean: {time_mean:.2f}\n")
            f.write(f"- Appears continuous: Yes, representing seconds elapsed.\n")
            f.write(f"- Approximate duration represented by the dataset: {duration_hours:.2f} hours (approx {duration_days:.2f} days)\n\n")

        # 10. Amount Analysis
        f.write("## Amount Analysis\n\n")
        if 'Amount' in df.columns:
            amt_min = df['Amount'].min()
            amt_max = df['Amount'].max()
            amt_mean = df['Amount'].mean()
            amt_median = df['Amount'].median()
            amt_std = df['Amount'].std()
            q1 = df['Amount'].quantile(0.25)
            q3 = df['Amount'].quantile(0.75)
            zero_amt = (df['Amount'] == 0).sum()
            skewness = df['Amount'].skew()
            
            f.write(f"- Minimum: {amt_min}\n")
            f.write(f"- Maximum: {amt_max}\n")
            f.write(f"- Mean: {amt_mean:.4f}\n")
            f.write(f"- Median: {amt_median}\n")
            f.write(f"- Standard Deviation: {amt_std:.4f}\n")
            f.write(f"- Quartiles: Q1={q1}, Q2={amt_median}, Q3={q3}\n")
            f.write(f"- Number of zero-amount transactions: {zero_amt}\n")
            f.write(f"- Skewness: {skewness:.4f}\n")
            if skewness > 1 or skewness < -1:
                f.write("- The distribution appears highly skewed.\n\n")
            else:
                f.write("- The distribution does not appear highly skewed.\n\n")

        # 12. Outlier Investigation
        f.write("## Outlier Investigation\n\n")
        f.write("Statistical observations only (Potential statistical outliers based on IQR = 1.5):\n\n")
        for col in ['Amount']:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                f.write(f"- {col} potential statistical outliers: {outliers} (out of {len(df)})\n")
                if df[col].skew() > 3:
                    f.write(f"- {col} is a heavily skewed feature.\n")
                f.write(f"- {col} has an extreme range: {df[col].min()} to {df[col].max()}.\n")
        f.write("\n")

        # 13. Fraud vs Legitimate Descriptive Comparison
        f.write("## Fraud vs Legitimate Descriptive Comparison\n\n")
        if target_col and 'Amount' in df.columns and 'Time' in df.columns:
            class_0 = df[df[target_col] == 0]
            class_1 = df[df[target_col] == 1]
            
            f.write("### Amount\n\n")
            f.write("#### Class = 0 (Legitimate)\n")
            f.write(f"- Count: {len(class_0)}\n")
            f.write(f"- Mean: {class_0['Amount'].mean():.4f}\n")
            f.write(f"- Median: {class_0['Amount'].median()}\n")
            f.write(f"- Standard Deviation: {class_0['Amount'].std():.4f}\n")
            f.write(f"- Quartiles: Q1={class_0['Amount'].quantile(0.25)}, Q3={class_0['Amount'].quantile(0.75)}\n\n")
            
            f.write("#### Class = 1 (Fraud)\n")
            f.write(f"- Count: {len(class_1)}\n")
            f.write(f"- Mean: {class_1['Amount'].mean():.4f}\n")
            f.write(f"- Median: {class_1['Amount'].median()}\n")
            f.write(f"- Standard Deviation: {class_1['Amount'].std():.4f}\n")
            f.write(f"- Quartiles: Q1={class_1['Amount'].quantile(0.25)}, Q3={class_1['Amount'].quantile(0.75)}\n\n")

            f.write("### Time\n\n")
            f.write("#### Class = 0 (Legitimate)\n")
            f.write(f"- Count: {len(class_0)}\n")
            f.write(f"- Mean: {class_0['Time'].mean():.4f}\n")
            f.write(f"- Median: {class_0['Time'].median()}\n")
            f.write(f"- Standard Deviation: {class_0['Time'].std():.4f}\n")
            f.write(f"- Quartiles: Q1={class_0['Time'].quantile(0.25)}, Q3={class_0['Time'].quantile(0.75)}\n\n")

            f.write("#### Class = 1 (Fraud)\n")
            f.write(f"- Count: {len(class_1)}\n")
            f.write(f"- Mean: {class_1['Time'].mean():.4f}\n")
            f.write(f"- Median: {class_1['Time'].median()}\n")
            f.write(f"- Standard Deviation: {class_1['Time'].std():.4f}\n")
            f.write(f"- Quartiles: Q1={class_1['Time'].quantile(0.25)}, Q3={class_1['Time'].quantile(0.75)}\n\n")

        # 14. Data Quality Findings
        f.write("## Data Quality Findings\n\n")
        f.write("### Confirmed issues\n")
        f.write(f"- Duplicate rows: {duplicate_count} duplicated rows found.\n")
        f.write(f"- Imbalanced classes: High class imbalance present.\n\n")
        
        f.write("### Things requiring further investigation\n")
        f.write(f"- Zero-amount transactions: {zero_amt} transactions have an amount of 0. Are these valid authorizations or errors?\n")
        f.write("- Meaning of V1-V28 features: No semantic meaning provided, but scale suggests PCA transformations.\n\n")

        # 15. Decisions Deferred
        f.write("## Decisions Deferred\n\n")
        f.write("The following will be decided in later phases:\n")
        f.write("- cleaning decisions\n")
        f.write("- missing-value strategy\n")
        f.write("- outlier treatment\n")
        f.write("- scaling\n")
        f.write("- feature engineering\n")
        f.write("- class imbalance strategy\n")
        f.write("- train/test splitting\n")

if __name__ == '__main__':
    analyze()
    print("Data understanding analysis completed successfully.")
