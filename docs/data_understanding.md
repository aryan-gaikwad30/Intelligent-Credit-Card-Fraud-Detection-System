# Phase 1: Data Understanding

## Dataset Overview

Number of rows: 284807
Number of columns: 31

## Schema

| Column | Data Type | Non-null Count | Missing Count | Unique Count |
| ------ | --------- | -------------: | ------------: | -----------: |
| Time | float64 | 284807 | 0 | 124592 |
| V1 | float64 | 284807 | 0 | 275663 |
| V2 | float64 | 284807 | 0 | 275663 |
| V3 | float64 | 284807 | 0 | 275663 |
| V4 | float64 | 284807 | 0 | 275663 |
| V5 | float64 | 284807 | 0 | 275663 |
| V6 | float64 | 284807 | 0 | 275663 |
| V7 | float64 | 284807 | 0 | 275663 |
| V8 | float64 | 284807 | 0 | 275663 |
| V9 | float64 | 284807 | 0 | 275663 |
| V10 | float64 | 284807 | 0 | 275663 |
| V11 | float64 | 284807 | 0 | 275663 |
| V12 | float64 | 284807 | 0 | 275663 |
| V13 | float64 | 284807 | 0 | 275663 |
| V14 | float64 | 284807 | 0 | 275663 |
| V15 | float64 | 284807 | 0 | 275663 |
| V16 | float64 | 284807 | 0 | 275663 |
| V17 | float64 | 284807 | 0 | 275663 |
| V18 | float64 | 284807 | 0 | 275663 |
| V19 | float64 | 284807 | 0 | 275663 |
| V20 | float64 | 284807 | 0 | 275663 |
| V21 | float64 | 284807 | 0 | 275663 |
| V22 | float64 | 284807 | 0 | 275663 |
| V23 | float64 | 284807 | 0 | 275663 |
| V24 | float64 | 284807 | 0 | 275663 |
| V25 | float64 | 284807 | 0 | 275663 |
| V26 | float64 | 284807 | 0 | 275663 |
| V27 | float64 | 284807 | 0 | 275663 |
| V28 | float64 | 284807 | 0 | 275663 |
| Amount | float64 | 284807 | 0 | 32767 |
| Class | int64 | 284807 | 0 | 2 |

## Class Distribution

Target column: Class
Target data type: int64
Target unique values: [0, 1]

Class 0:
Count: 284315
Percentage: 99.8273%

Class 1:
Count: 492
Percentage: 0.1727%

Imbalance ratio (majority_count / minority_count): 577.88

## Missing Values

No missing values detected in the raw dataset.

## Duplicate Analysis

Total duplicate rows: 1081
Percentage of duplicate rows: 0.3796%

## Numerical Summary

Summary statistics for all numerical features:

|        |   count |            mean |           std |        min |           25% |            50% |            75% |          max |
|:-------|--------:|----------------:|--------------:|-----------:|--------------:|---------------:|---------------:|-------------:|
| Time   |  284807 | 94813.9         | 47488.1       |    0       | 54201.5       | 84692          | 139320         | 172792       |
| V1     |  284807 |     1.17516e-15 |     1.9587    |  -56.4075  |    -0.920373  |     0.0181088  |      1.31564   |      2.45493 |
| V2     |  284807 |     3.38497e-16 |     1.65131   |  -72.7157  |    -0.59855   |     0.0654856  |      0.803724  |     22.0577  |
| V3     |  284807 |    -1.37954e-15 |     1.51626   |  -48.3256  |    -0.890365  |     0.179846   |      1.0272    |      9.38256 |
| V4     |  284807 |     2.09485e-15 |     1.41587   |   -5.68317 |    -0.84864   |    -0.0198465  |      0.743341  |     16.8753  |
| V5     |  284807 |     1.02188e-15 |     1.38025   | -113.743   |    -0.691597  |    -0.0543358  |      0.611926  |     34.8017  |
| V6     |  284807 |     1.4945e-15  |     1.33227   |  -26.1605  |    -0.768296  |    -0.274187   |      0.398565  |     73.3016  |
| V7     |  284807 |    -5.62033e-16 |     1.23709   |  -43.5572  |    -0.554076  |     0.0401031  |      0.570436  |    120.589   |
| V8     |  284807 |     1.14961e-16 |     1.19435   |  -73.2167  |    -0.20863   |     0.022358   |      0.327346  |     20.0072  |
| V9     |  284807 |    -2.41419e-15 |     1.09863   |  -13.4341  |    -0.643098  |    -0.0514287  |      0.597139  |     15.595   |
| V10    |  284807 |     2.23855e-15 |     1.08885   |  -24.5883  |    -0.535426  |    -0.0929174  |      0.453923  |     23.7451  |
| V11    |  284807 |     1.72442e-15 |     1.02071   |   -4.79747 |    -0.762494  |    -0.0327574  |      0.739593  |     12.0189  |
| V12    |  284807 |    -1.24542e-15 |     0.999201  |  -18.6837  |    -0.405571  |     0.140033   |      0.618238  |      7.84839 |
| V13    |  284807 |     8.2389e-16  |     0.995274  |   -5.79188 |    -0.648539  |    -0.0135681  |      0.662505  |      7.12688 |
| V14    |  284807 |     1.21348e-15 |     0.958596  |  -19.2143  |    -0.425574  |     0.0506013  |      0.49315   |     10.5268  |
| V15    |  284807 |     4.8667e-15  |     0.915316  |   -4.49894 |    -0.582884  |     0.0480715  |      0.648821  |      8.87774 |
| V16    |  284807 |     1.43622e-15 |     0.876253  |  -14.1299  |    -0.468037  |     0.0664133  |      0.523296  |     17.3151  |
| V17    |  284807 |    -3.76818e-16 |     0.849337  |  -25.1628  |    -0.483748  |    -0.0656758  |      0.399675  |      9.25353 |
| V18    |  284807 |     9.70785e-16 |     0.838176  |   -9.49875 |    -0.49885   |    -0.00363631 |      0.500807  |      5.04107 |
| V19    |  284807 |     1.03625e-15 |     0.814041  |   -7.21353 |    -0.456299  |     0.00373482 |      0.458949  |      5.59197 |
| V20    |  284807 |     6.41868e-16 |     0.770925  |  -54.4977  |    -0.211721  |    -0.0624811  |      0.133041  |     39.4209  |
| V21    |  284807 |     1.62862e-16 |     0.734524  |  -34.8304  |    -0.228395  |    -0.0294502  |      0.186377  |     27.2028  |
| V22    |  284807 |    -3.57658e-16 |     0.725702  |  -10.9331  |    -0.54235   |     0.00678194 |      0.528554  |     10.5031  |
| V23    |  284807 |     2.61857e-16 |     0.62446   |  -44.8077  |    -0.161846  |    -0.0111929  |      0.147642  |     22.5284  |
| V24    |  284807 |     4.47391e-15 |     0.605647  |   -2.83663 |    -0.354586  |     0.0409761  |      0.439527  |      4.58455 |
| V25    |  284807 |     5.1094e-16  |     0.521278  |  -10.2954  |    -0.317145  |     0.0165935  |      0.350716  |      7.51959 |
| V26    |  284807 |     1.6861e-15  |     0.482227  |   -2.60455 |    -0.326984  |    -0.0521391  |      0.240952  |      3.51735 |
| V27    |  284807 |    -3.6614e-16  |     0.403632  |  -22.5657  |    -0.0708395 |     0.00134215 |      0.0910451 |     31.6122  |
| V28    |  284807 |    -1.22745e-16 |     0.330083  |  -15.4301  |    -0.0529598 |     0.0112438  |      0.07828   |     33.8478  |
| Amount |  284807 |    88.3496      |   250.12      |    0       |     5.6       |    22          |     77.165     |  25691.2     |
| Class  |  284807 |     0.00172749  |     0.0415272 |    0       |     0         |     0          |      0         |      1       |

Important observations:
- Features V1 to V28 appear to be standardized (mean near 0).
- Time and Amount features are on completely different scales compared to V1-V28.

## Time Analysis

- Minimum: 0.0
- Maximum: 172792.0
- Median: 84692.0
- Mean: 94813.86
- Appears continuous: Yes, representing seconds elapsed.
- Approximate duration represented by the dataset: 48.00 hours (approx 2.00 days)

## Amount Analysis

- Minimum: 0.0
- Maximum: 25691.16
- Mean: 88.3496
- Median: 22.0
- Standard Deviation: 250.1201
- Quartiles: Q1=5.6, Q2=22.0, Q3=77.16499999999999
- Number of zero-amount transactions: 1825
- Skewness: 16.9777
- The distribution appears highly skewed.

## Outlier Investigation

Statistical observations only (Potential statistical outliers based on IQR = 1.5):

- Amount potential statistical outliers: 31904 (out of 284807)
- Amount is a heavily skewed feature.
- Amount has an extreme range: 0.0 to 25691.16.

## Fraud vs Legitimate Descriptive Comparison

### Amount

#### Class = 0 (Legitimate)
- Count: 284315
- Mean: 88.2910
- Median: 22.0
- Standard Deviation: 250.1051
- Quartiles: Q1=5.65, Q3=77.05

#### Class = 1 (Fraud)
- Count: 492
- Mean: 122.2113
- Median: 9.25
- Standard Deviation: 256.6833
- Quartiles: Q1=1.0, Q3=105.89

### Time

#### Class = 0 (Legitimate)
- Count: 284315
- Mean: 94838.2023
- Median: 84711.0
- Standard Deviation: 47484.0158
- Quartiles: Q1=54230.0, Q3=139333.0

#### Class = 1 (Fraud)
- Count: 492
- Mean: 80746.8069
- Median: 75568.5
- Standard Deviation: 47835.3651
- Quartiles: Q1=41241.5, Q3=128483.0

## Data Quality Findings

### Confirmed issues
- Duplicate rows: 1081 duplicated rows found.
- Imbalanced classes: High class imbalance present.

### Things requiring further investigation
- Zero-amount transactions: 1825 transactions have an amount of 0. Are these valid authorizations or errors?
- Meaning of V1-V28 features: No semantic meaning provided, but scale suggests PCA transformations.

## Decisions Deferred

The following will be decided in later phases:
- cleaning decisions
- missing-value strategy
- outlier treatment
- scaling
- feature engineering
- class imbalance strategy
- train/test splitting
