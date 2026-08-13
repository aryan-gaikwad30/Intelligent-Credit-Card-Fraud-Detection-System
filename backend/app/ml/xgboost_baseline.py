import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    accuracy_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

SEED = 42

EXPECTED_COLUMNS = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'time_of_day_sin', 'time_of_day_cos', 'amount_log1p', 'Class'
]

def load_and_verify_data(train_path, val_path):
    print("Loading and verifying data schema...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    for name, df in [('train', train_df), ('validation', val_df)]:
        # Verify columns exist and match exactly in order
        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Schema mismatch in {name} dataset. Expected {EXPECTED_COLUMNS}, got {list(df.columns)}")
        
        # Verify no NaN or Inf
        if df.isnull().values.any():
            raise ValueError(f"NaN values found in {name} dataset.")
        if np.isinf(df.values).any():
            raise ValueError(f"Infinite values found in {name} dataset.")

    X_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']

    X_val = val_df.drop(columns=['Class'])
    y_val = val_df['Class']

    return X_train, y_train, X_val, y_val

def calculate_scale_pos_weight(y_train):
    print("Calculating scale_pos_weight from training data...")
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    if positive_count == 0:
        raise ValueError("No positive class samples in training data.")
    scale_pos_weight = negative_count / positive_count
    print(f"Negative: {negative_count}, Positive: {positive_count}, Ratio: {scale_pos_weight}")
    return scale_pos_weight, negative_count, positive_count

def train_xgboost(X_train, y_train, scale_pos_weight):
    print("Training XGBoost baseline...")
    # As explicitly instructed: no eval_set, no tuning.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        importance_type='gain',
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def predict_probabilities(model, X):
    return model.predict_proba(X)[:, 1]

def evaluate_model(y_true, y_prob, threshold=0.5):
    print("Evaluating XGBoost model...")
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
        'false_positive_rate': fpr,
        'false_negative_rate': fnr
    }
    return metrics

def calculate_improvements(xgb_metrics, lr_metrics_path):
    print("Loading Logistic Regression metrics for comparison...")
    if not os.path.exists(lr_metrics_path):
        print("LR metrics not found, skipping comparison.")
        return {}
    
    with open(lr_metrics_path, 'r') as f:
        lr_metrics = json.load(f)
        
    xgb_pr_auc = xgb_metrics['pr_auc']
    lr_pr_auc = lr_metrics['pr_auc']
    xgb_roc_auc = xgb_metrics['roc_auc']
    lr_roc_auc = lr_metrics['roc_auc']
    
    pr_auc_abs_change = xgb_pr_auc - lr_pr_auc
    pr_auc_rel_change = pr_auc_abs_change / lr_pr_auc if lr_pr_auc > 0 else 0.0
    
    roc_auc_abs_change = xgb_roc_auc - lr_roc_auc
    roc_auc_rel_change = roc_auc_abs_change / lr_roc_auc if lr_roc_auc > 0 else 0.0
    
    comparison = {
        'lr_pr_auc': lr_pr_auc,
        'xgb_pr_auc': xgb_pr_auc,
        'pr_auc_abs_change': pr_auc_abs_change,
        'pr_auc_rel_change': pr_auc_rel_change,
        'lr_roc_auc': lr_roc_auc,
        'xgb_roc_auc': xgb_roc_auc,
        'roc_auc_abs_change': roc_auc_abs_change,
        'roc_auc_rel_change': roc_auc_rel_change,
        'lr_precision': lr_metrics['precision'],
        'lr_recall': lr_metrics['recall'],
        'lr_f1': lr_metrics['f1'],
        'lr_accuracy': lr_metrics['accuracy'],
        'lr_fp': lr_metrics['false_positive'],
        'lr_fn': lr_metrics['false_negative'],
        'lr_fpr': lr_metrics['false_positive_rate'],
        'lr_fnr': lr_metrics['false_negative_rate']
    }
    return comparison

def save_artifacts(model, metrics, metrics_path, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def generate_visualizations(y_true, y_prob, metrics, comparison, feature_importance_df, lr_metrics_path):
    print("Generating figures...")
    os.makedirs('docs/figures', exist_ok=True)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('XGBoost Validation Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('docs/figures/xgboost_validation_confusion_matrix.png')
    plt.close()

    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"XGBoost PR-AUC = {metrics['pr_auc']:.4f}", color='blue')
    plt.title('Precision-Recall Curve (XGBoost Validation)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/figures/xgboost_validation_precision_recall_curve.png')
    plt.close()

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"XGBoost ROC-AUC = {metrics['roc_auc']:.4f}", color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('ROC Curve (XGBoost Validation)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/figures/xgboost_validation_roc_curve.png')
    plt.close()

    # 4. Feature Importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis', hue='feature', legend=False)
    plt.title('Top 15 XGBoost Feature Importances (Gain)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('docs/figures/xgboost_feature_importance.png')
    plt.close()
    
    # 5. Model Comparison
    if comparison:
        metrics_names = ['PR-AUC', 'ROC-AUC']
        lr_vals = [comparison['lr_pr_auc'], comparison['lr_roc_auc']]
        xgb_vals = [comparison['xgb_pr_auc'], comparison['xgb_roc_auc']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - width/2, lr_vals, width, label='Logistic Regression', color='lightblue')
        ax.bar(x + width/2, xgb_vals, width, label='XGBoost', color='blue')
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        
        for i, (l, x_v) in enumerate(zip(lr_vals, xgb_vals)):
            ax.text(i - width/2, l + 0.01, f"{l:.3f}", ha='center', va='bottom', fontsize=10)
            ax.text(i + width/2, x_v + 0.01, f"{x_v:.3f}", ha='center', va='bottom', fontsize=10)
            
        plt.tight_layout()
        plt.savefig('docs/figures/model_comparison.png')
        plt.close()

def extract_feature_importance(model, feature_names):
    # Retrieve feature importance using 'gain' which was set in XGBClassifier
    importances = model.feature_importances_
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    df = df.sort_values(by='importance', ascending=False)
    os.makedirs('docs', exist_ok=True)
    df.to_csv('docs/xgboost_feature_importance.csv', index=False)
    return df

def main():
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/validation.csv'
    lr_metrics_path = 'models/logistic_baseline_metrics.json'

    # 1. Verify and Load
    X_train, y_train, X_val, y_val = load_and_verify_data(train_path, val_path)

    # 2. Scale pos weight
    scale_pos_weight, neg_count, pos_count = calculate_scale_pos_weight(y_train)

    # 3. Train
    model = train_xgboost(X_train, y_train, scale_pos_weight)

    # 4. Predict
    y_prob = predict_probabilities(model, X_val)

    # 5. Evaluate
    metrics = evaluate_model(y_val, y_prob, threshold=0.5)
    
    # 6. Compare
    comparison = calculate_improvements(metrics, lr_metrics_path)
    
    # Enrich metrics
    metrics['model_name'] = 'XGBoost Baseline'
    metrics['xgboost_version'] = xgboost.__version__
    metrics['dataset_version'] = 'Phase 2B Output'
    metrics['random_seed'] = SEED
    metrics['n_estimators'] = 300
    metrics['max_depth'] = 5
    metrics['learning_rate'] = 0.05
    metrics['subsample'] = 0.8
    metrics['colsample_bytree'] = 0.8
    metrics['scale_pos_weight'] = scale_pos_weight
    metrics['training_positive_count'] = pos_count
    metrics['training_negative_count'] = neg_count
    metrics['importance_type'] = 'gain'
    
    metrics.update(comparison)

    # 7. Feature Importance
    feature_importance_df = extract_feature_importance(model, X_train.columns)

    # 8. Visualizations
    generate_visualizations(y_val, y_prob, metrics, comparison, feature_importance_df, lr_metrics_path)

    # 9. Save
    save_artifacts(model, metrics, 'models/xgboost_baseline_metrics.json', 'models/xgboost_baseline.joblib')

    print("Phase 4 complete.")

if __name__ == '__main__':
    main()
