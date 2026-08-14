import pandas as pd
import numpy as np
import os
import json
import joblib
import time
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

def load_data(val_path="data/processed/validation.csv"):
    val_df = pd.read_csv(val_path)
    X_val = val_df.drop('Class', axis=1)
    y_val = val_df['Class']
    return X_val, y_val

def calculate_pr_auc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def get_metrics_for_threshold(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    
    # Handle zero division
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'threshold': round(threshold, 2),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'FPR': float(fpr),
        'FNR': float(fnr)
    }

def run_threshold_analysis():
    print("Starting Phase 6: Threshold Analysis & Final Model Selection")
    start_time = time.time()
    
    # 1. Load Data
    X_val, y_val = load_data()
    
    val_row_count = len(y_val)
    val_fraud_count = sum(y_val == 1)
    
    print(f"Validation rows: {val_row_count}")
    print(f"Validation fraud count: {val_fraud_count}")
    
    # 2. Load Model
    model = joblib.load('models/xgboost_baseline.joblib')
    
    # 3. Generate Probabilities
    y_probs = model.predict_proba(X_val)[:, 1]
    
    pr_auc = float(calculate_pr_auc(y_val, y_probs))
    roc_auc = float(roc_auc_score(y_val, y_probs))
    
    # 4. Threshold Range (0.01 to 0.99)
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    
    results = []
    for t in thresholds:
        metrics = get_metrics_for_threshold(y_val, y_probs, t)
        results.append(metrics)
        
    df_results = pd.DataFrame(results)
    
    # 5. Extract Candidates
    baseline_metrics = next(item for item in results if item["threshold"] == 0.50)
    
    f1_max_idx = df_results['f1'].idxmax()
    f1_max_threshold = df_results.loc[f1_max_idx, 'threshold']
    
    prec_max_idx = df_results['precision'].idxmax()
    prec_max_threshold = df_results.loc[prec_max_idx, 'threshold']
    
    rec_max_idx = df_results['recall'].idxmax()
    rec_max_threshold = df_results.loc[rec_max_idx, 'threshold']
    
    # ~80% recall threshold (closest absolute difference to 0.80)
    df_results['recall_diff'] = (df_results['recall'] - 0.80).abs()
    approx_80_idx = df_results['recall_diff'].idxmin()
    approx_80_threshold = df_results.loc[approx_80_idx, 'threshold']
    
    # 6. Final Threshold Selection Rule
    eligible_df = df_results[df_results['recall'] >= 0.80].copy()
    
    if len(eligible_df) > 0:
        # Find max precision among eligible
        max_prec_val = eligible_df['precision'].max()
        best_candidates = eligible_df[eligible_df['precision'] == max_prec_val].copy()
        
        if len(best_candidates) == 1:
            selected_row = best_candidates.iloc[0]
        else:
            # Tie breaking
            best_candidates['dist_to_050'] = (best_candidates['threshold'] - 0.50).abs()
            # Sort by Recall descending, then F1 descending, then dist to 0.50 ascending
            best_candidates = best_candidates.sort_values(
                by=['recall', 'f1', 'dist_to_050'], 
                ascending=[False, False, True]
            )
            selected_row = best_candidates.iloc[0]
    else:
        print("WARNING: No threshold satisfies Recall >= 0.80. Falling back to max F1.")
        selected_row = df_results.loc[f1_max_idx]
        
    final_threshold = selected_row['threshold']
    
    # 7. Save Artifacts
    os.makedirs('docs', exist_ok=True)
    
    # Clean up temp column
    df_results = df_results.drop(columns=['recall_diff'], errors='ignore')
    
    # Save CSV
    df_results.to_csv('docs/xgboost_threshold_analysis.csv', index=False)
    
    # Create precision-recall curve plot
    os.makedirs('docs/figures', exist_ok=True)
    precisions, recalls, thresholds_prc = precision_recall_curve(y_val, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'PR Curve (AUC = {pr_auc:.4f})', color='blue', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Validation)')
    
    # Annotate final threshold on the curve
    final_prec = selected_row['precision']
    final_rec = selected_row['recall']
    plt.scatter([final_rec], [final_prec], color='red', s=100, zorder=5, label=f'Selected (Thresh={final_threshold})')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('docs/figures/xgboost_threshold_precision_recall_curve.png')
    plt.close()
    
    # Create tradeoff plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['threshold'], df_results['precision'], label='Precision', color='green')
    plt.plot(df_results['threshold'], df_results['recall'], label='Recall', color='red')
    plt.plot(df_results['threshold'], df_results['f1'], label='F1 Score', color='blue')
    plt.axvline(x=final_threshold, color='black', linestyle='--', label=f'Selected ({final_threshold})')
    plt.axvline(x=0.5, color='gray', linestyle=':', label='Baseline (0.50)')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Trade-off Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/figures/xgboost_threshold_tradeoff.png')
    plt.close()
    
    # Confusion matrix comparison
    # baseline vs selected
    b_tn, b_fp, b_fn, b_tp = baseline_metrics['TN'], baseline_metrics['FP'], baseline_metrics['FN'], baseline_metrics['TP']
    s_tn, s_fp, s_fn, s_tp = selected_row['TN'], selected_row['FP'], selected_row['FN'], selected_row['TP']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_base = np.array([[b_tn, b_fp], [b_fn, b_tp]])
    cm_sel = np.array([[s_tn, s_fp], [s_fn, s_tp]])
    
    # Simple heatmap plotting
    for i, (ax, cm, title) in enumerate(zip(axes, [cm_base, cm_sel], [f'Baseline (Thr=0.50)', f'Selected (Thr={final_threshold})'])):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Legit (0)', 'Fraud (1)'])
        ax.set_yticklabels(['Legit (0)', 'Fraud (1)'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Loop over data dimensions and create text annotations.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.tight_layout()
    plt.savefig('docs/figures/xgboost_threshold_confusion_matrix.png')
    plt.close()
    
    # 8. Create Final Config
    final_config = {
        "model_name": "Phase 4 XGBoost Baseline",
        "model_artifact": "models/xgboost_baseline.joblib",
        "threshold": float(final_threshold),
        "random_state": 42,
        "feature_columns": list(X_val.columns),
        "selection_reason": "Satisfies Recall >= 0.80 and maximizes precision",
        "validation_pr_auc": pr_auc,
        "validation_roc_auc": roc_auc,
        "validation_precision": float(selected_row['precision']),
        "validation_recall": float(selected_row['recall']),
        "validation_f1": float(selected_row['f1']),
        "validation_accuracy": float(selected_row['accuracy']),
        "TN": int(selected_row['TN']),
        "FP": int(selected_row['FP']),
        "FN": int(selected_row['FN']),
        "TP": int(selected_row['TP']),
        "FPR": float(selected_row['FPR']),
        "FNR": float(selected_row['FNR'])
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/final_model_config.json', 'w') as f:
        json.dump(final_config, f, indent=4)
        
    end_time = time.time()
    runtime = end_time - start_time
    
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print("1. Selected model: Phase 4 XGBoost Baseline")
    print(f"2. Validation row count: {val_row_count}")
    print(f"3. Validation fraud count: {val_fraud_count}")
    print(f"4. Baseline threshold: 0.50")
    print(f"5. Baseline precision: {baseline_metrics['precision']:.4f}")
    print(f"6. Baseline recall: {baseline_metrics['recall']:.4f}")
    print(f"7. Baseline F1: {baseline_metrics['f1']:.4f}")
    print(f"8. Baseline FP: {baseline_metrics['FP']}")
    print(f"9. Baseline FN: {baseline_metrics['FN']}")
    print(f"10. F1-maximizing threshold: {f1_max_threshold}")
    print(f"11. Precision-maximizing threshold: {prec_max_threshold}")
    print(f"12. Recall-maximizing threshold: {rec_max_threshold}")
    print(f"13. ~80%-recall threshold: {approx_80_threshold}")
    print(f"14. Selected final threshold: {final_threshold}")
    print(f"15. Final precision: {selected_row['precision']:.4f}")
    print(f"16. Final recall: {selected_row['recall']:.4f}")
    print(f"17. Final F1: {selected_row['f1']:.4f}")
    print(f"18. Final accuracy: {selected_row['accuracy']:.4f}")
    print(f"19. Final TN: {selected_row['TN']}")
    print(f"20. Final FP: {selected_row['FP']}")
    print(f"21. Final FN: {selected_row['FN']}")
    print(f"22. Final TP: {selected_row['TP']}")
    print(f"23. Final FPR: {selected_row['FPR']:.4f}")
    print(f"24. Final FNR: {selected_row['FNR']:.4f}")
    print(f"25. PR-AUC: {pr_auc:.4f}")
    print(f"26. ROC-AUC: {roc_auc:.4f}")
    print("27. Business interpretation: Real-world costs depend on fraud-loss vs customer-friction policies. An FN means financial loss, an FP means annoying a legitimate customer.")
    print("28. Selection rationale: Satisfies Recall >= 0.80 and maximizes precision among eligible candidates.")
    print("29. Test-set isolation confirmation: CONFIRMED. Test set not loaded.")
    print("30. Test count/result: (Check pytest output)")
    print(f"31. Runtime: {runtime:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    run_threshold_analysis()
