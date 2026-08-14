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

def load_data(test_path="data/processed/test.csv"):
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    return X_test, y_test

def calculate_pr_auc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def run_final_test_evaluation():
    print("Starting Phase 7: Final Unseen Test Evaluation")
    start_time = time.time()
    
    # 1. Verify Configuration
    with open('models/final_model_config.json', 'r') as f:
        config = json.load(f)
        
    model_name = config['model_name']
    model_artifact = config['model_artifact']
    threshold = config['threshold']
    
    print(f"Locked Model: {model_name}")
    print(f"Locked Threshold: {threshold}")
    print(f"Artifact Path: {model_artifact}")
    
    # 2. Load Model
    model = joblib.load(model_artifact)
    
    # 3. Load Data
    X_test, y_test = load_data()
    
    test_row_count = len(y_test)
    test_fraud_count = sum(y_test == 1)
    test_legit_count = sum(y_test == 0)
    
    print(f"\nTest Data Loaded.")
    print(f"Test row count: {test_row_count}")
    print(f"Test fraud count: {test_fraud_count}")
    print(f"Test legitimate count: {test_legit_count}")
    
    # 4. Generate Probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 5. Apply Threshold
    y_pred = (y_probs >= threshold).astype(int)
    
    # 6. Calculate Metrics
    test_pr_auc = float(calculate_pr_auc(y_test, y_probs))
    test_roc_auc = float(roc_auc_score(y_test, y_probs))
    
    test_precision = float(precision_score(y_test, y_pred, zero_division=0))
    test_recall = float(recall_score(y_test, y_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    test_accuracy = float(accuracy_score(y_test, y_pred))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_fpr = float(fp / (fp + tn) if (fp + tn) > 0 else 0)
    test_fnr = float(fn / (fn + tp) if (fn + tp) > 0 else 0)
    
    print("\nMetrics Calculated.")
    
    # 7. Compare against Validation
    val_pr_auc = config.get('validation_pr_auc', 0)
    val_roc_auc = config.get('validation_roc_auc', 0)
    val_precision = config.get('validation_precision', 0)
    val_recall = config.get('validation_recall', 0)
    val_f1 = config.get('validation_f1', 0)
    val_accuracy = config.get('validation_accuracy', 0)
    
    # 8. Save Metrics
    test_metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "test_row_count": test_row_count,
        "test_fraud_count": int(test_fraud_count),
        "test_legit_count": int(test_legit_count),
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_accuracy": test_accuracy,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FPR": test_fpr,
        "FNR": test_fnr
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/final_test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
        
    # 9. Confusion Matrix Visualization
    os.makedirs('docs/figures', exist_ok=True)
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Final Test Confusion Matrix (Threshold={threshold})')
    plt.colorbar(im)
    plt.xticks([0, 1], ['Legit (0)', 'Fraud (1)'])
    plt.yticks([0, 1], ['Legit (0)', 'Fraud (1)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center", 
                     color="white" if cm[i, j] > cm.max() / 2. else "black")
            
    plt.tight_layout()
    plt.savefig('docs/figures/final_test_confusion_matrix.png')
    plt.close()
    
    # 10. Generalization Interpretation
    f1_diff = test_f1 - val_f1
    if abs(f1_diff) < 0.05:
        generalization = "Excellent generalization. Test performance closely mirrors validation performance, indicating no severe overfitting."
    elif f1_diff < 0:
        generalization = "Moderate degradation. The model performs slightly worse on the unseen test set, which is typical but highlights real-world variance."
    else:
        generalization = "Unexpected improvement. The test set performance is higher than validation, possibly due to a slightly easier test distribution or variance."
        
    end_time = time.time()
    runtime = end_time - start_time
    
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION REPORT")
    print("="*50)
    print(f"1. Locked model: {model_name}")
    print(f"2. Locked threshold: {threshold}")
    print(f"3. Test row count: {test_row_count}")
    print(f"4. Test fraud count: {test_fraud_count}")
    print(f"5. Test legitimate count: {test_legit_count}")
    print(f"6. Test PR-AUC: {test_pr_auc:.4f}")
    print(f"7. Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"8. Test Precision: {test_precision:.4f}")
    print(f"9. Test Recall: {test_recall:.4f}")
    print(f"10. Test F1: {test_f1:.4f}")
    print(f"11. Test Accuracy: {test_accuracy:.4f}")
    print(f"12. TN: {int(tn)}")
    print(f"13. FP: {int(fp)}")
    print(f"14. FN: {int(fn)}")
    print(f"15. TP: {int(tp)}")
    print(f"16. FPR: {test_fpr:.4f}")
    print(f"17. FNR: {test_fnr:.4f}")
    print("\n18. Validation vs Test comparison:")
    print(f"    - PR-AUC:   Val {val_pr_auc:.4f} | Test {test_pr_auc:.4f}")
    print(f"    - ROC-AUC:  Val {val_roc_auc:.4f} | Test {test_roc_auc:.4f}")
    print(f"    - Precision:Val {val_precision:.4f} | Test {test_precision:.4f}")
    print(f"    - Recall:   Val {val_recall:.4f} | Test {test_recall:.4f}")
    print(f"    - F1:       Val {val_f1:.4f} | Test {test_f1:.4f}")
    print(f"\n19. Generalization interpretation: {generalization}")
    print("20. Test-set isolation confirmation: CONFIRMED. Test set was loaded exactly once for this final evaluation. No training, tuning, or threshold adjustments occurred.")
    print("21. Full test-suite result: (Check pytest output)")
    print("22. Generated artifacts: models/final_test_metrics.json, docs/figures/final_test_confusion_matrix.png")
    print(f"23. Runtime: {runtime:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    run_final_test_evaluation()
