import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    accuracy_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

import sys
sys.path.append(os.path.abspath('backend'))

SEED = 42

def load_data(train_path, val_path):
    """Loads training and validation data, separating features and targets."""
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']

    X_val = val_df.drop(columns=['Class'])
    y_val = val_df['Class']

    return X_train, y_train, X_val, y_val

def load_preprocessor(filepath):
    """Loads the fitted preprocessor pipeline."""
    print(f"Loading preprocessor from {filepath}...")
    pipeline = joblib.load(filepath)
    # The pipeline is: [('engineer', FeatureEngineer()), ('scaler', RobustScaler())]
    # We explicitly extract the scaler that was fitted on training data in Phase 2B.
    scaler = pipeline.named_steps['scaler']
    return scaler

def prepare_features(X_train, X_val, scaler):
    """Applies the pre-fitted scaler to the feature matrices."""
    print("Scaling features...")
    # We DO NOT call fit() or fit_transform() here. We only use transform().
    # This prevents double preprocessing and leakage.
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    return X_train_scaled, X_val_scaled

def train_baseline(X_train, y_train):
    """Trains the Logistic Regression baseline model."""
    print("Training Logistic Regression baseline...")
    model = LogisticRegression(
        random_state=SEED,
        class_weight='balanced',
        max_iter=1000  # Increased max_iter to ensure convergence
    )
    model.fit(X_train, y_train)
    return model

def predict_probabilities(model, X):
    """Predicts fraud probabilities using the trained model."""
    return model.predict_proba(X)[:, 1]

def evaluate_model(y_true, y_prob, threshold=0.5):
    """Evaluates the model on validation data using a baseline threshold."""
    print("Evaluating model...")
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate rates safely
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

def save_model(model, filepath):
    """Saves the trained model to disk."""
    print(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def save_metrics(metrics, filepath):
    """Saves evaluation metrics to a JSON file."""
    print(f"Saving metrics to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_confusion_matrix(y_true, y_pred, filepath):
    """Generates and saves the confusion matrix plot."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Validation Confusion Matrix (Threshold 0.5)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, filepath, pr_auc):
    """Generates and saves the Precision-Recall curve."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}', color='blue')
    plt.title('Precision-Recall Curve (Validation)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_roc_curve(y_true, y_prob, filepath, roc_auc):
    """Generates and saves the ROC curve."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.title('ROC Curve (Validation)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def save_coefficients(model, feature_names, filepath):
    """Extracts and saves the model coefficients sorted by absolute magnitude."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    coefficients = model.coef_[0]
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'absolute_coefficient': np.abs(coefficients)
    })
    coef_df = coef_df.sort_values(by='absolute_coefficient', ascending=False)
    coef_df.to_csv(filepath, index=False)
    return coef_df

def main():
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/validation.csv'
    preprocessor_path = 'models/preprocessing/logistic_preprocessor.joblib'

    # 1. Load Data
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)

    # 2. Load Preprocessor (Scaler)
    scaler = load_preprocessor(preprocessor_path)

    # 3. Prepare Features (Transform only)
    X_train_scaled, X_val_scaled = prepare_features(X_train, X_val, scaler)

    # 4. Train Model
    model = train_baseline(X_train_scaled, y_train)

    # 5. Predict
    y_prob = predict_probabilities(model, X_val_scaled)

    # 6. Evaluate
    metrics = evaluate_model(y_val, y_prob, threshold=0.5)
    
    # Add metadata to metrics
    metrics['model_name'] = 'Logistic Regression'
    metrics['dataset_version'] = 'Phase 2B Output'
    metrics['random_seed'] = SEED
    
    # 7. Save Model and Metrics
    save_model(model, 'models/logistic_baseline.joblib')
    save_metrics(metrics, 'models/logistic_baseline_metrics.json')

    # 8. Visualizations
    print("Generating figures...")
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_val, y_pred, 'docs/figures/logistic_validation_confusion_matrix.png')
    plot_precision_recall_curve(y_val, y_prob, 'docs/figures/logistic_validation_precision_recall_curve.png', metrics['pr_auc'])
    plot_roc_curve(y_val, y_prob, 'docs/figures/logistic_validation_roc_curve.png', metrics['roc_auc'])

    # 9. Save Coefficients
    save_coefficients(model, X_train_scaled.columns, 'docs/logistic_coefficients.csv')

    print("Phase 3 complete.")

if __name__ == '__main__':
    # Make sure we are not importing this script directly to run the main logic, unless intended
    main()
