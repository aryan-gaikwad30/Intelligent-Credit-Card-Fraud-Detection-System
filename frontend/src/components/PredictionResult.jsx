import React from 'react';
import '../styles/Dashboard.css';

export default function PredictionResult({ result, onReset }) {
  if (!result) return null;

  const { is_fraud, fraud_probability, threshold, model_name } = result;
  
  // Do NOT hard-code threshold logic. Use what API returns.
  const percentage = (fraud_probability * 100).toFixed(2);
  const thresholdPercentage = (threshold * 100).toFixed(2);
  
  return (
    <div className={`result-container ${is_fraud ? 'fraud-alert' : 'legit-pass'}`}>
      <div className="result-header">
        <h2 className="result-title">
          {is_fraud ? '⚠️ HIGH RISK DETECTED' : '✅ LOW RISK'}
        </h2>
        <span className="result-badge">{is_fraud ? 'FRAUDULENT' : 'LEGITIMATE'}</span>
      </div>
      
      <div className="result-stats">
        <div className="stat-box">
          <span className="stat-label">Fraud Probability</span>
          <span className="stat-value">{percentage}%</span>
        </div>
        <div className="stat-box">
          <span className="stat-label">Operating Threshold</span>
          <span className="stat-value">{thresholdPercentage}%</span>
        </div>
        <div className="stat-box">
          <span className="stat-label">Active Model</span>
          <span className="stat-value model-name">{model_name}</span>
        </div>
      </div>
      
      <div className="result-interpretation">
        <p>
          {is_fraud 
            ? `The transaction probability (${percentage}%) exceeds the required risk threshold (${thresholdPercentage}%). This transaction has been flagged for immediate review.` 
            : `The transaction probability (${percentage}%) falls below the risk threshold (${thresholdPercentage}%). The transaction appears normal.`}
        </p>
      </div>
      
      <div className="result-actions">
        <button onClick={onReset} className="reset-button">Analyze Another Transaction</button>
      </div>
    </div>
  );
}
