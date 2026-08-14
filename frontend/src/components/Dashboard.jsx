import React, { useState } from 'react';
import StatusIndicator from './StatusIndicator';
import TransactionForm from './TransactionForm';
import PredictionResult from './PredictionResult';
import { predictFraud } from '../api/client';
import '../styles/Dashboard.css';

export default function Dashboard() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState('');

  const handleAnalyze = async (transactionData) => {
    setLoading(true);
    setApiError('');
    setResult(null);
    try {
      const response = await predictFraud(transactionData);
      setResult(response);
    } catch (err) {
      setApiError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setApiError('');
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-content">
          <h1>Intelligent Fraud Detection</h1>
          <p>Real-time transaction risk analysis powered by XGBoost</p>
        </div>
        <StatusIndicator />
      </header>

      <main className="dashboard-main">
        {apiError && (
          <div className="api-error-banner" role="alert">
            <strong>Analysis Failed:</strong> {apiError}
          </div>
        )}

        {!result ? (
          <TransactionForm onSubmit={handleAnalyze} isLoading={loading} />
        ) : (
          <PredictionResult result={result} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}
