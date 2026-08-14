import React, { useState } from 'react';
import '../styles/Dashboard.css';

export default function TransactionForm({ onSubmit, isLoading }) {
  const [formData, setFormData] = useState({
    Time: 0,
    Amount: 0,
    V1: 0, V2: 0, V3: 0, V4: 0, V5: 0, V6: 0, V7: 0, V8: 0, V9: 0, V10: 0,
    V11: 0, V12: 0, V13: 0, V14: 0, V15: 0, V16: 0, V17: 0, V18: 0, V19: 0, V20: 0,
    V21: 0, V22: 0, V23: 0, V24: 0, V25: 0, V26: 0, V27: 0, V28: 0
  });
  
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    // Allow empty string for backspacing, otherwise coerce to float
    setFormData(prev => ({
      ...prev,
      [name]: value === '' ? '' : Number(value)
    }));
  };

  const validateForm = () => {
    // Check if any required field is empty or NaN
    for (const [key, value] of Object.entries(formData)) {
      if (value === '' || isNaN(value)) {
        return `Field ${key} must be a valid number.`;
      }
      if (!isFinite(value)) {
        return `Field ${key} must be a finite number.`;
      }
    }
    if (formData.Amount < 0) {
      return 'Amount cannot be negative.';
    }
    return null;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');
    
    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }
    
    onSubmit(formData);
  };

  return (
    <form className="transaction-form" onSubmit={handleSubmit}>
      {error && <div className="form-error" role="alert">{error}</div>}
      
      <div className="form-group-section">
        <h3>Transaction Metadata</h3>
        <div className="form-grid">
          <div className="input-group">
            <label htmlFor="Time">Time</label>
            <input type="number" step="any" id="Time" name="Time" value={formData.Time} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label htmlFor="Amount">Amount ($)</label>
            <input type="number" step="any" id="Amount" name="Amount" value={formData.Amount} onChange={handleChange} required />
          </div>
        </div>
      </div>

      <div className="form-group-section">
        <h3>Anonymized Features (V1 - V14)</h3>
        <div className="form-grid col-4">
          {[...Array(14)].map((_, i) => (
            <div className="input-group" key={`V${i+1}`}>
              <label htmlFor={`V${i+1}`}>V{i+1}</label>
              <input type="number" step="any" id={`V${i+1}`} name={`V${i+1}`} value={formData[`V${i+1}`]} onChange={handleChange} required />
            </div>
          ))}
        </div>
      </div>

      <div className="form-group-section">
        <h3>Anonymized Features (V15 - V28)</h3>
        <div className="form-grid col-4">
          {[...Array(14)].map((_, i) => (
            <div className="input-group" key={`V${i+15}`}>
              <label htmlFor={`V${i+15}`}>V{i+15}</label>
              <input type="number" step="any" id={`V${i+15}`} name={`V${i+15}`} value={formData[`V${i+15}`]} onChange={handleChange} required />
            </div>
          ))}
        </div>
      </div>

      <div className="form-actions">
        <button type="submit" className="primary-button" disabled={isLoading}>
          {isLoading ? 'Analyzing...' : 'Analyze Transaction'}
        </button>
      </div>
    </form>
  );
}
