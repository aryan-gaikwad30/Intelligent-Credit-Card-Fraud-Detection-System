import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import Dashboard from '../components/Dashboard';
import * as apiClient from '../api/client';

// Mock the API client
vi.mock('../api/client', () => ({
  checkHealth: vi.fn(),
  predictFraud: vi.fn(),
}));

describe('Fraud Detection Dashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiClient.checkHealth.mockResolvedValue({ status: 'ok', model_loaded: true });
  });

  it('renders dashboard with required fields', async () => {
    render(<Dashboard />);
    
    // Check header
    expect(screen.getByText('Intelligent Fraud Detection')).toBeInTheDocument();
    
    // Check fields
    expect(screen.getByLabelText('Time')).toBeInTheDocument();
    expect(screen.getByLabelText('Amount ($)')).toBeInTheDocument();
    
    // Check all V1-V28
    for (let i = 1; i <= 28; i++) {
      expect(screen.getByLabelText(`V${i}`)).toBeInTheDocument();
    }
  });

  it('triggers local validation when amount is negative', async () => {
    const user = userEvent.setup();
    render(<Dashboard />);
    
    const amountInput = screen.getByLabelText('Amount ($)');
    await user.clear(amountInput);
    await user.type(amountInput, '-50');
    
    const analyzeButton = screen.getByRole('button', { name: /analyze transaction/i });
    await user.click(analyzeButton);
    
    expect(screen.getByText('Amount cannot be negative.')).toBeInTheDocument();
    expect(apiClient.predictFraud).not.toHaveBeenCalled();
  });

  it('submits form and displays legitimate result correctly', async () => {
    const user = userEvent.setup();
    apiClient.predictFraud.mockResolvedValue({
      is_fraud: false,
      fraud_probability: 0.05,
      threshold: 0.31,
      model_name: 'Phase 4 XGBoost Baseline'
    });

    render(<Dashboard />);
    
    const analyzeButton = screen.getByRole('button', { name: /analyze transaction/i });
    await user.click(analyzeButton);
    
    // Wait for result
    await waitFor(() => {
      expect(screen.getByText('✅ LOW RISK')).toBeInTheDocument();
    });
    
    expect(screen.getByText('LEGITIMATE')).toBeInTheDocument();
    expect(screen.getByText('5.00%')).toBeInTheDocument(); // 0.05 * 100
    expect(screen.getByText('31.00%')).toBeInTheDocument(); // 0.31 * 100
    expect(screen.getByText(/The transaction appears normal/)).toBeInTheDocument();
  });

  it('submits form and displays fraud result correctly', async () => {
    const user = userEvent.setup();
    apiClient.predictFraud.mockResolvedValue({
      is_fraud: true,
      fraud_probability: 0.85,
      threshold: 0.31,
      model_name: 'Phase 4 XGBoost Baseline'
    });

    render(<Dashboard />);
    
    const analyzeButton = screen.getByRole('button', { name: /analyze transaction/i });
    await user.click(analyzeButton);
    
    await waitFor(() => {
      expect(screen.getByText('⚠️ HIGH RISK DETECTED')).toBeInTheDocument();
    });
    
    expect(screen.getByText('FRAUDULENT')).toBeInTheDocument();
    expect(screen.getByText('85.00%')).toBeInTheDocument();
    expect(screen.getByText(/exceeds the required risk threshold/)).toBeInTheDocument();
  });

  it('handles API error correctly without stack traces', async () => {
    const user = userEvent.setup();
    apiClient.predictFraud.mockRejectedValue(new Error('Validation error: Please ensure all transaction fields are correctly formatted numbers.'));

    render(<Dashboard />);
    
    const analyzeButton = screen.getByRole('button', { name: /analyze transaction/i });
    await user.click(analyzeButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Validation error: Please ensure all transaction fields are correctly formatted numbers/)).toBeInTheDocument();
    });
    
    // Form should still be visible, result not rendered
    expect(screen.queryByText('✅ LOW RISK')).not.toBeInTheDocument();
  });

  it('resets state when clicking reset button', async () => {
    const user = userEvent.setup();
    apiClient.predictFraud.mockResolvedValue({
      is_fraud: true,
      fraud_probability: 0.85,
      threshold: 0.31,
      model_name: 'Phase 4 XGBoost Baseline'
    });

    render(<Dashboard />);
    
    const analyzeButton = screen.getByRole('button', { name: /analyze transaction/i });
    await user.click(analyzeButton);
    
    await waitFor(() => {
      expect(screen.getByText('⚠️ HIGH RISK DETECTED')).toBeInTheDocument();
    });
    
    const resetButton = screen.getByRole('button', { name: /analyze another transaction/i });
    await user.click(resetButton);
    
    expect(screen.queryByText('⚠️ HIGH RISK DETECTED')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /analyze transaction/i })).toBeInTheDocument();
  });
});
