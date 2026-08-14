const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    throw new Error('Failed to connect to the fraud detection API.');
  }
}

export async function predictFraud(transactionData) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(transactionData),
    });

    if (!response.ok) {
      if (response.status === 422) {
        throw new Error('Validation error: Please ensure all transaction fields are correctly formatted numbers.');
      } else if (response.status === 503) {
        throw new Error('Service unavailable: The fraud detection model is currently not loaded.');
      } else {
        throw new Error('Internal server error occurred while analyzing the transaction.');
      }
    }
    
    return await response.json();
  } catch (error) {
    if (error.message.includes('Validation') || error.message.includes('Service') || error.message.includes('Internal')) {
      throw error;
    }
    throw new Error('Network error: Unable to reach the fraud detection API.');
  }
}
