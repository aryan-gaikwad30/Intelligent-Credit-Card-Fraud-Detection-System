# React Fraud Detection Dashboard

This directory (`/frontend`) contains the React-based frontend dashboard for the Intelligent Credit Card Fraud Detection System.

## Architecture

The frontend is a lightweight Single Page Application (SPA) built with:
- **React 18** for declarative UI components.
- **Vite** for rapid bundling and development.
- **Vanilla CSS** for a bespoke, premium fintech aesthetic without relying on massive UI libraries.
- **Vitest & React Testing Library** for robust, fast unit testing.

## Component Structure

- **`Dashboard.jsx`**: The main container component that manages state (loading, error, result) and controls the flow between the form and the result display.
- **`StatusIndicator.jsx`**: A header component that continually monitors the API (`/health`) and indicates whether the model is successfully loaded.
- **`TransactionForm.jsx`**: Form UI for entering the 30 strict transaction fields (Time, Amount, V1-V28). Contains client-side numeric validation to prevent obvious bad data from hitting the network.
- **`PredictionResult.jsx`**: A contextual view displaying fraud probabilities, exact thresholds, and a clear risk interpretation to the end-user.

## API Integration

All API calls are abstracted within `src/api/client.js`. 
The frontend expects a FastAPI endpoint and respects the strict separation of concerns—**no feature engineering** (like `time_of_day_sin` or `amount_log1p`) occurs on the frontend. The raw inputs are securely transmitted to the backend.

### Environment Variables
Configure the API endpoint URL via environment variables (or rely on the default).
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Running the Application

### Local Development
```bash
cd frontend
npm install
npm run dev
```

### Testing
Tests use Vitest to mock the API and ensure UI flows behave correctly without needing the real backend.
```bash
cd frontend
npm run test
```

### Production Build
```bash
cd frontend
npm run build
```
