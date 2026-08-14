import React, { useState, useEffect } from 'react';
import { checkHealth } from '../api/client';
import '../styles/Dashboard.css';

export default function StatusIndicator() {
  const [status, setStatus] = useState('checking');

  useEffect(() => {
    let mounted = true;
    const verifyStatus = async () => {
      try {
        const res = await checkHealth();
        if (mounted) {
          setStatus(res.model_loaded ? 'connected' : 'error');
        }
      } catch (err) {
        if (mounted) {
          setStatus('disconnected');
        }
      }
    };
    verifyStatus();
    return () => { mounted = false; };
  }, []);

  return (
    <div className="status-indicator" role="status" aria-label={`API Status: ${status}`}>
      <span className={`status-dot ${status}`}></span>
      <span className="status-text">
        {status === 'checking' && 'Connecting...'}
        {status === 'connected' && 'System Online (Model Loaded)'}
        {status === 'error' && 'System Error (Model Missing)'}
        {status === 'disconnected' && 'API Disconnected'}
      </span>
    </div>
  );
}
