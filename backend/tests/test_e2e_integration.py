import pytest
import subprocess
import time
import httpx
import socket
import json
import os
import signal
from pathlib import Path

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

@pytest.fixture(scope="module")
def api_server():
    # Dynamically select a free port
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    
    # Start uvicorn
    # Make sure we run the python executable from the current env
    import sys
    env = os.environ.copy()
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for health check to pass
    max_retries = 30
    server_ready = False
    
    with httpx.Client() as client:
        for _ in range(max_retries):
            try:
                resp = client.get(f"{base_url}/api/v1/health")
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    server_ready = True
                    break
            except httpx.RequestError:
                pass
            time.sleep(0.5)
            
    if not server_ready:
        process.kill()
        raise RuntimeError(f"Server failed to start on port {port}.")
        
    yield base_url
    
    # Teardown: gracefully terminate the subprocess
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

def test_health_check(api_server):
    with httpx.Client() as client:
        response = client.get(f"{api_server}/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

def test_predict_fraud_integration(api_server):
    # Load synthetic transaction
    fixture_path = Path(__file__).parent / "fixtures" / "synthetic_transaction.json"
    with open(fixture_path, "r") as f:
        transaction = json.load(f)
        
    with httpx.Client() as client:
        response = client.post(f"{api_server}/api/v1/predict", json=transaction)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify schema
        assert "prediction" in data
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "threshold" in data
        assert "model_name" in data
        
        # Verify values
        assert data["threshold"] == 0.31
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["model_name"] == "Phase 4 XGBoost Baseline"
        
        # Verify prediction agrees with threshold logic
        if data["fraud_probability"] >= data["threshold"]:
            assert data["is_fraud"] is True
            assert data["prediction"] == "fraud"
        else:
            assert data["is_fraud"] is False
            assert data["prediction"] == "legitimate"

def test_predict_invalid_schema(api_server):
    # Test controlled error responses
    invalid_transaction = {"Time": 0.0} # Missing all other fields
    
    with httpx.Client() as client:
        response = client.post(f"{api_server}/api/v1/predict", json=invalid_transaction)
        assert response.status_code == 422

def test_playwright_frontend_integration(api_server):
    # Run the playwright test suite, passing the dynamically allocated backend URL
    # so the frontend talks to the exact backend instance we just spun up.
    import subprocess
    import sys
    from pathlib import Path
    
    frontend_dir = Path(__file__).parent.parent.parent / "frontend"
    env = os.environ.copy()
    env["VITE_API_BASE_URL"] = api_server
    
    # Run Playwright tests headlessly
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
    process = subprocess.run(
        [npx_cmd, "playwright", "test", "--project=chromium"],
        cwd=str(frontend_dir),
        env=env,
        capture_output=True,
    )
    
    stdout = process.stdout.decode('utf-8', errors='replace')
    stderr = process.stderr.decode('utf-8', errors='replace')
    
    if process.returncode != 0:
        print("Playwright E2E Tests Failed!")
        print(stdout)
        print(stderr)
        
    assert process.returncode == 0, f"Playwright E2E Tests Failed: {stdout}"
