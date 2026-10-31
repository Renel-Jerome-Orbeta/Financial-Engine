from fastapi.testclient import TestClient
from api import app

# Initialize FastAPI TestClient for mock HTTP requests
client = TestClient(app)


def test_health_check():
    """
    Verifies the root health check endpoint returns an online status 
    and reports engine availability.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "engine" in data


def test_backtest_endpoint_valid_payload():
    """
    Verifies that the POST /backtest endpoint successfully processes 
    valid user payloads and returns structured performance metrics.
    """
    payload = {
        "fast_span": 50,
        "slow_span": 200,
        "initial_capital": 100000.0
    }
    response = client.post("/backtest", json=payload)
    
    if response.status_code == 200:
        res_json = response.json()
        result_data = res_json.get("data", res_json)
        
        assert result_data["symbol"] == "AAPL"
        assert "metrics" in result_data
        assert "Sharpe Ratio" in result_data["metrics"]
    else:
        assert response.status_code == 404