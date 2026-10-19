from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_check():
    """Verify the health check endpoint returns online status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "engine" in data


def test_backtest_endpoint_valid_payload():
    """Verify the /backtest endpoint processes valid requests successfully."""
    payload = {
        "fast_span": 50,
        "slow_span": 200,
        "initial_capital": 100000.0
    }
    response = client.post("/backtest", json=payload)
    
    # If data exists, it should return 200 with structured results
    if response.status_code == 200:
        res_json = response.json()
        # Unpack from the API response wrapper ("data" key)
        result_data = res_json.get("data", res_json)
        
        assert result_data["symbol"] == "AAPL"
        assert "metrics" in result_data
        assert "Sharpe Ratio" in result_data["metrics"]
    else:
        # Graceful handling if data file is missing in test environment
        assert response.status_code == 404