"""Tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.schemas import LoanApplication


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def valid_application():
    """Sample valid loan application"""
    return {
        "person_age": 35,
        "person_income": 50000,
        "person_home_ownership": "OWN",
        "person_emp_length": 10,
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "loan_amnt": 5000,
        "loan_int_rate": 0.08,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": 0,
        "cb_person_cred_hist_length": 8,
    }


class TestHealthCheck:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_check_model_loaded(self, client):
        """Test health check confirms model is loaded"""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestModelInfo:
    """Test model info endpoint"""

    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "threshold" in data
        assert "num_features" in data
        assert "features" in data

    def test_model_info_has_features(self, client):
        """Test model info endpoint returns features"""
        response = client.get("/model/info")
        data = response.json()
        assert len(data["features"]) > 0
        assert isinstance(data["features"], list)


class TestPredictEndpoint:
    """Test prediction endpoint"""

    def test_predict_valid_application(self, client, valid_application):
        """Test prediction with valid application"""
        response = client.post("/predict", json=valid_application)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability_default" in data
        assert "probability_non_default" in data
        assert "risk_level" in data
        assert "recommendation" in data

    def test_predict_returns_probabilities(self, client, valid_application):
        """Test prediction returns valid probabilities"""
        response = client.post("/predict", json=valid_application)
        data = response.json()
        assert 0 <= data["probability_default"] <= 1
        assert 0 <= data["probability_non_default"] <= 1
        assert (
            abs(data["probability_default"] + data["probability_non_default"] - 1.0)
            < 0.001
        )

    def test_predict_returns_valid_prediction(self, client, valid_application):
        """Test prediction returns 0 or 1"""
        response = client.post("/predict", json=valid_application)
        data = response.json()
        assert data["prediction"] in [0, 1]

    def test_predict_returns_risk_level(self, client, valid_application):
        """Test prediction returns valid risk level"""
        response = client.post("/predict", json=valid_application)
        data = response.json()
        assert data["risk_level"] in ["high_risk", "low_risk"]

    def test_predict_invalid_age(self, client, valid_application):
        """Test prediction fails with invalid age"""
        valid_application["person_age"] = 10
        response = client.post("/predict", json=valid_application)
        assert response.status_code == 422

    def test_predict_invalid_income(self, client, valid_application):
        """Test prediction fails with invalid income"""
        valid_application["person_income"] = -1000
        response = client.post("/predict", json=valid_application)
        assert response.status_code == 422

    def test_predict_missing_field(self, client, valid_application):
        """Test prediction fails with missing field"""
        del valid_application["person_age"]
        response = client.post("/predict", json=valid_application)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint"""

    def test_batch_predict_valid_applications(self, client, valid_application):
        """Test batch prediction with valid applications"""
        request_data = {"applications": [valid_application, valid_application]}
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "num_predictions" in data
        assert "predictions" in data

    def test_batch_predict_multiple_records(self, client, valid_application):
        """Test batch prediction with multiple records"""
        apps = [valid_application for _ in range(5)]
        request_data = {"applications": apps}
        response = client.post("/predict/batch", json=request_data)
        data = response.json()
        assert data["num_predictions"] == 5
        assert len(data["predictions"]) == 5

    def test_batch_predict_single_record(self, client, valid_application):
        """Test batch prediction with single record"""
        request_data = {"applications": [valid_application]}
        response = client.post("/predict/batch", json=request_data)
        data = response.json()
        assert data["num_predictions"] == 1

    def test_batch_predict_max_items(self, client, valid_application):
        """Test batch prediction with max items"""
        apps = [valid_application for _ in range(100)]
        request_data = {"applications": apps}
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["num_predictions"] == 100

    def test_batch_predict_exceeds_max_items(self, client, valid_application):
        """Test batch prediction fails when exceeding max items"""
        apps = [valid_application for _ in range(101)]
        request_data = {"applications": apps}
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list returns empty results"""
        request_data = {"applications": []}
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["num_predictions"] == 0
        assert data["predictions"] == []

    def test_batch_predict_invalid_record(self, client, valid_application):
        """Test batch prediction with invalid record"""
        invalid_app = valid_application.copy()
        invalid_app["person_age"] = 10
        request_data = {"applications": [valid_application, invalid_app]}
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422
