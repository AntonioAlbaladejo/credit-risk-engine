"""Tests for API schemas"""

import pytest
from pydantic import ValidationError
from src.api.schemas import (
    LoanApplication,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheck,
    ModelInfo,
    HomeOwnershipEnum,
    LoanIntentEnum,
    LoanGradeEnum,
)


class TestHomeOwnershipEnum:
    """Test HomeOwnershipEnum"""

    def test_valid_values(self):
        """Test valid home ownership values"""
        assert HomeOwnershipEnum.RENT.value == "RENT"
        assert HomeOwnershipEnum.OWN.value == "OWN"
        assert HomeOwnershipEnum.MORTGAGE.value == "MORTGAGE"
        assert HomeOwnershipEnum.OTHER.value == "OTHER"


class TestLoanIntentEnum:
    """Test LoanIntentEnum"""

    def test_valid_values(self):
        """Test valid loan intent values"""
        assert LoanIntentEnum.PERSONAL.value == "PERSONAL"
        assert LoanIntentEnum.EDUCATION.value == "EDUCATION"
        assert LoanIntentEnum.MEDICAL.value == "MEDICAL"
        assert LoanIntentEnum.VENTURE.value == "VENTURE"
        assert LoanIntentEnum.HOMEIMPROVEMENT.value == "HOMEIMPROVEMENT"
        assert LoanIntentEnum.DEBTCONSOLIDATION.value == "DEBTCONSOLIDATION"


class TestLoanApplicationSchema:
    """Test LoanApplication schema validation"""

    @pytest.fixture
    def valid_application(self):
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

    def test_valid_application(self, valid_application):
        """Test valid loan application"""
        app = LoanApplication(**valid_application)
        assert app.person_age == 35
        assert app.person_income == 50000
        assert app.person_home_ownership == "OWN"

    def test_age_too_young(self, valid_application):
        """Test age below minimum"""
        valid_application["person_age"] = 17
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_age_too_old(self, valid_application):
        """Test age above maximum"""
        valid_application["person_age"] = 101
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_negative_income(self, valid_application):
        """Test negative income"""
        valid_application["person_income"] = -1000
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_zero_income_valid(self, valid_application):
        """Test zero income is valid"""
        valid_application["person_income"] = 0
        # Should NOT raise
        app = LoanApplication(**valid_application)
        assert app.person_income == 0

    def test_loan_amount_below_minimum(self, valid_application):
        """Test loan amount below minimum"""
        valid_application["loan_amnt"] = 100
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_loan_amount_above_maximum(self, valid_application):
        """Test loan amount above maximum"""
        valid_application["loan_amnt"] = 150000
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_invalid_home_ownership(self, valid_application):
        """Test invalid home ownership"""
        valid_application["person_home_ownership"] = "INVALID"
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_invalid_loan_intent(self, valid_application):
        """Test invalid loan intent"""
        valid_application["loan_intent"] = "INVALID"
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_invalid_loan_grade(self, valid_application):
        """Test invalid loan grade"""
        valid_application["loan_grade"] = "H"
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_interest_rate_too_high(self, valid_application):
        """Test interest rate above maximum"""
        valid_application["loan_int_rate"] = 1.5
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)

    def test_loan_percent_income_above_max(self, valid_application):
        """Test loan percent income above maximum"""
        valid_application["loan_percent_income"] = 1.5
        with pytest.raises(ValidationError):
            LoanApplication(**valid_application)


class TestPredictionResponse:
    """Test PredictionResponse schema"""

    def test_valid_prediction_response(self):
        """Test valid prediction response"""
        response = PredictionResponse(
            prediction=1,
            probability_default=0.75,
            probability_non_default=0.25,
            risk_level="high_risk",
            threshold_used=0.5,
            recommendation="Reject application",
        )
        assert response.prediction == 1
        assert response.probability_default == 0.75
        assert response.risk_level == "high_risk"


class TestBatchPredictionRequest:
    """Test BatchPredictionRequest schema"""

    @pytest.fixture
    def valid_application(self):
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

    def test_valid_batch_request(self, valid_application):
        """Test valid batch request"""
        request = BatchPredictionRequest(
            applications=[LoanApplication(**valid_application)]
        )
        assert len(request.applications) == 1

    def test_batch_request_max_items(self, valid_application):
        """Test batch request respects max items"""
        apps = [LoanApplication(**valid_application) for _ in range(100)]
        request = BatchPredictionRequest(applications=apps)
        assert len(request.applications) == 100

    def test_batch_request_exceeds_max_items(self, valid_application):
        """Test batch request raises error when exceeding max items"""
        apps = [LoanApplication(**valid_application) for _ in range(101)]
        with pytest.raises(ValidationError):
            BatchPredictionRequest(applications=apps)


class TestHealthCheck:
    """Test HealthCheck schema"""

    def test_valid_health_check(self):
        """Test valid health check"""
        check = HealthCheck(status="healthy", model_loaded=True, version="1.0.0")
        assert check.status == "healthy"
        assert check.model_loaded is True


class TestModelInfo:
    """Test ModelInfo schema"""

    def test_valid_model_info(self):
        """Test valid model info"""
        features = ["age", "income", "loan_amount"]
        info = ModelInfo(
            model_type="XGBClassifier",
            threshold=0.5,
            num_features=3,
            features=features,
        )
        assert info.model_type == "XGBClassifier"
        assert info.num_features == 3
        assert len(info.features) == 3
