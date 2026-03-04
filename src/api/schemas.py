from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from src.config import (
    MIN_AGE,
    MAX_AGE,
    MIN_EMP_LENGTH,
    MAX_EMP_LENGTH,
    MIN_LOAN_AMOUNT,
    MAX_LOAN_AMOUNT,
)


class HomeOwnershipEnum(str, Enum):
    """Validd values for home ownership"""

    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanIntentEnum(str, Enum):
    """Valid values for loan intent"""

    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"


class LoanGradeEnum(str, Enum):
    """Valid values for loan grade"""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class LoanApplication(BaseModel):
    """Schema for an individual loan application with raw data"""

    person_age: int = Field(..., ge=MIN_AGE, le=MAX_AGE, description="Applicant's age")
    person_income: float = Field(..., gt=0, description="Annual income in dollars")
    person_emp_length: float = Field(
        ..., ge=MIN_EMP_LENGTH, le=MAX_EMP_LENGTH, description="Years of employment"
    )
    person_home_ownership: HomeOwnershipEnum = Field(
        ..., description="Home ownership status"
    )
    loan_intent: LoanIntentEnum = Field(..., description="Purpose of the loan")
    loan_grade: LoanGradeEnum = Field(
        ..., description="Loan grade assigned by the lender"
    )
    loan_amnt: float = Field(
        ...,
        ge=MIN_LOAN_AMOUNT,
        le=MAX_LOAN_AMOUNT,
        description="Requested loan amount in dollars",
    )
    loan_int_rate: float = Field(..., ge=0, le=1, description="Interest rate")
    loan_percent_income: float = Field(
        ..., ge=0, le=1, description="Percentage of income"
    )
    cb_person_default_on_file: int = Field(
        ..., ge=0, le=1, description="Historical default record"
    )
    cb_person_cred_hist_length: int = Field(
        ..., ge=0, description="Years of credit history"
    )

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Schema for the prediction response"""

    prediction: int = Field(..., description="1=Default, 0=No Default")
    probability_default: float = Field(..., description="Probability of default")
    probability_non_default: float = Field(..., description="Probability of no default")
    risk_level: str = Field(..., description="Risk level")
    threshold_used: float = Field(..., description="Threshold used")
    recommendation: str = Field(..., description="Decision recommendation")


class BatchPredictionRequest(BaseModel):
    """Schema for batch predictions"""

    applications: List[LoanApplication] = Field(..., max_items=100)


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""

    success: bool
    num_predictions: int
    predictions: List[PredictionResponse]


class HealthCheck(BaseModel):
    """Schema for health check"""

    status: str
    model_loaded: bool
    version: str


class ModelInfo(BaseModel):
    """Schema for model information"""

    model_type: str
    threshold: float
    num_features: int
    features: List[str]
