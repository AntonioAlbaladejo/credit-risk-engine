from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.config import (
    MAX_AGE,
    MAX_EMP_LENGTH,
    MAX_HYPOTHETICAL_PASSAGE_LENGTH,
    MAX_LOAN_AMOUNT,
    MAX_LOAN_INT_RATE,
    MAX_QUESTION_LENGTH,
    MIN_AGE,
    MIN_EMP_LENGTH,
    MIN_LOAN_AMOUNT,
    MIN_LOAN_INT_RATE,
)


class HomeOwnershipEnum(Enum):
    """Valid values for home ownership"""

    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanIntentEnum(Enum):
    """Valid values for loan intent"""

    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"


class LoanGradeEnum(Enum):
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
    person_income: float = Field(
        ..., ge=0, description="Annual income in dollars (0 allowed for no income)"
    )
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
    loan_int_rate: float = Field(
        ...,
        ge=MIN_LOAN_INT_RATE,
        le=MAX_LOAN_INT_RATE,
        description="Annual interest rate as a percentage, e.g. 11.5 for 11.5%",
    )
    loan_percent_income: float = Field(
        ..., ge=0, le=1, description="Percentage of income"
    )
    cb_person_default_on_file: int = Field(
        ..., ge=0, le=1, description="Historical default record"
    )

    # cb_person_cred_hist_length used to be required here. It never reached the
    # model -- it correlates 0.878 with person_age and the correlation filter
    # dropped it on every training run -- so callers were made to supply a value
    # that was always discarded. Clients still sending it are unaffected:
    # Pydantic ignores unknown fields.

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "person_age": 35,
                "person_income": 50000,
                "person_home_ownership": "OWN",
                "person_emp_length": 10,
                "loan_intent": "PERSONAL",
                "loan_grade": "A",
                "loan_amnt": 5000,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.1,
                "cb_person_default_on_file": 0,
            }
        }
    )


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

    applications: list[LoanApplication] = Field(..., max_length=100)


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""

    success: bool
    num_predictions: int
    predictions: list[PredictionResponse]


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
    features: list[str]


class RegulationSearchRequest(BaseModel):
    """Schema for a search over the regulatory corpus"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="The question, in the words the user asked it in",
    )
    hypothetical_passage: str = Field(
        "",
        max_length=MAX_HYPOTHETICAL_PASSAGE_LENGTH,
        description=(
            "The provision you would expect to find if the answer existed: "
            "two or three sentences in the register of EU legislation "
            "('shall', 'the controller'), with no invented article numbers. "
            "It is matched against the corpus in place of the question and "
            "roughly halves the passages the search misses. A wrong guess in "
            "the right register still retrieves better than the question "
            "alone, and it cannot make the search answer what it should not: "
            "`question` alone decides whether anything comes back."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Do we have to let someone contest an automated rejection?",
                "hypothetical_passage": (
                    "The data subject shall have the right to obtain human "
                    "intervention on the part of the controller, to express "
                    "his or her point of view and to contest the decision."
                ),
            }
        }
    )


class RegulationPassage(BaseModel):
    """Schema for one retrieved passage of legislation"""

    citation: str = Field(..., description="The provision this text comes from")
    text: str = Field(..., description="The passage itself, verbatim")
    source_url: str = Field(..., description="Where it was published")
    retrieved_on: str = Field(..., description="When it was ingested, ISO date")


class RegulationSearchResponse(BaseModel):
    """Schema for the regulation search response"""

    passages: list[RegulationPassage] = Field(
        ..., description="Best first. Empty when the corpus does not answer."
    )
    note: str | None = Field(
        None, description="Present only when `passages` is empty: why it is"
    )
