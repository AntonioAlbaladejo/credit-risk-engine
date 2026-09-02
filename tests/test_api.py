"""Tests for API endpoints"""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.main import app
from src.config import (
    MAX_QUESTION_LENGTH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)
from src.retriever import CorpusRetriever


@pytest.fixture(autouse=True)
def fresh_rate_limit(monkeypatch):
    """Give every test the full budget.

    The limiter keeps one counter for the whole process, so without this the
    suite spends its own allowance and later tests start seeing 429s.
    """
    monkeypatch.setattr(main, "_rate_counts", {})
    monkeypatch.setattr(main, "_rate_window_start", time.monotonic())


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
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": 0,
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

    def test_health_check_returns_503_when_model_unavailable(self, client, monkeypatch):
        """An instance that cannot serve predictions must fail its health check.

        Returning 200 here would let a load balancer route traffic to a broken
        container, which is how a fully broken deploy can look green.
        """

        def broken_predictor():
            raise RuntimeError("Model not loaded")

        monkeypatch.setattr(main, "get_predictor", broken_predictor)

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


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


# A retriever over three orthogonal chunks with a stub embedder: the endpoint
# is exercised without the `genai` group or a model download, which is how CI
# installs. The real model is covered in tests/test_retriever.py.
CORPUS_CHUNKS = [
    {
        "chunk_id": "gdpr:art_22#1",
        "citation": "GDPR, Article 22(1-4)",
        "text": "automated individual decision-making",
        "source_url": "https://example.invalid/gdpr",
        "retrieved_on": "2026-08-17",
    },
    {
        "chunk_id": "ai_act:anx_III#2",
        "citation": "AI Act, ANNEX III",
        "text": "creditworthiness of natural persons",
        "source_url": "https://example.invalid/ai-act",
        "retrieved_on": "2026-08-17",
    },
]


@pytest.fixture
def corpus(monkeypatch):
    """Install a stub-embedded retriever and hand back its aim."""
    vectors = np.eye(2, dtype=np.float32)

    def aim_at(index):
        retriever = CorpusRetriever(
            CORPUS_CHUNKS, vectors, lambda query: vectors[index]
        )
        monkeypatch.setattr(main, "_retriever", retriever)
        return retriever

    yield aim_at
    monkeypatch.setattr(main, "_retriever", None)


@pytest.fixture
def missing_corpus(monkeypatch):
    """Make the corpus impossible to build, as in an image without it."""
    monkeypatch.setattr(main, "_retriever", None)
    monkeypatch.setattr(
        main.CorpusRetriever,
        "from_files",
        classmethod(lambda cls: (_ for _ in ()).throw(FileNotFoundError("gone"))),
    )


class TestSearchRegulation:
    """Test the regulation search endpoint"""

    def test_returns_passages_with_their_citations(self, client, corpus):
        """A hit comes back shaped as the schema, carrying its provenance"""
        corpus(0)
        response = client.post(
            "/regulation/search", json={"question": "Can a decision be automated?"}
        )
        assert response.status_code == 200
        passages = response.json()["passages"]
        assert passages[0]["citation"] == "GDPR, Article 22(1-4)"
        assert passages[0]["text"] == "automated individual decision-making"
        assert passages[0]["source_url"] == "https://example.invalid/gdpr"
        assert passages[0]["retrieved_on"] == "2026-08-17"

    def test_the_ranking_score_never_leaves_the_service(self, client, corpus):
        """`score` and `chunk_id` carry unit weights a reader would misread"""
        corpus(0)
        response = client.post(
            "/regulation/search", json={"question": "Can a decision be automated?"}
        )
        assert set(response.json()["passages"][0]) == {
            "citation",
            "text",
            "source_url",
            "retrieved_on",
        }

    def test_nothing_close_enough_returns_an_empty_list_and_says_why(
        self, client, corpus
    ):
        """Silence is an answer, and the payload has to name which silence"""
        retriever = corpus(0)
        # Orthogonal to every chunk: nothing can clear the threshold.
        retriever.embed_query = lambda query: np.zeros(2, dtype=np.float32)
        response = client.post(
            "/regulation/search", json={"question": "What is our current AUC?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["passages"] == []
        assert "does not answer" in body["note"]

    def test_a_missing_corpus_is_unavailable_rather_than_a_server_error(
        self, client, missing_corpus
    ):
        """A deployment built without the corpus must say so, not throw 500"""
        response = client.post(
            "/regulation/search", json={"question": "Can a decision be automated?"}
        )
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]

    def test_scoring_does_not_require_the_corpus(
        self, client, valid_application, missing_corpus
    ):
        """The two halves of the service fail independently"""
        assert client.get("/health").json()["model_loaded"] is True
        assert client.post("/predict", json=valid_application).status_code == 200

    def test_a_corpus_that_fails_to_load_does_not_take_startup_down(
        self, missing_corpus
    ):
        """The warm-up runs at startup, so its failure must stay contained"""
        with TestClient(app) as started:
            assert started.get("/health").status_code == 200

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": ""},
            {"question": "x" * (MAX_QUESTION_LENGTH + 1)},
            {"question": "ok", "hypothetical_passage": "y" * 2001},
            {},
        ],
    )
    def test_the_bounds_are_enforced_at_the_edge(self, client, payload):
        """Unbounded text reaches an embedding model that truncates silently"""
        assert client.post("/regulation/search", json=payload).status_code == 422

    def test_two_callers_at_once_load_the_corpus_once(self, monkeypatch):
        """Each duplicate build costs another 240 MB of the task's 1 GB"""
        monkeypatch.setattr(main, "_retriever", None)
        builds = []

        def slow_build(cls):
            builds.append(1)
            time.sleep(0.05)
            return object()

        monkeypatch.setattr(main.CorpusRetriever, "from_files", classmethod(slow_build))
        threads = [threading.Thread(target=main.get_retriever) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(builds) == 1


class TestRateLimit:
    """Test the per-IP request budget"""

    def test_a_caller_over_the_budget_is_refused_with_a_retry_hint(self, client):
        """One caller must not be able to spend the task's CPU on searches"""
        for _ in range(RATE_LIMIT_REQUESTS):
            assert client.get("/").status_code == 200
        refused = client.get("/")
        assert refused.status_code == 429
        assert 0 < int(refused.headers["Retry-After"]) <= RATE_LIMIT_WINDOW_SECONDS + 1

    def test_health_is_never_limited(self, client):
        """ECS reads /health to decide the task lives; starving it restarts it"""
        for _ in range(RATE_LIMIT_REQUESTS + 5):
            client.get("/")
        assert client.get("/health").status_code == 200

    def test_each_caller_gets_its_own_budget(self):
        """A busy client must not lock everyone else out"""
        noisy = TestClient(app, client=("10.0.0.1", 5000))
        quiet = TestClient(app, client=("10.0.0.2", 5000))
        for _ in range(RATE_LIMIT_REQUESTS + 1):
            noisy.get("/")
        assert noisy.get("/").status_code == 429
        assert quiet.get("/").status_code == 200

    def test_the_budget_comes_back_when_the_window_rolls(self, client, monkeypatch):
        """A refusal is for a minute, not for the life of the task"""
        for _ in range(RATE_LIMIT_REQUESTS + 1):
            client.get("/")
        assert client.get("/").status_code == 429
        monkeypatch.setattr(
            main, "_rate_window_start", time.monotonic() - RATE_LIMIT_WINDOW_SECONDS - 1
        )
        assert client.get("/").status_code == 200
