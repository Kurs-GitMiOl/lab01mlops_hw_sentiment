from fastapi.testclient import TestClient
from inference import load_models, predict

from app import app


client = TestClient(app)


def test_predict_returns_valid_response() -> None:
    """Test /predict if returns  valid JSON response."""
    response = client.post(
        "/predict",
        json={"text": "What a great MLOps lecture, I am very satisfied"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], str)

    # Test if model return string and correct class form valid range not test if model predict right answer
    assert data["prediction"] in {"negative", "neutral", "positive"}


def test_predict_negative_text() -> None:
    """Test /predict for  negative text."""
    response = client.post(
        "/predict",
        json={"text": "Bad terrible horrible day and I am disappointed"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"negative", "neutral", "positive"}


def test_predict_strict_negative_text() -> None:
    """Test /predict for  negative text."""
    response = client.post(
        "/predict",
        json={"text": "Bad terrible horrible negative"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "negative"


def test_predict_neutral_text() -> None:
    """Test if /predict works for a neutral text."""
    response = client.post(
        "/predict",
        json={"text": "The day was okay, it was a mean day"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"negative", "neutral", "positive"}


def test_predict_strict_neutral_text() -> None:
    """Test if /predict works for a neutral text."""
    response = client.post(
        "/predict",
        json={"text": "Neutral, neutral, neutral"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "neutral"


def test_predict_rejects_missing_text_field() -> None:
    """Test validation fails when text field is empty."""
    response = client.post("/predict", json={})

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_not_string_text() -> None:
    """Test if validation fails when text is not a string."""
    response = client.post("/predict", json={"text": 123})

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_empty_text() -> None:
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_whitespace_text() -> None:
    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_model_loading() -> None:
    transformer, classifier = load_models()

    assert transformer is not None
    assert classifier is not None


def test_inference_function() -> None:
    transformer, classifier = load_models()

    result = predict(
        "I like it is very positive and amazing",
        transformer,
        classifier,
    )

    assert result in {"negative", "neutral", "positive"}
