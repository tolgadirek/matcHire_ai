"""
Tests for the Flask embedding/similarity API in main.py.

These tests monkeypatch the ModelService to avoid loading the real
SentenceTransformer model from disk. They use Flask's test client
to exercise the endpoints and validate responses and error handling.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pytest

import main


class FakeModelService:
    """Lightweight fake model service used for testing endpoints."""

    def __init__(self, model_path: str | None = None) -> None:
        # model_path argument is accepted to match the real ModelService constructor
        self.model_path = model_path

    def embed(self, text: str) -> np.ndarray:
        """
        Return deterministic embeddings for a few known inputs to allow
        precise similarity checks in tests.
        """
        if text == "x":
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if text == "y":
            return np.array([0.0, 1.0, 0.0], dtype=float)
        # generic deterministic vector based on length
        return np.array([float(len(text)), 0.0, 0.0], dtype=float)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Reuse simple cosine similarity calculation."""
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


@pytest.fixture
def client(monkeypatch) -> Any:
    """
    Create a Flask test client with the ModelService monkeypatched to the fake.

    This prevents the real SentenceTransformer model from being loaded.
    """
    # Replace the ModelService class in the main module before app creation
    monkeypatch.setattr(main, "ModelService", FakeModelService)
    app = main.create_app(model_path=None)
    app.testing = True
    with app.test_client() as client:
        yield client


def post_json(client: Any, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to POST JSON and return parsed JSON response."""
    resp = client.post(url, data=json.dumps(payload), content_type="application/json")
    return {"status": resp.status_code, "json": resp.get_json()}


def test_embed_success(client: Any) -> None:
    """POST /embed returns embedding vector for valid input."""
    result = post_json(client, "/embed", {"text": "hello"})
    assert result["status"] == 200
    emb = result["json"].get("embedding")
    assert isinstance(emb, list)
    # FakeModelService returns length-based vector for "hello" -> len=5 -> [5.0,0,0]
    assert emb == [5.0, 0.0, 0.0]


def test_embed_validation_errors(client: Any) -> None:
    """Invalid or missing 'text' field yields 400 with error message."""
    r1 = post_json(client, "/embed", {})  # missing
    assert r1["status"] == 400
    assert "error" in r1["json"]

    r2 = post_json(client, "/embed", {"text": ""})  # empty
    assert r2["status"] == 400
    assert "error" in r2["json"]


def test_similarity_same_text_returns_one(client: Any) -> None:
    """If both texts map to identical vectors, similarity should be ~1.0."""
    r = post_json(client, "/similarity", {"text1": "x", "text2": "x"})
    assert r["status"] == 200
    sim = r["json"].get("similarity")
    assert isinstance(sim, float)
    assert pytest.approx(sim, rel=1e-6) == 1.0


def test_similarity_different_texts(client: Any) -> None:
    """Known orthogonal vectors from FakeModelService yield similarity 0.0."""
    r = post_json(client, "/similarity", {"text1": "x", "text2": "y"})
    assert r["status"] == 200
    sim = r["json"].get("similarity")
    assert isinstance(sim, float)
    assert pytest.approx(sim, abs=1e-6) == 0.0


def test_similarity_validation_errors(client: Any) -> None:
    """Missing or non-string fields produce 400 responses."""
    r1 = post_json(client, "/similarity", {"text1": "x"})  # missing text2
    assert r1["status"] == 400
    assert "error" in r1["json"]

    r2 = post_json(client, "/similarity", {"text1": "", "text2": "y"})  # empty text1
    assert r2["status"] == 400
    assert "error" in r2["json"]