"""
Flask REST API for SentenceTransformer embeddings and similarity.

- Loads SentenceTransformer model once on startup from "../models/smart_job_model".
- POST /embed       -> {"text": "..."}  returns {"embedding": [...]}.
- POST /similarity  -> {"text1":"...","text2":"..."} returns {"similarity": float}.

For production: run with gunicorn:
    gunicorn -w 4 -b 0.0.0.0:8000 main:app
Or use waitress:
    waitress-serve --listen=*:8000 main:app
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from flask import Flask, json, jsonify, request, current_app
from sentence_transformers import SentenceTransformer
from model_service import ModelService

# Constants
MODEL_DIR = Path(__file__).parent.joinpath("..", "models", "smart_job_model").resolve()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def create_app(model_path: Optional[Path] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        model_path: Optional path to model directory. Defaults to MODEL_DIR.

    Returns:
        Configured Flask app with model service loaded.
    """
    app = Flask(__name__)
    model_path = Path(model_path) if model_path is not None else MODEL_DIR
    # Load model service once and attach to app
    app.config["MODEL_SERVICE"] = ModelService(model_path)

    @app.route("/embed", methods=["POST"])
    def embed_endpoint() -> Any:
        """
        POST JSON {"text": "..."} -> returns {"embedding": [...]}.
        """
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Field 'text' must be a non-empty string."}), 400
        try:
            service: ModelService = current_app.config["MODEL_SERVICE"]
            emb = service.embed(text)
            return jsonify({"embedding": emb.tolist()})
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("Failed to generate embedding: %s", exc)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/similarity", methods=["POST"])
    def similarity_endpoint() -> Any:
        """
        POST JSON {"cv_text": "...", "job_text": "..."} -> returns {"similarity": float}.
        """
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        cv_text = data.get("cv_text")
        #json to string for cv text
        cv_text = json.dumps(cv_text)
        #extract cv info using gemini
        job_text = data.get("job_text")
    

        print(data)
        if not all(isinstance(t, str) for t in (cv_text, job_text)):
            return jsonify(
                {"error": "Fields 'cv_text' and 'job_text' must be provided as strings."}
            ), 400
        if not cv_text.strip() or not job_text.strip():
            return (
                jsonify({"error": "'cv_text' and 'job_text' must be non-empty strings."}),
                400,
            )
        try:
            service: ModelService = current_app.config["MODEL_SERVICE"]
            score = service.cosine_similarity(cv_text=cv_text, job_text=job_text)
            return jsonify({"similarity": score})
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("Failed to compute similarity: %s", exc)
            return jsonify({"error": "Internal server error"}), 500

    return app


# Create app instance for WSGI servers (gunicorn, waitress, etc.)
app = create_app()

def main() -> None:
    """Run development server. For production use gunicorn or waitress."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()