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
from pdf_to_text import pdf_to_text
import os
import tempfile
from sentence_transformers import util

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
        POST JSON {"text": "..."} -> returns {"embedding": [...]}
        """
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        text = data.get("text")
        if text is None:
            return jsonify({"error": "Missing field: 'text'"}), 400
        if not isinstance(text, str):
            return jsonify({"error": "Field 'text' must be a string"}), 400
        if not text.strip():
            return jsonify({"error": "'text' cannot be empty"}), 400
        
        try:
            service: ModelService = current_app.config["MODEL_SERVICE"]
            embedding = service.embed(text)
            return jsonify({"embedding": embedding.tolist()})
        except Exception as exc:
            logger.exception("Embed failed: %s", exc)
            return jsonify({"error": "Internal server error while embedding"}), 500

    @app.route("/similarity", methods=["POST"])
    def similarity_endpoint():
        """
        Accepts:
        {
            "cvId": "...",
            "job_text": "...",
            "cv_path": "uploads/cv/.../abc.pdf"  (Node gönderebilir)
        }
        """

        data = request.get_json(silent=True) or {}

        cv_id = data.get("cvId")
        job_text = data.get("job_text")
        cv_path = data.get("cv_path")  # Node bunu gönderecek

        if not cv_path or not job_text:
            return jsonify({"error": "cv_path and job_text required"}), 400

        # PDF → metin
        try:
            cv_text = pdf_to_text(cv_path)
        except Exception as e:
            return jsonify({"error": f"PDF could not be read: {e}"}), 500

        # similarity hesapla
        service: ModelService = current_app.config["MODEL_SERVICE"]
        score = service.cosine_similarity(cv_text=cv_text, job_text=job_text)

        return jsonify({"similarity": score})
        
    @app.route("/batch_similarity", methods=["POST"])
    def batch_similarity_endpoint() -> Any:
        """
        POST JSON {
        "cv_list": ["...", "...", ...],
        "job_text": "..."
        }
        Returns: [{"cv_index":0, "similarity":0.87}, ...]
        """

        data = request.get_json(silent=True) or {}

        cv_list = data.get("cv_list")
        job_text = data.get("job_text", "")

        # --- Input validation ---
        if not isinstance(cv_list, list) or not cv_list:
            return jsonify({"error": "cv_list must be a non-empty array of strings"}), 400

        if not isinstance(job_text, str) or not job_text.strip():
            return jsonify({"error": "job_text must be a non-empty string"}), 400

        # Ensure all are strings
        if any(not isinstance(cv, str) or not cv.strip() for cv in cv_list):
            return jsonify({"error": "All CV entries must be non-empty strings"}), 400

        try:
            service: ModelService = current_app.config["MODEL_SERVICE"]
            cv_embeddings = service.embed_batch(cv_list)
            job_emb = service.embed(job_text)
            sims = util.cos_sim(cv_embeddings, job_emb).cpu().numpy().flatten()
            results = []
            for i, sim in enumerate(sims):
                results.append({
                    "cv_index": i,
                    "similarity": round(float(sim), 3)
                })

            return jsonify({"results": results})

        except Exception as exc:
            logger.exception("Failed batch similarity: %s", exc)
            return jsonify({"error": "Internal server error"}), 500
        
    @app.route("/explain_keywords", methods=["POST"])
    def explain_keywords():
        """
        POST JSON: {"cv_text": "...", "job_text": "..."}
        KeyBERT + spaCy + langdetect ile iş ilanındaki eksik anahtar kelimeleri tespit eder.
        Şimdilik sonuçları terminale basar.
        """
        data = request.get_json(silent=True) or {}
        cv_text = data.get("cv_text", "")
        job_text = data.get("job_text", "")

        if not isinstance(cv_text, str) or not cv_text.strip():
            return jsonify({"error": "cv_text must be non-empty string"}), 400
        if not isinstance(job_text, str) or not job_text.strip():
            return jsonify({"error": "job_text must be non-empty string"}), 400

        service: ModelService = current_app.config["MODEL_SERVICE"]
        missing = service.explain_missing_keywords(cv_text=cv_text, job_text=job_text)

        return jsonify({
            "message": "Eksik kelimeler terminale yazdırıldı.",
            "missing_keywords_count": len(missing)
        })

    return app


# Create app instance for WSGI servers (gunicorn, waitress, etc.)
app = create_app()

def main() -> None:
    """Run development server. For production use gunicorn or waitress."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()