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

from flask import Flask, jsonify, request, current_app
from model_service import ModelService
from pdf_to_text import pdf_to_text
import os
import tempfile
from sentence_transformers import util
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# append sys ../src path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from suggestion import generate_recommendations, format_final_report
from utils import filter_requirements, job_description_to_atoms
LOG_LEVEL = logging.INFO
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
        print("Yeni istek alındı similarity")
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
        
    @app.route("/explain_keywords", methods=["POST"])
    def explain_keywords():
        print("\n=== Yeni İstek Alındı: /explain_keywords ===")
        service: ModelService = current_app.config["MODEL_SERVICE"]

        cv_text = request.json.get("cv_text", "")
        cv_text = service.translate_if_needed(cv_text, target_lang="en") 
        job_text = request.json.get("job_text", "")
        atoms = job_description_to_atoms(job_text)
        real_requirements = filter_requirements(atoms)
        recommendations = generate_recommendations(real_requirements, cv_text)
        print("recommandations: ", recommendations)
        format_final_report(recommendations)


        return jsonify({"recommendations": [str(rec) for rec in recommendations]})

    return app


# Create app instance for WSGI servers (gunicorn, waitress, etc.)
app = create_app()

def main() -> None:
    """Run development server. For production use gunicorn or waitress."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()