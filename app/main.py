"""
Flask REST API for SentenceTransformer embeddings and similarity.
Production run:
    gunicorn -w 4 -b 0.0.0.0:8000 main:app
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, current_app
from model_service import ModelService
from pdf_to_text import pdf_to_text

from suggestion import generate_recommendations, format_final_report
from utils import filter_requirements, job_description_to_atoms

LOG_LEVEL = logging.INFO
# Model yolu
MODEL_DIR = Path(__file__).parent.joinpath("..", "models", "smart_job_model").resolve()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(model_path: Optional[Path] = None) -> Flask:
    app = Flask(__name__)
    model_path = Path(model_path) if model_path is not None else MODEL_DIR
    
    # Modeli bir kere yükle ve app config'e at
    app.config["MODEL_SERVICE"] = ModelService(model_path)

    @app.route("/similarity", methods=["POST"])
    def similarity_endpoint():
        """
        Body: { "cvId": "...", "job_text": "...", "cv_path": "..." }
        """
        data = request.get_json(silent=True) or {}
        job_text = data.get("job_text")
        cv_path = data.get("cv_path")

        if not cv_path or not job_text:
            return jsonify({"error": "cv_path and job_text required"}), 400

        try:
            cv_text = pdf_to_text(cv_path)
        except Exception as e:
            return jsonify({"error": f"PDF could not be read: {e}"}), 500

        service: ModelService = current_app.config["MODEL_SERVICE"]
        score = service.cosine_similarity(cv_text=cv_text, job_text=job_text)

        return jsonify({"similarity": score})
        
    @app.route("/explain_keywords", methods=["POST"])
    def explain_keywords():
        """
        Body: { "cv_text": "...", "job_text": "..." }
        """
        service: ModelService = current_app.config["MODEL_SERVICE"]

        cv_text = request.json.get("cv_text", "")
        # İngilizceye çeviri (ModelService içindeki fonksiyonu kullan)
        cv_text = service.translate_if_needed(cv_text, target_lang="en")
        
        job_text = request.json.get("job_text", "")
        
        # İş ilanı parçalama ve analiz
        atoms = job_description_to_atoms(job_text)
        real_requirements = filter_requirements(atoms)
        
        # DİKKAT: Modeli buradan parametre olarak gönderiyoruz, tekrar yüklemiyoruz!
        recommendations = generate_recommendations(
            real_requirements, 
            cv_text, 
            model=service._model
        )
        
        # Konsola rapor bas (Opsiyonel, debug için)
        format_final_report(recommendations)

        return jsonify({"recommendations": [str(rec) for rec in recommendations]})

    return app

app = create_app()

def main() -> None:
    app.run(host="0.0.0.0", port=8000, debug=False)

if __name__ == "__main__":
    main()