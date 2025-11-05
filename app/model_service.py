import logging
from pathlib import Path
import re
from typing import Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer , util

LOG_LEVEL = logging.INFO

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class ModelService:
    """Load SentenceTransformer once and provide embedding utilities."""

    def __init__(self, model_path: Path) -> None:
        """
        Initialize and load the SentenceTransformer model.

        Args:
            model_path: Path to a local SentenceTransformer model directory.
        """
        self.model_path = model_path.resolve()
        self._model: Optional[SentenceTransformer] = None
        self._load_model()
        self.alpha = 1.0  # Default alpha value

    def _load_model(self) -> None:
        """Load model from disk. Raises FileNotFoundError if missing."""
        if not self.model_path.exists():
            logger.error("Model path not found: %s", self.model_path)
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        logger.info("Loading SentenceTransformer model from %s", self.model_path)
        self._model = SentenceTransformer(str(self.model_path))
        logger.info("Model loaded successfully")

    def embed(self, text: str):
        return self._model.encode(text, convert_to_tensor=True)

    def embed_batch(self, texts: List[str]):
        return self._model.encode(texts, convert_to_tensor=True)

    def keyword_overlap(self, cv_text: str, job_text: str) -> float:
        """Basit kelime örtüşme oranı"""
        cv_tokens = set(re.findall(r'\b\w+\b', cv_text.lower()))
        job_tokens = set(re.findall(r'\b\w+\b', job_text.lower()))
        if not job_tokens:
            return 0.0
        return len(cv_tokens & job_tokens) / len(job_tokens)

    def cosine_similarity(self, cv_text: str, job_text: str) -> float:
        """Hybrid similarity: semantic + keyword overlap"""


        cv_emb = self._model.encode(cv_text, convert_to_tensor=True)
        job_emb = self._model.encode(job_text, convert_to_tensor=True)
        semantic = util.cos_sim(cv_emb, job_emb).item()
        keyword = self.keyword_overlap(cv_text, job_text)

        final_score = self.alpha * semantic + (1 - self.alpha) * keyword
        return round(float(final_score), 3)         
