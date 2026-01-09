import logging
from pathlib import Path
import re
from typing import Optional
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect

LOG_LEVEL = logging.INFO
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class ModelService:
    """SentenceTransformer modelini yöneten ve yardımcı fonksiyonlar sunan servis."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve()
        self._model: Optional[SentenceTransformer] = None
        self._load_model()
        self.alpha = 1.0  # Semantic ağırlığı (1.0 = tamamen anlamsal)

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.error("Model path not found: %s", self.model_path)
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        logger.info("Loading SentenceTransformer from %s", self.model_path)
        self._model = SentenceTransformer(str(self.model_path))
        logger.info("Model loaded successfully")

    def keyword_overlap(self, cv_text: str, job_text: str) -> float:
        """Basit kelime örtüşme oranı (yedek skorlama için)"""
        cv_tokens = set(re.findall(r'\b\w+\b', cv_text.lower()))
        job_tokens = set(re.findall(r'\b\w+\b', job_text.lower()))
        if not job_tokens:
            return 0.0
        return len(cv_tokens & job_tokens) / len(job_tokens)
    
    def translate_if_needed(self, text: str, target_lang: str = "en") -> str:
        """Türkçe metinleri İngilizceye çevirir."""
        if not text or not isinstance(text, str):
            return text

        try:
            # Hızlı dil tespiti
            detected_lang = detect(text)
            if detected_lang.startswith("tr"):
                translated = GoogleTranslator(source='tr', target=target_lang).translate(text)
                return translated
        except Exception as e:
            print(f"[WARN] Çeviri hatası: {e}")

        return text

    def cosine_similarity(self, cv_text: str, job_text: str) -> float:
        """Hybrid similarity: semantic + keyword overlap"""
        cv_text = self.translate_if_needed(cv_text)
        
        # Tensor'a çevirip hesapla
        cv_emb = self._model.encode(cv_text, convert_to_tensor=True)
        job_emb = self._model.encode(job_text, convert_to_tensor=True)
        
        semantic = util.cos_sim(cv_emb, job_emb).item()
        keyword = self.keyword_overlap(cv_text, job_text)

        final_score = self.alpha * semantic + (1 - self.alpha) * keyword
        return round(float(final_score), 3)