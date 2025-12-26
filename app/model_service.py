import logging
from pathlib import Path
import re
from typing import Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer , util
from deep_translator import GoogleTranslator
from langdetect import detect
from keybert import KeyBERT
from pdf_to_text import pdf_to_text
import spacy

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

    def keyword_overlap(self, cv_text: str, job_text: str) -> float:
        """Basit kelime örtüşme oranı"""
        cv_tokens = set(re.findall(r'\b\w+\b', cv_text.lower()))
        job_tokens = set(re.findall(r'\b\w+\b', job_text.lower()))
        if not job_tokens:
            return 0.0
        return len(cv_tokens & job_tokens) / len(job_tokens)
    
    def translate_if_needed(self, text: str, target_lang: str = "en") -> str:
        """
        Türkçe metinleri otomatik olarak İngilizceye çevirir.
        Diğer dillerdeyse olduğu gibi döner.
        """
        if not text or not isinstance(text, str):
            return text

        try:
            detected_lang = detect(text)
            if detected_lang.startswith("tr"):
                translated = GoogleTranslator(source='tr', target=target_lang).translate(text)
                print("[INFO] Türkçe metin İngilizceye çevrildi.")
                return translated
        except Exception as e:
            print(f"[WARN] Çeviri yapılırken hata oluştu: {e}")

        # Zaten İngilizce veya başka dilse çeviri yapma
        return text

    def cosine_similarity(self, cv_text: str, job_text: str) -> float:
        """Hybrid similarity: semantic + keyword overlap"""

        # CVde Türkçe metinleri İngilizceye çevir
        cv_text = self.translate_if_needed(cv_text)
        #job_text = self.translate_if_needed(job_text)

        cv_emb = self._model.encode(cv_text, convert_to_tensor=True)
        job_emb = self._model.encode(job_text, convert_to_tensor=True)
        semantic = util.cos_sim(cv_emb, job_emb).item()
        keyword = self.keyword_overlap(cv_text, job_text)

        final_score = self.alpha * semantic + (1 - self.alpha) * keyword
        return round(float(final_score), 3)
    
    def explain_missing_keywords(self, cv_text: str, job_text: str, top_n: int = 10):
        """
        İş ilanındaki anlamlı anahtar kelimeleri (KeyBERT + spaCy + filtreleme) çıkarır,
        CV'de olmayanları tespit eder.
        """
        from nltk.corpus import stopwords

        print("\n[INFO] Anahtar kelime analizi başlatılıyor...")

        # 1️⃣ Dil tespiti
        try:
            detected_lang = detect(job_text)
        except:
            detected_lang = "en"

        # 2️⃣ spaCy model yükleme (fallback)
        try:
            if detected_lang.startswith("tr"):
                print("[INFO] Türkçe model yüklenmeye çalışılıyor...")
                nlp = spacy.load("tr_core_news_sm")
            else:
                nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"[WARN] Türkçe model bulunamadı veya uyumsuz: {e}")
            print("[INFO] İngilizce model fallback olarak kullanılacak.")
            nlp = spacy.load("en_core_web_sm")

        # 3️⃣ POS filtreleme (yalnızca anlamlı kelimeler)
        doc = nlp(job_text)
        allowed_pos = {"NOUN", "PROPN", "ADJ"}
        filtered_tokens = []
        for token in doc:
            if token.pos_ not in allowed_pos or token.is_stop or not token.is_alpha:
                continue
            # Şirket veya kişi adlarını çıkar
            if token.pos_ == "PROPN" and token.ent_type_ in {"ORG", "PERSON"}:
                continue
            filtered_tokens.append(token.text.lower())

        filtered_text = " ".join(filtered_tokens)

        # 4️⃣ KeyBERT ile anahtar kelimeleri çıkar (MMR + çeşitlilik + 2-gram)
        kw_model = KeyBERT(model=self._model)
        raw_keywords = kw_model.extract_keywords(
            filtered_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=top_n * 3  # fazladan çıkar, sonra filtrele
        )

        # 5️⃣ Stopword, custom kelime ve anlam filtresi
        try:
            turkish_sw = stopwords.words("turkish")
        except:
            import nltk
            nltk.download("stopwords")
            turkish_sw = stopwords.words("turkish")
        english_sw = stopwords.words("english")
        STOPWORDS = set(turkish_sw + english_sw)
        CUSTOM_EXCLUDE = {"fitbul", "startup", "ekip", "ekipleriyle", "firmamız", "aranmaktadır", "pozisyon", "şirket", "katkı", "edici", "görev"}

        def is_meaningful(word: str) -> bool:
            if len(word) < 3:
                return False
            if re.search(r'\d', word):  # sayılar varsa
                return False
            bad_suffixes = ('lık', 'lik', 'lu', 'lü', 'cı', 'ci', 'cılar', 'cilik', 'sız', 'siz', 'ıcı', 'ici')
            if word.endswith(bad_suffixes):
                return False
            return True

        cleaned_keywords = []
        for kw, score in raw_keywords:
            if score < 0.4:  # anlam skor eşiği
                continue
            word = kw.lower().strip()
            if len(word) < 3 or word in STOPWORDS or word in CUSTOM_EXCLUDE:
                continue
            if not is_meaningful(word):
                continue
            cleaned_keywords.append(word)

        # 6️⃣ En anlamlı ilk N kelimeyi al
        job_keywords = list(dict.fromkeys(cleaned_keywords))[:top_n]
        print(f"[INFO] İş ilanı anahtar kelimeleri ({len(job_keywords)}): {job_keywords}")

        # 7️⃣ CV'deki kelimeleri normalize et
        doc_cv = nlp(cv_text)
        cv_tokens = {
            token.lemma_.lower()
            for token in doc_cv
            if not token.is_stop and token.is_alpha
        }

        # 8️⃣ Eksik kelimeleri bul
        missing = [kw for kw in job_keywords if kw not in cv_tokens]

        print("\n=== 🔍 Eksik Kelimeler ===")
        if not missing:
            print("✅ CV'de tüm anahtar kelimeler mevcut!")
        else:
            for kw in missing:
                print(f"❌ {kw}")

        print("[INFO] Karşılaştırma tamamlandı.\n")

        return missing

