import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from transformers import pipeline

# NLTK verisini indir (Eğer yoksa)
nltk.download('punkt_tab')

def job_description_to_atoms(text):
    # 1. Önce bullet pointleri (-, *, •) tespit edip satır başlarına göre ayır
    sentences = sent_tokenize(text)
    atoms = [sent.strip() for sent in sentences]
    #print(f"[INFO] İş ilanı atomları ({len(atoms)}): {atoms}")
    #print("\n=== 🔍 İş İlanı Atomları ===")
    #print(atoms)
    return atoms
# TEST
job_text = """
We are a leading tech company. 
Requirements:
* 3+ years of experience in Python.
* Strong knowledge of Django and PostgreSQL.
* Excellent communication skills.
We offer competitive salary and health insurance.
"""
"""
atoms = job_description_to_atoms(job_text)
print("Extracted Atoms:")
for atom in atoms:
    print(f"- {atom}")
"""

# Daha ağır ama çok zeki bir model: bart-large-mnli
# Eğer bilgisayarın kasarsa 'valhalla/distilbart-mnli-12-1' kullanabilirsin.
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

def filter_requirements(atoms):
    candidate_labels = [
    "requirements and qualifications expected from the candidate",
    "information describing the company, its culture or mission",
    "salary, benefits, compensation and employee perks"
    ]
    filtered_needs = []
    
    print("\n--- Sınıflandırma Başlıyor ---")
    for atom in atoms:
        # Model her atom için olasılık hesaplar
        res = classifier(atom, candidate_labels)
        label= res['labels'][0] # En yüksek olasılıklı etiket
        score = res['scores'][0] # O etiketin güven skoru
        
        if (
            label == "requirements and qualifications expected from the candidate" and
            score > 0.5
        ):
            filtered_needs.append(atom)
        #    print(f"[REK]: {atom[:50]}... (Güven: %{score*100:.1f})")
        #else:
        #    print(f"[GÜRÜLTÜ]: {atom[:50]}...")
        #    print(f"       Etiketler: {res['labels']} | Skorlar: {[f'%{s*100:.1f}' for s in res['scores']]}")
            
    return filtered_needs

# Adım 1'den gelen 'atoms' listesini buraya sokuyoruz
#real_requirements = filter_requirements(atoms)