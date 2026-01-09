import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# NLTK verisini indir
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def job_description_to_atoms(text):
    # Cümleleri parçala (bullet pointleri ve satırları ayırır)
    sentences = sent_tokenize(text)
    atoms = [sent.strip() for sent in sentences if sent.strip()]
    return atoms

# Zero-shot classification modeli
# Bu model RAM'de yer kaplar ama Flask her requestte yeniden yüklemesin diye globalde kalması iyidir.
# Eğer bellek sorunu yaşarsan bunu da main.py'da bir kez yükleyip fonksiyona parametre geçebilirsin.
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
        label = res['labels'][0] # En yüksek olasılıklı etiket
        score = res['scores'][0] # O etiketin güven skoru
        
        # Sadece adayla ilgili gereksinimleri al
        if (
            label == "requirements and qualifications expected from the candidate" and
            score > 0.5
        ):
            filtered_needs.append(atom)
            
    return filtered_needs