# 🧠 matcHire AI Service

**matcHire**, işe alım süreçlerini yapay zeka ile dönüştüren güçlü bir NLP (Doğal Dil İşleme) motorudur. Geleneksel anahtar kelime eşleşmesinin ötesine geçerek, adaylar ve işverenler için _anlamsal_ analizler sunar.

![Python](https://img.shields.io/badge/Python-gray)
![Flask](https://img.shields.io/badge/Framework-Flask-blue)
![AI](https://img.shields.io/badge/AI-Sentence_Transformers-orange)

## 🎯 Projenin Amacı ve Kullanım Senaryoları

Bu servis, **Node.js Backend** ve **Next.js Frontend** ile haberleşerek iki temel kullanıcı grubu için özel çözümler üretir:

### 👨‍💼 İşverenler İçin (Toplu Analiz & Sıralama)

Yüzlerce CV'yi tek tek okumak yerine, yapay zeka desteğiyle **toplu analiz** yapar.

- İlanın içeriği ile adayların yetkinliklerini anlamsal olarak karşılaştırır.
- Adayları **"En Uygun"dan "En Az Uygun"a** doğru puanlayarak sıralar (%95, %82 vb.).
- Bu sayede en doğru yeteneğe en kısa sürede ulaşılmasını sağlar.

### 👨‍💻 İş Arayanlar İçin (Eksik Analizi & Tavsiye)

Adayın kendi CV'sini ilana göre optimize etmesine yardımcı olur.

- CV'yi analiz eder ve ilanda istenen ancak CV'de bulunmayan yetkinlikleri tespit eder.
- **"Kritik Eksik"** veya **"Geliştirilmeli"** etiketleriyle, adayın hangi alanlara odaklanması gerektiğini raporlar.

## 🚀 Teknik Özellikler

- **📄 PDF Metin Madenciliği:** `PyMuPDF` kütüphanesi ile PDF formatındaki karmaşık CV yapılarını bozulmadan işlenebilir metne dönüştürür.
- **🌍 Akıllı Çoklu Dil Desteği:**
    _ Sistem, Türkçe ve İngilizce metinleri otomatik olarak algılar (`langdetect`).
    _ En yüksek model doğruluğu için Türkçe içerikleri arka planda İngilizceye çevirerek (`deep-translator`) global NLP modelleriyle işler.
- **⚖️ Hibrit Skorlama Algoritması:**
    _ **Anlamsal Benzerlik (Semantic Similarity):** `SentenceTransformer` ile metinlerin vektör uzayındaki bağlamsal yakınlığını ölçer.
    _ **Kelime Örtüşmesi (Keyword Overlap):** Teknik terimlerin ve sertifikaların varlığını kontrol eder.
- **💡 Zero-Shot Tavsiye Sistemi:**
    _ İş ilanını atomik parçalara (cümlelere) ayırır.
    _ Zero-Shot Classification kullanarak ilandaki cümlelerin bir "Gereksinim" mi yoksa "Genel Bilgi" mi olduğunu ayırt eder.

## 📂 Proje Yapısı

- `main.py`: Flask uygulamasının giriş noktası. API route'larını ve sunucu ayarlarını içerir.
- `model_service.py`: Sentence Transformer modelini yükleyen, cosine_similarity ve keyword_overlap hesaplamalarını yapan çekirdek sınıf.
- `suggestion.py`: İş ilanını analiz edip eksik yetkinlikleri bulan ve kullanıcıya tavsiye üreten mantık.
- `utils.py`: Metin temizleme, tokenization ve sınıflandırma (Zero-Shot) yardımcı araçları.
- `pdf_to_text.py`: PDF dosyasından metin ayıklama modülü.

## 🛠️ Kurulum ve Çalıştırma

Bu servisi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1️⃣ Repoyu Klonlayın

```bash
git clone https://github.com/tolgadirek/matchire_ai.git
cd matchire_ai
```

### 2️⃣ Gerekli zip dosyalarını indir ve klasörlere yerleştirin

- https://drive.google.com/drive/folders/163G2AnDdCDqP1a486vu3rOHvWQrhMuYi
- src klasörünün olduğu dizine yani kök dizine zip içerisindeki data ve models klasörlerini yerleştir.

> Zaten hazır fine tune edilmiş modeli kullanmak isterseniz models klasörü yeterlidir. Eğer modeli fine tune etmek isterseniz data klasörünü de koymanız gerekir.

### 3️⃣ Sanal ortam oluştur ve aktif edin

```bash
python -m venv venv
venv\Scripts\activate
```

### 4️⃣ Gerekli paketleri yükleyin

```bash
pip install -r requirements.txt
```

### 5️⃣ Oluşması gereken klasör yapısı:

- matchire_ai
  - app/
  - data/
  - models/
  - src/
  - venv/
  - requirements.txt
  - .gitignore

### 6️⃣ Uygulamayı başlat

```bash
python app/main.py
```

#### 🔗 İlgili Repolar

Tam çalışan bir sistem için aşağıdaki servislerin de ayakta olması gerekir:

💻 Backend: [matchire_backend](https://github.com/tolgadirek/matcHire_backend)

💻 Frontend: [matcHire_frontend](https://github.com/Jessitoii/matcHire_frontend)

## 👥 Ekip Üyeleri

| İsim Soyisim       | GitHub Profili                                 |
| :----------------- | :--------------------------------------------- |
| **Tolga Direk**    | [@tolgadirek](https://github.com/tolgadirek)   |
| **Alper Can Özer** | [@Jessitoii](https://github.com/Jessitoii) |
