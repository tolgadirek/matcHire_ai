# 🚀 Kurulum Adımları

### 1️⃣ Repo'yu klonla
```bash
git clone https://github.com/<senin-kullanici-adin>/matchire_ai.git
cd matchire_ai
```

### 2️⃣ Gerekli zip dosyalarını indir ve klasörlere yerleştir
- https://drive.google.com/drive/folders/163G2AnDdCDqP1a486vu3rOHvWQrhMuYi
- src klasörünün olduğu dizine yani kök dizine zip içerisindeki data ve models klasörlerini yerleştir.

### 3️⃣ Sanal ortam oluştur ve aktif et
```bash
python -m venv venv
venv\Scripts\activate
```

### 4️⃣ Gerekli paketleri yükle
```bash
pip install -r requirements.txt
```

### 5️⃣ .env dosyasını oluştur
İçerisine ```GEMINI_API_KEY=seninanahtarın``` yerleştir.

### Oluşması gereken proje yapısı:
- matchire_ai
  - data/
  - models/
  - src/
  - venv/
  - requirements.txt
  - .env
  - .gitignore
 
### 6️⃣ Uygulamayı başlat
```bash
cd src
python app_gui.py
```
### Model Train etme 
```bash
python ./src/train_model.py --data_dir ./data --train_csv train_pairs.csv --model_name BAAI/bge-base-en-v1.5 --model_out ./models/job_match_model --epochs 4 --batch_size 16 --balance --save_embeddings

Eğer ki ram yetmezse batch_size ı 8 e düşür öyle dene 
