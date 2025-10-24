# src/app_gui.py

from dotenv import load_dotenv
load_dotenv()

import tkinter as tk
from tkinter import filedialog, scrolledtext
import json
from pdf_to_text import pdf_to_text
from info_extractor_combined import extract_info
from matcher import calculate_similarity  # 🔹 Artık buradan geliyor

def process_cv():
    """PDF dosyasını seçip modeli test eder (sadece görsel amaçlı)."""
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        return

    job_text = job_entry.get("1.0", tk.END).strip()
    result_text.delete("1.0", tk.END)

    if not job_text:
        result_text.insert(tk.END, "⚠️ Lütfen iş ilanı açıklaması girin.\n")
        return

    try:
        # PDF → metin
        result_text.insert(tk.END, "📄 PDF okunuyor...\n")
        cv_text = pdf_to_text(pdf_path)

        # CV analizi
        result_text.insert(tk.END, "🧠 CV analiz ediliyor...\n")
        cv_info = extract_info(cv_text)

        # Skor hesapla
        result_text.insert(tk.END, "🎯 Skor hesaplanıyor...\n")
        score = calculate_similarity(cv_text, job_text)

        # GUI’ye yazdır
        result_text.insert(tk.END, f"\n✅ Benzerlik Skoru: {score}\n\n")
        result_text.insert(tk.END, "📊 Çıkarılan Bilgiler (JSON):\n")
        result_text.insert(tk.END, json.dumps(cv_info, indent=2, ensure_ascii=False))

    except Exception as e:
        result_text.insert(tk.END, f"\n❌ Hata: {str(e)}\n")

# === Basit GUI ===
root = tk.Tk()
root.title("Matchire AI – Model Test Arayüzü")
root.geometry("950x750")

tk.Label(root, text="İş İlanı Açıklaması", font=("Arial", 12, "bold")).pack(pady=5)
job_entry = scrolledtext.ScrolledText(root, height=8, width=110)
job_entry.pack(padx=10, pady=5)

tk.Button(root, text="📄 CV Seç ve Skoru Hesapla", command=process_cv,
          font=("Arial", 12, "bold"), bg="#4CAF50", fg="white").pack(pady=10)

result_text = scrolledtext.ScrolledText(root, height=28, width=110)
result_text.pack(padx=10, pady=10)

root.mainloop()
