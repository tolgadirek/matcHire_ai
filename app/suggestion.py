from sentence_transformers import util

# NOT: Modeli burada tekrar YÜKLEMİYORUZ. Main.py'dan parametre olarak alıyoruz.

def generate_recommendations(real_requirements, cv_text, model):
    """
    model: SentenceTransformer instance (passed from main.py)
    """
    cv_emb = model.encode(cv_text, convert_to_tensor=True)
    recommendations = []
    
    print("\n--- Eksik Analizi Başlıyor ---")
    for req in real_requirements:
        req_emb = model.encode(req, convert_to_tensor=True)
        # Gereksinim cümlesi ile CV arasındaki benzerliği ölç
        score = util.cos_sim(req_emb, cv_emb).item()
        
        # Skorlama Mantığı
        if score < 0.30:
            status = "EKSİK"
            advice = f"CV'nizde '{req}' beklentisine dair güçlü bir kanıt bulunamadı. Bu yeteneği projelerinizle örneklendirerek eklemelisiniz."
        elif 0.30 <= score < 0.45:
            status = "GELİŞTİRİLMELİ"
            advice = f"İş ilanındaki '{req}' şartı ile CV'niz sadece kısmen örtüşüyor. Bu konudaki tecrübenizi daha net vurgulayın."
        else:
            status = "TAMAM"
            advice = None
            
        if advice:
            recommendations.append({
                "requirement": req,
                "status": status,
                "score": round(score, 3),
                "advice": advice
            })
            
    return recommendations

def format_final_report(recommendations):
    print("\n" + "="*50)
    print("CV ANALİZ VE İYİLEŞTİRME RAPORU")
    print("="*50)
    
    for rec in recommendations:
        if rec['status'] == "EKSİK":
            emoji = "❌"
            prefix = "KRİTİK EKSİK:"
        else:
            emoji = "⚠️"
            prefix = "GELİŞTİRİLMELİ:"
            
        print(f"\n{emoji} {prefix} {rec['requirement']}")
        print(f"   💡 TAVSİYE: {rec['advice']}")
        print(f"   📊 Eşleşme Gücü: %{rec['score']*100:.1f}")