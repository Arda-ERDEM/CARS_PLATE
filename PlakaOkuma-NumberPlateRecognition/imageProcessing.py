import os
import cv2
import numpy as np
import urllib.request
import time
import warnings
import re
from PIL import Image
from dotenv import load_dotenv

from google import genai
from ultralytics import YOLO

# Ortam değişkenlerini yükle (GEMINI_API_KEY)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    gemini_client = genai.Client(api_key=api_key)
else:
    print("UYARI: GEMINI_API_KEY bulunamadı. Lütfen .env dosyasını oluşturun.")
    gemini_client = None

# ==========================================
# 1. ÖZEL YOLO (Plaka Tespiti)
# ==========================================
model_path = "license_plate_detector.pt"

if not os.path.exists(model_path):
    print(f"HATA: {model_path} bulunamadı!")
else:
    print(f"Özel YOLO modeli ({model_path}) diskten yükleniyor...")

yolo_model = YOLO(model_path)

# Modeli ısıtma
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
_ = yolo_model(dummy_img, conf=0.5, imgsz=640, verbose=False)


# --- TÜRK PLAKA FORMAT DÜZELTİCİ (Gemini hata yaparsa diye) ---
LETTER_TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'J': '1',
                   'Z': '2', 'B': '8', 'G': '6', 'S': '5', 'T': '7', 'A': '4'}
DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S',
                   '6': 'G', '7': 'T', '8': 'B', '9': 'P'}

def normalize_turkish_plate(raw: str) -> str:
    if not raw or len(raw) < 5: return raw
    text = raw.upper()
    result = list(text)
    
    for i in range(min(2, len(result))):
        if result[i].isalpha():
            result[i] = LETTER_TO_DIGIT.get(result[i], result[i])
            
    letter_start, letter_end = 2, 2
    for i in range(2, min(5, len(result))):
        ch = result[i]
        if ch.isalpha(): letter_end = i + 1
        elif ch.isdigit() and i < 5 and i == letter_start:
            look_ahead_has_letter = any(result[j].isalpha() for j in range(i+1, min(i+3, len(result))))
            if look_ahead_has_letter and ch in DIGIT_TO_LETTER:
                result[i] = DIGIT_TO_LETTER[ch]
                letter_end = i + 1
            else: break
        else: break
        
    for i in range(letter_start, letter_end):
        if result[i].isdigit(): result[i] = DIGIT_TO_LETTER.get(result[i], result[i])
    for i in range(letter_end, len(result)):
        if result[i].isalpha(): result[i] = LETTER_TO_DIGIT.get(result[i], result[i])
        
    corrected = ''.join(result)
    try:
        city_code = int(corrected[:2])
        if city_code < 1 or city_code > 81: return raw
    except ValueError: return raw
    return corrected

# ==========================================
# 2. GEMINI FLASH API (Plaka Okuma)
# ==========================================
def run_ocr(vehicle_bgr_image) -> str:
    """
    Kırpılmış araç (Araba/Tır vs.) resmini alıp Gemini Flash'a gönderir
    ve plakayı metin olarak döndürür.
    """
    if gemini_client is None:
        return "API_KEY_EKSİK"
    
    try:
        # OpenCV BGR to RGB
        rgb_image = cv2.cvtColor(vehicle_bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        prompt = (
            "You are a professional License Plate Recognition system. "
            "Look at this vehicle image and read the Turkish license plate on it. "
            "Respond ONLY with the exact alphanumeric string of the license plate without any spaces or extra text. "
            "If you absolutely cannot find or read any license plate, respond with 'YOK'."
        )
        
        response = gemini_client.models.generate_content(
            model='gemini-flash-latest',
            contents=[prompt, pil_image]
        )
        text = response.text.strip().upper()
        
        if "YOK" in text or len(text) < 4:
            return ""
            
        # Sadece harf ve rakamları tut
        clean_text = re.sub(r'[^A-Z0-9]', '', text)
        
        # Formatı düzenle (Eğer Gemini yanlış okuduysa düzelt)
        final_plate = normalize_turkish_plate(clean_text)
        
        return final_plate
        
    except Exception as e:
        print(f"Gemini API Hatası: {e}")
        return ""
