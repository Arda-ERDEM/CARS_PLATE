import os
import cv2
import numpy as np
import time
import re
import base64
import requests
from ultralytics import YOLO

# ==========================================
# 1. ARAÇ & PLAKA TESPİT MODELLERİ
# ==========================================
# Araç Tespiti: Bisiklet/Scooter (1), Araba (2), Motosiklet (3), Otobüs (5), Kamyon/Tır (7)
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]
vehicle_model_path = "yolov8n.pt"

print(f"Araç Tespit Modeli ({vehicle_model_path}) yükleniyor...")
try:
    vehicle_model = YOLO(vehicle_model_path)
    dummy_v = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = vehicle_model(dummy_v, classes=VEHICLE_CLASS_IDS, conf=0.3, imgsz=640, verbose=False)
    print("✅ Araç Tespit Modeli (YOLOv8n) Hazır!")
except Exception as e:
    print(f"⚠️ Araç modeli yükleme hatası: {e}")
    vehicle_model = None

# Plaka Tespiti: license_plate_detector.pt
model_path = "license_plate_detector.pt"

if not os.path.exists(model_path):
    print(f"HATA: {model_path} bulunamadı!")
    yolo_model = None
else:
    print(f"Özel Plaka YOLO Modeli ({model_path}) diskten yükleniyor...")
    yolo_model = YOLO(model_path)
    # Modeli ısıtma (Warmup)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = yolo_model(dummy_img, conf=0.5, imgsz=640, verbose=False)
    print("✅ Plaka Tespit Modeli Hazır!")


# ==========================================
# 2. GÖRÜNTÜ KALİTE DEĞERLENDİRME & İYİLEŞTİRME (3-Frame Best Selection)
# ==========================================
def calculate_sharpness_score(img_bgr) -> float:
    """
    Görüntünün netlik, kontrast ve çözünürlük kalitesini milisaniyeler içinde hesaplar.
    Laplacian varyansı (netlik) + Kontrast + Alan Faktörü (Yakındaki büyük plakalar en yüksek puanı alır).
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_std = float(np.std(gray))
    h, w = img_bgr.shape[:2]
    # Yakındaki ve okunabilir boyuttaki plakaları güçlü şekilde ödüllendir
    area_factor = (h * w) / (30.0 * 90.0) # 90x30 baz boyut
    return laplacian_var * (1.0 + contrast_std / 100.0) * max(0.5, area_factor)

def enhance_plate_image(img_bgr):
    """
    Kırpılmış plaka görüntüsünü güzelleştirir ve iyileştirir:
    1. Yetersiz çözünürlük için akıllı Bicubic Upscale (en az 110px yükseklik).
    2. LAB renk uzayında CLAHE adaptif kontrast ve ışık dengeleme (parlama/gölge giderme).
    3. Plaka karakterlerini belirginleştiren özel kenar keskinleştirme filtresi.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
        
    h, w = img_bgr.shape[:2]
    
    # 1. Çözünürlük İyileştirme (Upscaling)
    target_h = max(h, 110)
    target_w = int(w * (target_h / max(1, h)))
    if target_h > h:
        img_resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    else:
        img_resized = img_bgr.copy()
        
    # 2. LAB Renk Uzayında CLAHE Kontrast Dengeleme
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 3. Harf Belirginleştirme & Keskinleştirme
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.35, gaussian, -0.35, 0)
    
    return sharpened


# ==========================================
# 3. TÜRK PLAKA FORMAT DÜZELTİCİ & KESİN DOĞRULAMA (6 GEÇERLİ ŞABLON)
# ==========================================
LETTER_TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'J': '1',
                   'Z': '2', 'B': '8', 'G': '6', 'S': '5', 'T': '7', 'A': '4'}
DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S',
                   '6': 'G', '7': 'T', '8': 'B', '9': 'P'}

# Türkiye'de geçerli 6 kesin plaka şablonu:
# İl Kodu: 01-81
# 1) 2 harf + 3 rakam (Örn: 34 AB 123)
# 2) 1 harf + 4 rakam (Örn: 34 A 1234)
# 3) 3 harf + 2 rakam (Örn: 34 ABC 12)
# 4) 2 harf + 4 rakam (Örn: 34 AB 1234)
# 5) 1 harf + 5 rakam (Örn: 34 A 12345)
# 6) 3 harf + 3 rakam (Örn: 34 ABC 123)
STRICT_TR_PLATE_REGEX = re.compile(
    r'^(0[1-9]|[1-7][0-9]|8[0-1])'
    r'('
    r'[A-Z]{2}\d{3}|'
    r'[A-Z]{1}\d{4}|'
    r'[A-Z]{3}\d{2}|'
    r'[A-Z]{2}\d{4}|'
    r'[A-Z]{1}\d{5}|'
    r'[A-Z]{3}\d{3}'
    r')$'
)

def normalize_turkish_plate(raw: str) -> str:
    """
    Kullanıcının belirlediği 6 zorunlu Türkiye plaka kuralını uygular.
    İlk 2 karakter kesinlikle 01-81 arası il kodu rakamı olmak zorundadır.
    Ardından gelen kısım sadece izin verilen 6 şablondan birine uymalıdır.
    Uymayan her metin ELENİR ve boş string döner.
    """
    if not raw:
        return ""
        
    text = re.sub(r'[^A-Z0-9]', '', raw.upper())
    
    # Baştaki mavi şerit TR ibaresini temizle
    if text.startswith("TR") and len(text) >= 7:
        if text[2:4].isdigit() or (text[2] in LETTER_TO_DIGIT and text[3].isdigit()):
            text = text[2:]
            
    if len(text) < 6 or len(text) > 8:
        return ""
        
    result = list(text)
    
    # 1. İlk 2 karakter kesinlikle rakam olmak zorunda (01-81 İl Kodu)
    for i in range(2):
        if result[i].isalpha():
            result[i] = LETTER_TO_DIGIT.get(result[i], result[i])
            
    if not (result[0].isdigit() and result[1].isdigit()):
        return ""
        
    city_code = int(''.join(result[:2]))
    if city_code < 1 or city_code > 81:
        return ""
        
    raw_candidate = ''.join(result)
    if STRICT_TR_PLATE_REGEX.match(raw_candidate):
        return raw_candidate
        
    # 2. İl kodundan sonraki harf/rakam dizilimini 6 geçerli şablon için onar
    rest = ''.join(result[2:])
    rest_len = len(rest)
    
    if rest_len == 5:
        splits = [(2, 3), (1, 4), (3, 2)]
    elif rest_len == 6:
        splits = [(2, 4), (1, 5), (3, 3)]
    else:
        return ""
        
    for n_letters, n_digits in splits:
        cand_list = list(rest)
        for j in range(n_letters):
            if cand_list[j].isdigit():
                cand_list[j] = DIGIT_TO_LETTER.get(cand_list[j], cand_list[j])
        for k in range(n_letters, rest_len):
            if cand_list[k].isalpha():
                cand_list[k] = LETTER_TO_DIGIT.get(cand_list[k], cand_list[k])
                
        candidate_plate = ''.join(result[:2]) + ''.join(cand_list)
        if STRICT_TR_PLATE_REGEX.match(candidate_plate):
            return candidate_plate
            
    return ""


# ==========================================
# 4. SOTA ÖZEL PLAKA OCR MODELİ (FastPlateOCR - Compact Convolutional Transformer)
# ==========================================
from fast_plate_ocr import LicensePlateRecognizer

print("SOTA Plaka OCR Modeli (cct-xs-v2-global-model) yükleniyor...")
try:
    plate_ocr_model = LicensePlateRecognizer(hub_ocr_model="cct-xs-v2-global-model", device="auto")
    # Modeli ısıtma (Warmup)
    dummy_plate = np.full((64, 128, 3), 255, dtype=np.uint8)
    _ = plate_ocr_model.run(dummy_plate)
    print("✅ SOTA Plaka OCR Modeli Hazır! (3ms Süper Hızlı Tanıma)")
except Exception as e:
    print(f"⚠️ OCR Model Yükleme Hatası: {e}")
    plate_ocr_model = None

def check_ocr_health() -> bool:
    """OCR modelinin hazır olup olmadığını kontrol eder."""
    return plate_ocr_model is not None

def read_plate_fast_ocr(plate_bgr_image) -> str:
    """
    Kırpılmış plaka BGR görüntüsünü alır, SOTA CCT Plaka OCR modeliyle
    3 milisaniye içinde okur ve Türkiye 6 zorunlu şablonuna göre doğrulayarak döner.
    """
    if plate_bgr_image is None or plate_bgr_image.size == 0 or plate_ocr_model is None:
        return ""
        
    try:
        # BGR görüntüyü RGB formatına dönüştür
        rgb_image = cv2.cvtColor(plate_bgr_image, cv2.COLOR_BGR2RGB)
        
        # SOTA Plaka OCR Modeli ile Tanıma (~3 ms)
        predictions = plate_ocr_model.run(rgb_image)
        if not predictions:
            return ""
            
        raw_plate = predictions[0].plate.strip()
        if not raw_plate:
            return ""
            
        # Türkiye 6 zorunlu şablon kurallarıyla kesin doğrula
        normalized = normalize_turkish_plate(raw_plate)
        return normalized
            
    except Exception as e:
        print(f"Plaka OCR Okuma İstisnası: {e}")
        return ""

run_ocr = read_plate_fast_ocr
read_plate_paligemma = read_plate_fast_ocr
check_paligemma_health = check_ocr_health
