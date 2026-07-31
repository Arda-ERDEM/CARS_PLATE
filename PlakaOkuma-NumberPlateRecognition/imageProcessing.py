import cv2
import numpy as np
import save
import easyocr
import os
import urllib.request
import time
import warnings
import re
from collections import Counter
from ultralytics import YOLO
from paddleocr import PaddleOCR

# EasyOCR beamsearch overflow uyarısını gizle
warnings.filterwarnings("ignore", category=RuntimeWarning)

model_path = "license_plate_detector.pt"

# Model dosyası klasörde yoksa otomatik olarak GitHub'dan indir.
if not os.path.exists(model_path):
    print("YOLO plaka tespit modeli (Yapay Zeka) indiriliyor (bu işlem bir kez yapılır, ~20MB)...")
    url = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
    urllib.request.urlretrieve(url, model_path)
    print("Model başarıyla indirildi!")
else:
    print("YOLO modeli diskten yükleniyor...")

yolo_model = YOLO(model_path)

print("PaddleOCR metin okuyucu başlatılıyor...")
# PaddleOCR sessiz modda çalışsın (gereksiz logları gizle)
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
reader = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)

# Zamansal Oylama (Temporal Consensus) Değişkenleri
plate_buffer = []
last_logged_plate = ""
last_logged_time = 0

def rec(img, is_video=True):
    global plate_buffer, last_logged_plate, last_logged_time
    
    if img is None:
        return img
    
    # YOLO modeli ile plaka tespiti (%40 güven seviyesi)
    results = yolo_model(img, conf=0.4, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Ekranda plakanın etrafına yeşil kutu çizimi
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # YOLO genellikle tam plakayı bulur. Fazladan pay (margin) bırakmak arabanın 
            # tamponunu veya plaka çerçevesindeki yazıları (ve kenardaki siyah çizgileri '1' sanmasını) tetikler.
            # Bu yüzden margin'i çok küçültüyoruz (Sadece 2 piksel pay).
            margin = 2
            h, w, _ = img.shape
            crop_x1 = max(0, x1 - margin)
            crop_y1 = max(0, y1 - margin)
            crop_x2 = min(w, x2 + margin)
            crop_y2 = min(h, y2 + margin)
            
            # Türkiye ve Avrupa standart uzun plakalarında (Genişlik/Yükseklik oranı > 3.0) 
            # sol taraftaki MAVİ ŞERİT (TR/EU), EasyOCR'ın ilk karakteri (özellikle 3'ü B, S'yi F) 
            # yanlış okumasına sebep olan en büyük düşmandır.
            box_w = crop_x2 - crop_x1
            box_h = crop_y2 - crop_y1
            if box_h > 0 and (box_w / box_h) > 3.0:
                # Plakanın solundan %8'lik kısmı (Mavi Şeridi) fiziksel olarak kesip çöpe atıyoruz.
                crop_x1 = crop_x1 + int(box_w * 0.08)
            
            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped.size > 0:
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                # BORDER_CONSTANT (Saf beyaz) eklemek, gri plakalarda yapay bir sınır çizgisi oluşturup 
                # yapay zekanın "1" veya "I" halüsinasyonu görmesine yol açıyordu. 
                # Bunun yerine BORDER_REPLICATE ile kenar renklerini dışa doğru uzatıyoruz.
                padded = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
                
                # PaddleOCR 3 kanallı (RGB/BGR) resim beklediği için gri resmi tekrar BGR'a çeviriyoruz
                padded_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
                
                # PaddleOCR ile metin okuma
                text_results = reader.ocr(padded_bgr)
                
                if text_results and len(text_results) > 0:
                    ocr_result = text_results[0]
                    if 'rec_texts' in ocr_result and 'rec_polys' in ocr_result:
                        texts = ocr_result['rec_texts']
                        polys = ocr_result['rec_polys']
                        
                        # 1. KONUMSAL FİLTRELEME (Boyut Analizi)
                        max_height = 0
                        
                        # En büyük fontun (asıl plakanın) yüksekliğini bul
                        for bbox in polys:
                            min_y = min([p[1] for p in bbox])
                            max_y = max([p[1] for p in bbox])
                            height = max_y - min_y
                            if height > max_height:
                                max_height = height
                                
                        valid_texts = []
                        # Sadece yeterince büyük olan (çöp olmayan) yazıları al (%60 kuralı)
                        for i, bbox in enumerate(polys):
                            text = texts[i]
                            
                            min_y = min([p[1] for p in bbox])
                            max_y = max([p[1] for p in bbox])
                            height = max_y - min_y
                            
                            if height >= max_height * 0.6:
                                # Sadece harf ve rakamlara izin ver
                                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                                if clean_text:
                                    valid_texts.append(clean_text)
                                
                        if valid_texts:
                            full_text = "".join(valid_texts)
                            text = full_text
                        
                        if text.startswith("TR"):
                            text = text[2:]
                        elif text.endswith("TR"):
                            text = text[:-2]
                        
                        if len(text) >= 5:
                            if not is_video:
                                # Tekil resim modunda beklemeden direkt ekrana yazdır (Oylama yapma)
                                print(f"👉 Tespit Edilen Plaka: {text}")
                                cv2.putText(img, text, (x1, max(0, y1 - 10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                # 2. ZAMANSAL OYLAMA (Temporal Consensus) - Sadece videoda aktif
                                plate_buffer.append(text)
                                
                                if len(plate_buffer) > 20: # Havuzu son 20 kare ile sınırla
                                    plate_buffer.pop(0)
                                    
                                counter = Counter(plate_buffer)
                                most_common_plate, count = counter.most_common(1)[0]
                                
                                # Eğer aynı plaka en az 5 kez okunduysa (Kesin onay)
                                if count >= 5:
                                    current_time = time.time()
                                    
                                    # Aynı aracı art arda 5 saniye içinde spam olarak kaydetme
                                    if most_common_plate != last_logged_plate or (current_time - last_logged_time) > 5.0:
                                        save.write(most_common_plate)
                                        print(f"\n============================================")
                                        print(f"✅ ONAYLANAN GİRİŞ YAPAN PLAKA: {most_common_plate}")
                                        print(f"============================================\n")
                                        
                                        last_logged_plate = most_common_plate
                                        last_logged_time = current_time
                                        
                                        # Havuzu temizle ki yeniden oylamaya başlasın
                                        plate_buffer.clear()
                                    
                                    # Ekrana her zaman en güvendiği (oylanan) plakayı yazdır
                                    cv2.putText(img, most_common_plate, (x1, max(0, y1 - 10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                            
    return img