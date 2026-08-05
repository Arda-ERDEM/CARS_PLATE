"""
ui.py — ALPR Otonom Güvenlik İstasyonu
=======================================
Thread Mimarisi (FPS drop sıfır):

  CameraThread          → sadece kare okur, 30 FPS, HİÇBİR işlem yapmaz
       ↓ (frame_lock)
  InferenceThread       → sadece YOLO çalıştırır (~30ms GPU), kırpılmış plakaları
                           ocr_queue'ya atar ve HEMEN devam eder
       ↓ (ocr_queue — maxsize=2, dolu ise kare düşürülür, BLOKLANMAZ)
  OCRWorkerThread       → PaddleOCR çalıştırır (~200ms), konsensus tamponu tutar,
                           onaylanan plakayı db_queue'ya atar
       ↓ (db_queue — maxsize=10)
  DatabaseWorkerThread  → Excel I/O yapar (~100ms), sonucu pyqtSignal ile UI'a bildirir
       ↓ (Qt queued connection — thread-safe)
  MainWindow            → UI güncellenir

Hiçbir aşama bir öncekini BLOKLAMAZ. Kamera her zaman 30 FPS akar.
"""

import sys
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2
import time
import queue as queue_module
import threading
from collections import Counter
from datetime import datetime
import re

import numpy as np
import torch
import torch.nn as nn
from paddleocr import PaddleOCR

# --- 1. Karakter Sözlüğü ---
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
char_to_idx = {c: i+1 for i, c in enumerate(chars)} # 0'ı CTC boşluk (blank) token'ı için ayırıyoruz
idx_to_char = {i: c for c, i in char_to_idx.items()}
num_classes = len(chars) + 1

# --- 2. Model Mimarisi (CRNN) ---
class CRNN(nn.Module):
    def __init__(self, img_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), 
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), 
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU() 
        )
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)
        conv_out = conv_out.mean(dim=2) 
        conv_out = conv_out.permute(0, 2, 1) 
        rnn_out, _ = self.rnn(conv_out)
        output = self.fc(rnn_out) 
        output = output.log_softmax(2)
        return output

# --- Çözücü Fonksiyon ---
def decode_crnn(preds):
    preds = preds.permute(1, 0, 2)
    _, preds_index = preds.max(2)
    preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
    
    char_list = []
    for i in range(len(preds_index)):
        if preds_index[i] != 0 and (not (i > 0 and preds_index[i - 1] == preds_index[i])):
            char_list.append(idx_to_char[preds_index[i].item()])
    return ''.join(char_list)

def clean_plate(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

tr_plate_pattern = re.compile(r'^(0[1-9]|[1-7][0-9]|8[0-1])[A-Z]{1,3}\d{2,4}$')

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QTextEdit, QFrame, QSizePolicy, QPushButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

import imageProcessing as imgprocess
import database_manager as dbm


# ── Kuyruk boyutları ──────────────────────────────────────────────────────────
OCR_QUEUE_MAXSIZE = 2   # Dolu olunca kare düşürülür (bloklama yok)
DB_QUEUE_MAXSIZE  = 10  # Onaylanan plakalar


# ═══════════════════════════════════════════════════════════════════════════════
# 1) KAMERA THREAD — Sadece kare okur, 30 FPS
# ═══════════════════════════════════════════════════════════════════════════════
class CameraThread(QThread):
    new_frame_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True
        self.latest_frame = None
        self.frame_lock    = threading.Lock()
        self.overlay_boxes = []          # [(x1, y1, x2, y2, text), ...]
        self.overlay_lock  = threading.Lock()
        self.recent_plates = []          # [(cx, cy, text, timestamp), ...]
        self.recent_plates_lock = threading.Lock()
        self.target_zoom_delta = 0       # Kameranın optik zoom miktarını değiştirmek için

    def run(self):
        # # --- IP KAMERA (şu an devre dışı) ---
        camera_url = "rtsp://planlama:pl@ma85123@192.6.3.205:554/Streaming/Channels/601" 
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            print(f"Uyarı: {camera_url} açılamadı, web kamerasına dönülüyor...")
            cap = cv2.VideoCapture(0)
        
        # --- KONFTEL USB KAMERA ---
        # DirectShow testlerinde Index 0 = Konftel Cam10, Index 1 = Dahili Webcam olduğu kesinleşti.
        # MSMF hatalarını (Error: -1072875772) önlemek için DirectShow (CAP_DSHOW) kullanıyoruz
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # if not cap.isOpened():
        #     cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        # if cap.isOpened():
        #     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.eptz_zoom_level = 0  # 0: Orijinal Tam Açı, 1-10: ePTZ Yakınlaştırma Seviyeleri

        while self.running:
            # Butondan gelen ePTZ komutlarını işle
            if self.target_zoom_delta != 0:
                self.eptz_zoom_level = max(0, min(10, self.eptz_zoom_level + self.target_zoom_delta))
                self.target_zoom_delta = 0
                
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # # ── ROI (BÖLGE KESİMİ - SADECE İŞARETLİ KAPI) ─────────────────────
            # # Kullanıcının işaretlediği sol kapı bölgesine odaklanıyoruz
            # h, w = frame.shape[:2]
            # # Kırmızı çizgiye göre ayarlandı
            # x1 = 0
            # x2 = int(w * 0.42)
            # y1 = int(h * 0.15)
            # y2 = int(h * 0.70)
            
            # frame = frame[y1:y2, x1:x2]
            # ------------------------------------------------------------------

            # # --- GÖRÜNTÜ İYİLEŞTİRME ---
            # # 1) Unsharp Mask ile keskinleştirme (bulanıklığı giderir)
            # blurred = cv2.GaussianBlur(frame, (0, 0), 3)
            # frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
            
            # # 2) CLAHE ile kontrast iyileştirme (plaka harflerini daha belirgin yapar)
            # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            # l, a, b = cv2.split(lab)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # l = clahe.apply(l)
            # frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            # # ----------------------------

            with self.frame_lock:
                self.latest_frame = frame

            display = frame.copy()
            with self.overlay_lock:
                for (x1, y1, x2, y2, text) in self.overlay_boxes:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    if text:
                        cv2.putText(display, text, (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.new_frame_signal.emit(qt_img.copy())

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# 2) INFERENCE THREAD — Sadece YOLO, sonucu OCR kuyruğuna atar
# ═══════════════════════════════════════════════════════════════════════════════
class InferenceThread(QThread):

    def __init__(self, camera_thread: CameraThread, ocr_queue: queue_module.Queue):
        super().__init__()
        self.camera_thread = camera_thread
        self.ocr_queue     = ocr_queue
        self.running       = True

    def run(self):
        while self.running:
            frame = None
            with self.camera_thread.frame_lock:
                if self.camera_thread.latest_frame is not None:
                    frame = self.camera_thread.latest_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            # ── YOLO (~30ms GPU, GIL bırakılır çünkü C++ uzantısı) ──────────
            # imgsz=1280: Uzaktaki küçük plakaları algılamak için yüksek çözünürlük
            # conf=0.3: Uzaktaki bulanık plakaları da yakalaması için düşük güven eşiği
            results = imgprocess.yolo_model(frame, conf=0.3, imgsz=1280, verbose=False)

            new_boxes     = []
            crops_to_send = []

            current_time = time.time()
            with self.camera_thread.recent_plates_lock:
                self.camera_thread.recent_plates = [p for p in self.camera_thread.recent_plates if current_time - p[3] < 2.0]

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    closest_text = ""
                    min_dist = 500000
                    with self.camera_thread.recent_plates_lock:
                        for pcx, pcy, ptext, pts in self.camera_thread.recent_plates:
                            dist = (cx - pcx)**2 + (cy - pcy)**2
                            if dist < min_dist:
                                min_dist = dist
                                closest_text = ptext

                    new_boxes.append((int(x1), int(y1), int(x2), int(y2), closest_text))

                    # Plakayı biraz pay bırakarak kırp (OCR doğruluğu için)
                    margin = 15
                    h, w, _ = frame.shape
                    cx1 = max(0, int(x1) - margin)
                    cy1 = max(0, int(y1) - margin)
                    cx2 = min(w, int(x2) + margin)
                    cy2 = min(h, int(y2) + margin)
                    
                    cropped_vehicle = frame[cy1:cy2, cx1:cx2]
                    if cropped_vehicle.size > 0:
                        # Uzaktaki küçük plakaları OCR okuyabilsin diye büyüt
                        ch, cw = cropped_vehicle.shape[:2]
                        if ch < 40 or cw < 100:
                            scale = max(40 / max(ch, 1), 100 / max(cw, 1))
                            cropped_vehicle = cv2.resize(cropped_vehicle, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        crops_to_send.append((cropped_vehicle, cx, cy))

            with self.camera_thread.overlay_lock:
                self.camera_thread.overlay_boxes = new_boxes

            # ── OCR Kuyruğuna AT — BLOKLANMA YOK ────────────────────────────
            if crops_to_send:
                try:
                    self.ocr_queue.put_nowait(crops_to_send)
                except queue_module.Full:
                    pass   # Kuyruk dolu → bu kareyi atla

            time.sleep(0.01)   # ~100 inference/sn üst sınırı

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# 3) OCR WORKER THREAD — PaddleOCR + Konsensus tamponu
# ═══════════════════════════════════════════════════════════════════════════════
class OCRWorkerThread(QThread):
    """
    ocr_queue'dan görüntü alır, PaddleOCR çalıştırır.
    5 üzeri eşleşme olduğunda plakayı db_queue'ya atar.
    Bu thread kendi başına çalışır; ne kamera ne de UI thread'ini bloklamaz.
    """
    plate_confirmed = pyqtSignal(str)   # DatabaseWorkerThread'e bağlanır

    def __init__(self, ocr_queue: queue_module.Queue, camera_thread: CameraThread):
        super().__init__()
        self.ocr_queue   = ocr_queue
        self.camera_thread = camera_thread
        self.running     = True
        self._buffer     = []
        self._last_plates = {}
        self._COOLDOWN   = 5.0   # sn — aynı plaka tekrar tetiklenmesi önlemi

    def run(self):
        print("Modeller Yükleniyor (OCRWorkerThread GPU)...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 1. PaddleOCR Yükle (main.py ile birebir aynı ayarlar)
        try:
            ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, use_gpu=True)
            print("✅ PaddleOCR başarıyla yüklendi!")
        except Exception as e:
            print(f"❌ PaddleOCR Yükleme Hatası: {e}")
            ocr_engine = None
        
        # 2. CRNN Yükle (Grayscale)
        crnn_model = None
        try:
            crnn_model = CRNN(img_channels=1, num_classes=num_classes)
            crnn_model.to(device)
            weights_path = "best_alpr_model.pth"
            if os.path.exists(weights_path):
                crnn_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
                crnn_model.eval()
                print("✅ CRNN başarıyla yüklendi!")
            else:
                print(f"⚠️ {weights_path} bulunamadı, CRNN devre dışı.")
                crnn_model = None
        except Exception as e:
            print(f"❌ CRNN Yükleme Hatası: {e}")
            crnn_model = None

        while self.running:
            try:
                crops_batch = self.ocr_queue.get(timeout=0.1)
            except queue_module.Empty:
                continue

            for vehicle_bgr, cx, cy in crops_batch:
              try:
                plate_crop = vehicle_bgr
                
                # Çok küçük kırpımları atla (main.py ile aynı)
                if plate_crop.shape[0] < 15 or plate_crop.shape[1] < 45:
                    continue
                
                crnn_text = ""
                paddle_text = ""
                
                # --- MOTOR 1: CRNN (main.py ile birebir aynı) ---
                if crnn_model is not None:
                    img = cv2.resize(plate_crop, (256, 64))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img.astype('float32') / 255.0
                    mean = 0.5
                    std = 0.5
                    img = (img - mean) / std
                    img = np.expand_dims(img, axis=0)
                    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        preds = crnn_model(img_tensor)
                    # main.py'daki decode fonksiyonu
                    preds_np = preds[:, 0, :].cpu().detach().numpy()
                    char_list = []
                    for i in range(preds_np.shape[0]):
                        char_list.append(np.argmax(preds_np[i, :]))
                    decoded = []
                    for i in range(len(char_list)):
                        if char_list[i] != 0 and (not (i > 0 and char_list[i - 1] == char_list[i])):
                            decoded.append(idx_to_char[char_list[i]])
                    crnn_text = clean_plate("".join(decoded))
                
                # --- MOTOR 2: PaddleOCR (main.py ile birebir aynı: det=False) ---
                if ocr_engine is not None:
                    ocr_result = ocr_engine.ocr(plate_crop, det=False, cls=False)
                    if ocr_result and ocr_result[0]:
                        paddle_text = clean_plate(ocr_result[0][0][0])
                
                # --- KONSENSÜS (main.py ile birebir aynı) ---
                final_text = ""
                crnn_match = tr_plate_pattern.match(crnn_text)
                paddle_match = tr_plate_pattern.match(paddle_text)
                
                if crnn_text and paddle_text and crnn_text == paddle_text:
                    final_text = crnn_text
                    print(f"[MUTABIK] CRNN: {crnn_text} | Paddle: {paddle_text}")
                elif crnn_match and not paddle_match:
                    final_text = crnn_text
                    print(f"[CRNN Seçildi] CRNN: {crnn_text} | Paddle: {paddle_text}")
                elif paddle_match and not crnn_match:
                    final_text = paddle_text
                    print(f"[Paddle Seçildi] CRNN: {crnn_text} | Paddle: {paddle_text}")
                else:
                    final_text = paddle_text if paddle_text else crnn_text
                    if final_text:
                        print(f"[KARARSIZ] CRNN: {crnn_text} | Paddle: {paddle_text} -> Sonuç: {final_text}")
                
                plate_text = final_text
                if not plate_text or len(plate_text) < 4:
                    continue
                
                with self.camera_thread.recent_plates_lock:
                    found = False
                    for i, (pcx, pcy, ptext, pts) in enumerate(self.camera_thread.recent_plates):
                        if (cx - pcx)**2 + (cy - pcy)**2 < 500000:
                            self.camera_thread.recent_plates[i] = (cx, cy, plate_text, time.time())
                            found = True
                            break
                    if not found:
                        self.camera_thread.recent_plates.append((cx, cy, plate_text, time.time()))

                # --- TEMPORAL KONSENSÜS (3 TEKRAR GEREKLİ) ---
                self._buffer.append(plate_text)
                if len(self._buffer) > 15:
                    self._buffer = self._buffer[-15:]
                
                count = self._buffer.count(plate_text)
                if count >= 3:
                    now = time.time()
                    if plate_text not in self._last_plates or (now - self._last_plates[plate_text]) > self._COOLDOWN:
                        self._last_plates[plate_text] = now
                        self.plate_confirmed.emit(plate_text)
                        print(f"✅ ONAYLANDI ({count} tekrar): {plate_text}")
                        self._buffer.clear()
              except Exception as e:
                print(f"❌ OCR İşleme Hatası: {e}")
                import traceback
                traceback.print_exc()

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# 4) DATABASE WORKER THREAD — Excel I/O (arka planda, kimseyi bloklamaz)
# ═══════════════════════════════════════════════════════════════════════════════
class DatabaseWorkerThread(QThread):
    """
    db_queue'dan plaka alır, database_manager.process_plate() çağırır.
    Excel I/O bu thread'de olur; in-memory işlem hızlı, disk yazımı async.
    Sonucu Qt sinyaliyle UI thread'ine iletir.
    """
    # plaka, status, mesaj, surucu, firma, amac, giris, cikis, sure
    result_ready = pyqtSignal(str, str, str, str, str, str, str, str, str)

    def __init__(self, db_queue: queue_module.Queue):
        super().__init__()
        self.db_queue = db_queue
        self.running  = True

    def run(self):
        while self.running:
            try:
                plate_text = self.db_queue.get(timeout=0.1)
            except queue_module.Empty:
                continue

            result = dbm.process_plate(plate_text)
            self.result_ready.emit(
                result["plaka"],
                result["status"],
                result["mesaj"],
                result["surucu"],
                result["firma"],
                result["amac"],
                result["giris"],
                result["cikis"],
                result["sure"],
            )

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# ANA PENCERE
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALPR Otonom Güvenlik İstasyonu")
        self.resize(1280, 860)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ── HEADER ────────────────────────────────────────────────────────────
        header_layout = QHBoxLayout()

        self.logo_label = QLabel()
        logo_px = QPixmap("logo.png")
        if not logo_px.isNull():
            self.logo_label.setPixmap(
                logo_px.scaledToHeight(150, Qt.TransformationMode.SmoothTransformation)
            )
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        header_layout.addWidget(self.logo_label, 1)

        header_title = QLabel("PLAKA TANIMA SİSTEMİ")
        header_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_title.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #D32F2F; padding: 10px;"
        )
        header_layout.addWidget(header_title, 0, Qt.AlignmentFlag.AlignCenter)

        header_layout.addWidget(QLabel(), 1)   # sağ dengeleyici
        main_layout.addLayout(header_layout)

        # ── ORTA: Kamera + Bilgi Paneli ───────────────────────────────────────
        middle = QHBoxLayout()

        self.video_label = QLabel("Kamera Başlatılıyor...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #E0E0E0; border: 2px solid #BDBDBD; "
            "border-radius: 8px; color: #555; font-size: 16px;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setMinimumSize(640, 480)
        middle.addWidget(self.video_label, stretch=7)

        # ── Bilgi Paneli ──────────────────────────────────────────────────────
        info_panel = QFrame()
        info_panel.setStyleSheet(
            "background-color: #F5F5F5; border: 1px solid #E0E0E0; border-radius: 8px;"
        )
        info_layout = QVBoxLayout(info_panel)
        info_layout.setSpacing(8)

        lbl_title = QLabel("SON TESPİT EDİLEN ARAÇ")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet(
            "font-size: 14px; color: #555; font-weight: bold; "
            "border-bottom: 1px solid #CCC; padding-bottom: 6px;"
        )
        info_layout.addWidget(lbl_title)

        self.plate_label = QLabel("---")
        self.plate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plate_label.setStyleSheet(
            "font-size: 44px; font-weight: bold; color: #333; padding: 16px; "
            "background-color: #FFF; border: 2px solid #CCC; border-radius: 10px;"
        )
        info_layout.addWidget(self.plate_label)

        self.status_label = QLabel("Bekleniyor...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #666; "
            "padding: 6px; border-radius: 6px;"
        )
        info_layout.addWidget(self.status_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #DDD;")
        info_layout.addWidget(sep)

        # ── ePTZ Zoom Kontrolleri ──
        zoom_lbl = QLabel("🔍 Kamera Zoom Kontrolü (ePTZ)")
        zoom_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #555;")
        info_layout.addWidget(zoom_lbl)

        zoom_lay = QHBoxLayout()
        self.btn_zoom_out = QPushButton("➖ Uzaklaş")
        self.btn_zoom_out.setStyleSheet("font-size:14px; font-weight:bold; background-color:#EEE; border: 1px solid #CCC; border-radius:4px; padding:6px;")
        self.btn_zoom_in = QPushButton("➕ Yakınlaş")
        self.btn_zoom_in.setStyleSheet("font-size:14px; font-weight:bold; background-color:#EEE; border: 1px solid #CCC; border-radius:4px; padding:6px;")
        
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        
        zoom_lay.addWidget(self.btn_zoom_out)
        zoom_lay.addWidget(self.btn_zoom_in)
        info_layout.addLayout(zoom_lay)
        
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #DDD;")
        info_layout.addWidget(sep2)

        def detail_row(icon, default="—"):
            w = QWidget()
            w.setStyleSheet("background: transparent; border: none;")
            h = QHBoxLayout(w)
            h.setContentsMargins(4, 2, 4, 2)
            h.setSpacing(6)
            ic = QLabel(icon)
            ic.setStyleSheet("font-size: 16px; border:none; background:transparent;")
            h.addWidget(ic)
            val = QLabel(default)
            val.setWordWrap(True)
            val.setStyleSheet("font-size: 13px; color:#444; border:none; background:transparent;")
            h.addWidget(val, stretch=1)
            return w, val

        rw_s, self.surucu_val = detail_row("👤")
        rw_f, self.firma_val  = detail_row("🏢")
        rw_a, self.amac_val   = detail_row("📋")
        rw_g, self.giris_val  = detail_row("🟢 Giriş")
        rw_c, self.cikis_val  = detail_row("🔴 Çıkış")
        rw_t, self.sure_val   = detail_row("⏱ Süre")

        for rw in [rw_s, rw_f, rw_a, rw_g, rw_c, rw_t]:
            info_layout.addWidget(rw)

        info_layout.addStretch()
        middle.addWidget(info_panel, stretch=3)
        main_layout.addLayout(middle, stretch=7)

        # ── LOG ───────────────────────────────────────────────────────────────
        log_hdr = QLabel("📋  HAREKET KAYITLARI")
        log_hdr.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; padding: 4px 0;")
        main_layout.addWidget(log_hdr)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            "background-color: #FAFAFA; color: #333; font-family: Consolas; "
            "font-size: 13px; border: 1px solid #CCC; border-radius: 5px;"
        )
        main_layout.addWidget(self.log_console, stretch=3)

        self.log_message("🟡 Sistem başlatıldı. Kamera bağlantısı kuruluyor...", "#B8860B")

        # ── THREAD BAŞLATMA ───────────────────────────────────────────────────
        self._ocr_queue = queue_module.Queue(maxsize=OCR_QUEUE_MAXSIZE)
        self._db_queue  = queue_module.Queue(maxsize=DB_QUEUE_MAXSIZE)

        self.camera_thread   = CameraThread()
        self.inference_thread = InferenceThread(self.camera_thread, self._ocr_queue)
        self.ocr_thread      = OCRWorkerThread(self._ocr_queue, self.camera_thread)
        self.db_thread       = DatabaseWorkerThread(self._db_queue)

        self.camera_thread.new_frame_signal.connect(self.update_video)
        self.db_thread.result_ready.connect(self.update_info)

        # OCRWorkerThread onaylı plakayı db_queue'ya atar (Qt queued bağlantısı)
        self.ocr_thread.plate_confirmed.connect(self._on_plate_confirmed)

        self.camera_thread.start()
        self.inference_thread.start()
        self.ocr_thread.start()
        self.db_thread.start()

    # ── Sinyal işleyiciler ────────────────────────────────────────────────────

    def _on_plate_confirmed(self, plate: str):
        """OCR thread'den gelen onaylı plakayı DB kuyruğuna atar."""
        try:
            self._db_queue.put_nowait(plate)
        except queue_module.Full:
            pass

    def log_message(self, message: str, color: str = "#006400"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_console.append(
            f"<span style='color:#888;'>[{ts}]</span> "
            f"<span style='color:{color};'>{message}</span>"
        )
        self.log_console.verticalScrollBar().setValue(
            self.log_console.verticalScrollBar().maximum()
        )

    def zoom_in(self):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.target_zoom_delta = 1
            lvl = min(10, getattr(self.camera_thread, 'eptz_zoom_level', 0) + 1)
            self.log_message(f"🔍 ePTZ Yakınlaştırma yapıldı (Seviye {lvl}/10)", "#006400")

    def zoom_out(self):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.target_zoom_delta = -1
            lvl = max(0, getattr(self.camera_thread, 'eptz_zoom_level', 0) - 1)
            self.log_message(f"🔍 ePTZ Uzaklaştırma yapıldı (Seviye {lvl}/10)", "#006400")

    def _set(self, lbl: QLabel, text: str):
        lbl.setText(text if text else "—")

    def update_video(self, qt_image: QImage):
        px = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(
            px.scaled(self.video_label.size(),
                      Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.FastTransformation)
        )

    def update_info(self, plate, status, mesaj, surucu, firma, amac, giris, cikis, sure):
        self.plate_label.setText(plate)
        self._set(self.surucu_val, surucu)
        self._set(self.firma_val,  firma)
        self._set(self.amac_val,   amac)
        self._set(self.giris_val,  giris)
        self._set(self.cikis_val,  cikis)
        self._set(self.sure_val,   sure)

        if status == "giris":
            self.plate_label.setStyleSheet(
                "font-size: 44px; font-weight: bold; color: #FFF; padding: 16px; "
                "background-color: #2E7D32; border-radius: 10px; border: 2px solid #4CAF50;"
            )
            self.status_label.setText("🟢  GİRİŞ YAPTI")
            self.status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #FFF; padding: 6px; "
                "border-radius: 6px; background-color: #388E3C;"
            )
            self.log_message(
                f"🟢 GİRİŞ &nbsp;|&nbsp; <b>{plate}</b> &nbsp;|&nbsp; "
                f"{surucu} &nbsp;|&nbsp; {firma} &nbsp;|&nbsp; {amac} &nbsp;|&nbsp; ⏰ {giris}",
                "#2E7D32",
            )

        elif status == "cikis":
            self.plate_label.setStyleSheet(
                "font-size: 44px; font-weight: bold; color: #FFF; padding: 16px; "
                "background-color: #E65100; border-radius: 10px; border: 2px solid #FF9800;"
            )
            self.status_label.setText("🔴  ÇIKIŞ YAPTI")
            self.status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #FFF; padding: 6px; "
                "border-radius: 6px; background-color: #BF360C;"
            )
            sure_txt = f" &nbsp;|&nbsp; ⏱ {sure}" if sure else ""
            self.log_message(
                f"🔴 ÇIKIŞ &nbsp;|&nbsp; <b>{plate}</b> &nbsp;|&nbsp; "
                f"{surucu} &nbsp;|&nbsp; {firma}{sure_txt}",
                "#E65100",
            )

        elif status == "kara_liste":
            self.plate_label.setStyleSheet(
                "font-size: 44px; font-weight: bold; color: #FFF; padding: 16px; "
                "background-color: #212121; border-radius: 10px; border: 2px solid #757575;"
            )
            self.status_label.setText("⛔  KARA LİSTE")
            self.status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #FFF; padding: 6px; "
                "border-radius: 6px; background-color: #212121;"
            )
            self.log_message(
                f"⛔ KARA LİSTE &nbsp;|&nbsp; <b>{plate}</b> &nbsp;|&nbsp; "
                f"{surucu} &nbsp;|&nbsp; {firma} &nbsp;— ERİŞİM REDDEDİLDİ",
                "#212121",
            )

        else:  # yabanci / hata
            self.plate_label.setStyleSheet(
                "font-size: 44px; font-weight: bold; color: #FFF; padding: 16px; "
                "background-color: #C62828; border-radius: 10px; border: 2px solid #F44336;"
            )
            self.status_label.setText("🚨  YABANCI ARAÇ")
            self.status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #FFF; padding: 6px; "
                "border-radius: 6px; background-color: #B71C1C;"
            )
            self.log_message(
                f"🚨 YABANCI ARAÇ &nbsp;|&nbsp; <b>{plate}</b> &nbsp;— Veritabanında kayıtlı değil",
                "#C62828",
            )

    def closeEvent(self, event):
        self.inference_thread.stop()
        self.ocr_thread.stop()
        self.db_thread.stop()
        self.camera_thread.stop()
        event.accept()


# ── Giriş noktası ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow { background-color: #FFFFFF; }
        QWidget      { color: #333333; }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
