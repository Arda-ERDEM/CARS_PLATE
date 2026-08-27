"""
ui.py — ALPR Otonom Güvenlik İstasyonu (Qwen2-VL / PaliGemma Entegrasyonu)
===================================================================
Thread Mimarisi (Sıfır FPS Drop):

  CameraThread          → Sadece kare okur, 30 FPS, HİÇBİR işlem yapmaz
       ↓ (frame_lock)
  InferenceThread       → YOLO çalıştırır (license_plate_detector.pt),
                          Bbox tespit edildiği AN (0 ms gecikmeyle) sağ alttaki ekrana kırpımı yansıtır,
                          kırpılmış plakayı ocr_queue'ya atar.
       ↓ (ocr_queue — maxsize=2, dolu ise kare düşürülür, BLOKLANMAZ)
  OCRWorkerThread       → VLM Docker API üzerinden plaka okur,
                          6 Kesin Türkiye Plaka Şablonu kontrolünden geçirir,
                          onaylanan plakayı db_queue'ya atar.
       ↓ (db_queue — maxsize=10)
  DatabaseWorkerThread  → Excel I/O ve loglama yapar, sonucu pyqtSignal ile UI'a bildirir.
       ↓ (Qt queued connection — thread-safe)
  MainWindow            → UI güncellenir (Giriş/Çıkış durumu, Sürücü, Firma, Hareket Kaydı).
"""

import sys
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2
import time
import queue as queue_module
import threading
from datetime import datetime
import re
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
OCR_QUEUE_MAXSIZE = 10  # Dolu olunca kare düşürülür (bloklama yok)
DB_QUEUE_MAXSIZE  = 20  # Onaylanan plakalar


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
        self.target_zoom_delta = 0       # Kameranın zoom miktarını değiştirmek için

    def run(self):
        # ── TÜM TEST VİDEOLARINI SIRAYLA OYNATMA (DÖNGÜ) [ŞUANLIK YORUM SATIRI] ──
        video_dir = os.path.join(os.path.dirname(__file__), "video")
        if not os.path.exists(video_dir):
            video_dir = r"C:\Users\Arda\Desktop\PlakaOkuma-NumberPlateRecognition\PlakaOkuma-NumberPlateRecognition\video"
        
        video_files = []
        if os.path.exists(video_dir):
            video_files = sorted([
                os.path.join(video_dir, f) for f in os.listdir(video_dir)
                if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
            ])
        
        current_v_idx = 0
        cap = None
        
        if video_files:
            print(f"🎬 Video Oynatma Listesi Hazır ({len(video_files)} video bulundu).")
            cap = cv2.VideoCapture(video_files[current_v_idx])
            print(f"▶️ [1/{len(video_files)}] Oynatılıyor: {os.path.basename(video_files[current_v_idx])}")
        else:
            print("⚠️ Video klasöründe dosya bulunamadı, web kamerasına dönülüyor...")
            cap = cv2.VideoCapture(0)

        # ── CANLI RTSP KAMERA BAĞLANTISI ──
        # rtsp_url = "rtsp://planlama:pl@ma85123@192.6.3.205:554/Streaming/Channels/601"
        # print(f"📡 Canlı RTSP Kamera Akışına Bağlanılıyor: {rtsp_url}")
        # cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        # if not cap.isOpened():
        #     print("⚠️ CAP_FFMPEG ile açılamadı, standart mod deneniyor...")
        #     cap = cv2.VideoCapture(rtsp_url)

        self.eptz_zoom_level = 0  # 0: Orijinal Tam Açı, 1-10: ePTZ Yakınlaştırma Seviyeleri

        while self.running:
            # Butondan gelen ePTZ komutlarını işle
            if self.target_zoom_delta != 0:
                self.eptz_zoom_level = max(0, min(10, self.eptz_zoom_level + self.target_zoom_delta))
                self.target_zoom_delta = 0
                
            ret, frame = cap.read()
            
            # Video bittiğinde sıradaki videoya geç (veya liste bitince başa dön)
            if not ret or frame is None:
                if video_files and len(video_files) > 0:
                    current_v_idx = (current_v_idx + 1) % len(video_files)
                    next_video = video_files[current_v_idx]
                    print(f"🔄 Sıradaki Video [{current_v_idx + 1}/{len(video_files)}]: {os.path.basename(next_video)}")
                    cap.release()
                    cap = cv2.VideoCapture(next_video)
                    ret, frame = cap.read()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

            # ── ePTZ Dijital Yakınlaştırma (Digital Zoom) ────────────────────
            if self.eptz_zoom_level > 0:
                h, w = frame.shape[:2]
                scale = 1.0 + (self.eptz_zoom_level * 0.25)
                new_w = int(w / scale)
                new_h = int(h / scale)
                x1 = (w - new_w) // 2
                y1 = (h - new_h) // 2
                cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
                frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

            with self.frame_lock:
                self.latest_frame = frame

            display = frame.copy()
            with self.overlay_lock:
                for box_info in self.overlay_boxes:
                    x1, y1, x2, y2, text = box_info[0], box_info[1], box_info[2], box_info[3], box_info[4]
                    color = box_info[5] if len(box_info) > 5 else (0, 255, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                    if text:
                        cv2.putText(display, text, (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.new_frame_signal.emit(qt_img.copy())

            # Video kare hızı senkronizasyonu (30 FPS)
            time.sleep(0.033)

        if cap is not None:
            cap.release()

    def stop(self):
        self.running = False
        self.wait()


def compute_box_iou(box1, box2):
    """İki sınır kutusu arasındaki Kesişim/Birleşim (IoU) oranını hesaplar."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 2) INFERENCE THREAD — 0ms Gecikmesiz Anlık Plaka Tespiti & Canlı Önizleme
# ═══════════════════════════════════════════════════════════════════════════════
class InferenceThread(QThread):
    plate_snapshot_signal = pyqtSignal(QImage)

    def __init__(self, camera_thread: CameraThread, ocr_queue: queue_module.Queue):
        super().__init__()
        self.camera_thread = camera_thread
        self.ocr_queue     = ocr_queue
        self.running       = True
        # {track_id: {"last_seen": float, "last_cx": int, "last_cy": int, "last_box": tuple, "confirmed_plate": str}}
        self.plate_trackers = {}
        self._next_track_id = 0

    def run(self):
        while self.running:
            frame = None
            with self.camera_thread.frame_lock:
                if self.camera_thread.latest_frame is not None:
                    frame = self.camera_thread.latest_frame.copy()

            if frame is None or imgprocess.yolo_model is None:
                time.sleep(0.005)
                continue

            # ── 1. Adım: SADECE ARAÇ TÜREVLERİ TESPİTİ (yolov8n.pt) ──
            # Araba, Motosiklet, Otobüs, Kamyon/Tır, Scooter harici tüm nesneler (çiçek, bitki, vb.) engellenir
            vehicle_boxes = []
            if getattr(imgprocess, 'vehicle_model', None) is not None:
                vehicle_results = imgprocess.vehicle_model(
                    frame, 
                    classes=imgprocess.VEHICLE_CLASS_IDS, 
                    conf=0.15, 
                    imgsz=640, 
                    verbose=False
                )
                for vr in vehicle_results:
                    for vb in vr.boxes:
                        vx1, vy1, vx2, vy2 = vb.xyxy[0].cpu().numpy().astype(int)
                        vehicle_boxes.append((vx1, vy1, vx2, vy2))

            # Ekranda HİÇBİR araç yoksa plaka aramaya girme, çiçek/bitki/arka planı asla okuma
            if getattr(imgprocess, 'vehicle_model', None) is not None and not vehicle_boxes:
                with self.camera_thread.overlay_lock:
                    self.camera_thread.overlay_boxes = []
                time.sleep(0.005)
                continue

            # ── 2. Adım: Plaka Tespiti (license_plate_detector.pt) ──
            results = imgprocess.yolo_model(frame, conf=0.25, imgsz=640, verbose=False)

            new_boxes     = []
            crops_to_send = []
            current_time  = time.time()

            # 1.0 saniyeden uzun süredir görülmeyen trackleri temizle
            self.plate_trackers = {
                tid: info for tid, info in self.plate_trackers.items()
                if current_time - info["last_seen"] < 1.0
            }

            h, w, _ = frame.shape

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cur_box = (int(x1), int(y1), int(x2), int(y2))

                    # ── 3. Adım: Plakanın tespit edilen bir ARACIN üzerinde olduğunu doğrula ──
                    if vehicle_boxes:
                        is_on_vehicle = False
                        for vx1, vy1, vx2, vy2 in vehicle_boxes:
                            if (vx1 - 60) <= cx <= (vx2 + 60) and (vy1 - 60) <= cy <= (vy2 + 60):
                                is_on_vehicle = True
                                break
                        if not is_on_vehicle:
                            continue  # Araç dışındaki sahte nesneleri (çiçek, çalı vb.) kesinlikle ele

                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    aspect_ratio = bw / float(bh)

                    # ── Geometrik Plaka Filtresi (1.4 - 7.0) ──
                    if not (1.4 <= aspect_ratio <= 7.0) or bw < 30 or bh < 8:
                        continue

                    margin = 12
                    cx1 = max(0, int(x1) - margin)
                    cy1 = max(0, int(y1) - margin)
                    cx2 = min(w, int(x2) + margin)
                    cy2 = min(h, int(y2) + margin)
                    cropped_plate = frame[cy1:cy2, cx1:cx2]

                    if cropped_plate.size == 0 or cropped_plate.shape[0] < 10 or cropped_plate.shape[1] < 25:
                        continue

                    # ── IoU + Centroid Tabanlı Araç Eşleme ─────────────────────
                    matched_tid = None
                    best_match_score = 0.0

                    for tid, info in self.plate_trackers.items():
                        iou = compute_box_iou(cur_box, info["last_box"])
                        dist = (cx - info["last_cx"])**2 + (cy - info["last_cy"])**2

                        if iou >= 0.15 or (dist < 6000 and iou > 0.03):
                            score = iou * 2.0 + (1.0 / (1.0 + dist * 0.001))
                            if score > best_match_score:
                                best_match_score = score
                                matched_tid = tid

                    if matched_tid is None:
                        # Yeni araç/plaka tespit edildi
                        matched_tid = self._next_track_id
                        self._next_track_id += 1
                        self.plate_trackers[matched_tid] = {
                            "first_seen": current_time,
                            "last_seen": current_time,
                            "last_cx": cx,
                            "last_cy": cy,
                            "last_box": cur_box,
                            "confirmed_plate": ""
                        }
                    else:
                        tracker = self.plate_trackers[matched_tid]
                        tracker["last_seen"] = current_time
                        tracker["last_cx"] = cx
                        tracker["last_cy"] = cy
                        tracker["last_box"] = cur_box

                    tracker = self.plate_trackers[matched_tid]

                    # ── KURAL: Plaka algılandığı AN (0 ms) SAĞ ALTA YANSIT ──
                    enhanced_img = imgprocess.enhance_plate_image(cropped_plate)

                    crop_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                    ch, cw, cch = crop_rgb.shape
                    crop_qt = QImage(crop_rgb.data, cw, ch, cch * cw, QImage.Format.Format_RGB888)
                    self.plate_snapshot_signal.emit(crop_qt.copy())

                    # ── ŞİPŞAK OCR İÇİN ANINDA KUYRUĞA GÖNDER ─────────────────
                    crops_to_send.append((enhanced_img, cx, cy, matched_tid))

                    # ── KUTU VE ETİKET (Durum/bekleme parametresi olmadan temiz) ──
                    conf_plate = tracker.get("confirmed_plate", "")
                    if conf_plate:
                        box_label = conf_plate
                    else:
                        box_label = ""  # Bekleme veya okunuyor yazısı yok, temiz çerçeve

                    box_color = (0, 255, 0)
                    new_boxes.append((int(x1), int(y1), int(x2), int(y2), box_label, box_color))

            with self.camera_thread.overlay_lock:
                self.camera_thread.overlay_boxes = new_boxes

            # ── OCR Kuyruğuna Anında İlet ────────────────────────────────────
            if crops_to_send:
                try:
                    self.ocr_queue.put_nowait(crops_to_send)
                except queue_module.Full:
                    pass

            time.sleep(0.005)

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# 3) OCR WORKER THREAD — SOTA Özel Plaka OCR (3ms) & Zorunlu 6 TR Şablon Filtresi
# ═══════════════════════════════════════════════════════════════════════════════
class OCRWorkerThread(QThread):
    """
    ocr_queue'dan kırpılmış plaka görüntüsünü alır,
    SOTA FastPlateOCR modeliyle 3ms'de okur, TR kurallarına uymayanları eler,
    sadece geçerli plakaları onaylayarak db_queue'ya iletir.
    """
    plate_confirmed = pyqtSignal(str, object)

    def __init__(self, ocr_queue: queue_module.Queue, camera_thread: CameraThread, inference_thread=None):
        super().__init__()
        self.ocr_queue        = ocr_queue
        self.camera_thread    = camera_thread
        self.inference_thread = inference_thread
        self.running          = True
        self._last_plates     = {}
        self._COOLDOWN        = 3.0    # sn — Veritabanı ve galeri için çift kayıt önleme süresi

    def run(self):
        print("SOTA Plaka OCR Worker başlatıldı...")
        if imgprocess.check_ocr_health():
            print("✅ SOTA Plaka OCR Motoru Aktif! (3ms Süper Hızlı Tanıma)")
        else:
            print("⚠️ UYARI: Plaka OCR modeli yüklenemedi!")

        while self.running:
            try:
                crops_batch = self.ocr_queue.get(timeout=0.02)
            except queue_module.Empty:
                continue

            for crop_item in crops_batch:
                plate_crop = crop_item[0]
                cx         = crop_item[1]
                cy         = crop_item[2]
                tid        = crop_item[3] if len(crop_item) > 3 else None
                try:
                    if plate_crop is None or plate_crop.shape[0] < 12 or plate_crop.shape[1] < 30:
                        continue

                    # ── SOTA OCR İle Plakayı Oku (Zorunlu 6 TR Şablon Kontrolünden Geçer) ─
                    plate_text = imgprocess.read_plate_fast_ocr(plate_crop)

                    # Kurala uymayan veya elenen plakalar boş döner -> Kesinlikle Atlansın
                    if not plate_text or len(plate_text) < 6:
                        continue

                    # İlgili aracın tracker'ına okunan plakayı doğrudan bağla
                    if tid is not None and self.inference_thread is not None:
                        if tid in self.inference_thread.plate_trackers:
                            self.inference_thread.plate_trackers[tid]["confirmed_plate"] = plate_text

                    # ── Onay & Cooldown Mekanizması ──────────────────────────
                    now = time.time()
                    if plate_text not in self._last_plates or (now - self._last_plates[plate_text]) > self._COOLDOWN:
                        self._last_plates[plate_text] = now
                        self.plate_confirmed.emit(plate_text, plate_crop)
                        print(f"✅ SOTA OCR ONAYLADI (6 TR Şablonuna Uygun): {plate_text}")

                except Exception as e:
                    print(f"❌ OCR Worker Hatası: {e}")

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# 4) DATABASE WORKER THREAD — Excel I/O (Arka planda çalışır)
# ═══════════════════════════════════════════════════════════════════════════════
class DatabaseWorkerThread(QThread):
    # plaka, status, mesaj, surucu, firma, amac, giris, cikis, sure, crop_img
    result_ready = pyqtSignal(str, str, str, str, str, str, str, str, str, object)

    def __init__(self, db_queue: queue_module.Queue):
        super().__init__()
        self.db_queue = db_queue
        self.running  = True

    def run(self):
        while self.running:
            try:
                item = self.db_queue.get(timeout=0.1)
            except queue_module.Empty:
                continue

            if isinstance(item, tuple):
                plate_text, crop_img = item
            else:
                plate_text, crop_img = item, None

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
                crop_img
            )

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# CANLI PLAKA AKIŞ GALERİSİ KARTI (MINI CARD WIDGET)
# ═══════════════════════════════════════════════════════════════════════════════
class PlateCardWidget(QFrame):
    """Canlı Plaka Akış Galerisindeki tek bir aracın mini kartı."""
    def __init__(self, crop_img, plate_text: str, status: str, surucu: str, timestamp: str):
        super().__init__()
        self.setStyleSheet("""
            QFrame#PlateCard {
                background-color: #FFFFFF;
                border: 1px solid #CED4DA;
                border-radius: 8px;
            }
            QFrame#PlateCard:hover {
                border: 1.5px solid #007BFF;
                background-color: #F8F9FA;
            }
        """)
        self.setObjectName("PlateCard")
        self.setFixedHeight(85)
        self.setMinimumWidth(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # Sol: Kırpılmış Plaka Görseli
        img_lbl = QLabel()
        img_lbl.setFixedSize(85, 45)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setStyleSheet("background-color: #121416; border-radius: 4px; border: 1px solid #444;")
        
        if crop_img is not None:
            if isinstance(crop_img, np.ndarray):
                rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                px = QPixmap.fromImage(qimg)
            elif isinstance(crop_img, QImage):
                px = QPixmap.fromImage(crop_img)
            elif isinstance(crop_img, QPixmap):
                px = crop_img
            else:
                px = QPixmap()
            if not px.isNull():
                img_lbl.setPixmap(px.scaled(85, 45, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                img_lbl.setText("📷")
        else:
            img_lbl.setText("📷")

        # Sol Alt Zaman
        left_v = QVBoxLayout()
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(2)
        left_v.addWidget(img_lbl)
        time_lbl = QLabel(timestamp)
        time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_lbl.setStyleSheet("font-size: 9px; color: #888; border: none; background: transparent;")
        left_v.addWidget(time_lbl)
        layout.addLayout(left_v)

        # Sağ: Bilgiler
        info_lay = QVBoxLayout()
        info_lay.setContentsMargins(0, 0, 0, 0)
        info_lay.setSpacing(2)

        # Plaka Başlığı
        p_lbl = QLabel(plate_text)
        p_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #212529; border: none; background: transparent;")
        info_lay.addWidget(p_lbl)

        # Durum Rozeti (Badge)
        status_badges = {
            "giris": ("#D1E7DD", "#0F5132", "🟢 GİRİŞ"),
            "cikis": ("#F8D7DA", "#842029", "🔴 ÇIKIŞ"),
            "yabanci": ("#FFF3CD", "#664D03", "🟡 YABANCI"),
            "kara_liste": ("#212529", "#FF5252", "⛔ KARA LİSTE")
        }
        bg, fg, label_txt = status_badges.get(status, ("#E2E3E5", "#41464B", status.upper()))
        s_lbl = QLabel(label_txt)
        s_lbl.setStyleSheet(f"background-color: {bg}; color: {fg}; font-size: 10px; font-weight: bold; border-radius: 4px; padding: 1px 4px; border: none;")
        info_lay.addWidget(s_lbl)

        # Sürücü İsmi
        driver_text = surucu if surucu else ("Kayıtsız Araç" if status == "yabanci" else "—")
        d_lbl = QLabel(driver_text)
        d_lbl.setStyleSheet("font-size: 10px; color: #495057; border: none; background: transparent;")
        info_lay.addWidget(d_lbl)

        layout.addLayout(info_lay, stretch=1)


# ═══════════════════════════════════════════════════════════════════════════════
# ANA PENCERE (GUI)
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALPR Otonom Güvenlik İstasyonu")
        self.resize(1340, 880)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # ── HEADER ────────────────────────────────────────────────────────────
        header_layout = QHBoxLayout()

        self.logo_label = QLabel()
        logo_px = QPixmap("logo.png")
        if not logo_px.isNull():
            self.logo_label.setPixmap(
                logo_px.scaledToHeight(120, Qt.TransformationMode.SmoothTransformation)
            )
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        header_layout.addWidget(self.logo_label, 1)

        header_title = QLabel("PLAKA TANIMA SİSTEMİ")
        header_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_title.setStyleSheet(
            "font-size: 26px; font-weight: bold; color: #D32F2F; padding: 5px;"
        )
        header_layout.addWidget(header_title, 0, Qt.AlignmentFlag.AlignCenter)

        header_layout.addWidget(QLabel(), 1)
        main_layout.addLayout(header_layout)

        # ── ORTA: Kamera + Bilgi Paneli & Sağ Alt Snapshot Ekranı ───────────────
        middle = QHBoxLayout()

        # Sol Ana Kamera Ekranı
        self.video_label = QLabel("Kamera Başlatılıyor...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #1E1E1E; border: 2px solid #333; "
            "border-radius: 8px; color: #AAA; font-size: 16px;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setMinimumSize(640, 480)
        middle.addWidget(self.video_label, stretch=7)

        # ── Sağ Bilgi Paneli ──────────────────────────────────────────────────
        info_panel = QFrame()
        info_panel.setStyleSheet(
            "background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 8px;"
        )
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(8)

        lbl_title = QLabel("SON TESPİT EDİLEN ARAÇ")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet(
            "font-size: 13px; color: #555; font-weight: bold; "
            "border-bottom: 1px solid #CCC; padding-bottom: 4px;"
        )
        info_layout.addWidget(lbl_title)

        # Plaka Metni
        self.plate_label = QLabel("---")
        self.plate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plate_label.setStyleSheet(
            "font-size: 38px; font-weight: bold; color: #222; padding: 12px; "
            "background-color: #FFF; border: 2px solid #CCC; border-radius: 8px;"
        )
        info_layout.addWidget(self.plate_label)

        # Durum Bannerı
        self.status_label = QLabel("Bekleniyor...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #666; "
            "padding: 5px; border-radius: 6px; background-color: #E9ECEF;"
        )
        info_layout.addWidget(self.status_label)

        def detail_row(icon, default="—"):
            w = QWidget()
            w.setStyleSheet("background: transparent; border: none;")
            h = QHBoxLayout(w)
            h.setContentsMargins(4, 1, 4, 1)
            h.setSpacing(6)
            ic = QLabel(icon)
            ic.setStyleSheet("font-size: 14px; border:none; background:transparent;")
            h.addWidget(ic)
            val = QLabel(default)
            val.setWordWrap(True)
            val.setStyleSheet("font-size: 12px; color:#444; border:none; background:transparent; font-weight: 500;")
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

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #DDD;")
        info_layout.addWidget(sep)

        # ── SAĞ ALT: ANLIK KIRPILAN PLAKA ÖNİZLEME PENCERESİ ──────────────────
        snap_box = QFrame()
        snap_box.setStyleSheet(
            "background-color: #212529; border: 2px solid #495057; border-radius: 6px; padding: 4px;"
        )
        snap_layout = QVBoxLayout(snap_box)
        snap_layout.setContentsMargins(4, 4, 4, 4)
        snap_layout.setSpacing(2)

        snap_title = QLabel("📸 ANLIK PLAKA GÖRÜNTÜSÜ")
        snap_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        snap_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #00E676; border: none;")
        snap_layout.addWidget(snap_title)

        self.crop_preview_label = QLabel("Plaka Algılanınca Burada Gözükecek")
        self.crop_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crop_preview_label.setStyleSheet(
            "color: #868E96; font-size: 11px; border: 1px dashed #6C757D; border-radius: 4px; background-color: #121416;"
        )
        self.crop_preview_label.setFixedHeight(105)
        self.crop_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        snap_layout.addWidget(self.crop_preview_label)

        info_layout.addWidget(snap_box)

        # ── ePTZ Zoom Kontrolleri ──
        zoom_lay = QHBoxLayout()
        self.btn_zoom_out = QPushButton("➖ Uzaklaş")
        self.btn_zoom_out.setStyleSheet("font-size:12px; font-weight:bold; background-color:#EEE; border: 1px solid #CCC; border-radius:4px; padding:4px;")
        self.btn_zoom_in = QPushButton("➕ Yakınlaş")
        self.btn_zoom_in.setStyleSheet("font-size:12px; font-weight:bold; background-color:#EEE; border: 1px solid #CCC; border-radius:4px; padding:4px;")
        
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        
        zoom_lay.addWidget(self.btn_zoom_out)
        zoom_lay.addWidget(self.btn_zoom_in)
        info_layout.addLayout(zoom_lay)

        middle.addWidget(info_panel, stretch=3)
        main_layout.addLayout(middle, stretch=6)

        # ── CANLI PLAKA AKIŞ GALERİSİ (SON 5 GEÇEN ARAÇ ŞERİDİ) ───────────────
        gallery_frame = QFrame()
        gallery_frame.setStyleSheet(
            "background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 8px; padding: 4px;"
        )
        gallery_v_layout = QVBoxLayout(gallery_frame)
        gallery_v_layout.setContentsMargins(6, 4, 6, 4)
        gallery_v_layout.setSpacing(4)

        gallery_header = QLabel("🚗  CANLI PLAKA AKIŞ GALERİSİ (SON GEÇEN ARAÇLAR)")
        gallery_header.setStyleSheet("font-size: 11px; font-weight: bold; color: #495057; border: none; background: transparent;")
        gallery_v_layout.addWidget(gallery_header)

        self.gallery_cards_layout = QHBoxLayout()
        self.gallery_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_cards_layout.setSpacing(8)
        self.gallery_cards = []

        self.gallery_placeholder = QLabel("Henüz araç geçişi kaydedilmedi. Algılanan plakalar canlı olarak burada akacaktır...")
        self.gallery_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gallery_placeholder.setStyleSheet("color: #ADB5BD; font-size: 11px; font-style: italic; border: none; padding: 12px;")
        self.gallery_cards_layout.addWidget(self.gallery_placeholder)

        gallery_v_layout.addLayout(self.gallery_cards_layout)
        main_layout.addWidget(gallery_frame)

        # ── LOG KONSOLLARI ────────────────────────────────────────────────────
        log_hdr = QLabel("📋  HAREKET VE SİSTEM KAYITLARI")
        log_hdr.setStyleSheet("font-size: 12px; font-weight: bold; color: #555; padding: 2px 0;")
        main_layout.addWidget(log_hdr)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            "background-color: #FAFAFA; color: #333; font-family: Consolas; "
            "font-size: 12px; border: 1px solid #CCC; border-radius: 5px;"
        )
        self.log_console.setFixedHeight(120)
        main_layout.addWidget(self.log_console)

        self.log_message("🟡 Sistem başlatıldı. SOTA OCR ve Canlı Akış Galerisi hazır.", "#B8860B")

        # ── THREAD BAŞLATMA ───────────────────────────────────────────────────
        self._ocr_queue = queue_module.Queue(maxsize=OCR_QUEUE_MAXSIZE)
        self._db_queue  = queue_module.Queue(maxsize=DB_QUEUE_MAXSIZE)

        self.camera_thread    = CameraThread()
        self.inference_thread = InferenceThread(self.camera_thread, self._ocr_queue)
        self.ocr_thread       = OCRWorkerThread(self._ocr_queue, self.camera_thread, self.inference_thread)
        self.db_thread        = DatabaseWorkerThread(self._db_queue)

        # Sinyal bağlantıları
        self.camera_thread.new_frame_signal.connect(self.update_video)
        self.inference_thread.plate_snapshot_signal.connect(self.update_crop_preview)
        self.ocr_thread.plate_confirmed.connect(self._on_plate_confirmed)
        self.db_thread.result_ready.connect(self.update_info)

        self.camera_thread.start()
        self.inference_thread.start()
        self.ocr_thread.start()
        self.db_thread.start()

    # ── Sinyal İşleyiciler ────────────────────────────────────────────────────

    def update_crop_preview(self, qt_crop_img: QImage):
        """YOLO plaka tespit ettiği ANDA anında sağ alttaki ekrana yansıtır."""
        px = QPixmap.fromImage(qt_crop_img)
        self.crop_preview_label.setPixmap(
            px.scaled(
                self.crop_preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
        )

    def _on_plate_confirmed(self, plate: str, crop_img=None):
        """Onaylı plakayı ve kırpılmış görüntüyü DB kuyruğuna atar."""
        try:
            self._db_queue.put_nowait((plate, crop_img))
        except queue_module.Full:
            pass

    def add_gallery_card(self, crop_img, plate: str, status: str, surucu: str, timestamp: str):
        """Canlı plaka akış galerisine yeni araç kartı ekler (Maks 5 kart tutar)."""
        if hasattr(self, 'gallery_placeholder') and self.gallery_placeholder is not None:
            self.gallery_cards_layout.removeWidget(self.gallery_placeholder)
            self.gallery_placeholder.deleteLater()
            self.gallery_placeholder = None

        card = PlateCardWidget(crop_img, plate, status, surucu, timestamp)
        self.gallery_cards_layout.insertWidget(0, card)
        self.gallery_cards.insert(0, card)

        # En fazla 5 kart tut, 5'ten fazlaysa en eskiyi kaldır
        if len(self.gallery_cards) > 5:
            old_card = self.gallery_cards.pop()
            self.gallery_cards_layout.removeWidget(old_card)
            old_card.deleteLater()

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
            mag = 1.0 + (lvl * 0.25)
            self.btn_zoom_in.setText(f"➕ Yakınlaş ({mag:.2f}x)")
            self.btn_zoom_out.setText(f"➖ Uzaklaş")
            self.log_message(f"🔍 ePTZ Dijital Zoom: {mag:.2f}x (Seviye {lvl}/10)", "#006400")

    def zoom_out(self):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.target_zoom_delta = -1
            lvl = max(0, getattr(self.camera_thread, 'eptz_zoom_level', 0) - 1)
            mag = 1.0 + (lvl * 0.25)
            if lvl == 0:
                self.btn_zoom_in.setText("➕ Yakınlaş")
                self.btn_zoom_out.setText("➖ Uzaklaş (1.0x)")
            else:
                self.btn_zoom_in.setText(f"➕ Yakınlaş ({mag:.2f}x)")
                self.btn_zoom_out.setText(f"➖ Uzaklaş")
            self.log_message(f"🔍 ePTZ Dijital Zoom: {mag:.2f}x (Seviye {lvl}/10)", "#006400")

    def _set(self, lbl: QLabel, text: str):
        lbl.setText(text if text else "—")

    def update_video(self, qt_image: QImage):
        px = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(
            px.scaled(self.video_label.size(),
                      Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.FastTransformation)
        )

    def update_info(self, plate, status, mesaj, surucu, firma, amac, giris, cikis, sure, crop_img=None):
        self.plate_label.setText(plate)
        self._set(self.surucu_val, surucu)
        self._set(self.firma_val,  firma)
        self._set(self.amac_val,   amac)
        self._set(self.giris_val,  giris)
        self._set(self.cikis_val,  cikis)
        self._set(self.sure_val,   sure)

        # Sağ alt önizleme görselini güncelle
        if crop_img is not None:
            if isinstance(crop_img, np.ndarray):
                rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.update_crop_preview(qimg)
            elif isinstance(crop_img, QImage):
                self.update_crop_preview(crop_img)

        # Canlı akış galerisine mini kart ekle
        self.add_gallery_card(crop_img, plate, status, surucu, datetime.now().strftime("%H:%M:%S"))

        if status == "giris":
            self.plate_label.setStyleSheet(
                "font-size: 38px; font-weight: bold; color: #FFF; padding: 12px; "
                "background-color: #2E7D32; border-radius: 8px; border: 2px solid #4CAF50;"
            )
            self.status_label.setText("🟢  GİRİŞ YAPTI")
            self.status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #FFF; padding: 5px; "
                "border-radius: 6px; background-color: #388E3C;"
            )
            self.log_message(
                f"🟢 GİRİŞ &nbsp;|&nbsp; <b>{plate}</b> &nbsp;|&nbsp; "
                f"{surucu} &nbsp;|&nbsp; {firma} &nbsp;|&nbsp; {amac} &nbsp;|&nbsp; ⏰ {giris}",
                "#2E7D32",
            )

        elif status == "cikis":
            self.plate_label.setStyleSheet(
                "font-size: 38px; font-weight: bold; color: #FFF; padding: 12px; "
                "background-color: #E65100; border-radius: 8px; border: 2px solid #FF9800;"
            )
            self.status_label.setText("🔴  ÇIKIŞ YAPTI")
            self.status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #FFF; padding: 5px; "
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
                "font-size: 38px; font-weight: bold; color: #FFF; padding: 12px; "
                "background-color: #212121; border-radius: 8px; border: 2px solid #757575;"
            )
            self.status_label.setText("⛔  KARA LİSTE")
            self.status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #FFF; padding: 5px; "
                "border-radius: 6px; background-color: #212121;"
            )
            self.log_message(
                f"⛔ KARA LİSTE &nbsp;|&nbsp; <b>{plate}</b> &nbsp;|&nbsp; "
                f"{surucu} &nbsp;|&nbsp; {firma} &nbsp;— ERİŞİM REDDEDİLDİ",
                "#212121",
            )

        else:  # yabanci / tanimsiz
            self.plate_label.setStyleSheet(
                "font-size: 38px; font-weight: bold; color: #FFF; padding: 12px; "
                "background-color: #C62828; border-radius: 8px; border: 2px solid #F44336;"
            )
            self.status_label.setText("🚨  YABANCI ARAÇ")
            self.status_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #FFF; padding: 5px; "
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


# ── Giriş Noktası ─────────────────────────────────────────────────────────────
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
