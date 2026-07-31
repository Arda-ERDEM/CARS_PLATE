import numpy as np
import cv2
import imageProcessing as imgprocess

cap = cv2.VideoCapture(0)

# Kameranın okuma çözünürlüğünü maksimuma (Full HD / 1080p) zorluyoruz. 
# Böylece uzaktaki küçük plakalar kırpıldığında bile OCR için devasa kalır.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Kamera açılamadı! Lütfen bir kamera bağlı olduğundan emin olun.")
else:
    print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break
    img = imgprocess.rec(frame)
    cv2.imshow('Screen', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()