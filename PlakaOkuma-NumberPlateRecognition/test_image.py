import cv2
import imageProcessing as imgprocess
import sys

import os

# Örnek kullanım: python test_image.py C:\Users\Arda\Desktop\araba_fotoğrafları
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "test_araba.jpg"

if not os.path.exists(path):
    print(f"HATA: '{path}' bulunamadı.")
    sys.exit()

if os.path.isdir(path):
    # Eğer klasörse, içindeki tüm resimleri bul
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"HATA: '{path}' klasöründe hiç resim bulunamadı.")
        sys.exit()
        
    print(f"'{path}' klasöründeki {len(image_files)} resim işleniyor...")
    
    for img_path in image_files:
        print(f"\n--- İşleniyor: {os.path.basename(img_path)} ---")
        img = cv2.imread(img_path)
        if img is not None:
            result_img = imgprocess.rec(img, is_video=False)
            cv2.imshow("Plaka Okuma Sonucu", result_img)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
else:
    # Eğer tek bir dosyaysa
    print(f"'{path}' işleniyor...")
    img = cv2.imread(path)
    if img is not None:
        result_img = imgprocess.rec(img, is_video=False)
        cv2.imshow("Plaka Okuma Sonucu", result_img)
        print("Çıkmak için herhangi bir tuşa basın...")
        cv2.waitKey(0)

cv2.destroyAllWindows()
