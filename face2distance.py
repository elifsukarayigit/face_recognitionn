import cv2
import numpy as np
import math

# Yüz Algılama için Haar Cascade sınıfı
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def calculate_distance(point1, point2):
    """İki nokta arasındaki öklidyen mesafeyi hesaplar."""
    
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Video akışını başlat
cap = cv2.VideoCapture(0)

# Odak noktası (örneğin, ekranın ortası)
focus_point = (320, 240)  # (x, y) formatında, ekranın ortası

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Grayscale'e dönüştür, gri donusum
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla kac tane 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Algılanan yüzler için 
    for (x, y, w, h) in faces:
         # Bounding box çizimi (x, y) sol üst köşe, (x+w, y+h) sağ alt köşe
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Yüzün merkezi burun kısmı
        face_center = (x + w // 2, y + h // 2)
        
        # Mesafeyi hesapla
        distance = calculate_distance(focus_point, face_center)
        
        # Merkezi ve mesafeyi görüntüle
        cv2.circle(frame, face_center, 5, (0, 255, 0), -1)
        cv2.line(frame, focus_point, face_center, (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f} px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Odak noktasını göster
    cv2.circle(frame, focus_point, 5, (0, 0, 255), -1)
    cv2.putText(frame, "Focus Point", (focus_point[0] + 10, focus_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Görüntüyü göster
    cv2.imshow("Face Distance Measurement", frame)

    # Çıkış için 'q' tuş
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
