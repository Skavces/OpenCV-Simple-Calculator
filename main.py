import cv2
import mediapipe as mp
import time

# Kamera ve Mediapipe eller modülü başlatma
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Tam ekran için genişlik
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Tam ekran için yükseklik

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Buton ayarları
buttons = [
    {'label': '1', 'x': 50, 'y': 50},
    {'label': '2', 'x': 130, 'y': 50},
    {'label': '3', 'x': 210, 'y': 50},
    {'label': '+', 'x': 290, 'y': 50},
    {'label': '4', 'x': 50, 'y': 130},
    {'label': '5', 'x': 130, 'y': 130},
    {'label': '6', 'x': 210, 'y': 130},
    {'label': '-', 'x': 290, 'y': 130},
    {'label': '7', 'x': 50, 'y': 210},
    {'label': '8', 'x': 130, 'y': 210},
    {'label': '9', 'x': 210, 'y': 210},
    {'label': '*', 'x': 290, 'y': 210},
    {'label': 'C', 'x': 50, 'y': 290},
    {'label': '0', 'x': 130, 'y': 290},
    {'label': '=', 'x': 210, 'y': 290},
    {'label': '/', 'x': 290, 'y': 290},
]
button_width, button_height = 60, 60  # Buton boyutları

# Hesap makinesi ekranı ve yazı ayarları
calc_display_x, calc_display_y = 50, 400
calc_display_width, calc_display_height = 300, 80
font = cv2.FONT_HERSHEY_SIMPLEXq
font_scale = 0.7
font_thickness = 2

# Hesap makinesi için başlangıç durumu
calc_input = ""
last_position = None
stay_duration = 1  # Parmağın aynı pozisyonda kalma süresi (saniye)
start_time = None

while True:
    success, img = cap.read()
    if not success:
        break

    # Görüntüyü ters çevir ve tam ekran ayarla
    img = cv2.flip(img, 1)

    # Görüntüyü RGB formatına dönüştür
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Elleri algıla
    result = hands.process(img_rgb)

    # Hesap makinesi ekranı çiz
    cv2.rectangle(img, (calc_display_x, calc_display_y), 
                  (calc_display_x + calc_display_width, calc_display_y + calc_display_height), 
                  (50, 50, 50), cv2.FILLED)
    cv2.putText(img, calc_input, (calc_display_x + 10, calc_display_y + calc_display_height // 2 + 10), 
                font, font_scale, (255, 255, 255), font_thickness)

    # Butonları çiz
    for button in buttons:
        cv2.rectangle(img, (button['x'], button['y']), 
                      (button['x'] + button_width, button['y'] + button_height), 
                      (100, 100, 100), cv2.FILLED)
        cv2.putText(img, button['label'], 
                    (button['x'] + 15, button['y'] + 40), font, font_scale, (255, 255, 255), font_thickness)

    # Elleri ve işaret parmağı uçlarını kontrol et
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # İşaret parmağı ucu (8 numaralı landmark) koordinatları
            index_finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1])
            index_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0])

            # Parmağın pozisyonunu kontrol et
            current_position = (index_finger_tip_x, index_finger_tip_y)
            if last_position is None or (abs(current_position[0] - last_position[0]) > 10 or abs(current_position[1] - last_position[1]) > 10):
                last_position = current_position
                start_time = time.time()  # Yeni pozisyon için süreyi başlat
            else:
                if time.time() - start_time > stay_duration:
                    # Parmağın bir buton üzerinde 2 saniye kaldığını kontrol et
                    for button in buttons:
                        if (button['x'] < index_finger_tip_x < button['x'] + button_width and 
                            button['y'] < index_finger_tip_y < button['y'] + button_height):
                            label = button['label']
                            if label == 'C':
                                calc_input = ""  # Temizle
                            elif label == '=':
                                try:
                                    calc_input = str(eval(calc_input))  # Hesapla
                                except:
                                    calc_input = "Error"
                            else:
                                calc_input += label  # Buton değerini ekle
                            last_position = None  # Yeni işlem için sıfırla
                            break

    # Görüntüyü göster
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
