import cv2
import mediapipe as mp
import numpy as np
import math

# --- MediaPipe ve Modeller ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 

# ... (Kafa yönü tespiti için gerekli sabitler - AYNI)
NOSE_TIP = 1
CHIN = 199
LEFT_EYE_LEFT_CORNER = 33
RIGHT_EYE_RIGHT_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# 3D Yüz Modeli Referans Noktaları (solvePnP için)
ANCHOR_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Burun Ucu
    (0.0, 90.0, -25.0),          # Ağız
    (-65.0, -65.0, -40.0),       # Sol Göz
    (65.0, -65.0, -40.0),        # Sağ Göz
    (0.0, -150.0, -50.0)         # Alın
], dtype=np.float32)

model_points = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), 
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# --- Kafa yönü fonksiyonları ve HeadDirectionFilter sınıfı - AYNI ---

def get_head_direction_improved(face_mesh_model, image):
    (h, w) = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False 
    results = face_mesh_model.process(image_rgb)
    if not results.multi_face_landmarks:
        return "YÜZ TESPİT EDİLEMEDİ"
    face_landmarks = results.multi_face_landmarks[0]
    focal_length = w
    camera_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float32)
    lm_list = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
    image_points = np.array([
        lm_list[NOSE_TIP], lm_list[CHIN], lm_list[LEFT_EYE_LEFT_CORNER], 
        lm_list[RIGHT_EYE_RIGHT_CORNER], lm_list[LEFT_MOUTH_CORNER], lm_list[RIGHT_MOUTH_CORNER]
    ], dtype=np.float32)
    model_points_reshaped = model_points.reshape((-1, 1, 3))
    image_points_reshaped = image_points.reshape((-1, 1, 2))
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points_reshaped, image_points_reshaped, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return "POZ TESPİT EDİLEMEDİ"
        rmat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat([rmat, translation_vector])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        yaw = euler_angles[1, 0]
        pitch = euler_angles[0, 0]
        YAW_THRESHOLD = 25
        PITCH_THRESHOLD = 20
        if yaw < -YAW_THRESHOLD:
            return "SOL"
        elif yaw > YAW_THRESHOLD:
            return "SAĞ"
        elif abs(pitch) > PITCH_THRESHOLD:
            return "YUKARI/AŞAĞI"
        else:
            return "DÜZ"
    except Exception as e:
        return f"HESAPLAMA HATASI: {str(e)}"

def get_head_direction_simple(face_mesh_model, image):
    (h, w) = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False 
    results = face_mesh_model.process(image_rgb)
    if not results.multi_face_landmarks:
        return "YÜZ TESPİT EDİLEMEDİ"
    face_landmarks = results.multi_face_landmarks[0]
    lm_list = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
    left_eye_left = lm_list[33]   
    right_eye_right = lm_list[263] 
    nose_tip = lm_list[1]         
    face_width = right_eye_right[0] - left_eye_left[0]
    face_center_x = left_eye_left[0] + face_width / 2
    nose_offset_ratio = (nose_tip[0] - face_center_x) / (face_width / 2)
    THRESHOLD = 0.15
    if nose_offset_ratio < -THRESHOLD:
        return "RIGHT"
    elif nose_offset_ratio > THRESHOLD:
        return "LEFT"
    else:
        return "CENTER"

def get_head_turn_ratio(face_mesh_model, image):
    """
    Kafanın ne kadar sağa/sola döndüğünü oransal bir değer (-1.0 ile 1.0 arası) olarak döndürür.
    Ayrıca görselleştirme için kullanılan noktaları da döndürür.
    """
    (h, w) = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False 
    results = face_mesh_model.process(image_rgb)
    if not results.multi_face_landmarks:
        return 0.0, [] # Yüz yoksa hareket yok, boş liste döndür
    face_landmarks = results.multi_face_landmarks[0]
    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
    
    left_eye_left = lm_list[33]   
    right_eye_right = lm_list[263]
    nose_tip = lm_list[1]         
    
    # Görselleştirme için daha uygun noktalar seçelim
    forehead_tip = lm_list[10]
    left_eye_right = lm_list[133] # Sol göz iç köşe
    right_eye_left = lm_list[362] # Sağ göz iç köşe
    mouth_center = lm_list[13]    # Ağız ortası (üst dudak)

    # Noktaları mantıksal bir sırayla döndür
    debug_points = [forehead_tip, left_eye_right, right_eye_left, nose_tip, mouth_center]

    face_width = right_eye_right[0] - left_eye_left[0]
    if face_width == 0: return 0.0, debug_points

    face_center_x = left_eye_left[0] + face_width / 2
    nose_offset_ratio = (nose_tip[0] - face_center_x) / (face_width / 2)
    
    return nose_offset_ratio, debug_points

def get_head_angles(face_mesh_model, image):
    """
    Kafanın pitch ve yaw açılarını derece cinsinden döndürür.
    Pitch: Yukarı/Aşağı (Negatif = Yukarı, Pozitif = Aşağı)
    Yaw: Sağa/Sola (Negatif = Sola, Pozitif = Sağa)
    """
    h, w, _ = image.shape
    
    # Kamera matrisi
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Görüntüyü işle
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False 
    results = face_mesh_model.process(image_rgb)
    image_rgb.flags.writeable = True

    if not results.multi_face_landmarks:
        return None, None # Yüz bulunamadı

    face_landmarks = results.multi_face_landmarks[0]
    lm_list = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
    
    # solvePnP için 2D noktaları hazırla (ANCHOR_POINTS_3D ile aynı sırada)
    input_points_2d = np.array([
        lm_list[1],   # Burun Ucu
        lm_list[13],  # Ağız Ortası
        lm_list[133], # Sol Göz İç
        lm_list[362], # Sağ Göz İç
        lm_list[10]   # Alın
    ], dtype=np.float32)

    try:
        success, rvec, tvec = cv2.solvePnP(
            ANCHOR_POINTS_3D, input_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            return None, None

        rmat, _ = cv2.Rodrigues(rvec)
        proj_mat = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)
        return euler_angles[0, 0], euler_angles[1, 0] # pitch, yaw
    except Exception:
        return None, None

class HeadDirectionFilter:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.direction_buffer = []
    
    def add_direction(self, direction):
        self.direction_buffer.append(direction)
        if len(self.direction_buffer) > self.buffer_size:
            self.direction_buffer.pop(0)
    
    def get_filtered_direction(self):
        if not self.direction_buffer:
            return "BELIRSIZ"
        from collections import Counter
        most_common = Counter(self.direction_buffer).most_common(1)
        return most_common[0][0]

def hand_detection(hands, image):
    hand_results = hands.process(image)
    return hand_results

# -----------------------------------------------------
# GÜNCELLENMİŞ UZAKLIK FONKSİYONU VE EL JEST TESPİTİ
# ELİN ARKA TARAFI İÇİN MESAFE TABANLI KONTROL
# -----------------------------------------------------

def calculate_distance(p1, p2):
    """İki MediaPipe noktasının 2D (X, Y) Öklid uzaklığını hesaplar."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

# MediaPipe Hand Landmarks İndeksleri
FINGER_TIPS = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]
FINGER_PIPS = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
               mp_hands.HandLandmark.PINKY_PIP]
FINGER_MCPS = [mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.INDEX_FINGER_MCP,
               mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
               mp_hands.HandLandmark.PINKY_MCP]
WRIST = mp_hands.HandLandmark.WRIST

def is_palm_facing_camera(hand_landmarks):
    """
    Elin avuç içinin kameraya dönük olup olmadığını tespit eder.
    Daha güvenilir bir yöntem: işaret ve serçe parmak MCP'lerinin karşılaştırması.
    """
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Avuç içi kameraya dönükse, işaret parmağı MCP'si serçe parmağı MCP'sinden daha solda olur
    return index_mcp.x < pinky_mcp.x

def get_finger_open_status(hand_landmarks, finger_tip_index, finger_pip_index, finger_mcp_index, palm_facing_camera):
    """
    Bir parmağın açık olup olmadığını belirler.
    Hem ön hem de arka taraf için çalışan mesafe tabanlı yöntem.
    """
    tip = hand_landmarks.landmark[finger_tip_index]
    pip = hand_landmarks.landmark[finger_pip_index]
    mcp = hand_landmarks.landmark[finger_mcp_index]
    
    if palm_facing_camera:
        # AVUÇ İÇİ KAMERAYA DÖNÜK - Geleneksel Yöntem
        return tip.y < pip.y
    else:
        # ELİN ARKASI KAMERAYA DÖNÜK - Mesafe Tabanlı Yöntem
        # TIP ile MCP arasındaki mesafe, PIP ile MCP arasındaki mesafeden büyükse parmak açık
        tip_to_mcp_distance = calculate_distance(tip, mcp)
        pip_to_mcp_distance = calculate_distance(pip, mcp)
        
        # Eşik değeri: PIP-MCP mesafesinin 1.3 katı
        return tip_to_mcp_distance > (pip_to_mcp_distance * 1.3)

def get_thumb_open_status(hand_landmarks, palm_facing_camera, is_left_hand):
    """
    Baş parmağın açık olup olmadığını belirler.
    """
    thumb_tip = hand_landmarks.landmark[FINGER_TIPS[0]]
    thumb_ip = hand_landmarks.landmark[FINGER_PIPS[0]]
    thumb_mcp = hand_landmarks.landmark[FINGER_MCPS[0]]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    
    # Mesafe kontrolleri
    thumb_tip_ip_distance = calculate_distance(thumb_tip, thumb_ip)
    thumb_open_distance = calculate_distance(thumb_tip, thumb_mcp)
    thumb_base_distance = calculate_distance(thumb_cmc, thumb_mcp)
    
    # Eşik değeri: temel uzunluğun 1.2 katı
    distance_threshold = thumb_base_distance * 1.2
    
    if palm_facing_camera:
        # AVUÇ İÇİ KAMERAYA DÖNÜK
        if is_left_hand:
            is_thumb_open_position = thumb_tip.x < thumb_mcp.x
        else:
            is_thumb_open_position = thumb_tip.x > thumb_mcp.x
    else:
        # ELİN ARKASI KAMERAYA DÖNÜK - Mesafe tabanlı
        # TIP ile MCP arasındaki mesafe yeterince büyükse açık
        is_thumb_open_position = thumb_open_distance > distance_threshold
    
    return is_thumb_open_position and (thumb_open_distance > distance_threshold)

def is_fist(hand_landmarks):
    """
    Yumruk hareketini daha güvenilir bir şekilde tespit eder.
    Tüm parmak uçlarının bileğe olan mesafesini kontrol eder.
    """
    try:
        wrist = hand_landmarks.landmark[WRIST]
        
        # Bilek ile orta parmak kökü (MCP) arasındaki mesafeyi referans alalım
        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        base_distance = calculate_distance(wrist, middle_mcp)
        
        # Her parmak ucunun bileğe olan mesafesini kontrol et
        for tip_index in FINGER_TIPS:
            tip = hand_landmarks.landmark[tip_index]
            distance_to_wrist = calculate_distance(tip, wrist)
            # Eğer parmak ucu, referans mesafesinden daha uzaktaysa, yumruk değildir.
            if distance_to_wrist > base_distance * 1.2: # 1.2'lik bir tolerans payı
                return False
        return True # Tüm parmaklar yeterince yakınsa yumruktur
    except:
        return False

def get_hand_gesture(hand_landmarks):
    """
    Tespit edilen el noktalarına göre basit jestleri (sayıları) belirler.
    Hem ön hem de arka taraf için çalışır.
    """
    
    # Elin avuç içi kameraya dönük mü?
    palm_facing_camera = is_palm_facing_camera(hand_landmarks)
    
    # Elin yönünü belirle (sol el mi sağ el mi)
    wrist = hand_landmarks.landmark[WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    is_left_hand = wrist.x > middle_mcp.x
    
    # 1. BAŞ PARMAK KONTROLÜ
    is_thumb_open = get_thumb_open_status(hand_landmarks, palm_facing_camera, is_left_hand)
    
    # 2. DİĞER PARMAKLARIN KONTROLÜ
    fingers_open = []
    
    # İşaret, Orta, Yüzük, Serçe parmaklar
    for i in range(1, 5):
        is_finger_open = get_finger_open_status(
            hand_landmarks, 
            FINGER_TIPS[i], 
            FINGER_PIPS[i], 
            FINGER_MCPS[i],
            palm_facing_camera
        )
        fingers_open.append(is_finger_open)
    
    # Açık parmak sayısı
    open_finger_count = sum(fingers_open)
    total_open_fingers = open_finger_count + (1 if is_thumb_open else 0)
    
    # DEBUG: Parmak durumlarını göster
    debug_info = f"P:{'A' if palm_facing_camera else 'S'}"
    debug_info += f" B:{1 if is_thumb_open else 0}"
    for i, open_status in enumerate(fingers_open):
        debug_info += f" {i+1}:{1 if open_status else 0}"
    
    # SAYI 1
    if total_open_fingers == 1:
        if is_thumb_open and open_finger_count == 0:
            return "SAYI: 1 (Baş Parmak)"
        elif fingers_open[0] and not is_thumb_open and open_finger_count == 1:
            return "SAYI: 1 (İşaret)"
        else:
            return f"SAYI: 1? {debug_info}"
    
    # SAYI 2
    elif total_open_fingers == 2:
        if fingers_open[0] and fingers_open[1] and not is_thumb_open:
            return "SAYI: 2 / BARIŞ"
        else:
            return f"SAYI: 2 {debug_info}"
    
    # SAYI 3
    elif total_open_fingers == 3:
        if fingers_open[0] and fingers_open[1] and fingers_open[2] and not is_thumb_open:
            return "SAYI: 3"
        else:
            return f"SAYI: 3 {debug_info}"
    
    # SAYI 4
    elif total_open_fingers == 4:
        if all(fingers_open) and not is_thumb_open:
            return "SAYI: 4"
        else:
            return f"SAYI: 4 {debug_info}"
    
    # SAYI 5
    elif total_open_fingers == 5:
        return "SAYI: 5 (AÇIK EL)"
    
    # DİĞER JESTLER
    elif is_thumb_open and fingers_open[0] and not any(fingers_open[1:]):
        return "L (İşaret ve Baş)"
    elif is_thumb_open and fingers_open[3] and not any(fingers_open[0:3]):
        return "TELEFON (ROCK)"
    
    # TANIMLANAMAYAN
    else:
        return f"DİĞER ({debug_info})"

## Ana program
if __name__ == "__main__":
    
    # --- MediaPipe Modelleri Başlatılıyor ---
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    hands = mp_hands.Hands(
        model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1
    )
    
    cap = cv2.VideoCapture(0)
    direction_filter = HeadDirectionFilter(buffer_size=7)
    
    print("Geliştirilmiş Kafa ve El Tespiti Başladı. Çıkmak için 'q' tuşuna basın...")
    print("NOT: Elin arka tarafı için mesafe tabanlı algılama")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False 
        
        hand_results = hand_detection(hands, image_rgb)

        image.flags.writeable = True

        hand_gesture = "EL TESPİT EDİLEMEDİ"
        hand_orientation = ""
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                
                # El yönünü tespit et
                palm_facing = is_palm_facing_camera(hand_landmarks)
                hand_orientation = "AVUÇ İÇİ" if palm_facing else "ELİN ARKASI"
                
                # JEST TESPİTİ YAPILIYOR
                hand_gesture = get_hand_gesture(hand_landmarks)
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        # --- Kafa Yönü Tespiti ---
        direction2 = get_head_direction_simple(face_mesh, image)
        direction_filter.add_direction(direction2)
        filtered_direction = direction_filter.get_filtered_direction()
        
        # Ekrana yazdır (Konsol)
        print(f"\rKafa Yon: {filtered_direction:15} | Jest: {hand_gesture:45}", end="", flush=True)
        
        # Görselleştirme (Görüntü üzerinde)
        cv2.putText(image, f"Kafa Yonu: {filtered_direction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # El Hareketi bilgisini ekle
        cv2.putText(image, f"Jest: {hand_gesture}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # El yönünü ekle
        cv2.putText(image, f"El Yonu: {hand_orientation}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                   
        cv2.imshow('Kafa ve El Tespiti', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # --- Kapanış ---
    face_mesh.close()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram sonlandırıldı.")