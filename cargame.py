import pygame
import random
import sys
import os
import cv2
import math
import numpy as np
from collections import Counter
import mediapipe as mp
try:
    import pygame.gfxdraw
except ImportError:
    pygame.gfxdraw = None
from get_movement import get_head_direction_simple, HeadDirectionFilter, get_head_turn_ratio, get_head_angles

pygame.init()
# Ses için mixer'ı başlat (daha uyumlu parametrelerle)
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
except pygame.error as e:
    print(f"Mixer başlatılamadı: {e}. Sesler devre dışı.")

# EKRAN AYARLARI
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Car Game (İki Şeritli)")
clock = pygame.time.Clock()

# --- KAMERA AYARLARI (OpenCV) ---
CAMERA_ACTIVE = False
cap = None
try:
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    
    if cap.isOpened():
        CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        CAMERA_ACTIVE = True
        print(f"Kamera aktif: {CAM_W}x{CAM_H}")
    else:
        raise IOError("Kamera açılamadı!")
    
except IOError as e:
    CAM_W = 320 
    CAM_H = 240
    print(f"HATA: {e}. Klavye kontrolüne geçildi.")

# KLASÖR YÖNETİMİ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def ASSET(p):
    return os.path.join(BASE_DIR, "assets", p)
HIGH_SCORE_FILE = os.path.join(BASE_DIR, "highscore.txt")
# RENKLER
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (40, 40, 40) 
BLUE = (0, 150, 255)
ASPHALT_COLOR = (20, 20, 20) 
GOLD = (255, 215, 0) # Altın rengi eklendi
RED = (255, 0, 0)

# HAREKET SENSÖRÜ AYARLARI
CENTER_DEADZONE = 40 
CENTER_X = CAM_W // 2
BRIGHTNESS_THRESHOLD = 8 

# HIZ ve BOYUT AYARLARI
# ŞERİTLER 2'ye İNDİRİLDİ. Şeritler ortalandı.
CAR_WIDTH, CAR_HEIGHT = 75, 120
LANES = [50, 125, 200, 275] # 4 Şerit için X koordinatları
HITBOX_FACTOR = 0.7
PLAYER_HORIZONTAL_SPEED = 12 # Serbest hareket için MAKSİMUM hız sabiti
MIN_ENEMY_SPAWN_RATE = 15   # Trafiğin en yoğun olduğu an (daha sık araç)
MAX_ENEMY_SPAWN_RATE = 100  # Trafiğin en sakin olduğu an (daha seyrek araç)
SPAWN_WAVE_PERIOD = 900     # Yoğunluk dalgasının periyodu (frame cinsinden)
# PITCH KONTROL AYARLARI (Yukarı bakma hızlanma, aşağı bakma yavaşlama)
PITCH_THRESHOLD = -10  # Derece. pitch > bu değer ise hızlanır (yukarı bakma)
PITCH_ACCEL_SENSITIVITY = 0.03   # Hızlanma hassasiyeti (yukarı bakarken)
PITCH_DECEL_SENSITIVITY = 0.1   # Yavaşlama hassasiyeti (aşağı bakarken)
MIN_SPEED = 2.0              # Ulaşılabilecek minimum hız
MAX_SPEED = 12.0             # Ulaşılabilecek maksimum hız

# İvme Sınırı (3 km/s / 60 FPS)
MAX_ACCEL_PER_FRAME = 0.05
MAX_DECEL_PER_FRAME = 0.2

FOLLOWING_DISTANCE = 20 # Araçların birbirini takip etme mesafesi (pixel)
COIN_RADIUS = 15
COIN_SPAWN_RATE = 90 # Altınların belirme sıklığı (daha yüksek sayı = daha seyrek)
STAR_SIZE = (35, 35)
STAR_SPAWN_RATE = 500 # Yıldızların belirme sıklığı (çok daha nadir)

CAM_VIEW_SIZE = (100, 75) # Köşede gösterilecek kamera görüntüsünün boyutu
# YOL ÇİZGİSİ AYARLARI
LINE_WIDTH = 10
LINE_HEIGHT = 50
LINE_SPACING = 30
ROAD_X = 50 
ROAD_W = WIDTH - 2 * ROAD_X 

# ARABA GÖRSELLERİ
try:
    # Ana görselleri yükle
    player_img = pygame.image.load(ASSET("player.png")).convert_alpha()
    enemy_img = pygame.image.load(ASSET("enemy.png")).convert_alpha()
    truck_img = pygame.image.load(ASSET("truck.png")).convert_alpha()
    sport_img = pygame.image.load(ASSET("sport.png")).convert_alpha()
    star_img = pygame.image.load(ASSET("star.png")).convert_alpha()
    
    # Görselleri yeniden boyutlandır
    player_img = pygame.transform.scale(player_img, (CAR_WIDTH, CAR_HEIGHT))
    enemy_img = pygame.transform.scale(enemy_img, (CAR_WIDTH, CAR_HEIGHT))
    truck_img = pygame.transform.scale(truck_img, (CAR_WIDTH, int(CAR_HEIGHT * 1.5))) # Kamyon daha uzun
    sport_img = pygame.transform.scale(sport_img, (CAR_WIDTH, CAR_HEIGHT))
    star_img = pygame.transform.scale(star_img, STAR_SIZE)

except pygame.error as e:
    print(f"Görsel yükleme hatası: {e}. Renkli dörtgenler kullanılacak.")
    player_img = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
    player_img.fill(BLUE) 
    enemy_img = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
    enemy_img.fill(WHITE)
    truck_img = pygame.Surface((CAR_WIDTH, int(CAR_HEIGHT * 1.5)), pygame.SRCALPHA)
    truck_img.fill((150, 75, 0)) # Kahverengi kamyon
    sport_img = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
    sport_img.fill((255, 0, 0)) # Kırmızı spor araba
    star_img = pygame.Surface(STAR_SIZE, pygame.SRCALPHA)
    star_img.fill(GOLD) # Geçici olarak altın rengi

# DÜŞMAN TİPLERİ
ENEMY_TYPES = {
    'normal': {'image': enemy_img, 'speed_mult': 1.2, 'size': (CAR_WIDTH, CAR_HEIGHT), 'chance': 0.60},
    'truck': {'image': truck_img, 'speed_mult': 1.1, 'size': (CAR_WIDTH, int(CAR_HEIGHT * 1.5)), 'chance': 0.15},
    'sport': {'image': sport_img, 'speed_mult': 1.5, 'size': (CAR_WIDTH, CAR_HEIGHT), 'chance': 0.25}
}

# --- YENİ BEZIER YÜZ MODELİ (bezier curve.py dosyasından) ---
ANCHOR_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Burun Ucu
    (0.0, 90.0, -25.0),          # Ağız
    (-65.0, -65.0, -40.0),       # Sol Göz
    (65.0, -65.0, -40.0),        # Sağ Göz
    (0.0, -150.0, -50.0)         # Alın
], dtype=np.float32)

def _get_bezier_curve(p0, p1, p2, segments=20):
    """3 Noktalı (Kuadratik) Bezier Eğrisi"""
    pts = []
    for t in np.linspace(0, 1, segments):
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        z = (1-t)**2 * p0[2] + 2*(1-t)*t * p1[2] + t**2 * p2[2]
        pts.append((x, y, z))
    return np.array(pts, dtype=np.float32)

def _get_cubic_bezier(p0, p1, p2, p3, segments=30):
    """4 Noktalı (Kübik) Bezier Eğrisi"""
    pts = []
    for t in np.linspace(0, 1, segments):
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        z = (1-t)**3*p0[2] + 3*(1-t)**2*t*p1[2] + 3*(1-t)*t**2*p2[2] + t**3*p3[2]
        pts.append((x, y, z))
    return np.array(pts, dtype=np.float32)

def _get_ellipse_points(center, rx, ry, segments=30, z_offset=0):
    """Elips şeklinde noktalar üretir"""
    pts = []
    for i in range(segments + 1):
        theta = 2.0 * math.pi * i / segments
        x = center[0] + rx * math.cos(theta)
        y = center[1] + ry * math.sin(theta)
        z = center[2] + z_offset
        pts.append((x, y, z))
    return np.array(pts, dtype=np.float32)

BEZIER_MODEL_FEATURES = {
    'left_eye': _get_ellipse_points((-65, -65, -40), 28, 14),
    'right_eye': _get_ellipse_points((65, -65, -40), 28, 14),
    'left_eyebrow': _get_bezier_curve((-95, -90, -30), (-65, -110, -20), (-35, -90, -30)),
    'right_eyebrow': _get_bezier_curve((35, -90, -30), (65, -110, -20), (95, -90, -30)),
    'upper_lip': _get_bezier_curve((-35, 100, -30), (0, 90, -35), (35, 100, -30)),
    'lower_lip': _get_bezier_curve((-35, 100, -30), (0, 120, -30), (35, 100, -30)),
    'nose': _get_bezier_curve((0, -80, -30), (0, -40, -10), (0, 0, 0)),
    'nose_base': _get_bezier_curve((-20, -10, -20), (0, 5, 0), (20, -10, -20)),
    'left_ear': _get_cubic_bezier((-140, -40, -100), (-180, -70, -140), (-170, 60, -140), (-145, 80, -110)),
    'right_ear': _get_cubic_bezier((140, -40, -100), (180, -70, -140), (170, 60, -140), (145, 80, -110)),
    'jaw': np.concatenate((
        _get_bezier_curve((-145, 80, -110), (-130, 160, -80), (0, 170, -50), segments=25), # Sol çene
        _get_bezier_curve((0, 170, -50), (130, 160, -80), (145, 80, -110), segments=25)  # Sağ çene
    ))
}


# SES EFEKTLERİ
try:
    coin_sound = pygame.mixer.Sound(ASSET("coin.wav"))
    # powerup_sound = pygame.mixer.Sound(ASSET("powerup.wav")) # Yıldız toplama sesi
    powerup_sound = None # Yıldız toplama sesi
    crash_sound = pygame.mixer.Sound(ASSET("crash.wav"))
    coin_sound.set_volume(0.5)
    crash_sound.set_volume(0.6)
    # Arka plan müziği
    # pygame.mixer.music.load(ASSET("music.wav"))
    # pygame.mixer.music.set_volume(0.3)
except pygame.error as e:
    print(f"Ses dosyası yükleme hatası: {e}. Sesler olmadan devam edilecek.")
    coin_sound = None
    powerup_sound = None
    crash_sound = None

# YÜKSEK SKOR YÖNETİMİ
def load_high_score():
    """highscore.txt dosyasından en yüksek skoru yükler."""
    try:
        with open(HIGH_SCORE_FILE, 'r') as f:
            return int(f.read())
    except (IOError, ValueError):
        return 0

def save_high_score(score):
    """Yeni en yüksek skoru dosyaya kaydeder."""
    with open(HIGH_SCORE_FILE, 'w') as f:
        f.write(str(score))

## PLAYER SINIFI (YUMUŞAK HAREKET İÇİN)
class Player:
    def __init__(self):
        # Başlangıç pozisyonu yolun ortası
        self.x = ROAD_X + (ROAD_W / 2) - (CAR_WIDTH / 2)
        self.y = HEIGHT - 120
        self.hitbox_w = int(CAR_WIDTH * HITBOX_FACTOR)
        self.hitbox_h = int(CAR_HEIGHT * HITBOX_FACTOR)
        self.offset_x = (CAR_WIDTH - self.hitbox_w) // 2
        self.offset_y = (CAR_HEIGHT - self.hitbox_h) // 2
        self.shield_active = False
        self.shield_end_time = 0
        self.mask = pygame.mask.from_surface(player_img) # Çarpışma için maske

    def get_hitbox(self):
        return pygame.Rect(
            self.x + self.offset_x, 
            self.y + self.offset_y, 
            self.hitbox_w, 
            self.hitbox_h
        )
    
    def activate_shield(self, duration_ms):
        if not self.shield_active:
            self.shield_active = True
            self.shield_end_time = pygame.time.get_ticks() + duration_ms

# Button çizme fonksiyonu
def draw_button(text, x, y, w, h, color, txt_color=WHITE):
    pygame.draw.rect(screen, color, (x, y, w, h), border_radius=8)
    font = pygame.font.SysFont(None, 32)
    label = font.render(text, True, txt_color)
    screen.blit(label, (x + w // 2 - label.get_width() // 2,
                            y + h // 2 - label.get_height() // 2))


# --- PITCH EYLEM TAMPONU SINIFI ---
class PitchActionBuffer:
    def __init__(self, buffer_size=20):
        self.buffer_size = buffer_size
        self.action_buffer = []

    def _classify_pitch(self, raw_pitch):
        if raw_pitch is None: 
            return 'neutral'
        
        if raw_pitch > PITCH_THRESHOLD: # Yukarı bakma (hızlanma)
            return 'accelerate'
        else: # Aşağı bakma (yavaşlama)
            return 'decelerate'

    def add_pitch(self, raw_pitch):
        action = self._classify_pitch(raw_pitch)
        self.action_buffer.append(action)
        if len(self.action_buffer) > self.buffer_size: self.action_buffer.pop(0)

    def get_buffered_action(self):
        if not self.action_buffer: 
            return 'neutral'
        action_counts = Counter(self.action_buffer)
        return action_counts.most_common(1)[0][0]


# Ana oyun fonksiyonu
def game(difficulty_speed, mode='endless'):
    player = Player() # Oyuncuyu şeritsiz başlat
    enemy_cars = [] 
    coins = []
    powerups = [] # Yıldız gibi güçlendirmeler için liste
    spawn_timer = 0
    star_spawn_timer = 0 # Yıldız spawn zamanlayıcısı
    coin_spawn_timer = 0 # Altın spawn zamanlayıcısı
    actual_speed = difficulty_speed # Anlık hızı tutan değişken
    frame_count = 0 # Harmonik dalga için sayaç
    road_lines = []
    for i in range(HEIGHT // (LINE_HEIGHT + LINE_SPACING) + 2): 
        road_lines.append(i * (LINE_HEIGHT + LINE_SPACING) - (LINE_HEIGHT + LINE_SPACING)) 
    score = 0 # Puan değişkeni
    distance_traveled = 0.0 # Kat edilen mesafeyi tutar

    is_timed_mode = (mode == 'timed')
    if is_timed_mode:
        game_duration = 60000  # 60 saniye (ms)
        start_time = pygame.time.get_ticks()

    # --- 3D Rehber için Matrisler ---
    # Gerçek kamera için solvePnP matrisi
    focal_length_cam = CAM_W
    camera_matrix_cam = np.array([
        [focal_length_cam, 0, CAM_W / 2],
        [0, focal_length_cam, CAM_H / 2],
        [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Sanal kamera için projectPoints matrisi (Oyun ekranına çizim için)
    focal_length_virtual = WIDTH * 1.5
    camera_matrix_virtual = np.array([
        [focal_length_virtual, 0, WIDTH / 2],
        [0, focal_length_virtual, HEIGHT * 0.3], # Dikey konumu ayarla
        [0, 0, 1]], dtype=np.float32)

    last_rvec = None # Yumuşatma için son rotasyon vektörü
    SMOOTHING_FACTOR = 0.2 # Yumuşatma faktörü (düşük değer = daha fazla yumuşatma)

    # PitchActionBuffer'ı başlat
    pitch_action_buffer = PitchActionBuffer()
    score_font = pygame.font.SysFont(None, 36) # Puan için font
    paused = False
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands # EKLENDİ: El algılama

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    direction_filter = HeadDirectionFilter(buffer_size=7)  # Filtre
    while True:
        current_time = pygame.time.get_ticks() 
        frame_count += 1
        clock.tick(60)
        
        # --- 0. KAMERA GİRDİSİ AL ---
        success, image = cap.read()
        if not success:
            continue

        # Görüntüyü yatayda çevirerek ayna yansıması efekti oluştur.
        # Bu, kafa hareketlerinin daha sezgisel ve doğru algılanmasına yardımcı olur.
        image = cv2.flip(image, 1)
        
        # Hassas kafa dönüş oranını al
        head_turn_ratio, debug_points = get_head_turn_ratio(face_mesh, image)
        pitch, _ = get_head_angles(face_mesh, image) # Hız kontrolü için pitch açısını al

        # Kafa takibi için kullanılan noktaları video üzerine çiz
        if len(debug_points) == 5:
            # Noktaları çiz
            for point in debug_points:
                cv2.circle(image, point, 3, (0, 0, 255), -1)
            
            # Üstteki 4 noktayı birleştirerek dörtgen oluştur
            cv2.line(image, debug_points[0], debug_points[1], (0, 255, 255), 1) # Alın -> Sol Göz İç
            cv2.line(image, debug_points[1], debug_points[3], (0, 255, 255), 1) # Sol Göz İç -> Burun
            cv2.line(image, debug_points[3], debug_points[2], (0, 255, 255), 1) # Burun -> Sağ Göz İç
            cv2.line(image, debug_points[2], debug_points[0], (0, 255, 255), 1) # Sağ Göz İç -> Alın
            # Burun ile ağız noktasını birleştir
            cv2.line(image, debug_points[3], debug_points[4], (255, 0, 255), 1)
        
        # --- HIZ KONTROLÜ (PITCH VE TAMPON İLE) ---
        if pitch is not None:
            pitch_action_buffer.add_pitch(pitch)
            buffered_action = pitch_action_buffer.get_buffered_action()

            if buffered_action == 'accelerate': # Yukarı bakma (hızlanma)
                speed_change = math.copysign(PITCH_ACCEL_SENSITIVITY, pitch - PITCH_THRESHOLD)
                speed_change = max(speed_change, MAX_ACCEL_PER_FRAME) # Maksimum hızlanma sınırı
                actual_speed += speed_change
            elif buffered_action == 'decelerate': # Aşağı bakma (yavaşlama)
                speed_change = math.copysign(PITCH_DECEL_SENSITIVITY, pitch - PITCH_THRESHOLD)
                speed_change = min(speed_change, MAX_DECEL_PER_FRAME) # Maksimum yavaşlama sınırı
                actual_speed += speed_change
            else: # Nötr veya deadzone içinde, hızı yavaşça başlangıç hızına döndür
                if actual_speed > difficulty_speed: actual_speed = max(difficulty_speed, actual_speed - 0.05)
                elif actual_speed < difficulty_speed: actual_speed = min(difficulty_speed, actual_speed + 0.05)
        else: # Pitch değeri alınamıyorsa hızı yavaşça başlangıç hızına döndür
            if actual_speed > difficulty_speed: actual_speed = max(difficulty_speed, actual_speed - 0.05)
            elif actual_speed < difficulty_speed: actual_speed = min(difficulty_speed, actual_speed + 0.05)
        actual_speed = max(MIN_SPEED, min(actual_speed, MAX_SPEED)) # Hızı sınırlar içinde tut

        # Kat edilen mesafeyi güncelle
        distance_traveled += actual_speed

        # Süreli mod kontrolü
        if is_timed_mode:
            elapsed_time = current_time - start_time
            remaining_time_ms = game_duration - elapsed_time
            if remaining_time_ms <= 0:
                game_over_screen(score, reason='timeout') # Süre doldu, oyunu bitir
                return

        # Kalkan süresini kontrol et
        if player.shield_active and current_time > player.shield_end_time:
            player.shield_active = False
        
        # --- 1. YOL ÇİZİMİ ---
        screen.fill(GRAY) 
        pygame.draw.rect(screen, ASPHALT_COLOR, (ROAD_X, 0, ROAD_W, HEIGHT)) 
        
        # Üç şeritli yol için iki çizgi çizer
        line_x1 = ROAD_X + ROAD_W // 4
        line_x2 = ROAD_X + ROAD_W * 2 // 4
        line_x3 = ROAD_X + ROAD_W * 3 // 4
        for i in range(len(road_lines)):
            road_lines[i] += actual_speed 
            # Sol çizgi
            pygame.draw.rect(screen, WHITE, (line_x1 - LINE_WIDTH // 2, road_lines[i], LINE_WIDTH, LINE_HEIGHT))
            # Sağ çizgi
            pygame.draw.rect(screen, WHITE, (line_x2 - LINE_WIDTH // 2, road_lines[i], LINE_WIDTH, LINE_HEIGHT))
            # Yeni sağ çizgi
            pygame.draw.rect(screen, WHITE, (line_x3 - LINE_WIDTH // 2, road_lines[i], LINE_WIDTH, LINE_HEIGHT))
            if road_lines[i] > HEIGHT:
                road_lines[i] = -LINE_HEIGHT - LINE_SPACING 

        # EVENTLER
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if CAMERA_ACTIVE:
                    cap.release()
                pygame.quit()
                sys.exit()

            # PAUSE VE KLAVYE KONTROLLERİ
            if event.type == pygame.KEYDOWN:
                # P tuşu ile duraklatma/başlatma
                if event.key == pygame.K_p:
                    paused = not paused
                
            # MOUSE İLE PAUSE KONTROLÜ
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Pause butonunun konumunu ve boyutunu tanımla
                pause_button_w, pause_button_h = 40, 40
                pause_button_x = (WIDTH - CAM_VIEW_SIZE[0] - 10) + (CAM_VIEW_SIZE[0] - pause_button_w) // 2
                pause_button_y = 10 + CAM_VIEW_SIZE[1] + 10

                mouse = pygame.mouse.get_pos()
                if pause_button_x < mouse[0] < pause_button_x + pause_button_w and pause_button_y < mouse[1] < pause_button_y + pause_button_h:
                    paused = not paused
                if paused and 120 < mouse[0] < 280 and 260 < mouse[1] < 310:
                    paused = False

        # --- PAUSE DURUMU ---
        if paused:
            font = pygame.font.SysFont(None, 48)
            txt = font.render("PAUSED", True, WHITE)
            screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 180))
            draw_button("DEVAM ET", 120, 260, 160, 50, BLUE)
            pygame.draw.rect(screen, WHITE, (350, 10, 40, 40))
            pygame.draw.rect(screen, BLACK, (360, 20, 5, 20))
            pygame.draw.rect(screen, BLACK, (375, 20, 5, 20))
            pygame.display.update()
            continue

        # --- KLAVYE KONTROLÜ (SÜREKLİ HAREKET) ---
        if not paused:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player.x -= PLAYER_HORIZONTAL_SPEED
            if keys[pygame.K_RIGHT]:
                player.x += PLAYER_HORIZONTAL_SPEED

        # --- KAMERA KONTROL MANTIĞI (SERBEST HAREKET) ---
        if CAMERA_ACTIVE and not paused:
            # Kafa dönüş oranını -1.0 ile 1.0 arasında sınırla (clamping).
            # Bu, yüz kaybolduğunda oluşabilecek ani değer sıçramalarını engeller.
            clamped_ratio = max(-1.0, min(1.0, head_turn_ratio))
            # Görüntü çevrildiği için artık oran doğru yönü veriyor.
            player.x += clamped_ratio * PLAYER_HORIZONTAL_SPEED

        # --- OYUNCUYU YOL İÇİNDE TUTMA ---
        player.x = max(ROAD_X, player.x) # Sol sınır
        player.x = min(ROAD_X + ROAD_W - CAR_WIDTH, player.x) # Sağ sınır


        # ENEMY SPAWN
        spawn_timer += 1
        coin_spawn_timer += 1
        star_spawn_timer += 1

        # Harmonik dalgaya göre anlık spawn oranını hesapla
        # math.sin'in çıktısı (-1, 1) arasındadır. Bunu (0, 1) arasına çekiyoruz.
        wave = (math.sin(frame_count * 2 * math.pi / SPAWN_WAVE_PERIOD) + 1) / 2
        current_spawn_rate = MIN_ENEMY_SPAWN_RATE + (MAX_ENEMY_SPAWN_RATE - MIN_ENEMY_SPAWN_RATE) * wave

        if spawn_timer > current_spawn_rate:
            spawn_timer = 0
            
            # Olasılıklara göre bir düşman tipi seç
            enemy_type_name = random.choices(
                list(ENEMY_TYPES.keys()), 
                weights=[v['chance'] for v in ENEMY_TYPES.values()], 
                k=1
            )[0]
            enemy_type = ENEMY_TYPES[enemy_type_name]

            # Seçilen tipe göre özellikleri ayarla
            lane = random.choice([0, 1, 2, 3])
            enemy_w, enemy_h = enemy_type['size']
            enemy_x_draw = LANES[lane]
            enemy_y_draw = -enemy_h
            
            hitbox_w = int(enemy_w * HITBOX_FACTOR)
            hitbox_h = int(enemy_h * HITBOX_FACTOR)
            offset_x = (enemy_w - hitbox_w) // 2
            offset_y = (enemy_h - hitbox_h) // 2
            
            hitbox = pygame.Rect(
                enemy_x_draw + offset_x, 
                enemy_y_draw + offset_y, 
                hitbox_w, 
                hitbox_h)
            
            enemy_dict = {'hitbox': hitbox, 'draw_x': enemy_x_draw, 'draw_y': enemy_y_draw, 'type': enemy_type, 'mask': pygame.mask.from_surface(enemy_type['image'])}
            
            enemy_cars.append(enemy_dict)

        # COIN SPAWN (Altın oluşturma)
        if coin_spawn_timer > COIN_SPAWN_RATE:
            lane = random.choice([0, 1, 2, 3])
            # Altının X pozisyonu şeridin ortası olacak
            coin_x = LANES[lane] + CAR_WIDTH // 2
            coin_y = -COIN_RADIUS
            
            coin_rect = pygame.Rect(coin_x - COIN_RADIUS, coin_y - COIN_RADIUS, COIN_RADIUS * 2, COIN_RADIUS * 2)
            coins.append(coin_rect)
            coin_spawn_timer = 0

        # STAR SPAWN (Yıldız oluşturma)
        if star_spawn_timer > STAR_SPAWN_RATE:
            lane = random.choice([0, 1, 2, 3])
            star_x = LANES[lane] + (CAR_WIDTH // 2) - (STAR_SIZE[0] // 2)
            star_y = -STAR_SIZE[1]
            powerups.append(pygame.Rect(star_x, star_y, STAR_SIZE[0], STAR_SIZE[1]))
            star_spawn_timer = 0


        # ENEMY MOVE
        # Araçların birbirine girmesini önleyen ve takip mesafesini koruyan mantık
        for i, car1 in enumerate(enemy_cars):
            # Her aracın potansiyel hızını hesapla
            current_speed = actual_speed * car1['type']['speed_mult']

            # Bu aracın önündeki araçları kontrol et
            for j, car2 in enumerate(enemy_cars):
                if i == j:
                    continue

                # Aynı şeritteler mi ve car2, car1'in önünde mi?
                if car1['draw_x'] == car2['draw_x'] and car2['draw_y'] > car1['draw_y']:
                    # Araçların ön ve arka tamponları arasındaki mesafe
                    distance = car2['draw_y'] - (car1['draw_y'] + car1['type']['size'][1])

                    # Eğer mesafe çok azaldıysa ve arkadaki araç daha hızlıysa
                    if distance < FOLLOWING_DISTANCE:
                        # Öndeki aracın hızını al
                        front_car_speed = actual_speed * car2['type']['speed_mult']
                        # Arkadaki aracın hızını, öndekinin hızından daha yüksek olmayacak şekilde ayarla
                        current_speed = min(current_speed, front_car_speed)
            
            # Hesaplanan nihai hız ile aracı hareket ettir
            car1['hitbox'].y += current_speed
            car1['draw_y'] += current_speed
            
        # COIN MOVE (Altınları hareket ettir)
        for coin in coins:
            coin.y += actual_speed

        # POWERUP MOVE (Yıldızları hareket ettir)
        for p in powerups:
            p.y += actual_speed

        enemy_cars = [e for e in enemy_cars if e['hitbox'].y < HEIGHT + 50]
        coins = [c for c in coins if c.y < HEIGHT + 50] # Ekran dışına çıkan altınları sil
        powerups = [p for p in powerups if p.y < HEIGHT + 50] # Ekran dışına çıkan yıldızları sil

        # ÇARPIŞMA KONTROL
        player_hitbox = player.get_hitbox()
        for e in enemy_cars:
            # Maske tabanlı çarpışma kontrolü
            offset_x = e['draw_x'] - player.x
            offset_y = e['draw_y'] - player.y
            # overlap metodu, iki maske arasında kesişim varsa True döner
            if player.mask.overlap(e['mask'], (offset_x, offset_y)) and not player.shield_active:
                if crash_sound:
                    crash_sound.play()
                game_over_screen(score, reason='crash') # Oyunu bitirirken skoru gönder
                return

        # ALTIN TOPLAMA KONTROLÜ
        collected_coins = []
        for coin in coins:
            if player_hitbox.colliderect(coin):
                score += 10 # Puanı artır
                if coin_sound:
                    coin_sound.play()
                collected_coins.append(coin)
        coins = [c for c in coins if c not in collected_coins]

        # YILDIZ TOPLAMA KONTROLÜ
        collected_powerups = []
        for p in powerups:
            if player_hitbox.colliderect(p):
                player.activate_shield(5000) # 5 saniyelik kalkan/ölümsüzlük
                if powerup_sound:
                    powerup_sound.play()
                collected_powerups.append(p)
        powerups = [p for p in powerups if p not in collected_powerups]

        # ÇİZİM
        screen.blit(player_img, (player.x, player.y))
        # Kalkan aktifse görsel bir efekt ekle
        if player.shield_active:
            pygame.draw.rect(screen, (0, 200, 255, 100), player.get_hitbox(), 4, border_radius=10)
        
        # Altınları çiz
        for coin in coins:
            pygame.draw.circle(screen, GOLD, coin.center, COIN_RADIUS)

        # Yıldızları çiz
        for p in powerups:
            screen.blit(star_img, p.topleft)

        for e in enemy_cars:
            # Her düşmanı kendi görseliyle çiz
            screen.blit(e['type']['image'], (e['draw_x'], e['draw_y']))

        # PUANI EKRANA YAZDIR
        score_text = score_font.render(f"SKOR: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        # HIZ GÖSTERGESİ
        speed_display = int(actual_speed * 15) # Hızı daha anlamlı bir sayıya çevir
        speed_text = score_font.render(f"HIZ: {speed_display} KM/H", True, WHITE)
        screen.blit(speed_text, (10, 40))

        # MESAFE GÖSTERGESİ
        display_distance = distance_traveled / 5000 # Değeri KM'ye benzetmek için ölçekle
        distance_text = score_font.render(f"MESAFE: {display_distance:.1f} KM", True, WHITE)
        screen.blit(distance_text, (10, 70))

        # PITCH GÖSTERGESİ
        if pitch is not None:
            pitch_text = score_font.render(f"PITCH: {pitch:.1f}", True, WHITE)
            screen.blit(pitch_text, (10, 100))

        # SÜRE GÖSTERGESİ (Süreli modda)
        if is_timed_mode:
            remaining_seconds = max(0, remaining_time_ms // 1000)
            
            # Son 10 saniyede rengi kırmızı yap
            timer_color = RED if remaining_seconds <= 10 else WHITE
            
            timer_text = score_font.render(f"{remaining_seconds}s", True, timer_color)
            text_rect = timer_text.get_rect(center=(WIDTH // 2, 25))
            screen.blit(timer_text, text_rect)


        # EKRANIN ORTASINDA HAREKET EDEN 3D BEZIER REHBERİ
        if len(debug_points) == 5:
            # Gelen noktaları solvePnP'nin beklediği sıraya diz
            # Beklenen: Burun, Ağız, Sol Göz, Sağ Göz, Alın
            # Gelen: Alın(0), Sol Göz(1), Sağ Göz(2), Burun(3), Ağız(4)
            input_points_2d = np.array([
                debug_points[3], # Burun
                debug_points[4], # Ağız
                debug_points[1], # Sol Göz
                debug_points[2], # Sağ Göz
                debug_points[0]  # Alın
            ], dtype=np.float32)

            # Kafa pozunu (rotasyon) hesapla
            success, rvec, tvec = cv2.solvePnP(
                ANCHOR_POINTS_3D, input_points_2d, camera_matrix_cam, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )

            if success:
                # --- Rotasyon Vektörünü Yumuşatma ---
                if last_rvec is None:
                    last_rvec = rvec # İlk başarılı tespitte ata
                
                # Mevcut ve önceki rotasyon arasında yumuşak bir geçiş yap (Lerp)
                smoothed_rvec = last_rvec * (1.0 - SMOOTHING_FACTOR) + rvec * SMOOTHING_FACTOR
                last_rvec = smoothed_rvec # Bir sonraki kare için güncelle

                # Sanal bir tvec ile modeli ekranın önüne yerleştir
                tvec_virtual = np.array([[0.0], [0.0], [1200.0]])
                line_color = (0, 255, 255) # Saydamlık kaldırıldı, renk daha belirgin

                # Modeldeki her bir parçayı sanal kameraya göre ekrana yansıt ve çiz
                for part_name, points_3d in BEZIER_MODEL_FEATURES.items():
                    projected_2d, _ = cv2.projectPoints(points_3d, smoothed_rvec, tvec_virtual, camera_matrix_virtual, dist_coeffs)
                    guide_points = projected_2d.reshape(-1, 2).astype(int)
                    
                    if len(guide_points) > 2:
                        # Gözler gibi kapalı şekilleri belirle
                        is_closed = 'eye' in part_name and 'eyebrow' not in part_name
                        pygame.draw.lines(screen, line_color, is_closed, guide_points, 2) # Kalınlık 2 olarak ayarlandı

        # KAMERA GÖRÜNTÜSÜNÜ ÇİZ
        if CAMERA_ACTIVE:
            # Görüntüyü Pygame için hazırla
            cam_img_resized = cv2.resize(image, CAM_VIEW_SIZE)
            cam_img_rgb = cv2.cvtColor(cam_img_resized, cv2.COLOR_BGR2RGB)
            cam_surface = pygame.image.frombuffer(cam_img_rgb.tobytes(), CAM_VIEW_SIZE, "RGB")

            # Sağ üst köşeye yerleştir
            screen.blit(cam_surface, (WIDTH - CAM_VIEW_SIZE[0] - 10, 10))

            # PAUSE IKONUNU KAMERANIN ALTINA ÇİZ
            pause_button_w, pause_button_h = 40, 40
            pause_button_x = (WIDTH - CAM_VIEW_SIZE[0] - 10) + (CAM_VIEW_SIZE[0] - pause_button_w) // 2
            pause_button_y = 10 + CAM_VIEW_SIZE[1] + 10
            pygame.draw.rect(screen, WHITE, (pause_button_x, pause_button_y, pause_button_w, pause_button_h))
            pygame.draw.rect(screen, BLACK, (pause_button_x + 10, pause_button_y + 10, 5, 20))
            pygame.draw.rect(screen, BLACK, (pause_button_x + 25, pause_button_y + 10, 5, 20))

        pygame.display.update()


# GAME OVER EKRANI, ANA MENÜ ve Diğer Fonksiyonlar (Aynı kalır)
def game_over_screen(final_score, reason='crash'): # Bitiş nedeni parametresi eklendi
    high_score = load_high_score()
    new_high_score = False
    if final_score > high_score:
        high_score = final_score
        save_high_score(high_score)
        new_high_score = True

    while True:
        screen.fill(BLACK)
        font = pygame.font.SysFont(None, 60)
        score_font = pygame.font.SysFont(None, 48)

        # Bitiş nedenine göre mesajı belirle
        message = "SÜRE DOLDU!" if reason == 'timeout' else "OYUN BİTTİ"

        txt = font.render(message, True, WHITE)
        screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 180))
        draw_button("TEKRAR OYNA", 120, 300, 160, 60, BLUE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if CAMERA_ACTIVE and cap is not None:
                    cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse = pygame.mouse.get_pos()
                if 120 < mouse[0] < 280 and 300 < mouse[1] < 360:
                    main_menu()
        
        # Final skoru göster
        score_text = score_font.render(f"Skorun: {final_score}", True, WHITE)
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 230))

        # En yüksek skoru göster
        high_score_color = GOLD if new_high_score else WHITE
        high_score_label = "Yeni Rekor!" if new_high_score else "En Yüksek Skor"
        high_score_text = score_font.render(f"{high_score_label}: {high_score}", True, high_score_color)
        screen.blit(high_score_text, (WIDTH // 2 - high_score_text.get_width() // 2, 270))

        pygame.display.update()

def main_menu():
    # Arka plan müziğini sadece çalmıyorsa başlat
    if not pygame.mixer.music.get_busy():
        try:
            pygame.mixer.music.play(loops=-1)
        except pygame.error:
            print("Arka plan müziği çalınamadı.")
    while True:
        screen.fill(GRAY) 
        font = pygame.font.SysFont(None, 48)
        title = font.render("2D CAR GAME", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 80))
        draw_button("SÜRELİ MOD (1 DK)", 80, 220, 240, 60, BLUE)
        draw_button("SÜRESİZ MOD", 80, 300, 240, 60, BLUE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if CAMERA_ACTIVE and cap is not None: 
                    cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if 80 < x < 320:
                    if 220 < y < 280: # SÜRELİ MOD
                        game(difficulty_speed=6, mode='timed')
                    if 300 < y < 360: # SÜRESİZ MOD
                        game(difficulty_speed=6, mode='endless')
        pygame.display.update()

main_menu()