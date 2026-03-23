import cv2
import numpy as np
import math

class ArtisticFaceModel:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        
        # Kamera ve Lens Ayarları
        self.focal_length = self.width
        self.center = (self.width / 2, self.height / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))

        # --- 1. REFERANS NOKTALARI (SolvePnP) ---
        # 5 Nokta (Burun, Ağız, Gözler, Alın)
        self.anchor_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Burun Ucu
            (0.0, 90.0, -25.0),          # Ağız
            (-65.0, -65.0, -40.0),       # Sol Göz
            (65.0, -65.0, -40.0),        # Sağ Göz
            (0.0, -150.0, -50.0)         # Alın
        ], dtype=np.float32)

        self.model_features = {}
        self._build_high_res_model()

    def _get_bezier_curve(self, p0, p1, p2, segments=20):
        """3 Noktalı (Kuadratik) Bezier Eğrisi"""
        pts = []
        for t in np.linspace(0, 1, segments):
            # B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
            x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
            y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
            z = (1-t)**2 * p0[2] + 2*(1-t)*t * p1[2] + t**2 * p2[2]
            pts.append((x, y, z))
        return np.array(pts, dtype=np.float32)

    def _get_cubic_bezier(self, p0, p1, p2, p3, segments=30):
        """4 Noktalı (Kübik) Bezier Eğrisi - Daha karmaşık formlar için"""
        pts = []
        for t in np.linspace(0, 1, segments):
            # Kübik Bezier Formülü
            x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
            y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
            z = (1-t)**3*p0[2] + 3*(1-t)**2*t*p1[2] + 3*(1-t)*t**2*p2[2] + t**3*p3[2]
            pts.append((x, y, z))
        return np.array(pts, dtype=np.float32)

    def _get_ellipse_points(self, center, rx, ry, segments=30, z_offset=0):
        pts = []
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            x = center[0] + rx * math.cos(theta)
            y = center[1] + ry * math.sin(theta)
            z = center[2] + z_offset
            pts.append((x, y, z))
        return np.array(pts, dtype=np.float32)

    # --- YENİ: SOL KULAK FONKSİYONU ---
    def _create_left_ear(self):
        pts = []
        # Kulak Dış Hattı (Helix)
        # Başlangıç (Favori), Tepe, Arka, Kulak Memesi
        p0 = (-140, -40, -100) # Üst bağlantı
        p1 = (-180, -70, -140) # Dış Tepe noktası (Geriye doğru)
        p2 = (-170, 60, -140)  # Alt dış kavis
        p3 = (-145, 80, -110)  # Kulak memesi (Çene bağlantısına yakın)
        
        outer_rim = self._get_cubic_bezier(p0, p1, p2, p3, segments=20)
        pts.extend(outer_rim)

        # Kulak İçi Detayı (Tragus)
        inner_curve = self._get_bezier_curve((-150, 0, -130), (-140, 20, -135), (-150, 40, -130), segments=10)
        pts.extend(inner_curve)
        
        return np.array(pts, dtype=np.float32)

    # --- YENİ: SAĞ KULAK FONKSİYONU ---
    def _create_right_ear(self):
        pts = []
        # Sol kulağın X ekseninde simetriği (+X)
        p0 = (140, -40, -100)
        p1 = (180, -70, -140)
        p2 = (170, 60, -140)
        p3 = (145, 80, -110) # Sağ kulak memesi
        
        outer_rim = self._get_cubic_bezier(p0, p1, p2, p3, segments=20)
        pts.extend(outer_rim)

        # İç detay
        inner_curve = self._get_bezier_curve((150, 0, -130), (140, 20, -135), (150, 40, -130), segments=10)
        pts.extend(inner_curve)

        return np.array(pts, dtype=np.float32)

    def _build_high_res_model(self):
        # 1. Gözler ve Kaşlar
        self.model_features['left_eye'] = self._get_ellipse_points((-65, -65, -40), 28, 14)
        self.model_features['left_iris'] = self._get_ellipse_points((-65, -65, -40), 10, 10, z_offset=2)
        self.model_features['right_eye'] = self._get_ellipse_points((65, -65, -40), 28, 14)
        self.model_features['right_iris'] = self._get_ellipse_points((65, -65, -40), 10, 10, z_offset=2)
        
        self.model_features['left_eyebrow'] = self._get_bezier_curve((-95, -90, -30), (-65, -110, -20), (-35, -90, -30))
        self.model_features['right_eyebrow'] = self._get_bezier_curve((35, -90, -30), (65, -110, -20), (95, -90, -30))

        # 2. Ağız (Daha estetik)
        mouth_y = 100
        mouth_z = -30
        self.model_features['upper_lip'] = self._get_bezier_curve((-35, mouth_y, mouth_z), (0, mouth_y - 10, mouth_z-5), (35, mouth_y, mouth_z))
        self.model_features['lower_lip'] = self._get_bezier_curve((-35, mouth_y, mouth_z), (0, mouth_y + 20, mouth_z), (35, mouth_y, mouth_z))
        self.model_features['middle_lip'] = self._get_bezier_curve((-35, mouth_y, mouth_z), (0, mouth_y + 3, mouth_z-2), (35, mouth_y, mouth_z))

        # 3. Burun
        self.model_features['nose'] = self._get_bezier_curve((0, -80, -30), (0, -40, -10), (0, 0, 0))
        self.model_features['nose_base'] = self._get_bezier_curve((-20, -10, -20), (0, 5, 0), (20, -10, -20))

        # 4. KULAKLAR (Ayrı Fonksiyonlardan Çağırılıyor)
        self.model_features['left_ear'] = self._create_left_ear()
        self.model_features['right_ear'] = self._create_right_ear()

        # 5. ÇENE HATTI (YENİLENMİŞ "U" ŞEKLİ)
        # Mantık: Sol Kulak Memesi -> Sol Çene Köşesi -> Çene Ucu -> Sağ Çene Köşesi -> Sağ Kulak Memesi
        
        # Sol Parça (Kulaktan Çeneye)
        # P0: Sol Kulak Memesi (-145, 80, -110)
        # P1: Kontrol Noktası (Çene köşesini belirler, aşağı ve içe doğru) -> (-130, 150, -80)
        # P2: Çene Ucu (Tam orta, ağzın biraz altı) -> (0, 160, -50)
        jaw_left = self._get_bezier_curve(
            (-145, 80, -110),   # Başlangıç: Sol Kulak Memesi
            (-130, 160, -80),   # Kontrol: Sol çene kemiği dönüşü
            (0, 170, -50),      # Bitiş: Çene ucu (Ağzın altı)
            segments=25
        )

        # Sağ Parça (Çeneden Kulağa)
        jaw_right = self._get_bezier_curve(
            (0, 170, -50),      # Başlangıç: Çene ucu
            (130, 160, -80),    # Kontrol: Sağ çene kemiği dönüşü
            (145, 80, -110),    # Bitiş: Sağ Kulak Memesi
            segments=25
        )
        
        # İkisini birleştir
        self.model_features['jaw'] = np.concatenate((jaw_left, jaw_right))

    def process(self, input_5_points, canvas):
        input_points_f = np.array(input_5_points, dtype=np.float32)

        # Poz Kestirimi
        success, rvec, tvec = cv2.solvePnP(
            self.anchor_points_3d, 
            input_points_f, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success: return canvas

        # Açı Hesaplama
        rmat, _ = cv2.Rodrigues(rvec)
        yaw = math.atan2(rmat[0, 2], rmat[2, 2]) * 180 / math.pi

        for name, points_3d in self.model_features.items():
            # Gizleme Mantığı
            if name == 'left_ear' and yaw < -60: continue
            if name == 'right_ear' and yaw > 60: continue

            projected_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            projected_2d = projected_2d.reshape(-1, 2).astype(int)

            thickness = 2
            if 'iris' in name or 'nose' in name: thickness = 1
            if 'jaw' in name: thickness = 2

            is_closed = 'eye' in name and 'brow' not in name
            cv2.polylines(canvas, [projected_2d], is_closed, (10, 10, 10), thickness, lineType=cv2.LINE_AA)

        return canvas

# --- TEST ---
img = np.full((800, 800, 3), 230, dtype=np.uint8) # Hafif gri
# Test Verisi (Merkezi)
sample_points = [(400, 420), (400, 530), (320, 340), (480, 340), (400, 240)]

artist = ArtisticFaceModel(width=800, height=800)
result = artist.process(sample_points, img)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.imshow(result)
plt.axis('off')
plt.show()