# Dự Án Nhận Dạng Khuôn Mặt Sử Dụng Deep Learning và TensorFlow

## Tổng quan
Dự án này xây dựng một hệ thống nhận dạng khuôn mặt hoàn chỉnh với độ chính xác cao, sử dụng các kỹ thuật deep learning hiện đại. Hệ thống bao gồm:
- Thu thập và tiền xử lý dữ liệu
- Huấn luyện mô hình sâu với kiến trúc tiên tiến
- Tối ưu hóa mô hình cho thiết bị edge
- Triển khai thực tế

## Cấu trúc thư mục
```
face_recognition_project/
├── src/
│   ├── data/
│   │   ├── collect_data.py        # Thu thập dữ liệu
│   │   └── preprocess.py          # Tiền xử lý dữ liệu
│   ├── models/
│   │   ├── model.py              # Định nghĩa kiến trúc mô hình
│   │   └── training.py           # Huấn luyện mô hình
│   └── utils/
│       ├── augmentation.py       # Augmentation dữ liệu
│       └── metrics.py            # Các metrics đánh giá
├── configs/
│   └── config.yaml              # Cấu hình tham số
├── notebooks/
│   └── analysis.ipynb           # Phân tích kết quả
└── requirements.txt             # Thư viện cần thiết
```

## 1. Thu thập dữ liệu (collect_data.py)
```python
import cv2
import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Class thu thập dữ liệu khuôn mặt từ camera hoặc video
    với nhiều tính năng nâng cao và xử lý lỗi tốt hơn.
    """
    def __init__(
        self,
        output_dir: str,
        img_size: Tuple[int, int] = (224, 224),  # Kích thước phổ biến cho các mô hình hiện đại
        min_face_size: Tuple[int, int] = (30, 30),
        camera_warmup_time: float = 2.0
    ):
        """
        Khởi tạo DataCollector với các tham số cấu hình.
        
        Args:
            output_dir: Thư mục lưu ảnh
            img_size: Kích thước ảnh đầu ra
            min_face_size: Kích thước tối thiểu của khuôn mặt
            camera_warmup_time: Thời gian khởi động camera (giây)
        """
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.min_face_size = min_face_size
        self.camera_warmup_time = camera_warmup_time
        
        # Tạo thư mục output nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo bộ phát hiện khuôn mặt Haar Cascade
        # (nhanh hơn MTCNN cho giai đoạn thu thập dữ liệu)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(cascade_path)

    def setup_camera(self, camera_index: int = 0) -> Optional[cv2.VideoCapture]:
        """
        Khởi tạo và kiểm tra camera.
        
        Args:
            camera_index: Chỉ số camera (default: 0 cho webcam)
            
        Returns:
            VideoCapture object hoặc None nếu có lỗi
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Không thể mở camera {camera_index}")
            return None
            
        # Thiết lập các thông số camera tối ưu
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Thời gian khởi động camera
        time.sleep(self.camera_warmup_time)
        
        return cap

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Phát hiện khuôn mặt trong frame với các cải tiến.
        
        Args:
            frame: Ảnh đầu vào (BGR)
            
        Returns:
            List các bbox khuôn mặt [(x, y, w, h)]
        """
        # Chuyển sang ảnh xám để tăng tốc độ xử lý
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Cân bằng histogram để cải thiện độ tương phản
        gray = cv2.equalizeHist(gray)
        
        # Phát hiện khuôn mặt với các tham số tối ưu
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces

    def collect_data(
        self,
        person_name: str,
        num_images: int = 100,
        delay: float = 0.5,
        camera_index: int = 0,
        show_preview: bool = True
    ) -> None:
        """
        Thu thập ảnh khuôn mặt với nhiều cải tiến.
        
        Args:
            person_name: Tên thư mục lưu ảnh
            num_images: Số lượng ảnh cần thu thập
            delay: Độ trễ giữa các ảnh (giây)
            camera_index: Chỉ số camera
            show_preview: Hiển thị preview hay không
        """
        # Khởi tạo camera
        cap = self.setup_camera(camera_index)
        if cap is None:
            return
            
        # Tạo thư mục lưu ảnh cho person
        person_dir = self.output_dir / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        try:
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Lỗi đọc frame từ camera")
                    break
                    
                # Phát hiện khuôn mặt
                faces = self.detect_faces(frame)
                
                # Vẽ bbox và thông tin
                frame_display = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0,255,0), 2)
                
                if show_preview:
                    cv2.putText(
                        frame_display,
                        f"Collected: {count}/{num_images}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2
                    )
                    cv2.imshow("Collecting Data (Press 'q' to quit)", frame_display)
                
                # Chỉ lưu ảnh khi phát hiện đúng 1 khuôn mặt
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    
                    # Cắt và resize khuôn mặt
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, self.img_size)
                    
                    # Cải thiện chất lượng ảnh
                    face = cv2.detailEnhance(face, sigma_s=10, sigma_r=0.15)
                    
                    # Lưu ảnh với timestamp để tránh trùng tên
                    timestamp = int(time.time() * 1000)
                    filename = person_dir / f"{person_name}_{timestamp}.jpg"
                    cv2.imwrite(str(filename), face)
                    
                    logger.info(f"Đã lưu: {filename}")
                    count += 1
                    time.sleep(delay)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Người dùng dừng thu thập dữ liệu")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Thu thập dữ liệu bị dừng")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Khởi tạo collector
    collector = DataCollector(
        output_dir="dataset/raw",
        img_size=(224, 224),
        min_face_size=(30, 30)
    )
    
    # Thu thập dữ liệu cho một người
    collector.collect_data(
        person_name="person1",
        num_images=100,
        delay=0.5,
        show_preview=True
    )
```

Các cải tiến chính trong phiên bản này:

1. Cấu trúc code theo hướng OOP rõ ràng hơn
2. Xử lý lỗi toàn diện với logging
3. Sử dụng Haar Cascade thay vì MTCNN cho tốc độ nhanh hơn
4. Cải thiện chất lượng ảnh với detail enhancement
5. Thêm camera warmup time để ổn định
6. Type hints và docstrings đầy đủ
7. Sử dụng pathlib thay vì os.path
8. Lưu ảnh với timestamp để tránh trùng lặp

## 2. Tiền xử lý dữ liệu (preprocess.py)

```python
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacePreprocessor:
    """
    Class tiền xử lý dữ liệu khuôn mặt với nhiều cải tiến
    để tăng chất lượng và độ chính xác.
    """
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        min_face_size: int = 20,
        min_confidence: float = 0.95
    ):
        """
        Khởi tạo FacePreprocessor với các tham số tối ưu.
        
        Args:
            input_dir: Thư mục chứa ảnh gốc
            output_dir: Thư mục lưu ảnh đã xử lý
            img_size: Kích thước ảnh đầu ra
            min_face_size: Kích thước tối thiểu của khuôn mặt
            min_confidence: Ngưỡng tin cậy tối thiểu cho MTCNN
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        
        # Khởi tạo MTCNN với các tham số tối ưu
        self.detector = MTCNN(
            min_face_size=min_face_size,
            steps_threshold=[0.6, 0.7, self.min_confidence]
        )
        
        # Tạo thư mục output
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def align_face(
        self, 
        face: np.ndarray, 
        landmarks: dict
    ) -> np.ndarray:
        """
        Căn chỉnh khuôn mặt dựa trên landmarks.
        
        Args:
            face: Ảnh khuôn mặt
            landmarks: Dictionary chứa landmarks từ MTCNN
            
        Returns:
            Ảnh khuôn mặt đã căn chỉnh
        """
        # Lấy tọa độ mắt
        left_eye = np.mean(landmarks['left_eye'], axis=0)
        right_eye = np.mean(landmarks['right_eye'], axis=0)
        
        # Tính góc cần xoay
        angle = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ))
        
        # Tạo ma trận xoay
        center = (face.shape[1]//2, face.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Xoay ảnh
        aligned_face = cv2.warpAffine(
            face, 
            rotation_matrix, 
            (face.shape[1], face.shape[0