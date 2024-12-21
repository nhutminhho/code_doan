# Phần 8: Triển Khai và Đánh Giá Hệ Thống

## 8.1 Triển Khai Thời Gian Thực (realtime_inference.py)

```python
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
from typing import List, Tuple, Dict
import logging
from collections import deque
from threading import Thread
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """
    Hệ thống nhận dạng khuôn mặt thời gian thực với nhiều tính năng nâng cao:
    - Multi-threading để tăng FPS
    - Tracking để ổn định kết quả
    - Xử lý các trường hợp đặc biệt
    - Tối ưu hiệu năng
    """
    def __init__(
        self,
        model_path: str,
        label_map: Dict[int, str],
        detection_threshold: float = 0.9,
        recognition_threshold: float = 0.7,
        queue_size: int = 30,
        use_gpu: bool = True
    ):
        """
        Khởi tạo hệ thống.
        
        Args:
            model_path: Đường dẫn đến model TFLite
            label_map: Dict ánh xạ index -> tên người
            detection_threshold: Ngưỡng cho face detection
            recognition_threshold: Ngưỡng cho face recognition
            queue_size: Kích thước queue cho tracking
            use_gpu: Có sử dụng GPU không
        """
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.label_map = label_map
        self.queue_size = queue_size
        
        # Khởi tạo face detector
        self.face_detector = self._init_face_detector()
        
        # Load model TFLite
        self.interpreter = self._init_tflite_model(model_path, use_gpu)
        
        # Lấy thông tin input/output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Khởi tạo queues cho multi-threading
        self.input_queue = Queue(maxsize=queue_size)
        self.output_queue = Queue(maxsize=queue_size)
        
        # Dict lưu tracking history cho mỗi khuôn mặt
        self.face_trackers = {}
        
        # Thread cho inference
        self.inference_thread = Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def _init_face_detector(self):
        """
        Khởi tạo face detector với RetinaFace hoặc MTCNN.
        """
        from retinaface import RetinaFace
        detector = RetinaFace(
            quality="normal",
            threshold=self.detection_threshold
        )
        return detector

    def _init_tflite_model(
        self,
        model_path: str,
        use_gpu: bool
    ) -> tf.lite.Interpreter:
        """
        Khởi tạo TFLite interpreter với các tùy chọn tối ưu.
        """
        # Delegate options
        if use_gpu:
            delegates = [
                tf.lite.experimental.load_delegate('libedgetpu.so.1')
            ]
        else:
            delegates = []
            
        # Load và allocate tensors
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=delegates,
            num_threads=4
        )
        interpreter.allocate_tensors()
        
        return interpreter

    def preprocess_face(
        self,
        face: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Tiền xử lý ảnh khuôn mặt.
        """
        # Resize
        face = cv2.resize(face, target_size)
        
        # Chuẩn hóa
        face = face.astype(np.float32) / 255.0
        
        # Thêm batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face

    def _inference_loop(self):
        """
        Thread loop cho việc inference.
        """
        while True:
            # Lấy frame từ input queue
            frame, faces = self.input_queue.get()
            if frame is None:
                break
                
            results = []
            for face, bbox in faces:
                # Tiền xử lý
                processed_face = self.preprocess_face(face)
                
                # Set input tensor
                self.interpreter.set_tensor(
                    self.input_details[0]['index'],
                    processed_face
                )
                
                # Inference
                self.interpreter.invoke()
                
                # Lấy kết quả
                embedding = self.interpreter.get_tensor(
                    self.output_details[0]['index']
                )
                
                results.append((embedding[0], bbox))
                
            # Đưa kết quả vào output queue
            self.output_queue.put((frame, results))

    def track_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        label: str,
        confidence: float
    ):
        """
        Cập nhật tracking history cho một khuôn mặt.
        """
        if face_id not in self.face_trackers:
            self.face_trackers[face_id] = {
                'embeddings': deque(maxlen=self.queue_size),
                'labels': deque(maxlen=self.queue_size),
                'confidences': deque(maxlen=self.queue_size)
            }
            
        tracker = self.face_trackers[face_id]
        tracker['embeddings'].append(embedding)
        tracker['labels'].append(label)
        tracker['confidences'].append(confidence)
        
        # Lấy predicted label phổ biến nhất
        from collections import Counter
        label_counts = Counter(tracker['labels'])
        predicted_label = label_counts.most_common(1)[0][0]
        
        # Tính confidence trung bình
        avg_confidence = np.mean(tracker['confidences'])
        
        return predicted_label, avg_confidence

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Xử lý một frame: phát hiện, nhận dạng và tracking.
        
        Args:
            frame: Frame BGR từ camera
            
        Returns:
            Frame đã vẽ và list kết quả nhận dạng
        """
        # Phát hiện khuôn mặt
        faces = self.face_detector.detect(frame)
        
        if not faces:
            return frame, []
            
        # Chuẩn bị faces cho inference
        face_crops = []
        for face in faces:
            bbox = face['bbox'].astype(int)
            x1, y1, x2, y2 = bbox
            face_img = frame[y1:y2, x1:x2]
            face_crops.append((face_img, bbox))
            
        # Đưa vào input queue
        self.input_queue.put((frame, face_crops))
        
        # Lấy kết quả từ output queue
        frame, results = self.output_queue.get()
        
        # Xử lý từng khuôn mặt
        recognition_results = []
        for embedding, bbox in results:
            # Tạo face ID từ vị trí (đơn giản)
            face_id = f"{bbox[0]}_{bbox[1]}"
            
            # Tìm label gần nhất
            distances = []
            for idx, label_emb in enumerate(self.label_embeddings):
                dist = np.linalg.norm(embedding - label_emb)
                distances.append((dist, idx))
                
            min_dist, best_idx = min(distances)
            confidence = 1 / (1 + min_dist)
            
            if confidence > self.recognition_threshold:
                label = self.label_map[best_idx]
                
                # Tracking
                predicted_label, avg_confidence = self.track_face(
                    face_id, embedding, label, confidence
                )
                
                # Vẽ bbox và label
                x1, y1, x2, y2 = bbox
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"{predicted_label} ({avg_confidence:.2f})",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                result = {
                    'bbox': bbox,
                    'label': predicted_label,
                    'confidence': avg_confidence
                }
                recognition_results.append(result)
                
        return frame, recognition_results

    def run_camera(
        self,
        camera_id: int = 0,
        display: bool = True
    ):
        """
        Chạy nhận dạng thời gian thực từ camera.
        """
        cap = cv2.VideoCapture(camera_id)
        fps = FPS()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                fps.start()
                
                # Xử lý frame
                frame, results = self.process_frame(frame)
                
                fps.stop()
                
                # Hiển thị FPS
                cv2.putText(
                    frame,
                    f"FPS: {fps.fps():.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )
                
                if display:
                    cv2.imshow("Face Recognition", frame)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Dừng inference thread
            self.input_queue.put((None, None))
            self.inference_thread.join()

class FPS:
    """
    Class tính FPS.
    """
    def __init__(self):
        self._start = None
        self._end = None
        self._num_frames = 0
        
    def start(self):
        self._start = time.time()
        
    def stop(self):
        self._end = time.time()
        self._num_frames += 1
        
    def fps(self):
        return self._num_frames / (self._end - self._start)
```

## 8.2 Đánh Giá Chi Tiết (evaluation.py)

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

class FaceRecognitionEvaluator:
    """
    Class đánh giá toàn diện hệ thống nhận dạng khuôn mặt.
    """
    def __init__(
        self,
        groundtruth_file: str,
        prediction_file: str
    ):
        """
        Khởi tạo evaluator.
        
        Args:
            groundtruth_file: File CSV chứa ground truth
            prediction_file: File CSV chứa predictions
        """
        # Load data
        self.gt_df = pd.read_csv(groundtruth_file)
        self.pred_df = pd.read_csv(prediction_file)
        
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Tính toán các metrics chính.
        
        Returns:
            Dict các metrics và giá trị
        """
        y_true = self.gt_df['label']
        y_pred = self.pred_df['predicted_label']
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='weighted'
        )
        
        # ROC-AUC
        y_score = self.pred_df['confidence']
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        return metrics
        
    def analyze_errors(self) -> pd.DataFrame:
        """
        Phân tích chi tiết các trường hợp nhận dạng sai.
        
        Returns:
            DataFrame chứa thông tin lỗi
        """
        # Lọc ra các cases dự đoán sai
        errors_df = self.gt_df[
            self.gt_df['label'] != self.pred_df['predicted_label']
        ].copy()
        
        # Thêm thông tin dự đoán
        errors_df['predicted_label'] = self.pred_df['predicted_label']
        errors_df['confidence'] = self.pred_df['confidence']
        
        # Phân loại lỗi
        def categorize_error(row):
            if row['confidence'] < 0.5:
                return 'Low Confidence'
            elif row['illumination'] < 0.3:
                return 'Poor Lighting'
            elif row['pose_angle'] > 30:
                return 'Extreme Pose'
            else:
                return 'Other'
                
        errors_df['error_type'] = errors_df.apply(categorize_error, axis=1)
        
        return errors_df
        
    def plot_confusion_matrix(
        self,
        