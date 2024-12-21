## 2. Tiền xử lý dữ liệu (preprocess.py) - tiếp theo

```python
            # Tiếp tục từ phương thức align_face
        )), (face.shape[1], face.shape[0]))
        
        return aligned_face

    def enhance_face(self, face: np.ndarray) -> np.ndarray:
        """
        Cải thiện chất lượng ảnh khuôn mặt.
        
        Args:
            face: Ảnh khuôn mặt đầu vào
            
        Returns:
            Ảnh đã được cải thiện chất lượng
        """
        # Chuẩn hóa độ sáng và độ tương phản
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Giảm nhiễu với Non-local Means Denoising
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return enhanced

    def process_image(self, image_path: Path) -> Optional[Tuple[np.ndarray, Path]]:
        """
        Xử lý một ảnh: phát hiện, căn chỉnh và cải thiện chất lượng.
        
        Args:
            image_path: Đường dẫn ảnh đầu vào
            
        Returns:
            Tuple (ảnh đã xử lý, đường dẫn lưu) hoặc None nếu thất bại
        """
        try:
            # Đọc ảnh
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Không thể đọc ảnh: {image_path}")
                return None
                
            # Chuyển sang RGB cho MTCNN
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Phát hiện khuôn mặt
            results = self.detector.detect_faces(rgb_img)
            
            if not results or len(results) == 0:
                logger.warning(f"Không tìm thấy khuôn mặt trong: {image_path}")
                return None
                
            # Lấy kết quả có confidence cao nhất
            best_result = max(results, key=lambda x: x['confidence'])
            
            if best_result['confidence'] < self.min_confidence:
                logger.warning(f"Confidence thấp ({best_result['confidence']:.2f}) trong: {image_path}")
                return None
                
            # Cắt khuôn mặt
            x, y, w, h = best_result['box']
            face = img[y:y+h, x:x+w]
            
            # Căn chỉnh khuôn mặt
            face = self.align_face(face, best_result['keypoints'])
            
            # Cải thiện chất lượng
            face = self.enhance_face(face)
            
            # Resize về kích thước chuẩn
            face = cv2.resize(face, self.img_size)
            
            # Tạo đường dẫn lưu
            rel_path = image_path.relative_to(self.input_dir)
            output_path = self.output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return face, output_path
            
        except Exception as e:
            logger.error(f"Lỗi xử lý {image_path}: {str(e)}")
            return None

    def process_dataset(self, num_workers: int = 4):
        """
        Xử lý toàn bộ dataset với đa luồng.
        
        Args:
            num_workers: Số lượng worker threads
        """
        # Lấy danh sách ảnh cần xử lý
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(self.input_dir.rglob(ext))
            
        logger.info(f"Tìm thấy {len(image_paths)} ảnh cần xử lý")
        
        # Xử lý đa luồng với progress bar
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_image, image_paths),
                total=len(image_paths),
                desc="Đang xử lý ảnh"
            ))
            
        # Lưu kết quả
        success_count = 0
        for result in results:
            if result is not None:
                face, output_path = result
                cv2.imwrite(str(output_path), face)
                success_count += 1
                
        logger.info(f"Đã xử lý thành công {success_count}/{len(image_paths)} ảnh")

if __name__ == "__main__":
    # Khởi tạo preprocessor
    preprocessor = FacePreprocessor(
        input_dir="dataset/raw",
        output_dir="dataset/processed",
        img_size=(224, 224),
        min_face_size=20,
        min_confidence=0.95
    )
    
    # Xử lý toàn bộ dataset
    preprocessor.process_dataset(num_workers=4)
```

## 3. Định nghĩa Model (model.py)

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
import efficientnet.tfkeras as efn
from typing import Tuple, Optional

class FaceEmbeddingModel:
    """
    Model trích xuất đặc trưng khuôn mặt sử dụng EfficientNet-B0
    với các cải tiến cho độ chính xác cao.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        embedding_dim: int = 512,
        dropout_rate: float = 0.3,
        l2_reg: float = 0.01
    ):
        """
        Khởi tạo model với các tham số tối ưu.
        
        Args:
            input_shape: Kích thước ảnh đầu vào
            embedding_dim: Số chiều vector embedding
            dropout_rate: Tỷ lệ dropout để chống overfitting
            l2_reg: Hệ số regularization L2
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Khởi tạo base model EfficientNet-B0
        self.base_model = efn.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Xây dựng model hoàn chỉnh
        self.model = self.build_model()

    def build_model(self) -> Model:
        """
        Xây dựng kiến trúc model với các layer tùy chỉnh.
        
        Returns:
            Model Keras hoàn chỉnh
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation (chỉ dùng khi training)
        augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        x = augmentation(inputs)
        
        # Chuẩn hóa đầu vào
        x = layers.Rescaling(1./255)(x)
        
        # Base model
        x = self.base_model(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout và BatchNorm để chống overfitting
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.BatchNormalization()(x)
        
        # Dense layer với L2 regularization
        x = layers.Dense(
            self.embedding_dim * 2,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        
        # Thêm Dropout và BatchNorm
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.BatchNormalization()(x)
        
        # Layer embedding cuối cùng
        outputs = layers.Dense(
            self.embedding_dim,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name='embedding'
        )(x)
        
        # Normalize embedding về độ dài 1
        outputs = tf.nn.l2_normalize(outputs, axis=1)
        
        return Model(inputs=inputs, outputs=outputs, name='face_embedding')

    def add_classification_head(
        self,
        num_classes: int,
        freeze_base: bool = True
    ) -> Model:
        """
        Thêm classification head cho fine-tuning.
        
        Args:
            num_classes: Số lượng classes
            freeze_base: Có đóng băng base model không
            
        Returns:
            Model với classification head mới
        """
        if freeze_base:
            self.model.trainable = False
            
        # Lấy embedding
        embedding = self.model.output
        
        # Thêm classification head
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(embedding)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.BatchNormalization()(x)
        
        outputs = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        
        return Model(self.model.input, outputs, name='face_classifier')

class ArcFaceLayer(layers.Layer):
    """
    ArcFace layer để cải thiện độ chính xác của face recognition.
    """
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        margin: float = 0.5,
        scale: float = 64.0,
        **kwargs
    ):
        """
        Khởi tạo ArcFace layer.
        
        Args:
            num_classes: Số lượng classes
            embedding_dim: Số chiều của embedding
            margin: Margin góc cho ArcFace
            scale: Hệ số scale
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        
        # Khởi tạo weights
        self.w = None
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.th = tf.math.cos(tf.constant(np.pi) - margin)
        self.mm = tf.math.sin(tf.constant(np.pi) - margin) * margin

    def build(self, input_shape):
        self.w = self.add_weight(
            name='arcface_weights',
            shape=(self.embedding_dim, self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, training=None):
        # Normalize weights
        weights = tf.nn.l2_normalize(self.w, axis=0)
        
        # Normalize embeddings
        x = tf.nn.l2_normalize(inputs, axis=1)
        
        # Tính cosine similarity
        cos_t = tf.matmul(x, weights)
        
        if training:
            # Áp dụng ArcFace margin khi training
            sin_t = tf.sqrt(1.0 - tf.square(cos_t))
            cos_t_m = cos_t * self.cos_m - sin_t * self.sin_m
            
            # Điều kiện cho stable training
            cos_t_m = tf.where(cos_t > self.th, cos_t_m, cos_t - self.mm)
            
            # One-hot encoding của labels
            mask = inputs[1]
            
            # Áp dụng margin chỉ cho positive pairs
            output = tf.where(mask, cos_t_m, cos_t)
            
            # Scale output
            output *= self.scale
            
        else:
            # Khi inference chỉ cần scale
            output = cos_t * self.scale
            
        return output

def build_arcface_model(
    embedding_model: Model,
    num_classes: int,
    margin: float = 0.5,
    scale: float = 64.0
) -> Model:
    """
    Xây dựng model với ArcFace loss.
    
    Args:
        embedding_model: Base model tạo embedding
        num_classes: Số lượng classes
        margin: ArcFace margin
        scale: ArcFace scale
        
    Returns:
        Model với ArcFace
    """
    embedding_dim = embedding_model.output_shape[-1]
    
    inputs = layers.Input(shape=embedding_model.input_shape[1:])
    labels = layers.Input(shape=(num_classes,))
    
    x = embedding_model(inputs)
    outputs = ArcFaceLayer(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=margin,
        scale=scale
    )([x, labels])
    
    return Model([inputs, labels], outputs)
```

Các cải tiến chính trong Model:

1. Sử dụng EfficientNet-B0 làm backbone - cân bằng giữa độ chính xác và tốc độ
2. Data augmentation tích hợp trong model
3. Kết