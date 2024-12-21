## 4. Huấn luyện Mô hình (training.py)

```python
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
import mlflow
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

from src.models.model import FaceEmbeddingModel, build_arcface_model
from src.utils.metrics import (
    calculate_verification_metrics,
    calculate_identification_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceModelTrainer:
    """
    Class quản lý quá trình huấn luyện model với nhiều tính năng nâng cao:
    - Tracking thực nghiệm với MLflow
    - Custom callbacks
    - Đánh giá đa chiều
    - Learning rate scheduling
    - Mixed precision training
    """
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        experiment_name: str = "face_recognition"
    ):
        """
        Khởi tạo trainer với config từ file YAML.
        
        Args:
            config_path: Đường dẫn file config
            experiment_name: Tên thực nghiệm MLflow
        """
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Thiết lập MLflow
        mlflow.set_experiment(experiment_name)
        
        # Khởi tạo model
        self.embedding_model = FaceEmbeddingModel(
            input_shape=tuple(self.config['model']['input_shape']),
            embedding_dim=self.config['model']['embedding_dim'],
            dropout_rate=self.config['model']['dropout_rate'],
            l2_reg=self.config['model']['l2_reg']
        )
        
        # Thư mục lưu checkpoints
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(
        self
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Chuẩn bị data pipeline hiệu quả với tf.data.
        
        Returns:
            train_ds, val_ds, test_ds: Các tf.data.Dataset
        """
        # Load danh sách ảnh và nhãn
        data_dir = Path(self.config['data']['processed_dir'])
        
        image_paths = []
        labels = []
        
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.jpg'):
                    image_paths.append(str(img_path))
                    labels.append(str(class_dir.name))
                    
        # Encode nhãn
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = [label_to_idx[l] for l in labels]
        
        # Chia dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            image_paths,
            encoded_labels,
            test_size=self.config['data']['test_split'],
            stratify=encoded_labels,
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=self.config['data']['val_split'],
            stratify=y_train_val,
            random_state=42
        )
        
        # Hàm load và xử lý ảnh
        def process_path(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.config['model']['input_shape'][:2])
            return img, label
            
        # Tạo tf.data.Dataset
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Map và cấu hình dataset
        train_ds = (train_ds
            .map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(1000)
            .batch(self.config['training']['batch_size'])
            .prefetch(tf.data.AUTOTUNE))
            
        val_ds = (val_ds
            .map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .batch(self.config['training']['batch_size'])
            .prefetch(tf.data.AUTOTUNE))
            
        test_ds = (test_ds
            .map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .batch(self.config['training']['batch_size'])
            .prefetch(tf.data.AUTOTUNE))
            
        return train_ds, val_ds, test_ds

    def get_callbacks(self) -> list:
        """
        Tạo các callback cho quá trình training.
        
        Returns:
            List các callback
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / "model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5"
        callbacks.append(ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ))
        
        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ))
        
        # TensorBoard
        log_dir = Path(self.config['training']['log_dir'])
        callbacks.append(TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            update_freq='epoch'
        ))
        
        return callbacks

    def train(self) -> None:
        """
        Huấn luyện model với mixed precision và MLflow tracking.
        """
        # Chuẩn bị data
        train_ds, val_ds, test_ds = self.prepare_data()
        
        # Lấy số lượng classes
        num_classes = len(set([label.numpy() for _, label in train_ds.unbatch()]))
        
        # Xây dựng model với ArcFace
        model = build_arcface_model(
            embedding_model=self.embedding_model.model,
            num_classes=num_classes,
            margin=self.config['model']['arcface_margin'],
            scale=self.config['model']['arcface_scale']
        )
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['training']['initial_lr']
        )
        
        # Bật mixed precision
        if self.config['training']['mixed_precision']:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            # Đảm bảo output layer là float32
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # MLflow tracking
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config['model'])
            mlflow.log_params(self.config['training'])
            
            # Training
            history = model.fit(
                train_ds,
                epochs=self.config['training']['epochs'],
                validation_data=val_ds,
                callbacks=self.get_callbacks(),
                verbose=1
            )
            
            # Log metrics
            for epoch in range(len(history.history['accuracy'])):
                mlflow.log_metrics({
                    'train_accuracy': history.history['accuracy'][epoch],
                    'train_loss': history.history['loss'][epoch],
                    'val_accuracy': history.history['val_accuracy'][epoch],
                    'val_loss': history.history['val_loss'][epoch]
                }, step=epoch)
                
            # Đánh giá trên test set
            test_loss, test_acc = model.evaluate(test_ds)
            mlflow.log_metrics({
                'test_accuracy': test_acc,
                'test_loss': test_loss
            })
            
            # Tính các metrics nâng cao
            embeddings, labels = [], []
            for img_batch, label_batch in test_ds:
                batch_embeddings = self.embedding_model.model.predict(img_batch)
                embeddings.extend(batch_embeddings)
                labels.extend(label_batch.numpy())
                
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            
            # Verification metrics (FAR, FRR, EER)
            ver_metrics = calculate_verification_metrics(embeddings, labels)
            mlflow.log_metrics(ver_metrics)
            
            # Identification metrics (Rank-1, Rank-5)
            id_metrics = calculate_identification_metrics(embeddings, labels)
            mlflow.log_metrics(id_metrics)
            
            # Log model
            mlflow.keras.log_model(model, "model")
            
            logger.info("Training completed successfully!")

if __name__ == "__main__":
    trainer = FaceModelTrainer()
    trainer.train()
```

## 5. Tính Toán Metrics (metrics.py)

```python
import numpy as np
from sklearn.metrics import roc_curve
from scipy.spatial.distance import cdist
from typing import Dict, Tuple

def calculate_verification_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    distances: str = 'cosine',
    thresholds: np.ndarray = np.arange(0, 1, 0.01)
) -> Dict[str, float]:
    """
    Tính các metrics cho face verification.
    
    Args:
        embeddings: Ma trận embeddings
        labels: Mảng nhãn tương ứng
        distances: Loại khoảng cách ('cosine' hoặc 'euclidean')
        thresholds: Mảng các ngưỡng để tính FAR/FRR
        
    Returns:
        Dict các metrics: FAR, FRR, EER
    """
    # Tính ma trận khoảng cách
    dist_matrix = cdist(embeddings, embeddings, metric=distances)
    
    # Tạo ma trận positive pairs
    pos_pairs = labels[:, np.newaxis] == labels[np.newaxis, :]
    np.fill_diagonal(pos_pairs, False)  # Loại bỏ self-pairs
    
    # Tạo ma trận negative pairs
    neg_pairs = ~pos_pairs
    
    # Lấy distances của positive và negative pairs
    pos_dists = dist_matrix[pos_pairs]
    neg_dists = dist_matrix[neg_pairs]
    
    # Tính FAR và FRR tại mỗi ngưỡng
    fars, frrs = [], []
    for threshold in thresholds:
        far = (neg_dists <= threshold).mean()
        frr = (pos_dists > threshold).mean()
        fars.append(far)
        frrs.append(frr)
        
    fars = np.array(fars)
    frrs = np.array(frrs)
    
    # Tìm EER
    eer_threshold = thresholds[np.argmin(np.abs(fars - frrs))]
    eer = (fars[np.argmin(np.abs(fars - frrs))] + 
           frrs[np.argmin(np.abs(fars - frrs))]) / 2
           
    # Tính AUC
    fpr, tpr, _ = roc_curve(
        y_true=np.hstack([np.ones_like(pos_dists), np.zeros_like(neg_dists)]),
        y_score=np.hstack([-pos_dists, -neg_dists])
    )
    auc = -np.trapz(tpr, fpr)
    
    return {
        'FAR': fars[np.argmin(np.abs(fars - frrs))],
        'FRR': frrs[np.argmin(np.abs(fars - frrs))],
        'EER': eer,
        'EER_threshold': eer_threshold,
        'AUC': auc
    }

def calculate_identification_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    distances: str = 'cosine',
    ranks: Tuple[int] = (1, 5)
) -> Dict[str, float]:
    """
    Tính các metrics cho face identification.
    
    Args:
        embeddings: Ma trận embeddings
        labels: Mảng nhãn tương ứng
        distances: Loại khoảng cách
        ranks: Tuple các rank cần tính accuracy
        
    Returns:
        Dict các metrics Rank-N
    """
    # Tính ma trận khoảng cách
    dist_matrix = cdist(embeddings, embeddings, metric=distances)
    np.fill_diagonal(dist_matrix, np.inf)  # Loại bỏ self-pairs
    
    # Sắp xếp các indices theo khoảng cách tăng dần
    sorted_indices = np.argsort(dist_matrix, axis=1)
    
    # Tính accuracy cho mỗi rank
    metrics = {}
    for rank in ranks:
        # Lấy top-k predictions cho mỗi sample
        top_k_indices = sorted_indices[:, :rank]
        
        