## 5. Tính Toán Metrics (tiếp theo)

```python
        # Lấy nhãn dự đoán
        predicted_labels = labels[top_k_indices]
        
        # Kiểm tra xem nhãn thật có nằm trong top-k không
        correct = np.any(predicted_labels == labels[:, np.newaxis], axis=1)
        
        # Tính accuracy
        rank_acc = correct.mean()
        metrics[f'Rank_{rank}_Accuracy'] = float(rank_acc)
    
    return metrics

def calculate_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Tính các metrics đánh giá chất lượng embedding space.
    
    Args:
        embeddings: Ma trận embeddings
        labels: Mảng nhãn tương ứng
        
    Returns:
        Dict các metrics clustering
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    metrics = {
        'Silhouette_Score': float(silhouette_score(embeddings, labels)),
        'Calinski_Harabasz_Score': float(calinski_harabasz_score(embeddings, labels))
    }
    
    return metrics

def calculate_pair_metrics(
    pair_embeddings1: np.ndarray,
    pair_embeddings2: np.ndarray,
    pair_labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Tính các metrics cho face verification với pairs.
    
    Args:
        pair_embeddings1: Ma trận embeddings của ảnh 1
        pair_embeddings2: Ma trận embeddings của ảnh 2
        pair_labels: Mảng nhãn (1: same, 0: different)
        threshold: Ngưỡng quyết định
        
    Returns:
        Dict các metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Tính cosine similarity
    similarities = np.sum(
        pair_embeddings1 * pair_embeddings2,
        axis=1
    ) / (
        np.linalg.norm(pair_embeddings1, axis=1) *
        np.linalg.norm(pair_embeddings2, axis=1)
    )
    
    # Dự đoán dựa trên threshold
    predictions = (similarities >= threshold).astype(int)
    
    metrics = {
        'Pair_Accuracy': float(accuracy_score(pair_labels, predictions)),
        'Pair_Precision': float(precision_score(pair_labels, predictions)),
        'Pair_Recall': float(recall_score(pair_labels, predictions)),
        'Pair_F1': float(f1_score(pair_labels, predictions))
    }
    
    return metrics
```

## 6. Tối Ưu Hóa Mô Hình (optimize.py)

```python
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple
import tensorflow_model_optimization as tfmot

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Class tối ưu hóa mô hình cho triển khai thực tế:
    - Pruning để giảm số tham số
    - Quantization để giảm kích thước
    - Distillation để cải thiện performance
    """
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        output_dir: str = "optimized_models"
    ):
        """
        Khởi tạo optimizer.
        
        Args:
            model_path: Đường dẫn đến model gốc
            dataset_path: Đường dẫn đến calibration dataset
            output_dir: Thư mục lưu model đã tối ưu
        """
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model gốc
        self.model = tf.keras.models.load_model(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

    def apply_pruning(
        self,
        target_sparsity: float = 0.5,
        epochs: int = 5
    ) -> tf.keras.Model:
        """
        Áp dụng pruning để giảm số tham số.
        
        Args:
            target_sparsity: Tỷ lệ tham số cần loại bỏ
            epochs: Số epochs fine-tune sau pruning
            
        Returns:
            Model sau khi pruning
        """
        # Định nghĩa pruning schedule
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=epochs
            )
        }
        
        # Áp dụng pruning
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            **pruning_params
        )
        
        # Fine-tune
        model_for_pruning.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir="logs")
        ]
        
        model_for_pruning.fit(
            self._get_calibration_dataset(),
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Strip pruning wrapper
        pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        
        # Lưu model
        pruned_path = self.output_dir / "pruned_model.h5"
        pruned_model.save(pruned_path)
        logger.info(f"Saved pruned model to {pruned_path}")
        
        return pruned_model

    def apply_quantization(
        self,
        quantization_type: str = "int8",
        representative_dataset: Optional[callable] = None
    ) -> None:
        """
        Áp dụng quantization để giảm kích thước model.
        
        Args:
            quantization_type: Loại quantization ("int8" hoặc "float16")
            representative_dataset: Dataset cho calibration
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantization_type == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            if representative_dataset is None:
                representative_dataset = self._get_representative_dataset()
                
            converter.representative_dataset = representative_dataset
            
        elif quantization_type == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        else:
            raise ValueError("Unsupported quantization type")
            
        # Convert model
        tflite_model = converter.convert()
        
        # Lưu model
        tflite_path = self.output_dir / f"model_{quantization_type}.tflite"
        tflite_path.write_bytes(tflite_model)
        logger.info(f"Saved quantized model to {tflite_path}")

    def apply_distillation(
        self,
        student_model: tf.keras.Model,
        temperature: float = 3.0,
        alpha: float = 0.1,
        epochs: int = 10
    ) -> tf.keras.Model:
        """
        Áp dụng knowledge distillation để cải thiện performance.
        
        Args:
            student_model: Model nhỏ hơn cần train
            temperature: Temperature cho distillation
            alpha: Trọng số giữa distillation và student loss
            epochs: Số epochs training
            
        Returns:
            Student model sau khi distill
        """
        class DistillationModel(tf.keras.Model):
            def __init__(
                self,
                student_model,
                teacher_model,
                temperature,
                alpha
            ):
                super().__init__()
                self.student_model = student_model
                self.teacher_model = teacher_model
                self.temperature = temperature
                self.alpha = alpha

            def compile(self, optimizer, metrics):
                super().compile(optimizer=optimizer, metrics=metrics)
                self.distillation_loss_tracker = tf.keras.metrics.Mean(
                    name='distillation_loss'
                )
                self.student_loss_tracker = tf.keras.metrics.Mean(
                    name='student_loss'
                )

            def train_step(self, data):
                x, y = data
                
                # Forward pass qua teacher
                teacher_predictions = self.teacher_model(x, training=False)
                
                with tf.GradientTape() as tape:
                    # Forward pass qua student
                    student_predictions = self.student_model(x, training=True)
                    
                    # Tính distillation loss
                    distillation_loss = tf.keras.losses.categorical_crossentropy(
                        tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                        tf.nn.softmax(student_predictions / self.temperature, axis=1)
                    ) * (self.temperature ** 2)
                    
                    # Tính student loss
                    student_loss = tf.keras.losses.categorical_crossentropy(
                        y,
                        student_predictions
                    )
                    
                    # Tổng hợp loss
                    loss = (
                        self.alpha * student_loss +
                        (1 - self.alpha) * distillation_loss
                    )
                
                # Backprop
                trainable_vars = self.student_model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                
                # Update metrics
                self.distillation_loss_tracker.update_state(distillation_loss)
                self.student_loss_tracker.update_state(student_loss)
                
                return {
                    'distillation_loss': self.distillation_loss_tracker.result(),
                    'student_loss': self.student_loss_tracker.result()
                }
        
        # Tạo và train distillation model
        distillation_model = DistillationModel(
            student_model,
            self.model,
            temperature,
            alpha
        )
        
        distillation_model.compile(
            optimizer='adam',
            metrics=['accuracy']
        )
        
        distillation_model.fit(
            self._get_calibration_dataset(),
            epochs=epochs
        )
        
        # Lưu student model
        student_path = self.output_dir / "distilled_model.h5"
        student_model.save(student_path)
        logger.info(f"Saved distilled model to {student_path}")
        
        return student_model

    def _get_calibration_dataset(self) -> tf.data.Dataset:
        """
        Tạo dataset cho calibration từ thư mục.
        
        Returns:
            tf.data.Dataset
        """
        # Implement logic để load và xử lý dataset
        pass

    def _get_representative_dataset(self) -> callable:
        """
        Generator function cho int8 quantization.
        
        Returns:
            Generator function
        """
        def representative_dataset():
            dataset = self._get_calibration_dataset()
            for data in dataset.take(100):
                yield [data]
                
        return representative_dataset

    def evaluate_model(
        self,
        model: tf.keras.Model,
        test_dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """
        Đánh giá performance của model.
        
        Args:
            model: Model cần đánh giá
            test_dataset: Dataset test
            
        Returns:
            Dict các metrics
        """
        # Tính accuracy và loss
        test_loss, test_accuracy = model.evaluate(test_dataset)
        
        # Tính latency
        batch = next(iter(test_dataset))
        times = []
        for _ in range(100):
            start_time = time.time()
            _ = model.predict(batch)
            times.append(time.time() - start_time)
            
        avg_latency = np.mean(times) * 1000  # Convert to ms
        
        # Tính model size
        model_size = Path(model.name).stat().st_size / (1024 * 1024)  # MB
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'average_latency_ms': avg_latency,
            'model_size_mb': model_size
        }
```

## 7. File Cấu Hình (config.yaml)

```yaml
# Cấu hình dữ liệu
data:
  raw_dir: "dataset/raw"
  processed_dir: "dataset/processed"
  test_split: 0.2
  val_split: 0.2

# Cấu hình model
model:
  input_shape: [224, 224, 3]
  embedding_dim: 512
  dropout_rate: 0.3
  l2_reg: 0.01
  arcface_margin: 0.5
  arcface_scale: 64.0

# Cấu hình training
training:
  batch_size: 32
  epochs: 100
  initial_lr: 0.001
  early_stopping_patience: 10
  mixed_precision: true
  checkpoint_dir: "checkpoints"
  log_dir: "logs"

# Cấu hình tối ưu hóa
optimization:
  pruning:
    target_sparsity: 0.5
    fine_tune_epochs: 5
  quantization:
    type: "int8"
    calibration_size: 100