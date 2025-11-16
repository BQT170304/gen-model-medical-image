# AUROC Evaluation for Anomaly Detection

## Mô tả
Script này tính toán AUROC (Area Under ROC Curve) cho toàn bộ tập validation để đánh giá hiệu suất phát hiện bất thường của các mô hình LDM và VQ-VAE.

## Cấu trúc files
- `src/calculate_auroc.py`: Script chính để tính AUROC
- `configs/auroc_eval.yaml`: File cấu hình cho evaluation

## Cách sử dụng

### 1. Chạy AUROC evaluation với config mặc định:
```bash
cd /home/tqlong/qtung/gen-model-boilerplate
python src/calculate_auroc.py
```

### 2. Chạy với override các tham số:
```bash
# Thay đổi số samples và noise level
python src/calculate_auroc.py evaluation.max_samples=200 sampling.noise_level=500

# Tắt VQ-VAE evaluation
python src/calculate_auroc.py evaluation.use_vqvae=false

# Thay đổi classifier scale
python src/calculate_auroc.py classifier.scale=50.0

# Không skip normal samples
python src/calculate_auroc.py evaluation.skip_normal=false
```

### 3. Các tham số quan trọng trong config:

#### Evaluation settings:
- `evaluation.max_samples`: Số lượng samples tối đa để xử lý (100 mặc định)
- `evaluation.skip_normal`: Có bỏ qua samples bình thường hay không (true)
- `evaluation.use_vqvae`: Có đánh giá VQ-VAE hay không (true)
- `evaluation.save_dir`: Thư mục lưu kết quả

#### Sampling settings:
- `sampling.noise_level`: Mức noise L cho diffusion (200 mặc định)  
- `sampling.num_samples_per_image`: Số samples tạo cho mỗi ảnh (chọn tốt nhất)

#### Classifier settings:
- `classifier.use_classifier`: Có sử dụng classifier guidance hay không
- `classifier.scale`: Hệ số scale cho classifier guidance (100.0)
- `classifier.path`: Đường dẫn đến model classifier

## Kết quả
Script sẽ tạo ra:
1. **Console output**: Real-time progress và metrics
2. **auroc_results.txt**: File chứa tất cả kết quả chi tiết

### Các metrics được tính:
- **Global AUROC**: AUROC tính trên toàn bộ pixel của dataset (flatten tất cả)
- **Mean Dice Score**: Dice score trung bình trên tất cả samples
- **Mean IoU**: IoU trung bình trên tất cả samples
- **Individual results**: Kết quả chi tiết cho từng sample

## Phương pháp tính AUROC
Sử dụng phương pháp **Global AUROC**:
```python
# Flatten toàn bộ pixel của dataset
global_y_true = np.concatenate(all_gt_flat)  # Ground truth masks
global_y_score = np.concatenate(all_out_flat)  # Anomaly scores

# Tính AUROC duy nhất cho toàn dataset
auroc = roc_auc_score(global_y_true, global_y_score)
```

## Lưu ý
- Đảm bảo các model weights tồn tại tại đường dẫn được chỉ định
- Script tự động điều chỉnh device (GPU/CPU) dựa trên availability
- VQ-VAE evaluation có thể được tắt nếu không cần thiết
- Kết quả được lưu tự động vào thư mục được chỉ định