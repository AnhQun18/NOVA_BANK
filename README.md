# 🏦 Nova Bank — Dự báo rủi ro tín dụng cá nhân

Ứng dụng các mô hình học máy để dự báo khả năng vỡ nợ của khách hàng cá nhân, lựa chọn ngưỡng phân loại theo chi phí kinh doanh và giải thích kết quả bằng SHAP.

Dự án được thực hiện trong khuôn khổ đề án **“Ứng dụng mô hình phân loại dự báo rủi ro tín dụng khách hàng cá nhân tại Nova Bank”** tại Trường Đại học Kinh tế Quốc dân.

## Mục tiêu

- So sánh hiệu suất của Logistic Regression, Random Forest và XGBoost.
- Xây dựng quy trình tiền xử lý và huấn luyện bằng Pipeline.
- Tối ưu siêu tham số cho mô hình XGBoost.
- Lựa chọn ngưỡng quyết định nhằm giảm thiểu chi phí dự báo sai.
- Giải thích các yếu tố ảnh hưởng đến dự báo bằng SHAP.

## Nội dung repository

| File | Nội dung |
|---|---|
| [Tester.ipynb](Tester.ipynb) | Notebook chính: tiền xử lý, huấn luyện, tối ưu, đánh giá và trực quan hóa |
| [nova_bank_credit_risk_production.pkl](nova_bank_credit_risk_production.pkl) | Pipeline tiền xử lý và mô hình XGBoost đã huấn luyện |
| [Báo cáo đề án](Trần%20Anh%20Quân_11233244_Đề%20án.pdf) | Cơ sở lý thuyết, phương pháp nghiên cứu và phân tích kết quả |

## Dữ liệu

Nghiên cứu sử dụng bộ dữ liệu minh họa tổng hợp gồm **32.581 quan sát và 29 biến ban đầu**, mô tả thông tin khách hàng, đặc điểm khoản vay và lịch sử tín dụng.

Biến mục tiêu là `loan_status`:

- **0:** Không vỡ nợ.
- **1:** Vỡ nợ.

**Dữ liệu gốc không được đính kèm trong repository.** Notebook có lưu kết quả và biểu đồ để người đọc theo dõi quá trình phân tích.

## Phương pháp

### 1. Tiền xử lý dữ liệu

- Chuyển các giá trị tuổi và thâm niên không hợp lý thành giá trị thiếu.
- Loại bỏ biến định danh, biến có nguy cơ rò rỉ thông tin và một số biến nhạy cảm.
- Kiểm tra đa cộng tuyến bằng hệ số VIF.
- Xử lý ngoại lai bằng Winsorizing tại phân vị 1% và 99%.
- Điền khuyết bằng trung vị hoặc giá trị phổ biến nhất.
- Mã hóa biến phân loại bằng One-Hot Encoding.
- Chuẩn hóa biến số bằng StandardScaler.

### 2. Chia tập dữ liệu

Dữ liệu được chia có phân tầng theo biến mục tiêu với `random_state=42`.

| Tập dữ liệu | Tỷ lệ | Số quan sát |
|---|---:|---:|
| Train | 70% | 22.806 |
| Validation | 15% | 4.887 |
| Test | 15% | 4.888 |

### 3. Huấn luyện và tối ưu

Ba mô hình được so sánh bằng **Stratified 5-Fold Cross Validation**, sử dụng ROC-AUC làm tiêu chí chính.

XGBoost được lựa chọn để tối ưu bằng **RandomizedSearchCV**, kết hợp Early Stopping. Trọng số lớp được sử dụng để xử lý tình trạng mất cân bằng nhãn.

### 4. Lựa chọn ngưỡng quyết định

Ngưỡng được lựa chọn trên tập Validation theo hai tiêu chí:

- Tối đa hóa F1-score.
- Tối thiểu hóa chi phí kinh doanh với giả định:

**Chi phí = 5 × False Negative + 1 × False Positive**

Trong đó, False Negative là trường hợp khách hàng thực tế vỡ nợ nhưng được mô hình dự báo an toàn.

## Kết quả

### So sánh mô hình cơ sở

| Mô hình | ROC-AUC trung bình | Độ lệch chuẩn |
|---|---:|---:|
| Logistic Regression | 0.8516 | 0.0049 |
| Random Forest | 0.9071 | 0.0028 |
| XGBoost | **0.9263** | 0.0055 |

### XGBoost sau tối ưu

| Chỉ số | Kết quả |
|---|---:|
| ROC-AUC trên Train | 0.9807 |
| ROC-AUC trên Test | **0.9462** |
| Ngưỡng tối đa hóa F1-score | 0.670 |
| Ngưỡng tối thiểu hóa chi phí | **0.404** |
| Accuracy trên Test | 0.88 |
| Precision — lớp vỡ nợ | 0.67 |
| Recall — lớp vỡ nợ | **0.85** |
| F1-score — lớp vỡ nợ | 0.75 |

Tại ngưỡng chi phí được lựa chọn, mô hình nhận diện được khoảng **85% trường hợp vỡ nợ** trên tập Test.

Các số liệu trên là kết quả được lưu trong notebook và trình bày trong báo cáo. Ngưỡng hiển thị được làm tròn đến ba chữ số thập phân.

## Giải thích mô hình

Dự án sử dụng **SHAP TreeExplainer** để phân tích mức độ và chiều tác động của các đặc trưng đến đầu ra mô hình.

Notebook cung cấp các biểu đồ:

- Phân phối biến mục tiêu.
- Boxplot trước và sau Winsorizing.
- Ma trận tương quan.
- Đường cong ROC trên Train và Test.
- Đường cong Precision–Recall.
- Ma trận nhầm lẫn.
- Chi phí theo ngưỡng phân loại.
- SHAP Summary Plot.

## Công nghệ sử dụng

**Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · statsmodels · Matplotlib · Seaborn · joblib**

## Cách xem và chạy

Có thể mở `Tester.ipynb` trực tiếp trên GitHub hoặc tải về để xem bằng Jupyter Notebook, JupyterLab hay VS Code.

Để chạy lại toàn bộ quy trình, cần bổ sung dữ liệu đầu vào và cập nhật biến `file_path` trong notebook.

Cài đặt các thư viện cần thiết:

```bash
pip install jupyterlab pandas numpy scikit-learn xgboost shap statsmodels matplotlib seaborn joblib openpyxl
