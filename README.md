# EcoPredict Carbon – Final Python Files

Đề tài: **Ứng dụng học máy trong xây dựng hệ thống dự báo phát thải carbon của sản phẩm**  
Dataset: **The Carbon Catalogue – Product Level Data** (`carbon_catalogue.csv`)

## File chính

- `carbon_utils.py`: hàm dùng chung, xử lý dữ liệu, feature engineering, pipeline, metric.
- `carbon_eda.py`: EDA riêng, tạo biểu đồ phân phối PCF, missing values, boxplot theo ngành, correlation heatmap, KS-test.
- `train_advanced_models.py`: huấn luyện mô hình phân loại + hồi quy, CV, tuning, ROC, calibration, residual, permutation importance.
- `app.py`: giao diện Streamlit EcoPredict Carbon.
- `requirements.txt`: thư viện cần cài.

## Cách chạy trên Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Đặt file dữ liệu trong cùng thư mục và đổi tên thành:

```text
carbon_catalogue.csv
```

Chạy EDA:

```powershell
python carbon_eda.py
```

Huấn luyện mô hình:

```powershell
python train_advanced_models.py
```

Chạy web:

```powershell
python -m streamlit run app.py
```

## Các điểm đã cải thiện

- Tên hệ thống cố định: **EcoPredict Carbon – Hệ thống dự báo phát thải carbon của sản phẩm**.
- Dùng dữ liệu Carbon Catalogue thực tế, không còn logic của dataset mô phỏng 19 features.
- Có EDA riêng theo góp ý: phân phối PCF, missing, outlier, boxplot ngành, trend theo năm, heatmap, KS-test.
- Tạo nhãn Low/Medium/High bằng ngưỡng Q25/Q75 trên **train set** để tránh leakage.
- Loại `carbon_intensity` khỏi feature vì là biến dẫn xuất từ PCF/weight.
- Bổ sung feature: dominant_stage, weight_category, lifecycle_balance_std, interaction lifecycle × weight.
- Huấn luyện dual-task: classification + regression.
- Đánh giá: CV F1-macro, ROC-AUC, calibration curve, confusion matrix, predicted-vs-actual, residual distribution.
- Web sửa bố cục: “Giải thích yếu tố ảnh hưởng” nằm dưới “So sánh PCF với ngành”, chỉ hiện top 6 yếu tố, có bảng chi tiết.
