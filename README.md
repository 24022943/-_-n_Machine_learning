# EcoPredict Carbon 

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

