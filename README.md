# 🛒 Walmart Sales Forecasting

An end-to-end Time Series Forecasting project built using Python, ARIMA, Prophet, and LSTM to predict Walmart weekly sales.

---

## 📌 Project Overview

Accurate sales forecasting helps retailers optimize inventory management, workforce planning, and business decisions.

In this project, historical Walmart sales data was analyzed and multiple forecasting models were developed and compared:

- Baseline Forecasting
- ARIMA
- Prophet
- LSTM Neural Network

The best-performing model was selected based on forecasting metrics.

---

## 📂 Dataset

Dataset: Walmart Recruiting - Store Sales Forecasting

Files Used:

- train.csv
- features.csv
- stores.csv

Key Features:

- Store
- Department
- Weekly Sales
- Temperature
- Fuel Price
- CPI
- Unemployment
- Holiday Information
- Markdown Data

---

## 🔧 Project Pipeline

### 1. Exploratory Data Analysis (EDA)

- Data inspection
- Missing value analysis
- Sales distribution analysis
- Sales trend visualization

### 2. Feature Engineering

Created time-series features:

- Year
- Month
- Quarter
- Week

Created lag features:

- Lag_1
- Lag_4

Created rolling statistics:

- Rolling_Mean_4
- Rolling_STD_4

### 3. Forecasting Models

#### Baseline Model

Uses previous week's sales as prediction.

#### ARIMA

Classical statistical forecasting model.

#### Prophet

Facebook Prophet time-series forecasting model.

#### LSTM

Deep Learning forecasting model built using TensorFlow/Keras.

### 4. Backtesting

Walk-forward validation used to evaluate forecasting stability.

### 5. Streamlit Dashboard

Interactive dashboard for model comparison and visualization.

---

## 📊 Model Performance

| Model | MAE | RMSE |
|---------|---------:|---------:|
| Baseline | 2113.06 | 7282.82 |
| ARIMA | 5614.81 | 5956.78 |
| Prophet | 2414.88 | 4781.80 |
| LSTM | 2937.40 | 3639.54 |

### 🏆 Best Model

LSTM achieved the lowest RMSE:

RMSE = 3639.54

---

## 📈 Backtesting Results

Average MAE: 4292.61

Best MAE: 1009.72

Worst MAE: 12769.43

---

## 🖥️ Streamlit Dashboard

Run locally:

```bash
streamlit run app.py
```

Dashboard Features:

- Model comparison table
- RMSE visualization
- Forecasting performance comparison

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Statsmodels
- Prophet
- TensorFlow
- Keras
- Streamlit

---

## 📁 Project Structure

```text
walmart_sales_forecasting/
│
├── data/
│   ├── train.csv
│   ├── stores.csv
│   ├── features.csv
│
├── src/
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_baseline_model.py
│   ├── 04_arima_model.py
│   ├── 05_prophet_model.py
│   ├── 06_lstm_model.py
│   ├── 07_model_comparison.py
│   └── 08_backtesting.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Future Improvements

- Hyperparameter tuning
- XGBoost forecasting
- Transformer-based forecasting models
- Multi-store forecasting dashboard
- Cloud deployment

---

## 👨‍💻 Author

Pavan Pawar Patil

Aspiring Data Scientist | Python Developer | Machine Learning Enthusiast

LinkedIn:
www.linkedin.com/in/pavan-pawar-patil

GitHub:
