import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/features_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Step 1 - Select one series 

series_df = df[
    (df["Store"] == 1) &
    (df["Dept"] == 1) 
].copy()

print(series_df.shape)
print(series_df.head())

# Step 2 - Visualize the Series

plt.figure(figsize=(12, 6))

plt.plot(
    series_df["Date"],
    series_df["Weekly_Sales"]
)
plt.title("Store 1 Dept 1 Weekly Sales")
plt.xlabel("Data")
plt.ylabel("Weekly_Sales")

plt.grid()
plt.show()

# Step 3 - Create true time base split

split_idx = int(len(series_df) * 0.8)

train = series_df.iloc[:split_idx]
test = series_df.iloc[split_idx:]

print(train.shape)
print(test.shape)

# Step 4 - We Create first ARIMA Model

train_sales = train["Weekly_Sales"]
test_sales = test["Weekly_Sales"]

orders = [
    (1,1,1),
    (2,1,1),
    (1,1,2),
    (2,1,2)
]

for order in orders:

    print(f"\nTraining ARIMA{order}")

    model = ARIMA(
        train_sales,
        order=order
    )

    model_fit = model.fit()

    print(f"AIC: {model_fit.aic:.2f}")

forecast = model_fit.forecast(steps=len(test))

print(forecast.head())


# Step 5 - Evaluate ARIMA

mae = mean_absolute_error(test_sales, forecast)

rmse = np.sqrt(
    mean_squared_error(
        test_sales,
        forecast
    )
)
print(f"ARIMA MAE: {mae:.2f}")
print(f"ARIMA RMSE: {rmse:.2f}")

# Step 6 - Plot Actual vs Forecast

plt.figure(figsize=(12,6))

plt.plot(
    test["Date"],
    test_sales,
    label="Actual"
)

plt.plot(
    test["Date"],
    forecast,
    label="ARIMA Forecast"
)

plt.legend()

plt.title("ARIMA Forecast vs Actual")

plt.show()