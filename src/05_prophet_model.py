import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1 - Load Features_data file and Create Basic Prophet Script

df = pd.read_csv("data/features_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

series_df = df[
    (df["Store"] == 1) &
    (df["Dept"] == 1)
].copy()

print(series_df.shape)

# Step 2 - Prophet Data Format

prophet_df = series_df[["Date", "Weekly_Sales"]]
prophet_df = prophet_df.rename(
    columns={
        "Date": "ds",
        "Weekly_Sales": "y"
    }
)
print(prophet_df.head())


#Step 3 - Train Test Split

split_idx = int(len(prophet_df) * 0.8)

train = prophet_df.iloc[:split_idx]
test = prophet_df.iloc[split_idx:]

print("Train:", train.shape)
print("Test :", test.shape)

# Step 5 - Train Prophet

model = Prophet()
model.fit(train)


# Step 6 - Forecast the Test Period

future = model.make_future_dataframe(
    periods=len(test),
    freq="W"
)
forecast = model.predict(future)
print(
    forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]  
    ].tail()
)
# Prophet Components
model.plot_components(forecast)
plt.show()
# Column	    Meaning
# yhat	        Predicted sales
# yhat_lower	Lower confidence bound
# yhat_upper	Upper confidence bound

prophet_forecast = forecast.iloc[-len(test):]["yhat"]

print(prophet_forecast.head())

# Step 7 - Evaluate the Prophet
mae = mean_absolute_error(test["y"], prophet_forecast)

rmse = np.sqrt(
    mean_squared_error(test["y"],prophet_forecast)
)

mape = (np.mean
        (np.abs((test["y"] - prophet_forecast) / test["y"])) * 100
)
print(f"Prophet MAE: {mae:.2f}")
print(f"Prophet RMSE: {rmse:.2f}")
print(f"Prophet MAPE: {mape:.2f}%")

# Step 8 - Plot that

plt.figure(figsize=(12, 6))
plt.plot(test["ds"], test["y"], label="Actual")
plt.plot(test["ds"], prophet_forecast, label="Prophet_Forecast")
plt.legend()
plt.title("Prophet Forecast vs Actual")
plt.show()

# Step 9 - Create a comparison summary
results = {
    "Model": ["Baseline", "ARIMA", "Prophet"],
    "MAE": [2113.06, 5614.81, 2414.88],
    "RMSE": [7282.82, 5956.78, 4781.80]
}

results_df = pd.DataFrame(results)

print(results_df)