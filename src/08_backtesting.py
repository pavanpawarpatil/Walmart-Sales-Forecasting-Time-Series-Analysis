import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


# Step 1 - Load Data

df = pd.read_csv("data/features_data.csv")

series_df = df[
    (df["Store"] == 1) &
    (df["Dept"] == 1)
].copy()

series_df = series_df.sort_values("Date")

print(series_df.shape)

# Step 2 - Create Prophet Dataset

backtest_df = series_df[
    ["Date", "Weekly_Sales"]
].copy()

backtest_df.columns = ["ds", "y"]

print(backtest_df.head())

# Step 3 - Walk Forward Validation

window_size = 100
forecast_horizon = 4

mae_scores = []

# Step 4 - Backtesting Loop

for start in range(window_size, len(backtest_df) - forecast_horizon, forecast_horizon):

    train = backtest_df.iloc[:start]

    test = backtest_df.iloc[start:start + forecast_horizon]
    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(
        periods=forecast_horizon,
        freq="W"
    )

    forecast = model.predict(future)

    preds = forecast["yhat"].tail(
        forecast_horizon
    )

    mae = mean_absolute_error(
        test["y"],
        preds
    )

    mae_scores.append(mae)

    print(
        f"Window End={start} | MAE={mae:.2f}"
    )
    
# Step 5 - Summary

print("\nBacktesting Results")

print(
    f"Average MAE: {np.mean(mae_scores):.2f}"
)

print(
    f"Best MAE: {np.min(mae_scores):.2f}"
)

print(
    f"Worst MAE: {np.max(mae_scores):.2f}"
)