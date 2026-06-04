import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Step 1 - Load Features_data file

df = pd.read_csv("data/features_data.csv")

# print(df.shape)
# print(df.head())

# Step 2 - Sort Date

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Store", "Dept", "Date"]
)

# Step 3 - Create Baseline Prediction

df["Baseline_Prediction"] = (
    df.groupby(["Store", "Dept",])["Weekly_Sales"].shift(1)
)

# Step 4 - Remove Missing Values

df = df.dropna(subset=["Baseline_Prediction"])

# print(df[["Weekly_Sales", "Baseline_Prediction"]].head(10))
# print(df[["Store", "Dept", "Date", "Weekly_Sales", "Baseline_Prediction"]].head(10))


# Step 5 - Evaluate the Baseline

mae = mean_absolute_error(
    df["Weekly_Sales"],
    df["Baseline_Prediction"]
)

rmse = np.sqrt(
    mean_squared_error(
        df["Weekly_Sales"],
        df["Baseline_Prediction"]
    )
)


non_zero = df["Weekly_Sales"] != 0

mape = (
    np.mean(np.abs((df.loc[non_zero, "Weekly_Sales"] - df.loc[non_zero, "Baseline_Prediction"]) / df.loc[non_zero, "Weekly_Sales"])) * 100
)


print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

print((df["Weekly_Sales"] == 0).sum())

split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

print("Train Dates")
print(train_df["Date"].min())
print(train_df["Date"].max())

print("\nTest Dates")
print(test_df["Date"].min())
print(test_df["Date"].max())