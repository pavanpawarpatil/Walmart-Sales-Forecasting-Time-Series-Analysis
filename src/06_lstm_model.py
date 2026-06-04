import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

df = pd.read_csv("data/features_data.csv")
print(df.shape)

# Step 1 - Select one series

series_df = df[
    (df["Store"] == 1 ) &
    (df["Dept"] == 1)
]

print(series_df.shape)

# Step 2 - Keep Only Target
sales = series_df[["Weekly_Sales"]]
print(sales.head())

# Step 3 - Scale Data

scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales)
print(sales_scaled[:5])

# Step 4 - Check Shape

print(sales_scaled.shape)


# Step 5 - Create Sequences

x = []
y = []
window_size = 4
for i in range(window_size, len(sales_scaled)):
    x.append(sales_scaled[i-window_size:i, 0])
    y.append(sales_scaled[i, 0])

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

# Step 6 - Reshape for LSTM for 3D input
# (samples, timesteps, features)
# 135 samples, 4 time steps, 1 feature (sales)

x = x.reshape(
    x.shape[0],
    x.shape[1],
    1
)
print(x.shape)

# Step 7 - Train-Test Split

split_idx = int(len(x) * 0.8)
x_train = x[:split_idx]
x_test = x[split_idx:]

y_train = y[:split_idx]
y_test = y[split_idx:]

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

# Step 8 - Build the LSTM Model

model = Sequential()
model.add(
    LSTM(
        units = 50,
        activation = "relu",
        input_shape = (4, 1)
    )
)
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss = "mse"
)
model.summary()


# Step 9 - Train the Model
history = model.fit(
    x_train, y_train, epochs=50, batch_size=8, 
    validation_data = (x_test, y_test),
    verbose=1
)


# Step 10 - Plot that

# plt.figure(figsize=(10,5))

# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])

# plt.title("LSTM Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")

# plt.legend(["Train", "Validation"])

# plt.show()

# Step 11 - Make Predictions

predictions = model.predict(x_test)
# print(predictions[:5])

predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

print(predictions[:5])
print(y_test_actual[:5])


# Step 12 - Evaluate LSTM
mae = mean_absolute_error(
    y_test_actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        predictions
    )
)

print(f"LSTM MAE: {mae:.2f}")
print(f"LSTM RMSE: {rmse:.2f}")

# Step 13 - Plot Actual vs LSTM

plt.figure(figsize=(12,6))

plt.plot(
    y_test_actual,
    label="Actual"
)

plt.plot(
    predictions,
    label="LSTM Forecast"
)

plt.legend()

plt.title("LSTM Forecast vs Actual")

plt.show()