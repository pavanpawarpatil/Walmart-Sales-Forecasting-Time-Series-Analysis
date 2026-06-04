import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame({
    "Model": ["Baseline", "ARIMA", "Prophet", "LSTM"],
    "MAE": [2113.06, 5614.81, 2414.88, 2937.40],
    "RMSE": [7282.82, 5956.78, 4781.80, 3639.54]
})

print(results)


# RMSE Comparison Chart
plt.figure(figsize=(8,5))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE")

plt.show()