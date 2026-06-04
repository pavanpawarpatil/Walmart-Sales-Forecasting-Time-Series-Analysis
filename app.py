import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.title("Walmart Sales Forecasting")

results = pd.DataFrame({
    "Model": ["Baseline", "ARIMA", "Prophet", "LSTM"],
    "MAE": [2113.06, 5614.81, 2414.88, 2937.40],
    "RMSE": [7282.82, 5956.78, 4781.80, 3639.54]
})

st.subheader("Model Comparison")

st.dataframe(results)

fig, ax = plt.subplots()

ax.bar(
    results["Model"],
    results["RMSE"]
)

ax.set_title("RMSE Comparison")

st.pyplot(fig)