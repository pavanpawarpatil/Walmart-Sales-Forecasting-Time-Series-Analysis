import pandas as pd

train = pd.read_csv("data/train.csv")
features = pd.read_csv("data/features.csv")
stores = pd.read_csv("data/stores.csv")

# Handle missing values
features["CPI"] = features["CPI"].fillna(features["CPI"].median())
features["Unemployment"] = features["Unemployment"].fillna(features["Unemployment"].median())

markdown_cols = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
features[markdown_cols] = features[markdown_cols].fillna(0)

# Merge
df = train.merge(stores, on="Store", how="left")
df = df.merge(features, on=["Store","Date","IsHoliday"], how="left")

# Date conversion
df["Date"] = pd.to_datetime(df["Date"])

# Step 10 - Convert Date column
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["Week"] = df["Date"].dt.isocalendar().week

# print(df[["Date","Year","Month","Quarter","Week"]].head())

# Step 11 Create Lag Features

df = df.sort_values(["Store", "Dept", "Date"])
df["Lag_1"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
)

# print(df[["Store","Dept","Date","Weekly_Sales","Lag_1"]].head(10))

df["Lag_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
      .shift(4)
)

# print(df[["Date", "Weekly_Sales", "Lag_1", "Lag_4"]].head(10))

df["Rolling_Mean_4"] =(
    df.groupby(["Store","Dept"])["Weekly_Sales"].transform(lambda x: x.shift(1).rolling(4).mean())
)
# print(df[["Date","Weekly_Sales","Lag_1","Lag_4","Rolling_Mean_4"]].head(10))

df["Rolling_STD_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(lambda x: x.shift(1).rolling(4).std())
)
# print(df[["Date","Weekly_Sales","Rolling_Mean_4","Rolling_STD_4"]].head(10))

# print(df.isnull().sum())


df["Rolling_STD_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(lambda x: x.shift(1).rolling(4).std())
)
print(df.isnull().sum())

print(df.shape)
df = df.dropna()
print(df.shape)

df.to_csv("data/features_data.csv", index=False)