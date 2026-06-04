import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("data/train.csv")
features = pd.read_csv("data/features.csv")
stores = pd.read_csv("data/stores.csv")


print(train.shape)
print(features.shape)
print(stores.shape)
# print(train.columns)
# print(train.head())
# print(train.info())

# print("\nTrain Missing Values:")
# print(train.isnull().sum())

# print("\nfeatures Missing Values:")
# print(features.isnull().sum())

# print("\nstores Missing Values:")
# print(stores.isnull().sum())

# Step 1 - Handel Missing Values
features["CPI"] = features["CPI"].fillna(features["CPI"].median())
features["Unemployment"] = features["Unemployment"].fillna(features["Unemployment"].median())

markdown_cols = ["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
features[markdown_cols] = features[markdown_cols].fillna(0)

# print(features.isnull().sum())



# Step 2 - Merge the datasets
df = train.merge(stores, on="Store", how="left")
df = df.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

# print(df.shape)
# print(df.head())

# print(df.info())
# print(df.isnull().sum())



# Step 4 - Target Variable Analysis Weekly_Sales
plt.hist(df["Weekly_Sales"],bins=50)
plt.title("Weekly Sales Distribution")
plt.show()

# print(df["Weekly_Sales"].describe())

# Step 5 - Sales Trend Over Time
sales_trend = df.groupby("Date")["Weekly_Sales"].sum()

plt.figure(figsize=(15,6))
plt.plot(sales_trend)
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.show()

# Step 6 - Holiday Impact

holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].mean()
print(holiday_sales)

plt.figure(figsize=(6,4))
sns.boxplot(
    x = "IsHoliday",
    y = "Weekly_Sales",
    data=df    
)
plt.title("Holiday vs Non-Holiday Sales")
plt.show()

# Step 7 - Store-wise Analysis
top_stores = (df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False).head(10))

print(top_stores)

plt.figure(figsize=(10,5))
top_stores.plot(kind="bar")
plt.title("# Step 8 - Store-wise Analysis")
plt.show()

# Step 8 - Department-wise Analysis

top_depts = (
    df.groupby("Dept")["Weekly_Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print(top_depts)

plt.figure(figsize=(10,5))
top_depts.plot(kind="bar")
plt.title("Top 10 Departments by Sales")
plt.show()

# Step 9 - Correlation Analysis

corr = df[
    [
        "Weekly_Sales",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment"
    ]
].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)

plt.title("Correlation Matrix")
plt.show()