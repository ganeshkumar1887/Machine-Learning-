import pandas as pd
df = pd.read_csv("ml_homeprices.csv")
count = df.isnull().sum()
print(count)
df["bedrooms"].fillna(df["bedrooms"].mean(),inplace=True)
print(df)