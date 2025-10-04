import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("hiring.csv")
print(df)

print("Missing values:\n", df.isnull().sum())
df["experience"] = df["experience"].fillna(df["experience"].mean())
df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(df["test_score(out of 10)"].mean())
print(df)