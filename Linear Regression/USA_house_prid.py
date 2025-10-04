import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"C:\Users\91778\OneDrive\Desktop\machine learning\sallary recomadtion system\ml_homeprices.csv")
print("Missing values:\n", df.isnull().sum())



df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].mean())   # better way
X = df[['area','bedrooms','age']]
y = df['price']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = r2_score(y_train, y_train_pred)
test_accuracy = r2_score(y_test, y_test_pred)

print("Train Accuracy (R²):", train_accuracy)
print("Test Accuracy (R²):", test_accuracy)

area = float(input("Enter the area of house: "))
bedrooms = float(input("Enter the number of bedrooms: "))
age = float(input("Enter the age of house: "))


price = model.predict([[area, bedrooms, age]])
print(f"\nPredicted Price of the house: {price[0]:.2f}")
