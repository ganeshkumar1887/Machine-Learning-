import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\91778\OneDrive\Desktop\machine learning\house_price_prediction\area_price_data.csv")   # Make sure file has columns: Area, Price

X = data[['area']]  
y = data['price']    

model = LinearRegression()
model.fit(X, y)

area = float(input("Enter the area of the house (in sq.ft): "))

predicted_price = model.predict([[area]])
print(f"Predicted Price for {area} sq.ft = {predicted_price[0]:.2f}")

plt.scatter(X, y, color='blue', label='Actual Data')     
plt.plot(X, model.predict(X), color='red', label='Regression Line') 
plt.scatter(area, predicted_price, color='green', s=100, label='Predicted Point') 
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.legend()
plt.show()
