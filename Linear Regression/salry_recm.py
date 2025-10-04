import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"Salary_Data.csv")
print(df.head())
X = df[['YearsExperience']]  
y = df['Salary'] 


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)
model = LinearRegression()
model.fit(X, y)

salary= float(input("Enter your years experience:: "))
predict_sallary = model.predict(pd.DataFrame([[salary]], columns=['YearsExperience']))[0]
print(f"Predicted Salary for {salary} years experience is: {predict_sallary:.2f}")


# by solve using coef and intercept function
manual_pred = model.coef_[0] * salary + model.intercept_
print("Manual prediction:", manual_pred)

score=model.score(X_test,y_test)
print("Score",score)

# visualization on graph
plt.scatter(X, y, color='blue', label='Actual Data')     
plt.plot(X, model.predict(X), color='red', label='Regression Line') 
plt.scatter(X, y, color='green', s=100, label='Predicted Point') 
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Salary Prediction")
plt.legend()
plt.show()