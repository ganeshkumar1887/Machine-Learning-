import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\91778\OneDrive\Desktop\machine learning\weather prediction project\seattle-weather.csv")
print(df.head(15))

# Convert date to numeric
df['date'] = pd.to_datetime(df['date'])
df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())

# Features and target
X = df[["date_ordinal", "precipitation", "temp_max", "temp_min", "wind"]]
y = df["weather"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict new input
date_input = input("Enter day in yyyy-mm-dd format: ")
date_ordinal = datetime.strptime(date_input, "%Y-%m-%d").toordinal()

precipitation = float(input("Enter precipitation: "))
temp_max = float(input("Enter maximum temperature: "))
temp_min = float(input("Enter minimum temperature: "))
wind = float(input("Enter wind speed: "))

weather_prediction = model.predict([[date_ordinal, precipitation, temp_max, temp_min, wind]])
print("Predicted weather:", weather_prediction[0])
