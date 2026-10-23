import pandas as pd
import joblib

model = joblib.load("/content/model.pkl")
data = pd.read_csv("/content/unseen_passengers.csv")
passenger = data.iloc[[0]]

prediction = int(model.predict(passenger)[0])

print("Passenger details:")
print(passenger.to_string(index=False))
print("Prediction:", "Survived" if prediction == 1 else "Did Not Survive")