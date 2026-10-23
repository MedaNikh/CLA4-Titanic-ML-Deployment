import pandas as pd
import joblib
import time

model = joblib.load("/content/model.pkl")
data = pd.read_csv("/content/unseen_passengers.csv")
passengers = data.iloc[21:31]

for number, (_, row) in enumerate(passengers.iterrows(), start=1):
    prediction = int(model.predict(row.to_frame().T)[0])
    result = "Survived" if prediction == 1 else "Did Not Survive"
    print(f"Passenger {number}: {result}")
    time.sleep(1)

print("Streaming inference completed!")