import pandas as pd
import joblib

model = joblib.load("/content/model.pkl")
data = pd.read_csv("/content/unseen_passengers.csv")
passengers = data.iloc[1:21].copy()

passengers["Predicted_Survived"] = model.predict(passengers)
passengers["Prediction"] = passengers["Predicted_Survived"].map({
    1: "Survived", 0: "Did Not Survive"
})

passengers.to_csv("/content/batch_predictions.csv", index=False)
print(passengers.to_string(index=False))
print("Saved batch_predictions.csv")