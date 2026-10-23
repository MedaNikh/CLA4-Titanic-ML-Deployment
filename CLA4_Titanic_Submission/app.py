from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib

app = FastAPI(title="Titanic Prediction API")

model = joblib.load("/content/model.pkl")

class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float = Field(ge=0, le=100)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"]

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Titanic API is running"
    }

@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])
    prediction = int(model.predict(data)[0])

    return {
        "predicted_survived": prediction,
        "prediction": (
            "Survived" if prediction == 1
            else "Did Not Survive"
        )
    }
