from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

model = joblib.load("model.joblib")

@app.post("/predict")
def predict(data: InputData):
    input_features = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = model.predict(input_features)
    return {"prediction": prediction[0]}