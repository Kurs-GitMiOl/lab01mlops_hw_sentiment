from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: PredictRequest):
    return {"prediction": "some_text"}
