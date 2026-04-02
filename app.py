from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from inference import load_models, predict

# Create FastAPI app
app = FastAPI()

# Load models
transformer, classifier = load_models()


# input text
class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_input(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("Input text should be a non-empty string")
        return value


# Output prediction
class PredictResponse(BaseModel):
    prediction: str


# Endpoint /predict
@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest) -> PredictResponse:
    prediction = predict(request.text, transformer, classifier)
    return PredictResponse(prediction=prediction)
