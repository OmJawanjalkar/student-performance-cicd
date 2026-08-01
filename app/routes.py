from fastapi import APIRouter, HTTPException

from app.schemas import PredictionResponse, StudentRequest
from src.predict import StudentPerformancePredictor

router = APIRouter()

# Load the model once when the application starts
predictor = StudentPerformancePredictor()


@router.get("/")
def home():
    return {"message": "Student Performance Prediction API", "status": "Running"}


@router.get("/health")
def health():
    return {"status": "Healthy"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(student: StudentRequest):

    try:
        prediction = predictor.predict(student.model_dump())

        return PredictionResponse(predicted_score=round(prediction, 2))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
