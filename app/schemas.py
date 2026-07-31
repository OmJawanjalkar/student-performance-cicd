from pydantic import BaseModel, Field


class StudentRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24)
    attendance: float = Field(..., ge=0, le=100)
    assignments_completed: int = Field(..., ge=0)
    previous_score: float = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=24)
    internet_access: str


class PredictionResponse(BaseModel):
    predicted_score: float