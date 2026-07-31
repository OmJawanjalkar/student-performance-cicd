from src.predict import StudentPerformancePredictor

predictor = StudentPerformancePredictor()

student = {
    "hours_studied": 6.5,
    "attendance": 90,
    "assignments_completed": 18,
    "previous_score": 82,
    "sleep_hours": 7.5,
    "internet_access": "Yes",
}

prediction = predictor.predict(student)

print(f"Predicted Final Score: {prediction:.2f}")