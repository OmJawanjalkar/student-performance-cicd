from src.predict import StudentPerformancePredictor


def test_prediction():

    predictor = StudentPerformancePredictor()

    sample = {
        "hours_studied": 6.5,
        "attendance": 90,
        "assignments_completed": 18,
        "previous_score": 82,
        "sleep_hours": 7.5,
        "internet_access": "Yes",
    }

    prediction = predictor.predict(sample)

    assert isinstance(prediction, float)
