import joblib
import pandas as pd

from src.config import MODEL_PATH, PREPROCESSOR_PATH


class StudentPerformancePredictor:
    """
    Load the trained model and make predictions.
    """

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

    def predict(self, data: dict) -> float:
        """
        Predict the final score for a single student.

        Parameters
        ----------
        data : dict
            Dictionary containing student features.

        Returns
        -------
        float
            Predicted final score.
        """

        input_df = pd.DataFrame([data])

        transformed_data = self.preprocessor.transform(input_df)

        prediction = self.model.predict(transformed_data)

        return float(prediction[0])