import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

from src.config import MODEL_DIR
from src.preprocess import prepare_data


def train_model():
    """
    Train a Random Forest model and save it along with the preprocessor.
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = prepare_data()

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")

    print("\nModel saved successfully!")
    print(f"Model Path        : {MODEL_DIR / 'model.pkl'}")
    print(f"Preprocessor Path : {MODEL_DIR / 'preprocessor.pkl'}")


if __name__ == "__main__":
    train_model()
