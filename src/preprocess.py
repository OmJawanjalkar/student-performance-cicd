from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE


def load_data() -> pd.DataFrame:
    """
    Load the student dataset.
    """

    return pd.read_csv(DATA_PATH)


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features (X) and target (y).
    """

    X = df.drop(columns=["final_score"])
    y = df["final_score"]

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessing pipeline.
    """

    numerical_features = [
        "hours_studied",
        "attendance",
        "assignments_completed",
        "previous_score",
        "sleep_hours",
    ]

    categorical_features = [
        "internet_access",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def prepare_data():
    """
    Load, split, preprocess, and return train/test data.
    """

    df = load_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )