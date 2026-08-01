from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "students.csv"

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "model.pkl"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

TEST_SIZE = 0.2

RANDOM_STATE = 42
