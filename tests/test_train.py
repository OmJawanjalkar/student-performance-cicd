from pathlib import Path

from src.train import train_model


def test_training():

    train_model()

    assert Path("model/model.pkl").exists()

    assert Path("model/preprocessor.pkl").exists()