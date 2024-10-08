# test_train.py
import pytest
from train import train_model

def test_model_accuracy():
    accuracy = train_model()
    # Assert that the accuracy is above a certain threshold (e.g., 0.9)
    assert accuracy > 0.9, f"Expected accuracy to be above 0.9 but got {accuracy}"
