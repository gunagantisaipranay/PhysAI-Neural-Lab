"""Machine-learning subpackage: neural network that predicts SHM displacement."""

from .dataset import generate_dataset
from .model import NeuralOscillatorModel, TrainingResult
from .train import train_model
from .evaluate import evaluate_model

__all__ = [
    "generate_dataset",
    "NeuralOscillatorModel",
    "TrainingResult",
    "train_model",
    "evaluate_model",
]