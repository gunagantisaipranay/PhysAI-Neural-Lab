"""Physics subpackage: mass-spring oscillator analytical and numerical solvers."""

from .oscillator import (
    validate_parameters,
    angular_frequency,
    period,
    analytical_solution,
)
from .numerical_solver import rk4_solution

__all__ = [
    "validate_parameters",
    "angular_frequency",
    "period",
    "analytical_solution",
    "rk4_solution",
]