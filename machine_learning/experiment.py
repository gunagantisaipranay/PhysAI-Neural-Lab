
"""Shared experiment metadata and metric records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import json
from typing import Any, Mapping


@dataclass
class ExperimentRecord:
    """Reproducibility metadata and scalar metrics for one experiment."""

    name: str
    algorithm: str
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Experiment name must be a non-empty string.")
        if not isinstance(self.algorithm, str) or not self.algorithm.strip():
            raise ValueError("Algorithm must be a non-empty string.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a non-negative integer.")
        if self.schema_version != 1:
            raise ValueError("Unsupported experiment schema version.")

        # Copy caller-owned mappings so later external mutations do not
        # unexpectedly alter this record's top-level dictionaries.
        self.parameters = dict(self.parameters)
        self.dataset = dict(self.dataset)

        initial_metrics = dict(self.metrics)
        self.metrics = {}
        for key, value in initial_metrics.items():
            self.set_metric(key, value)

    def set_metric(self, name: str, value: float) -> None:
        """Store one finite numeric metric under a non-empty name."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric name must be a non-empty string.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Metric value must be numeric.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("Metric value must be finite.")
        self.metrics[name] = numeric_value

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        """Reconstruct and validate a record from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError("Experiment data must be a dictionary.")

        expected_keys = {
            "name",
            "algorithm",
            "seed",
            "parameters",
            "dataset",
            "metrics",
            "schema_version",
        }
        if set(data) != expected_keys:
            raise ValueError("Experiment data has missing or unexpected fields.")

        return cls(**data)