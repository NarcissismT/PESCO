"""PESCO experiment reporting and visualization.

The package deliberately keeps the data contract small and dependency-light.  A
runner can write either a JSON list, a JSON object containing ``runs`` (or
``records``/``results``), or one record per line in JSONL format.  The command
line entry point then produces machine-readable summaries and publication-ready
PNG/SVG figures.
"""

from .metrics import aggregate_metrics, load_records
from .adapters import trajectory_to_record, trajectories_to_records, write_trajectory_records

__all__ = [
    "aggregate_metrics",
    "load_records",
    "trajectory_to_record",
    "trajectories_to_records",
    "write_trajectory_records",
]
