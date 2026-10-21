"""Evaluation harness: ground-truth metrics, checks and reports."""
from .metrics import (
    EXPECTATIONS,
    Check,
    Expectation,
    FrameRecord,
    ScenarioResult,
    check_expectation,
    evaluate_scene,
    map_bodies_to_ground_truth,
)
from .report import markdown_report, markdown_table, summary_dict, write_json

__all__ = [
    "EXPECTATIONS",
    "Check",
    "Expectation",
    "FrameRecord",
    "ScenarioResult",
    "check_expectation",
    "evaluate_scene",
    "map_bodies_to_ground_truth",
    "markdown_report",
    "markdown_table",
    "summary_dict",
    "write_json",
]
