"""Report generation: JSON for machines, Markdown for humans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import Check, ScenarioResult


def result_to_dict(result: ScenarioResult, checks: list[Check] | None = None) -> dict[str, Any]:
    return {
        "scenario": result.name,
        "target_gt_id": result.target_gt_id,
        "metrics": result.metrics,
        "timeline": result.timeline(),
        "checks": [c.as_dict() for c in checks] if checks else [],
        "passed": all(c.ok for c in checks) if checks else None,
    }


def write_json(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def markdown_table(results: list[ScenarioResult]) -> str:
    header = (
        "| Scenario | Confirmed* | Temporary* | Unknown* | Conf. precision | "
        "Wrong-person frames | ID switches | Recovery (s) | Centring active |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for res in results:
        m = res.metrics
        frac = m.get("state_fraction_post_lock", m.get("state_fraction", {}))
        lines.append(
            "| {name} | {c:.0%} | {t:.0%} | {u:.0%} | {prec} | {wrong} | {sw} | {rec} | {cent:.0%} |".format(
                name=res.name,
                c=frac.get("confirmed", 0.0),
                t=frac.get("temporary", 0.0),
                u=frac.get("unknown", 0.0),
                prec=_fmt(m.get("confirmed_precision")),
                wrong=_fmt(m.get("wrong_person_frames")),
                sw=_fmt(m.get("identity_switches")),
                rec=_fmt(m.get("recovery_mean_s")),
                cent=m.get("centring_active_rate") or 0.0,
            )
        )
    return "\n".join(lines)


def markdown_report(results: list[ScenarioResult], checks: dict[str, list[Check]]) -> str:
    lines = ["# Identity continuity evaluation", ""]
    lines.append(markdown_table(results))
    lines.append("")
    for res in results:
        lines.append(f"## {res.name}")
        lines.append("")
        lines.append(f"Duration {res.metrics.get('duration_s')}s, "
                     f"{res.metrics.get('frames')} frames, "
                     f"faces visible in {res.metrics.get('face_visible_frames')} frames.")
        lines.append("")
        lines.append("| Window | State | Mean confidence |")
        lines.append("|---|---|---|")
        for seg in res.timeline():
            lines.append(
                f"| {seg['start']:.2f}-{seg['end']:.2f}s | {seg['state']} | {seg['confidence']:.2f} |"
            )
        lines.append("")
        for check in checks.get(res.name, []):
            mark = "PASS" if check.ok else "FAIL"
            lines.append(f"- **{mark}** {check.name}: {check.detail}")
        lines.append("")
    return "\n".join(lines)


def summary_dict(results: list[ScenarioResult], checks: dict[str, list[Check]]) -> dict[str, Any]:
    return {
        "scenarios": [
            result_to_dict(res, checks.get(res.name)) for res in results
        ],
        "overall_passed": all(
            all(c.ok for c in checks.get(res.name, [])) for res in results
        ),
    }


def results_from_dict(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a report into simple rows (used by the README generator)."""
    rows = []
    for scenario in data.get("scenarios", []):
        metrics = scenario.get("metrics", {})
        rows.append(
            {
                "name": scenario.get("scenario"),
                **asdict_simple(metrics),
                "passed": scenario.get("passed"),
            }
        )
    return rows


def asdict_simple(mapping: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if not isinstance(v, (dict, list))}
