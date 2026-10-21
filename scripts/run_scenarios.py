#!/usr/bin/env python
"""Evaluate identity continuity on hardware-free scripted scenes.

Runs the full pipeline (tracking -> face association -> identity -> centring)
over simulated detector streams with exact ground truth and checks that the
three-state model behaves as specified.

Examples
--------
    python scripts/run_scenarios.py
    python scripts/run_scenarios.py --scenarios head_turn occlusion
    python scripts/run_scenarios.py --sweep-decay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bodyfollowing.config import PipelineConfig  # noqa: E402
from bodyfollowing.evaluation.metrics import (  # noqa: E402
    EXPECTATIONS,
    check_expectation,
    evaluate_scene,
)
from bodyfollowing.evaluation.report import (  # noqa: E402
    markdown_report,
    markdown_table,
    summary_dict,
    write_json,
)
from bodyfollowing.sources.simulated import SCENARIOS, build_scenario  # noqa: E402


def run_all(names: list[str], config: PipelineConfig, seed: int | None = None):
    results, checks = [], {}
    for name in names:
        scene = build_scenario(name, seed=seed)
        result = evaluate_scene(scene, config=config)
        expectation = EXPECTATIONS.get(name)
        checks[name] = check_expectation(result, expectation) if expectation else []
        results.append(result)
        status = "PASS" if all(c.ok for c in checks[name]) else "FAIL"
        print(f"[{status}] {name:<14} "
              f"confirmed={result.metrics['state_fraction']['confirmed']:.0%} "
              f"temp={result.metrics['state_fraction']['temporary']:.0%} "
              f"unknown={result.metrics['state_fraction']['unknown']:.0%} "
              f"precision={result.metrics['confirmed_precision']} "
              f"switches={result.metrics['identity_switches']} "
              f"recovery={result.metrics['recovery_mean_s']}s")
    return results, checks


def render_figures(results, out_dir: Path) -> list[Path]:
    """Save one contact sheet per scenario (3 moments each)."""
    try:
        import cv2
    except Exception:
        print("[warn] OpenCV unavailable, skipping figures")
        return []

    from bodyfollowing.evaluation.visualize import contact_sheet, render_sim_frame

    figures: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        scene = build_scenario(result.name)
        frames = list(scene.frames())
        if not frames:
            continue
        picks = [int(len(frames) * f) for f in (0.15, 0.35, 0.55, 0.75)]
        images = []
        from bodyfollowing.pipeline import BodyFollowingPipeline

        cfg = PipelineConfig()
        cfg.auto_lock = False
        pipeline = BodyFollowingPipeline(cfg)
        wanted = set(picks)
        for sim in frames:
            frame_result = pipeline.step(
                sim.frame_array(), now=sim.timestamp, detections=sim.detections
            )
            if sim.index in wanted:
                images.append(render_sim_frame(scene, sim, frame_result))
        sheet = contact_sheet(images, cols=2, scale=0.55)
        path = out_dir / f"{result.name}.png"
        cv2.imwrite(str(path), sheet)
        figures.append(path)
    return figures


def robustness(names: list[str], seeds: int, out_dir: Path) -> dict:
    """Re-run every scenario under several detector-noise seeds."""
    summary = {}
    for name in names:
        passed = 0
        for seed in range(seeds):
            scene = build_scenario(name, seed=seed)
            result = evaluate_scene(scene)
            checks = check_expectation(result, EXPECTATIONS[name])
            passed += all(c.ok for c in checks)
        summary[name] = {"seeds": seeds, "passed": passed}
        print(f"  {name:<14} {passed}/{seeds} seeds passed")
    total = sum(v["passed"] for v in summary.values())
    runs = sum(v["seeds"] for v in summary.values())
    summary["_total"] = {"seeds": runs, "passed": total}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"robustness": summary}, out_dir / "robustness.json")
    print(f"  overall {total}/{runs} runs passed")
    return summary


def sweep_decay(out_dir: Path) -> dict:
    """Measure how long TEMPORARY survives for different decay constants.

    Compares the *measured* time from face loss to the UNKNOWN transition with
    the closed-form prediction ``tau * ln(conf0 / unknown_threshold)``.
    """
    import math

    from bodyfollowing.sources.simulated import ActorSpec, KinematicScene

    rows = []
    for tau in (0.5, 1.0, 2.0, 4.0, 8.0):
        cfg = PipelineConfig()
        cfg.auto_lock = False
        cfg.identity.decay_tau = tau
        # Give the probe enough time for the slow decay constants to finish.
        probe = tau * math.log(1.0 / cfg.identity.unknown_threshold)
        scene = KinematicScene(
            name="decay_probe",
            actors=[
                ActorSpec(
                    gt_id=1,
                    x0=40.0,
                    y0=300.0,
                    vx=20.0,
                    body_h=250.0,
                    face_hidden=((2.0, 2.0 + probe + 4.0),),
                )
            ],
            duration=2.0 + probe + 6.0,
        )
        result = evaluate_scene(scene, config=cfg)
        # First moment after the face disappears at t=2.0 s.
        first_unknown = next(
            (r.timestamp for r in result.records if r.timestamp > 2.0 and r.state == "unknown"),
            None,
        )
        conf_at_loss = next(
            (r.confidence for r in result.records if abs(r.timestamp - 2.0) < 1e-6),
            1.0,
        )
        predicted = tau * math.log(conf_at_loss / cfg.identity.unknown_threshold)
        rows.append(
            {
                "tau_s": tau,
                "half_life_s": round(tau * math.log(2), 2),
                "confidence_at_face_loss": round(conf_at_loss, 3),
                "first_unknown_s": round(first_unknown - 2.0, 2) if first_unknown else None,
                "predicted_s": round(predicted, 2),
                "error_s": (
                    round(abs((first_unknown - 2.0) - predicted), 2)
                    if first_unknown
                    else None
                ),
            }
        )
        print(
            f"tau={tau:>4}s  half-life={rows[-1]['half_life_s']:>5}s  "
            f"measured={rows[-1]['first_unknown_s']}s  "
            f"predicted={rows[-1]['predicted_s']}s"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"decay_sweep": rows}, out_dir / "decay_sweep.json")
    return {"decay_sweep": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--out", default="reports")
    parser.add_argument("--figures", action="store_true", help="render contact sheets")
    parser.add_argument("--sweep-decay", action="store_true", help="tau sweep")
    parser.add_argument("--seeds", type=int, default=0,
                        help="also re-run scenarios under N noise seeds")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig()
    results, checks = run_all(args.scenarios, config)

    text = markdown_table(results)
    print("\n" + text + "\n")
    for name, items in checks.items():
        for check in items:
            if not check.ok:
                print(f"  FAILED {name}: {check.name} -- {check.detail}")

    report = summary_dict(results, checks)
    write_json(report, out_dir / "scenarios.json")

    md = markdown_report(results, checks)
    (out_dir / "scenarios.md").write_text(md, encoding="utf-8")
    print(f"report -> {out_dir / 'scenarios.md'}")

    if args.figures:
        for path in render_figures(results, out_dir / "figures"):
            print(f"figure -> {path}")

    if args.sweep_decay:
        sweep_decay(out_dir)

    if args.seeds > 1:
        print(f"\nrobustness across {args.seeds} noise seeds:")
        robustness(args.scenarios, args.seeds, out_dir)

    if args.json:
        print(json.dumps(report, indent=2))

    return 0 if report.get("overall_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
