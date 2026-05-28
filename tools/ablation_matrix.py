from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    description: str
    config: Path
    output_dir: Path


def build_train_command(spec: ExperimentSpec, device: str = "cuda") -> str:
    return (
        f"python train.py --config {spec.config.as_posix()} "
        f"--device {device} --output-dir {spec.output_dir.as_posix()}"
    )


def build_eval_command(spec: ExperimentSpec, split: str = "test", device: str = "cuda") -> str:
    checkpoint = spec.output_dir / "checkpoints" / "csrnet_best.pth"
    prediction_dir = spec.output_dir / "predictions"
    return (
        f"python eval.py --config {spec.config.as_posix()} "
        f"--checkpoint {checkpoint.as_posix()} --split {split} --device {device} "
        f"--output-dir {prediction_dir.as_posix()}"
    )


def load_experiments(plan_path: Path) -> list[ExperimentSpec]:
    with plan_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)

    experiments: list[ExperimentSpec] = []
    for item in data.get("experiments", []):
        experiments.append(
            ExperimentSpec(
                experiment_id=str(item["id"]),
                description=str(item["description"]),
                config=Path(item["config"]),
                output_dir=Path(item["output_dir"]),
            )
        )
    if not experiments:
        raise ValueError(f"No experiments found in {plan_path}")
    return experiments


def render_commands(experiments: list[ExperimentSpec], device: str, split: str) -> str:
    lines: list[str] = []
    for spec in experiments:
        lines.append(f"# {spec.experiment_id}: {spec.description}")
        lines.append(build_train_command(spec, device=device))
        lines.append(build_eval_command(spec, split=split, device=device))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render reproducible Stage 8 ablation commands.")
    parser.add_argument("--plan", type=Path, default=Path("configs/ablations/ablation_plan.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = render_commands(load_experiments(args.plan), device=args.device, split=args.split)
    if args.output is None:
        print(commands, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(commands, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
