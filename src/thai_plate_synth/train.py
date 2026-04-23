"""Train a YOLOv8 character detector on a rendered synth dataset.

Thin wrapper over Ultralytics `YOLO.train()`. Defaults are tuned for the
synth dataset (clean plates, small characters, 54 classes); override via CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(
    data: Path,
    *,
    weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 480,
    batch: int = 32,
    project: str = "runs/detect",
    name: str = "synth_v1",
    device: str | int | None = None,
    seed: int = 42,
) -> None:
    from ultralytics import YOLO

    yaml = data / "dataset.yaml"
    if not yaml.is_file():
        raise FileNotFoundError(f"{yaml} not found — run `make synth` first.")

    model = YOLO(weights)
    model.train(
        data=str(yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        seed=seed,
        deterministic=True,
        exist_ok=True,
        verbose=True,
    )
    best = Path(project) / name / "weights" / "best.pt"
    print(f"\nbest weights: {best.resolve() if best.exists() else 'not produced'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLOv8 on synth plates.")
    ap.add_argument("--data", type=Path, default=Path("data/synth_v1"))
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="synth_v1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(
        args.data,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
