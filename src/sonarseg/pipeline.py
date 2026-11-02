"""Staged pipeline: train -> evaluate -> gate. Every run is logged to MLflow."""

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from . import __version__, baseline
from .metrics import confusion, iou_per_class
from .model import UNet, predict
from .simulate import CLASSES, NUM_CLASSES, Config, generate

VAL_SEEDS = range(100_000, 100_200)
TEST_SEEDS = range(200_000, 200_300)  # never used in training or model selection


class Simulated(Dataset):
    def __init__(self, seeds: range, config: Config):
        self.seeds, self.config = seeds, config

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int):
        image, label = generate(self.seeds[idx], self.config)
        return torch.from_numpy(image)[None], torch.from_numpy(label)


def score(model: UNet, seeds: range, config: Config) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for seed in seeds:
        image, label = generate(seed, config)
        matrix += confusion(predict(model, image), label, NUM_CLASSES)
    return matrix


def train(params: dict) -> tuple[UNet, float]:
    torch.manual_seed(params["seed"])
    config = Config()
    loader = DataLoader(
        Simulated(range(params["train_samples"]), config),
        batch_size=params["batch_size"], shuffle=True, num_workers=0,
    )
    model = UNet(params["base_channels"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, params["lr"], total_steps=params["epochs"] * len(loader)
    )
    # Objects and shadows are rare; weight them so they are not ignored.
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 4.0, 3.0])
    best, best_state = -1.0, None
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        losses = []
        for image, label in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(image), label, weight=weights)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        miou = float(np.nanmean(iou_per_class(score(model, VAL_SEEDS, config))))
        mlflow.log_metrics({"train_loss": float(np.mean(losses)), "val_miou": miou}, step=epoch)
        print(f"epoch {epoch} loss {np.mean(losses):.4f} val_miou {miou:.4f}", flush=True)
        if miou > best:
            best = miou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def evaluate(model: UNet, params: dict) -> dict:
    config = Config()
    fit_set = [generate(seed, config) for seed in range(300)]
    centres = baseline.fit([s[0] for s in fit_set], [s[1] for s in fit_set])
    base = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for seed in TEST_SEEDS:
        image, label = generate(seed, config)
        base += confusion(baseline.predict(image, centres), label, NUM_CLASSES)
    unet_iou = iou_per_class(score(model, TEST_SEEDS, config))
    base_iou = iou_per_class(base)

    robustness = {}
    for looks in (8.0, 4.0, 2.0, 1.0):
        noisy = Config(speckle_looks=looks)
        robustness[str(looks)] = round(
            float(np.nanmean(iou_per_class(score(model, TEST_SEEDS, noisy)))), 4
        )

    sample = generate(TEST_SEEDS[0], config)[0]
    started = time.perf_counter()
    for _ in range(30):
        predict(model, sample)
    return {
        "data": "synthetic (src/sonarseg/simulate.py)",
        "test_images": len(TEST_SEEDS),
        "unet_miou": round(float(np.nanmean(unet_iou)), 4),
        "baseline_miou": round(float(np.nanmean(base_iou)), 4),
        "per_class_iou": {
            name: {"unet": round(float(u), 4), "baseline": round(float(b), 4)}
            for name, u, b in zip(CLASSES, unet_iou, base_iou, strict=True)
        },
        "miou_by_speckle_looks": robustness,
        "cpu_ms_per_image": round((time.perf_counter() - started) / 30 * 1000, 1),
    }


def gate(report: dict, params: dict) -> None:
    """Fail the run if the model is not good enough to promote."""
    floor = params["min_test_miou"]
    margin = params["min_margin_over_baseline"]
    if report["unet_miou"] < floor:
        raise SystemExit(f"gate failed: mIoU {report['unet_miou']} is below {floor}")
    if report["unet_miou"] - report["baseline_miou"] < margin:
        raise SystemExit(f"gate failed: under {margin} better than baseline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()
    params = yaml.safe_load(args.config.read_text())

    mlflow.set_tracking_uri(params.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment("sonar-segmentation")
    with mlflow.start_run():
        mlflow.log_params(params)
        model, best = train(params)
        report = evaluate(model, params)
        Path("reports").mkdir(exist_ok=True)
        Path("reports/evaluation.json").write_text(json.dumps(report, indent=2))
        mlflow.log_metrics({"test_miou": report["unet_miou"], "best_val_miou": best})
        mlflow.log_artifact("reports/evaluation.json")
        gate(report, params)
        Path("models").mkdir(exist_ok=True)
        torch.save(
            {"state": model.state_dict(), "base": params["base_channels"], "version": __version__},
            "models/sonar_unet.pt",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
