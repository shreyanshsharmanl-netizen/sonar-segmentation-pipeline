"""Demo service: segment simulated or uploaded sonar images."""

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from . import __version__
from .metrics import confusion, iou_per_class
from .model import UNet, predict
from .simulate import CLASSES, NUM_CLASSES, Config, generate

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 5 * 1024 * 1024
PALETTE = np.array(
    [[8, 14, 28], [214, 184, 120], [120, 124, 134], [86, 120, 92], [232, 62, 52], [52, 92, 190]],
    np.uint8,
)


def png(array: np.ndarray, scale: int = 4) -> str:
    image = Image.fromarray(array).resize(
        (array.shape[1] * scale, array.shape[0] * scale), Image.NEAREST
    )
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def gray(image: np.ndarray) -> np.ndarray:
    return (np.sqrt(image) * 255).astype(np.uint8)  # sqrt keeps dark detail visible


@asynccontextmanager
async def lifespan(app: FastAPI):
    weights = Path(os.getenv("MODEL_PATH", ROOT / "models/sonar_unet.pt"))
    app.state.model, app.state.version = None, None
    if weights.exists():
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        app.state.model = UNet(checkpoint["base"])
        app.state.model.load_state_dict(checkpoint["state"])
        app.state.version = checkpoint["version"]
    yield


app = FastAPI(title="Sonar segmentation", version=__version__, lifespan=lifespan)


def require_model(request: Request) -> UNet:
    if request.app.state.model is None:
        raise HTTPException(503, "model not loaded")
    return request.app.state.model


def coverage(pred: np.ndarray) -> dict[str, float]:
    counts = np.bincount(pred.ravel(), minlength=NUM_CLASSES)
    return {n: round(float(c / pred.size * 100), 2) for n, c in zip(CLASSES, counts, strict=True)}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "static/index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(request: Request) -> dict:
    require_model(request)
    return {"status": "ready", "model_version": request.app.state.version}


@app.get("/v1/simulate")
def simulate(
    request: Request,
    seed: int = Query(0, ge=0, le=2**31),
    speckle_looks: float = Query(4.0, ge=1.0, le=16.0),
) -> dict:
    model = require_model(request)
    image, truth = generate(seed, Config(speckle_looks=speckle_looks))
    pred = predict(model, image)
    iou = iou_per_class(confusion(pred, truth, NUM_CLASSES))
    return {
        "seed": seed,
        "miou": round(float(np.nanmean(iou)), 4),
        "coverage_percent": coverage(pred),
        "input_png": png(gray(image)),
        "prediction_png": png(PALETTE[pred]),
        "truth_png": png(PALETTE[truth]),
    }


@app.post("/v1/segment")
async def segment(request: Request, file: UploadFile = File(...)) -> dict:
    model = require_model(request)
    payload = await file.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise HTTPException(413, "image is larger than 5 MB")
    try:
        image = Image.open(io.BytesIO(payload)).convert("L")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "file is not a readable image") from None
    array = np.array(image.resize((128, 128)), np.float32) / 255
    pred = predict(model, array)
    return {
        "warning": "Model was trained on simulated data only.",
        "coverage_percent": coverage(pred),
        "prediction_png": png(PALETTE[pred]),
    }


@app.get("/v1/classes")
def classes() -> list[dict]:
    return [
        {"name": n, "color": "#" + bytes(c.tolist()).hex()}
        for n, c in zip(CLASSES, PALETTE, strict=True)
    ]
