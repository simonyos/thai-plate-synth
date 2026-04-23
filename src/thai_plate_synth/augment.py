"""Plate-crop augmentations to close the synth→real gap.

Stage-2 input is a cropped plate from stage-1 detection. What makes real
crops hard: camera angle (imperfect plate pose → perspective), sensor
quality (noise, blur), lighting (brightness/contrast), and the fact that
stage-1 output is jpeg-re-encoded somewhere in the pipeline. Each of those
gets an augmentation here.

No rotation-flipping — Thai consonants are orientation-sensitive (ฆ / ม
look similar mirrored). No cutout / masking yet — real plates aren't
typically partially occluded on the registration line.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from thai_plate_synth.render import CharAnn


@dataclass(frozen=True)
class AugConfig:
    p_perspective: float = 0.75
    max_corner_shift: float = 0.12     # fraction of w/h each corner may move

    p_photometric: float = 0.8
    brightness_range: tuple[float, float] = (0.65, 1.25)
    contrast_range: tuple[float, float] = (0.75, 1.35)

    p_blur: float = 0.35
    max_blur_radius: float = 1.4

    p_noise: float = 0.5
    max_noise_sigma: float = 10.0

    p_jpeg: float = 0.6
    jpeg_quality_range: tuple[int, int] = (45, 92)


def _compute_coeffs(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list[float]:
    """Solve the 8 homography coefficients that map (dst → src) in PIL's convention.

    src[i] = H . dst[i] / (g*dst[i].x + h*dst[i].y + 1).
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (sx, sy), (dx, dy) in zip(src, dst, strict=True):
        rows.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        rows.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        rhs.extend((sx, sy))
    return np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64)).tolist()


def _apply_homography(pt: tuple[float, float], coeffs: list[float]) -> tuple[float, float]:
    a, b, c, d, e, f, g, h = coeffs
    x, y = pt
    den = g * x + h * y + 1.0
    return ((a * x + b * y + c) / den, (d * x + e * y + f) / den)


def perspective_warp(
    img: Image.Image,
    anns: list[CharAnn],
    rng: random.Random,
    *,
    max_shift: float = 0.12,
    fill: str = "white",
) -> tuple[Image.Image, list[CharAnn]]:
    w, h = img.size
    mx = int(w * max_shift)
    my = int(h * max_shift)
    src = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    dst = [
        (rng.uniform(0, mx), rng.uniform(0, my)),
        (rng.uniform(w - mx, w), rng.uniform(0, my)),
        (rng.uniform(w - mx, w), rng.uniform(h - my, h)),
        (rng.uniform(0, mx), rng.uniform(h - my, h)),
    ]
    # PIL wants dst→src (inverse mapping for resampling). Bboxes want src→dst.
    coeffs_pil = _compute_coeffs(src, dst)
    coeffs_fwd = _compute_coeffs(dst, src)
    warped = img.transform((w, h), Image.PERSPECTIVE, coeffs_pil, Image.BILINEAR, fillcolor=fill)

    new_anns: list[CharAnn] = []
    for a in anns:
        corners = [(a.x1, a.y1), (a.x2, a.y1), (a.x2, a.y2), (a.x1, a.y2)]
        warped_corners = [_apply_homography(c, coeffs_fwd) for c in corners]
        xs = [p[0] for p in warped_corners]
        ys = [p[1] for p in warped_corners]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(float(w), max(xs))
        y2 = min(float(h), max(ys))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue  # bbox warped off-canvas
        new_anns.append(CharAnn(cls=a.cls, glyph=a.glyph, x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)))
    return warped, new_anns


def photometric_jitter(
    img: Image.Image,
    rng: random.Random,
    *,
    brightness_range: tuple[float, float] = (0.65, 1.25),
    contrast_range: tuple[float, float] = (0.75, 1.35),
) -> Image.Image:
    out = ImageEnhance.Brightness(img).enhance(rng.uniform(*brightness_range))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(*contrast_range))
    return out


def gaussian_blur(img: Image.Image, rng: random.Random, *, max_radius: float = 1.4) -> Image.Image:
    r = rng.uniform(0.2, max_radius)
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def gaussian_noise(img: Image.Image, rng: random.Random, *, max_sigma: float = 10.0) -> Image.Image:
    sigma = rng.uniform(2.0, max_sigma)
    arr = np.asarray(img, dtype=np.int16)
    # Use numpy's default_rng with seed drawn from the deterministic rng
    noise_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noise = noise_rng.normal(0.0, sigma, size=arr.shape).astype(np.int16)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def jpeg_artifact(img: Image.Image, rng: random.Random, *, q_range: tuple[int, int] = (45, 92)) -> Image.Image:
    q = rng.randint(*q_range)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


def apply(
    img: Image.Image,
    anns: list[CharAnn],
    rng: random.Random,
    cfg: AugConfig = AugConfig(),
) -> tuple[Image.Image, list[CharAnn]]:
    """Apply the augmentation pipeline with each step's probability drawn from cfg."""
    if rng.random() < cfg.p_perspective:
        img, anns = perspective_warp(img, anns, rng, max_shift=cfg.max_corner_shift)
    if rng.random() < cfg.p_photometric:
        img = photometric_jitter(img, rng, brightness_range=cfg.brightness_range, contrast_range=cfg.contrast_range)
    if rng.random() < cfg.p_blur:
        img = gaussian_blur(img, rng, max_radius=cfg.max_blur_radius)
    if rng.random() < cfg.p_noise:
        img = gaussian_noise(img, rng, max_sigma=cfg.max_noise_sigma)
    if rng.random() < cfg.p_jpeg:
        img = jpeg_artifact(img, rng, q_range=cfg.jpeg_quality_range)
    return img, anns
