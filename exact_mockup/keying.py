"""Chroma-key helpers: soft alpha, region detection, lighting and spill removal."""
import cv2
import numpy as np

KEYS = ("green", "magenta")


def _channels(img):
    return [img[..., i].astype(np.float32) for i in range(3)]  # B, G, R


def key_alpha(img, key="green", low=14.0, span=None):
    """Soft 0..1 alpha of how strongly each pixel is the key color.

    AI renders rarely give a pure key color (dim or yellowish greens are common), so by
    default the ramp adapts to how saturated the keyed area actually is."""
    b, g, r = _channels(img)
    if key == "green":
        diff = g - np.maximum(b, r)
    elif key == "magenta":
        diff = np.minimum(b, r) - g
    else:
        raise ValueError(f"unknown key {key!r}, use one of {KEYS}")
    if span is None:
        keyed = diff[diff > low]
        strong = np.percentile(keyed, 90) if keyed.size > diff.size * 0.002 else low + 36
        span = float(np.clip((strong - low) * 0.6, 8, 36))
    return np.clip((diff - low) / span, 0, 1).astype(np.float32)


def key_channel(img, key):
    """The channel that carries the scene lighting inside a keyed area."""
    b, g, r = _channels(img)
    return g if key == "green" else (b + r) / 2


def regions(alpha, min_frac=0.01, threshold=0.6):
    """Connected keyed areas larger than min_frac of the image.

    Returns (labels, list of (label_id, area, x)), largest first."""
    h, w = alpha.shape
    k = max(3, int(round(min(h, w) / 120)) | 1)
    m = (alpha > threshold).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    found = [(i, st[i, 4], st[i, 0]) for i in range(1, n) if st[i, 4] > h * w * min_frac]
    return lab, sorted(found, key=lambda t: -t[1])


def lighting(img, mask, key, floor=0.85, sigma=None):
    """Shading map taken from the keyed surface itself (folds, shadows, falloff)."""
    h, w = mask.shape
    sigma = sigma or max(4.0, min(h, w) / 140)
    inside = (mask > 0.9).astype(np.float32)
    if inside.sum() < 50:
        return np.ones_like(mask)
    ch = key_channel(img, key) * inside
    num = cv2.GaussianBlur(ch, (0, 0), sigma)
    den = np.maximum(cv2.GaussianBlur(inside, (0, 0), sigma), 0.02)
    lum = num / den
    ref = np.percentile(lum[inside > 0], 98)
    return np.clip(lum / max(ref, 1e-3), floor, 1.0).astype(np.float32)


def despill(img, alpha, key, band_px=None):
    """Remove the key-color fringe, only in a thin band around the keyed area."""
    h, w = alpha.shape
    band_px = band_px or max(5, int(min(h, w) / 150))
    band = cv2.dilate((alpha > 0.02).astype(np.uint8), np.ones((band_px, band_px), np.uint8)) > 0
    out = img.astype(np.float32).copy()
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    if key == "green":
        g[band] = np.minimum(g[band], (b[band] + r[band]) * 0.55 + 4)
    else:
        excess = np.clip(np.minimum(b, r) - g - 4, 0, None) * 0.85
        b[band] -= excess[band]
        r[band] -= excess[band]
    return out
