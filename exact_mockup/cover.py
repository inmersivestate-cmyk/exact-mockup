"""Put a real cover (green face) and spine (magenta face) on a generated book."""
import cv2
import numpy as np

from .geometry import quad_from_mask, warp_to_quad
from .keying import despill, key_alpha, lighting, regions


def _face(photo, key):
    alpha = key_alpha(photo, key)
    lab, found = regions(alpha, min_frac=0.002)
    if not found:
        return alpha, None
    mask = (lab == found[0][0]).astype(np.uint8)
    return alpha * cv2.dilate(mask, np.ones((15, 15), np.uint8)), quad_from_mask(mask)


def _spine_color(cover):
    h, w = cover.shape[:2]
    strip = cover[h // 10:h - h // 10, :max(2, w // 40)]
    return np.median(strip.reshape(-1, 3), axis=0).astype(np.uint8)


def paste_cover(photo, cover, spine=None, light=True, grain=1.5, upscale=1.0):
    """photo: BGR render with a green front cover and an optional magenta spine."""
    if upscale != 1:
        photo = cv2.resize(photo, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    out = photo.astype(np.float32)
    total = np.zeros(photo.shape[:2], np.float32)
    faces = [("green", cover)]
    m_alpha, m_quad = _face(photo, "magenta")
    if m_quad is not None:
        if spine is None:
            q = m_quad
            sw = int(np.linalg.norm(q[1] - q[0])) or 1
            sh = int(np.linalg.norm(q[3] - q[0])) or 1
            spine = np.full((max(sh, 2), max(sw, 2), 3), _spine_color(cover), np.uint8)
        faces.append(("magenta", spine))
    elif spine is not None:
        print("warning: spine image given but no magenta area found")

    for key, img in faces:
        alpha, quad = (_face(photo, key) if key == "green" else (m_alpha, m_quad))
        if quad is None:
            raise SystemExit("no green cover area found in the photo")
        warped = warp_to_quad(img, quad, photo.shape).astype(np.float32)
        if light:
            warped *= lighting(photo, alpha, key, floor=0.8)[..., None]
        a = cv2.GaussianBlur(alpha, (3, 3), 0)[..., None]
        out = despill(out, alpha, key)
        out = out * (1 - a) + warped * a
        total = np.maximum(total, alpha)

    if grain:  # match the photo's noise so the cover does not look pasted
        rng = np.random.default_rng(7)
        noise = cv2.GaussianBlur(rng.normal(0, grain, out.shape).astype(np.float32), (0, 0), 0.7)
        out += noise * total[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
