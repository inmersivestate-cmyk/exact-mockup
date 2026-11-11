"""Builds synthetic demo inputs (no AI needed) and runs both commands on them.

usage: python examples/make_demo.py   (writes into examples/demo/)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exact_mockup import paste_cover, paste_pages  # noqa: E402
from exact_mockup.geometry import coons  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "demo")
rng = np.random.default_rng(3)


def bez(a, c, b, n=200):
    t = np.linspace(0, 1, n)[:, None]
    a, c, b = (np.array(x, float) for x in (a, c, b))
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b


def table(w, h):
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (70, 110, 150)  # warm wood, BGR
    grain = cv2.GaussianBlur(rng.normal(0, 18, (h, w)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=1.5)
    base += grain[..., None]
    yy, xx = np.mgrid[0:h, 0:w]
    vignette = 1 - 0.35 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 2
    return base * vignette[..., None]


def fill_patch(img, edges, color, gutter_x):
    top, right, bottom, left = edges
    outline = np.vstack([top, right, bottom[::-1], left[::-1]]).astype(np.int32)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [outline], 1, cv2.LINE_AA)
    xx = np.arange(img.shape[1], dtype=np.float32)
    shade = 1 - 0.28 * np.exp(-np.abs(xx - gutter_x) / 60)  # darker toward the gutter
    flat = np.array(color, np.float32)[None, None, :] * shade[None, :, None]
    img[:] = np.where(m[..., None] > 0, flat, img)


def page(title, w=850, h=1100):
    p = np.full((h, w, 3), 250, np.uint8)
    cv2.putText(p, title, (70, 150), cv2.FONT_HERSHEY_DUPLEX, 2.0, (40, 40, 40), 3, cv2.LINE_AA)
    cv2.line(p, (70, 185), (w - 70, 185), (60, 120, 200), 4)
    words = "Every word on this page is the real file, not an AI guess.".split()
    y = 260
    for i in range(18):
        line = " ".join(words[(i * 3) % len(words):] + words[:(i * 3) % len(words)])[:44]
        cv2.putText(p, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (55, 55, 55), 2, cv2.LINE_AA)
        y += 44
    cv2.putText(p, f"p. {title[-1]}", (w // 2 - 30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2, cv2.LINE_AA)
    return p


def open_book():
    w, h = 1600, 1200
    img = table(w, h)
    green = (40, 200, 50)
    left = (bez((180, 270), (480, 200), (800, 300)), bez((800, 300), (790, 620), (800, 960)),
            bez((170, 990), (480, 920), (800, 960)), bez((180, 270), (175, 630), (170, 990)))
    right = (bez((800, 300), (1120, 200), (1420, 270)), bez((1420, 270), (1425, 630), (1430, 990)),
             bez((800, 960), (1120, 920), (1430, 990)), bez((800, 300), (790, 620), (800, 960)))
    fill_patch(img, left, green, 800)
    fill_patch(img, right, green, 800)
    cv2.line(img, (800, 300), (800, 960), (20, 70, 25), 3, cv2.LINE_AA)
    # a thumb holding the page
    cv2.ellipse(img, (1405, 820), (70, 120), -20, 0, 360, (120, 150, 205), -1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def book_on_table():
    w, h = 1400, 1100
    img = table(w, h)
    front = np.array([[560, 180], [1080, 230], [1050, 930], [530, 900]], np.int32)
    spine = np.array([[470, 210], [560, 180], [530, 900], [440, 920]], np.int32)
    shadow = np.array([[455, 925], [1060, 940], [1120, 980], [480, 975]], np.int32)
    cv2.fillPoly(img, [shadow], (35, 55, 75), cv2.LINE_AA)
    cv2.fillPoly(img, [front], (40, 205, 45), cv2.LINE_AA)
    cv2.fillPoly(img, [spine], (200, 40, 205), cv2.LINE_AA)
    # soft light falloff across the cover
    yy, xx = np.mgrid[0:h, 0:w]
    fall = np.clip(1.0 - (xx - 560) / 520 * 0.18 - (yy - 180) / 750 * 0.08, 0.7, 1)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [front, spine], 1)
    img = np.where(m[..., None] > 0, img * fall[..., None], img)
    return np.clip(img, 0, 255).astype(np.uint8)


def cover_art(w=1000, h=1400):
    c = np.zeros((h, w, 3), np.uint8)
    c[:] = (90, 45, 30)
    cv2.circle(c, (w // 2, h // 2 + 80), 300, (60, 160, 235), -1, cv2.LINE_AA)
    cv2.putText(c, "THE EXACT", (95, 260), cv2.FONT_HERSHEY_DUPLEX, 3.3, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(c, "MOCKUP", (215, 400), cv2.FONT_HERSHEY_DUPLEX, 3.3, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(c, "every letter stays sharp", (170, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (235, 225, 210), 3, cv2.LINE_AA)
    return c


def spine_art(w=180, h=1400):
    s = np.zeros((w, h, 3), np.uint8)  # drawn sideways, then rotated
    s[:] = (90, 45, 30)
    cv2.putText(s, "THE EXACT MOCKUP", (180, 115), cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
    return cv2.rotate(s, cv2.ROTATE_90_CLOCKWISE)


def side_by_side(a, b):
    h = max(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    gap = np.full((h, 24, 3), 255, np.uint8)
    return np.hstack([a, gap, b])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    book = open_book()
    p1, p2 = page("Chapter 1"), page("Chapter 2")
    for name, im in [("pages_input.jpg", book), ("page_1.png", p1), ("page_2.png", p2)]:
        cv2.imwrite(os.path.join(OUT, name), im)
    res = paste_pages(book, [p1, p2])
    cv2.imwrite(os.path.join(OUT, "pages_output.jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(OUT, "pages_before_after.jpg"), side_by_side(book, res), [cv2.IMWRITE_JPEG_QUALITY, 90])

    scene, art, sp = book_on_table(), cover_art(), spine_art()
    for name, im in [("cover_input.jpg", scene), ("cover.png", art), ("spine.png", sp)]:
        cv2.imwrite(os.path.join(OUT, name), im)
    res = paste_cover(scene, art, spine=sp)
    cv2.imwrite(os.path.join(OUT, "cover_output.jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(OUT, "cover_before_after.jpg"), side_by_side(scene, res), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("demo written to", OUT)
