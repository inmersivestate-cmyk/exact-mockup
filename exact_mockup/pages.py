"""Paste real pages onto green sheets (flat, curved, in any perspective)."""
import cv2
import numpy as np

from .geometry import curve_len, sheet_edges, warp_to_patch
from .keying import despill, key_alpha, lighting, regions


def _split_in_two(img, mask):
    """Cut a single keyed blob (an open book seen as one area) along its darkest seam."""
    H, W = mask.shape
    rows = np.where(mask.sum(1) > W * 0.05)[0]
    if len(rows) < 10:
        return mask
    y0, y1 = rows.min(), rows.max()
    g = cv2.GaussianBlur(img[..., 1].astype(np.float32), (9, 1), 0)
    pts = []
    for y in range(y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4, 6):
        xs = np.where(mask[y])[0]
        if len(xs) < 10:
            continue
        mid, half = (xs.min() + xs.max()) // 2, (xs.max() - xs.min()) // 6
        a, b = max(mid - half, 0), min(mid + half, W)
        pts.append((y, a + int(g[y, a:b].argmin())))
    if len(pts) < 3:
        return mask
    pts = np.array(pts)
    coef = np.polyfit(pts[:, 0], pts[:, 1], 1)
    cut = mask.copy()
    t = max(2, W // 300)
    for y in range(H):
        x = int(np.polyval(coef, y))
        cut[y, max(x - t, 0):x + t + 1] = 0
    return cut


def _fit_page(page, ratio, mode):
    """Match the page to the sheet's height/width ratio without squashing text."""
    ph, pw = page.shape[:2]
    if mode == "stretch" or abs(np.log(ratio / (ph / pw))) < np.log(1.06):
        return page
    color = np.median(page[int(ph * .9):int(ph * .93), pw // 4:pw // 2].reshape(-1, 3), axis=0).astype(np.uint8)
    if ratio > ph / pw:  # sheet taller than the page: paper margin top and bottom
        extra = int(pw * ratio) - ph
        pad = np.full((extra // 2, pw, 3), color, np.uint8)
        return np.vstack([pad, page, pad])
    extra = int(ph / ratio) - pw
    pad = np.full((ph, extra // 2, 3), color, np.uint8)
    return np.hstack([pad, page, pad])


def _gutter_shadow(page, side, strength=0.24, width=0.08):
    pw = page.shape[1]
    ramp = np.linspace(0, 1, pw, dtype=np.float32)
    if side == "right":
        ramp = ramp[::-1]
    shade = 1 - strength * np.clip(1 - ramp / width, 0, 1) ** 2
    return np.clip(page.astype(np.float32) * shade[None, :, None], 0, 255).astype(np.uint8)


def paste_pages(photo, pages, key="green", fit="pad", gutter="auto", light=True, upscale=1.0):
    """photo: BGR image with green sheets. pages: list of BGR images, assigned left to right."""
    if upscale != 1:
        photo = cv2.resize(photo, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    H, W = photo.shape[:2]
    alpha = key_alpha(photo, key)
    lab, found = regions(alpha)
    if len(found) < len(pages) and len(pages) == 2 and found:
        cut = _split_in_two(photo, (alpha > 0.6).astype(np.uint8))
        lab, found = regions(cut.astype(np.float32))
    if not found:
        raise SystemExit(f"no {key} area found in the photo")
    if len(found) < len(pages):
        print(f"warning: {len(pages)} pages given but only {len(found)} {key} areas found")
    chosen = sorted(found[:len(pages)], key=lambda t: t[2])  # left to right

    base = photo.astype(np.float32)
    comp = base.copy()
    for k, ((lid, _, _), page) in enumerate(zip(chosen, pages)):
        sheet = (lab == lid).astype(np.uint8)
        edges = sheet_edges(sheet)
        top, right, bottom, left = edges
        ratio = (curve_len(left) + curve_len(right)) / (curve_len(top) + curve_len(bottom))
        pg = _fit_page(page, ratio, fit)
        side = gutter
        if gutter == "auto":
            side = None if len(chosen) == 1 else ("right" if k == 0 else "left" if k == len(chosen) - 1 else None)
        if side in ("left", "right"):
            pg = _gutter_shadow(pg, side)
        tw = int((curve_len(top) + curve_len(bottom)) / 2 * 1.05)
        if tw < pg.shape[1]:  # shrink with area averaging first so text does not alias
            pg = cv2.resize(pg, (tw, int(tw * pg.shape[0] / pg.shape[1])), interpolation=cv2.INTER_AREA)
        warped, (x0, y0, x1, y1) = warp_to_patch(pg, edges, photo.shape)
        warped = warped.astype(np.float32)
        if light:
            sheet_alpha = alpha * sheet
            warped *= lighting(photo, sheet_alpha, key)[y0:y1, x0:x1, None]
        grow = max(9, int(min(H, W) / 25)) | 1
        reg = cv2.dilate(sheet, np.ones((grow, grow), np.uint8))[y0:y1, x0:x1] > 0
        comp[y0:y1, x0:x1][reg] = warped[reg]

    a = cv2.GaussianBlur(alpha, (5, 5), 0)[..., None]
    base = despill(base, alpha, key)
    out = base * (1 - a) + comp * a
    return np.clip(out, 0, 255).astype(np.uint8)
