"""Geometry: corners, robust edge curves, Coons patches and warps."""
import cv2
import numpy as np
from scipy.interpolate import griddata


def order_quad(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(pts.sum(1).argmin()), axis=0)


def largest_contour(mask):
    cs = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return max(cs, key=cv2.contourArea)


def quad_from_mask(mask):
    """Best 4-corner polygon around a keyed area (falls back to the min-area box)."""
    c = largest_contour(mask)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.1, 40):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order_quad(ap[:, 0, :])
    return order_quad(cv2.boxPoints(cv2.minAreaRect(c)))


def contour_corners(pts):
    """Extreme corners of a roughly upright sheet contour: TL, TR, BR, BL."""
    s = pts.sum(1)
    d = pts[:, 0] - pts[:, 1]
    return pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]


def contour_between(c, a, b):
    """Contour points between two corners, along the shorter way round."""
    i = np.linalg.norm(c - a, axis=1).argmin()
    j = np.linalg.norm(c - b, axis=1).argmin()
    n = len(c)
    p1 = c[np.arange(i, i + ((j - i) % n) + 1) % n]
    p2 = c[np.arange(j, j + ((i - j) % n) + 1) % n][::-1]
    return p1 if len(p1) <= len(p2) else p2


def fit_edge(pts, a, b, deg, n=200):
    """Smooth curve from a to b through pts, ignoring bites (fingers, clips)."""
    a = a.astype(float)
    b = b.astype(float)
    ax = (b - a) / np.linalg.norm(b - a)
    nr = np.array([-ax[1], ax[0]])
    t = (pts - a) @ ax
    o = (pts - a) @ nr
    keep = np.ones(len(t), bool)
    deg = min(deg, max(1, len(t) - 1))
    for _ in range(6):
        coef = np.polyfit(t[keep], o[keep], deg)
        err = o - np.polyval(coef, t)
        keep = np.abs(err) < max(1.5, np.median(np.abs(err[keep])) * 2.5)
    T = np.linspace(0, (b - a) @ ax, n)
    return a + np.outer(T, ax) + np.outer(np.polyval(coef, T), nr)


def sheet_edges(mask, curve_deg=3):
    """Four fitted edges (top, right, bottom, left) of a possibly curved sheet."""
    c = largest_contour(mask)[:, 0, :].astype(float)
    tl, tr, br, bl = contour_corners(c)
    top = fit_edge(contour_between(c, tl, tr), tl, tr, curve_deg)
    bottom = fit_edge(contour_between(c, bl, br), bl, br, curve_deg)
    left = fit_edge(contour_between(c, tl, bl), tl, bl, 1)
    right = fit_edge(contour_between(c, tr, br), tr, br, 1)
    return top, right, bottom, left


def curve_len(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def coons(top, right, bottom, left, n=120, m=170):
    """Coons patch: grid of image points P for (u, v) in the unit square."""
    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, m)
    U, V = np.meshgrid(u, v)

    def at(curve, t):
        x = np.linspace(0, 1, len(curve))
        return np.stack([np.interp(t, x, curve[:, 0]), np.interp(t, x, curve[:, 1])], -1)

    T, B, L, R = at(top, U), at(bottom, U), at(left, V), at(right, V)
    tl, tr, bl, br = top[0], top[-1], bottom[0], bottom[-1]
    Uu, Vv = U[..., None], V[..., None]
    P = ((1 - Vv) * T + Vv * B + (1 - Uu) * L + Uu * R
         - ((1 - Uu) * (1 - Vv) * tl + Uu * (1 - Vv) * tr + (1 - Uu) * Vv * bl + Uu * Vv * br))
    return U, V, P


def warp_to_patch(src, edges, shape, margin=90):
    """Warp src onto the curved patch. Returns (warped crop, (x0, y0, x1, y1))."""
    H, W = shape[:2]
    U, V, P = coons(*edges)
    pts = P.reshape(-1, 2)
    x0, y0 = np.floor(pts.min(0)).astype(int) - margin
    x1, y1 = np.ceil(pts.max(0)).astype(int) + margin
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    X, Y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    uv = []
    for vals in (U.ravel(), V.ravel()):
        lin = griddata(pts, vals, (X, Y), method="linear")
        near = griddata(pts, vals, (X, Y), method="nearest")
        uv.append(np.where(np.isnan(lin), near, lin))
    sh, sw = src.shape[:2]
    out = cv2.remap(src, (uv[0] * (sw - 1)).astype(np.float32), (uv[1] * (sh - 1)).astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, (x0, y0, x1, y1)


def warp_to_quad(src, quad, shape):
    """Perspective-correct warp of src onto a 4-corner quad, pre-shrunk so text stays crisp."""
    quad = np.asarray(quad, np.float32)
    tw = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])) * 1.1)
    th = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])) * 1.1)
    if tw < src.shape[1] and th < src.shape[0]:
        src = cv2.resize(src, (max(tw, 1), max(th, 1)), interpolation=cv2.INTER_AREA)
    h, w = src.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
    return cv2.warpPerspective(src, M, (shape[1], shape[0]), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
