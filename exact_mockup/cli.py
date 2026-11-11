"""exact-mockup command line."""
import argparse
import sys

import cv2

from .cover import paste_cover
from .pages import paste_pages


def _read(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read image: {path}")
    return img


def _write(path, img):
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.lower().endswith((".jpg", ".jpeg")) else []
    if not cv2.imwrite(path, img, params):
        sys.exit(f"cannot write image: {path}")
    print(f"saved {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="exact-mockup",
                                description="Put real pages and covers into AI-generated mockups, with exact text.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("pages", help="paste page images onto green sheets")
    pg.add_argument("photo", help="mockup with green sheets")
    pg.add_argument("pages", nargs="+", help="page images, assigned to the sheets left to right")
    pg.add_argument("-o", "--out", required=True)
    pg.add_argument("--key", choices=["green", "magenta"], default="green")
    pg.add_argument("--fit", choices=["pad", "stretch"], default="pad",
                    help="pad: add paper margin when the sheet ratio differs (default); stretch: fill the sheet")
    pg.add_argument("--gutter", choices=["auto", "left", "right", "none"], default="auto",
                    help="shadow toward the book's spine")
    pg.add_argument("--no-light", action="store_true", help="do not carry the sheet's shading onto the page")
    pg.add_argument("--upscale", type=float, default=1.0, help="work (and save) at a larger size for sharper text")

    cv = sub.add_parser("cover", help="paste a cover on a green front face (and a spine on a magenta one)")
    cv.add_argument("photo", help="mockup with a green cover and optional magenta spine")
    cv.add_argument("cover", help="flat cover image")
    cv.add_argument("-o", "--out", required=True)
    cv.add_argument("--spine", help="flat spine image, top of the book at the top (default: cover's edge color)")
    cv.add_argument("--no-light", action="store_true")
    cv.add_argument("--grain", type=float, default=1.5, help="noise added over the cover, 0 to disable")
    cv.add_argument("--upscale", type=float, default=1.0)

    a = p.parse_args(argv)
    photo = _read(a.photo)
    if a.cmd == "pages":
        gutter = None if a.gutter == "none" else a.gutter
        out = paste_pages(photo, [_read(x) for x in a.pages], key=a.key, fit=a.fit, gutter=gutter,
                          light=not a.no_light, upscale=a.upscale)
    else:
        out = paste_cover(photo, _read(a.cover), spine=_read(a.spine) if a.spine else None,
                          light=not a.no_light, grain=a.grain, upscale=a.upscale)
    _write(a.out, out)


if __name__ == "__main__":
    main()
