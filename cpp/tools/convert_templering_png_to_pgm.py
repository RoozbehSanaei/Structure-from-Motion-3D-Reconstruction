#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
from PIL import Image

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: convert_templering_png_to_pgm.py <templering_root>")
        return 2
    root = Path(sys.argv[1]).expanduser().resolve()
    src_dir = root / "templeRing"
    out_dir = root / "templeRing_pgm"
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(src_dir.glob("templeR*.png"))
    if not pngs:
        print(f"No PNGs found in {src_dir}")
        return 1

    for p in pngs:
        im = Image.open(p).convert("L")
        out = out_dir / (p.stem + ".pgm")
        # PGM (P5) via raw save
        im.save(out, format="PPM")  # Pillow writes PGM when mode=L for PPM family
    print(f"Wrote {len(pngs)} PGM files to {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
