#!/usr/bin/env python3
# Owner: kevinrunescape1997
# Purpose: Convert pixel bitmaps to pixel-accurate SVG (<rect> per pixel) with per-row batched writes.

"""
Convert pixel bitmaps to pixel-accurate SVG using <rect> per pixel.

Primary loader: Pillow
Optional fallbacks (used if Pillow can't open a file):
- imageio (+ numpy) for many formats (EXR, HDR/RGBE, PFM, SGI RGB, DDS, KTX/KTX2, etc. depending on plugins)
- pydicom for DICOM (.dcm)
- astropy for FITS (.fits/.fit/.fts)
- rawpy for RAW camera formats
- Pillow plugins: pillow-heif, pillow-avif-plugin, pillow-jxl-plugin (registering decoders for HEIF/HEIC, AVIF, JXL)

If a format isn't readable due to missing dependencies, a clear error explains what to install.

Examples:
- python bitmap_svg_converter.py input.png -o output.svg
- python bitmap_svg_converter.py input.gif -o output.svg --frame 0
- python bitmap_svg_converter.py input.jpg -o output.svg --scale 2
- python bitmap_svg_converter.py input.tif -o output.svg --frame 0
- python bitmap_svg_converter.py input.webp -o output.svg
- python bitmap_svg_converter.py image.avif -o output.svg
- python bitmap_svg_converter.py image.heic -o output.svg
- python bitmap_svg_converter.py image.jxl -o output.svg
- python bitmap_svg_converter.py image.dcm -o output.svg
- python bitmap_svg_converter.py image.fits -o output.svg
- python bitmap_svg_converter.py image.exr -o output.svg
- python bitmap_svg_converter.py image.hdr -o output.svg
- python bitmap_svg_converter.py image.cr2 -o output.svg
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

# Try to register Pillow plugins for HEIF/AVIF/JXL if available
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
except Exception:
    pass
try:
    from pillow_avif_plugin import register_avif_opener  # type: ignore
    register_avif_opener()
except Exception:
    pass
try:
    from pillow_jxl_plugin import register_jxl_opener  # type: ignore
    register_jxl_opener()
except Exception:
    pass

# Optional backends
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# imageio v3 preferred
try:
    import imageio.v3 as iio  # type: ignore
except Exception:
    iio = None  # type: ignore

# pydicom
try:
    import pydicom  # type: ignore
except Exception:
    pydicom = None  # type: ignore

# astropy FITS
try:
    from astropy.io import fits  # type: ignore
except Exception:
    fits = None  # type: ignore

# rawpy
try:
    import rawpy  # type: ignore
except Exception:
    rawpy = None  # type: ignore


def rgba_to_hex(r: int, g: int, b: int) -> str:
    """Return #RRGGBB for the given RGB components."""
    return f"#{r:02x}{g:02x}{b:02x}"


def _require_numpy_for(msg_format: str):
    if np is None:
        raise RuntimeError(
            f"{msg_format} requires numpy. Please install: pip install numpy"
        )


def _normalize_to_uint8(arr):
    """Normalize numpy array of dtype float or >8-bit integers to uint8 [0,255]."""
    _require_numpy_for("Image conversion")
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo = float(np.nanpercentile(a, 1.0))
    hi = float(np.nanpercentile(a, 99.0))
    if hi <= lo:
        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
    if hi <= lo:
        hi = lo + 1.0
    a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    a = (a * 255.0).astype(np.uint8)
    return a


def _numpy_to_pil_rgba(arr) -> Image.Image:
    """Convert numpy array (H,W) or (H,W,C) to PIL RGBA."""
    _require_numpy_for("Image conversion")
    if arr.ndim == 2:
        a8 = _normalize_to_uint8(arr)
        im = Image.fromarray(a8, mode="L").convert("RGBA")
        return im
    elif arr.ndim == 3:
        h, w, c = arr.shape
        if c >= 4:
            a8 = _normalize_to_uint8(arr[:, :, :4])
            im = Image.fromarray(a8, mode="RGBA")
            return im
        elif c == 3:
            a8 = _normalize_to_uint8(arr)
            im = Image.fromarray(a8, mode="RGB").convert("RGBA")
            return im
        elif c == 2:
            base = _normalize_to_uint8(arr[:, :, 0])
            alpha = _normalize_to_uint8(arr[:, :, 1])
            rgb = np.stack([base, base, base, alpha], axis=-1)
            im = Image.fromarray(rgb, mode="RGBA")
            return im
        elif c == 1:
            a8 = _normalize_to_uint8(arr[:, :, 0])
            im = Image.fromarray(a8, mode="L").convert("RGBA")
            return im
        else:
            raise RuntimeError(f"Unsupported channel count: {c}")
    else:
        raise RuntimeError(f"Unsupported array shape: {arr.shape}")


def _read_with_imageio(path: str, frame_index: int) -> Optional[Image.Image]:
    """Try reading with imageio; returns PIL Image or None if not possible."""
    if iio is None:
        return None
    try:
        arr = iio.imread(path, index=frame_index)
        return _numpy_to_pil_rgba(arr)
    except Exception:
        try:
            arr = iio.imread(path)
            return _numpy_to_pil_rgba(arr)
        except Exception:
            return None


def _read_dicom(path: str) -> Optional[Image.Image]:
    if pydicom is None:
        return None
    try:
        ds = pydicom.dcmread(path)
        if not hasattr(ds, "pixel_array"):
            return None
        arr = ds.pixel_array
        if np is None:
            return None
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            return _numpy_to_pil_rgba(arr)
        else:
            return _numpy_to_pil_rgba(arr)
    except Exception:
        return None


def _read_fits(path: str) -> Optional[Image.Image]:
    if fits is None or np is None:
        return None
    try:
        with fits.open(path, memmap=True) as hdul:
            data = None
            if hdul and hdul[0].data is not None:
                data = hdul[0].data
            else:
                for hdu in hdul:
                    if hasattr(hdu, "data") and hdu.data is not None:
                        data = hdu.data
                        break
            if data is None:
                return None
            arr = np.asarray(data)
            if arr.ndim == 3:
                arr = arr[0]
            return _numpy_to_pil_rgba(arr)
    except Exception:
        return None


def _read_raw(path: str) -> Optional[Image.Image]:
    if rawpy is None or np is None:
        return None
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                output_bps=8,
                no_auto_bright=True,
                use_camera_wb=True,
                gamma=(1, 1),
            )
            return _numpy_to_pil_rgba(rgb)
    except Exception:
        return None


def open_image(path: str, frame_index: int = 0) -> Image.Image:
    """
    Open an image and return the RGBA frame specified.

    Strategy:
    1) Try Pillow (plus registered plugins). If multi-frame, seek frame_index.
    2) If Pillow fails, try specialized loaders based on file extension.
    3) If all fail, raise a helpful error indicating which package to install.
    """
    try:
        im = Image.open(path)
        try:
            if getattr(im, "is_animated", False):
                im.seek(frame_index)
        except EOFError:
            im.seek(0)
        return im.convert("RGBA")
    except (UnidentifiedImageError, OSError):
        pass

    ext = Path(path).suffix.lower()

    if ext == ".dcm":
        im = _read_dicom(path)
        if im is not None:
            return im
        raise RuntimeError("Failed to read DICOM (.dcm). Please install: pip install pydicom numpy")

    if ext in {".fits", ".fit", ".fts"}:
        im = _read_fits(path)
        if im is not None:
            return im
        raise RuntimeError("Failed to read FITS. Please install: pip install astropy numpy")

    raw_exts = {
        ".cr2", ".nef", ".arw", ".rw2", ".dng", ".orf", ".raf", ".sr2", ".pef", ".srw", ".rwl", ".nrw",
        ".3fr", ".fff", ".mef",
    }
    if ext in raw_exts:
        im = _read_raw(path)
        if im is not None:
            return im
        raise RuntimeError("Failed to read RAW file. Please install: pip install rawpy numpy")

    im = _read_with_imageio(path, frame_index)
    if im is not None:
        return im

    hints = {
        ".exr": "OpenEXR may require imageio with appropriate plugins. Try: pip install imageio numpy",
        ".hdr": "HDR/RGBE may require imageio FreeImage backend. Try: pip install imageio numpy",
        ".rgbe": "HDR/RGBE may require imageio FreeImage backend. Try: pip install imageio numpy",
        ".pfm": "PFM may require imageio support. Try: pip install imageio numpy",
        ".sgi": "SGI RGB may require imageio support. Try: pip install imageio numpy",
        ".rgb": "SGI RGB may require imageio support. Try: pip install imageio numpy",
        ".dds": "DDS may require imageio (FreeImage) support or conversion to PNG first. Try: pip install imageio numpy",
        ".ktx": "KTX may require imageio support. Try: pip install imageio numpy",
        ".ktx2": "KTX2 may require imageio support. Try: pip install imageio numpy",
        ".astc": "ASTC often requires specialized decoders; consider converting to PNG using external tools.",
        ".pvr": "PVR/PVRTC often requires specialized decoders; consider converting to PNG using external tools.",
        ".avif": "AVIF requires pillow-avif-plugin. Try: pip install pillow-avif-plugin",
        ".heif": "HEIF/HEIC requires pillow-heif. Try: pip install pillow-heif",
        ".heic": "HEIF/HEIC requires pillow-heif. Try: pip install pillow-heif",
        ".jxl": "JPEG XL requires pillow-jxl-plugin. Try: pip install pillow-jxl-plugin",
    }
    msg_hint = hints.get(ext, "Install imageio (pip install imageio numpy) or convert the image to PNG first.")
    raise RuntimeError(f"Could not open '{path}' with Pillow or fallbacks. Hint: {msg_hint}")


def emit_svg_header(f, svg_id: str, width: int, height: int, scale: int):
    """
    Write the SVG header with the requested id, dimensions, and viewBox.
    We keep 1 unit per pixel in the viewBox and scale rendered size via width/height.
    """
    write = f.write
    write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    write(
        f'<svg xmlns="http://www.w3.org/2000/svg" id="{svg_id}" '
        f'width="{width * scale}" height="{height * scale}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">'
    )


def emit_svg_footer(f):
    f.write('</svg>')


def write_rect(f, x: int, y: int, w: int, h: int, fill_hex: str, alpha: int):
    """
    Emit a single <rect> to the file.
    id uses '{x}-{y}' for single pixels.
    Opacity is kept as 0..255 to match your sample style exactly.
    """
    f.write(
        f'<rect id="{x}-{y}" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'shape-rendering="crispEdges" style="fill:{fill_hex};opacity:{alpha};"></rect>'
    )


def _rect_str(x: int, y: int, w: int, h: int, fill_hex: str, alpha: int) -> str:
    """Return the <rect> element string."""
    return (
        f'<rect id="{x}-{y}" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'shape-rendering="crispEdges" style="fill:{fill_hex};opacity:{alpha};"></rect>'
    )


def generate_svg_per_pixel(im: Image.Image, out_path: str, svg_id: str, scale: int, progress_cb: Optional[callable] = None):
    """
    Generate one rect per visible pixel with batched row writes to reduce I/O overhead.

    If progress_cb is provided, it will be called as progress_cb(rows_done, total_rows)
    once per row, allowing callers (e.g., GUI) to update a progress indicator.
    """
    width, height = im.size
    pixels = im.load()
    with open(out_path, "w", encoding="utf-8") as f:
        write = f.write
        emit_svg_header(f, svg_id, width, height, scale)
        for y in range(height):
            row_parts: list[str] = []
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                fill_hex = rgba_to_hex(r, g, b)
                row_parts.append(_rect_str(x, y, 1, 1, fill_hex, a))
            if row_parts:
                write("".join(row_parts))
            if progress_cb:
                try:
                    progress_cb(y + 1, height)
                except Exception:
                    pass
        emit_svg_footer(f)


def main():
    parser = argparse.ArgumentParser(description="Convert a bitmap to pixel-accurate SVG (per-pixel <rect>).")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("-o", "--output", help="Path to output SVG (default: input name with .svg)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to convert for multi-frame formats (default: 0)")
    parser.add_argument("--id", default=None, help='SVG root id (default: stem of input, e.g., "clocktower")')
    parser.add_argument("--scale", type=int, default=1, help="Output size multiplier (default: 1). Rects remain 1x1 in viewBox; width/height scaled.")
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output or (os.path.splitext(in_path)[0] + ".svg")
    svg_id = args.id or os.path.splitext(os.path.basename(in_path))[0]

    im = open_image(in_path, frame_index=args.frame)

    generate_svg_per_pixel(im, out_path, svg_id, args.scale, progress_cb=None)

    print(f"SVG written to: {out_path}")


if __name__ == "__main__":
    main()
