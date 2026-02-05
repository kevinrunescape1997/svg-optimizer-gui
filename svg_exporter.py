#!/usr/bin/env python3
from __future__ import annotations

# Owner: kevinrunescape1997
# Purpose: Convert SVG/SVGZ to EPS/PDF/TIFF/PNG via Inkscape preferred, CairoSVG/Pillow/Ghostscript fallback.
# Notes: Cache binary discovery to reduce repeated PATH lookups.

"""
svg_exporter.py

Convert SVG or SVGZ to one of: EPS, PDF, TIFF, PNG.

Backends:
- Preferred: Inkscape CLI (fast, robust vector output for PDF/EPS, PNG raster)
- Fallback: CairoSVG (svg2pdf/svg2png/svg2ps)
- For EPS (fallback): PDF -> EPS via Ghostscript (eps2write)
- For TIFF: PNG -> TIFF via Pillow
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Optional python libraries
try:
    import cairosvg  # type: ignore
except Exception:
    cairosvg = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore


SUPPORTED_FORMATS = {"eps", "pdf", "tiff", "png"}

# Cached binary discovery to avoid repeated PATH lookups
_INKSCAPE_BIN: Optional[str] = None
_GS_BIN: Optional[str] = None


def which_inkscape() -> Optional[str]:
    global _INKSCAPE_BIN
    if _INKSCAPE_BIN is not None:
        return _INKSCAPE_BIN
    _INKSCAPE_BIN = shutil.which("inkscape")
    return _INKSCAPE_BIN


def which_ghostscript() -> Optional[str]:
    """
    Find Ghostscript executable.

    Order:
    1) System PATH (gs, gswin64c, gswin32c, gs.exe)
    2) Bundled location next to the packaged app: <exe_dir>/ghostscript/bin/<gs*>
    """
    global _GS_BIN
    if _GS_BIN is not None:
        return _GS_BIN

    for cmd in ("gs", "gswin64c", "gswin32c", "gs.exe"):
        p = shutil.which(cmd)
        if p:
            _GS_BIN = p
            return _GS_BIN

    try:
        base = None
        if getattr(sys, "frozen", False):
            base = Path(sys.executable).resolve().parent
        else:
            base = Path(__file__).resolve().parent
        gs_bin = base / "ghostscript" / "bin"
        candidates = [
            gs_bin / "gswin64c.exe",
            gs_bin / "gswin32c.exe",
            gs_bin / "gs.exe",
            gs_bin / "gs",
        ]
        for c in candidates:
            try:
                if c.exists():
                    _GS_BIN = str(c)
                    return _GS_BIN
            except Exception:
                pass
    except Exception:
        pass
    return None


def ensure_parent_dir(p: Path) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _run(cmd: list[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (proc.returncode == 0)
        out = (proc.stdout + "\n" + proc.stderr).strip()
        return ok, out
    except Exception as e:
        return False, str(e)


def _inkscape_export(in_svg: Path, fmt: str, out_path: Path, dpi: Optional[int]) -> Tuple[bool, str]:
    """
    Use Inkscape CLI to export directly when possible.
    EPS, PDF, PNG supported directly; TIFF not directly supported (use PNG + Pillow).
    """
    inkscape = which_inkscape()
    if not inkscape:
        return False, "Inkscape not found in PATH."
    ensure_parent_dir(out_path)
    base_cmd = [inkscape, str(in_svg), "--export-filename", str(out_path)]
    type_map = {"eps": "eps", "pdf": "pdf", "png": "png"}
    if fmt in type_map:
        base_cmd.extend(["--export-type", type_map[fmt]])
        if fmt == "png" and dpi:
            base_cmd.extend(["--export-dpi", str(int(dpi))])
        ok, out = _run(base_cmd)
        if ok:
            return True, f"Inkscape OK: {fmt}"
        return False, f"Inkscape failed: {out}"
    return False, "Inkscape cannot directly export this format."


def _cairosvg_export(in_svg: Path, fmt: str, out_path: Path, dpi: Optional[int]) -> Tuple[bool, str]:
    """
    Use CairoSVG as fallback for PDF, PNG, PS (EPS via Ghostscript).
    """
    if cairosvg is None:
        return False, "CairoSVG not installed. pip install cairosvg"

    ensure_parent_dir(out_path)
    try:
        if fmt == "pdf":
            cairosvg.svg2pdf(url=str(in_svg), write_to=str(out_path))
            return True, "CairoSVG OK: pdf"
        elif fmt == "png":
            kw = {}
            if dpi and dpi > 0:
                kw["dpi"] = int(dpi)
            cairosvg.svg2png(url=str(in_svg), write_to=str(out_path), **kw)
            return True, "CairoSVG OK: png"
        elif fmt == "eps":
            with tempfile.TemporaryDirectory() as td:
                tmp_pdf = Path(td) / "tmp.pdf"
                cairosvg.svg2pdf(url=str(in_svg), write_to=str(tmp_pdf))
                ok, msg = _ghostscript_pdf_to_eps(tmp_pdf, out_path)
                if ok:
                    return True, "CairoSVG+Ghostscript OK: eps"
                if "Ghostscript not found" in msg:
                    return False, (
                        "Ghostscript not found. On Linux/macOS, install ghostscript "
                        "(e.g., apt-get install ghostscript or brew install ghostscript) "
                        "or install Inkscape and use its native EPS export. "
                        "On Windows builds, Ghostscript may be bundled in the zip under 'ghostscript/bin'."
                    )
                tmp_ps = Path(td) / "tmp.ps"
                cairosvg.svg2ps(url=str(in_svg), write_to=str(tmp_ps))
                shutil.copyfile(str(tmp_ps), str(out_path))
                return True, "CairoSVG PS fallback: wrote .eps (PS content)"
        elif fmt == "tiff":
            if Image is None:
                return False, "Pillow not installed. pip install pillow"
            with tempfile.TemporaryDirectory() as td:
                tmp_png = Path(td) / "tmp.png"
                kw = {}
                if dpi and dpi > 0:
                    kw["dpi"] = int(dpi)
                cairosvg.svg2png(url=str(in_svg), write_to=str(tmp_png), **kw)
                im = Image.open(str(tmp_png))
                im.save(str(out_path), format="TIFF")
                return True, "CairoSVG+Pillow OK: tiff"
        else:
            return False, f"Unsupported format: {fmt}"
    except Exception as e:
        return False, f"CairoSVG failed: {e}"


def _ghostscript_pdf_to_eps(pdf_path: Path, eps_out: Path) -> Tuple[bool, str]:
    gs = which_ghostscript()
    if not gs:
        return False, "Ghostscript not found."
    ensure_parent_dir(eps_out)
    cmd = [
        gs, "-dSAFER", "-dBATCH", "-dNOPAUSE",
        "-sDEVICE=eps2write", "-dEPSCrop",
        "-sOutputFile=" + str(eps_out),
        str(pdf_path),
    ]
    ok, out = _run(cmd)
    if ok:
        return True, "Ghostscript OK: eps2write"
    return False, f"Ghostscript failed: {out}"


def _pillow_png_to_tiff(png_path: Path, tiff_out: Path) -> Tuple[bool, str]:
    if Image is None:
        return False, "Pillow not installed. pip install pillow"
    ensure_parent_dir(tiff_out)
    try:
        im = Image.open(str(png_path))
        im.save(str(tiff_out), format="TIFF")
        return True, "Pillow OK: tiff"
    except Exception as e:
        return False, f"Pillow TIFF failed: {e}"


def convert_svg(in_svg: Path, fmt: str, out_path: Path, dpi: Optional[int] = None) -> Tuple[bool, str]:
    """
    Convert a single input SVG/SVGZ to the desired output format.
    """
    fmt = fmt.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {fmt}"

    if fmt in {"pdf", "png", "eps"}:
        ok, msg = _inkscape_export(in_svg, fmt, out_path, dpi)
        if ok:
            return True, msg

    if fmt == "tiff":
        inkscape = which_inkscape()
        if inkscape:
            with tempfile.TemporaryDirectory() as td:
                tmp_png = Path(td) / "tmp.png"
                ok_png, msg_png = _inkscape_export(in_svg, "png", tmp_png, dpi)
                if ok_png:
                    ok_tif, msg_tif = _pillow_png_to_tiff(tmp_png, out_path)
                    if ok_tif:
                        return True, f"Inkscape+Pillow OK: tiff"

    ok, msg = _cairosvg_export(in_svg, fmt, out_path, dpi)
    return ok, msg


def _suffix_for(fmt: str) -> str:
    return { "pdf": ".pdf", "png": ".png", "eps": ".eps", "tiff": ".tiff" }[fmt]


def default_output_path(inp: Path, fmt: str) -> Path:
    return inp.with_suffix(_suffix_for(fmt))


def main():
    ap = argparse.ArgumentParser(description="Convert SVG/SVGZ to EPS/PDF/TIFF/PNG using Inkscape (preferred) or CairoSVG/Pillow/Ghostscript fallbacks.")
    ap.add_argument("input", nargs="+", help="Input SVG/SVGZ file(s). Supports wildcards via shell.")
    ap.add_argument("--format", required=True, choices=sorted(SUPPORTED_FORMATS), help="Output format.")
    ap.add_argument("-o", "--output", help="Output path (only valid with a single input).")
    ap.add_argument("--dpi", type=int, default=None, help="Raster DPI for PNG/TIFF (default: backend default).")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.input]
    fmt = args.format.lower().strip()

    if args.output and len(inputs) != 1:
        raise SystemExit("Error: -o/--output can only be used with a single input file.")

    for inp in inputs:
        if not inp.exists():
            print(f"Skip (not found): {inp}")
            continue
        if inp.suffix.lower() not in {".svg", ".svgz"}:
            print(f"Skip (unsupported): {inp}")
            continue

        out_path = Path(args.output) if args.output else default_output_path(inp, fmt)
        try:
            ok, msg = convert_svg(inp, fmt, out_path, dpi=args.dpi)
            if ok:
                size = out_path.stat().st_size if out_path.exists() else 0
                print(f"Wrote: {out_path} | bytes: {size:,} | {msg}")
            else:
                if fmt == "eps" and "Ghostscript not found" in msg:
                    msg += " | Tip: Install Inkscape or Ghostscript to enable EPS export."
                print(f"Failed: {inp} -> {out_path} | {msg}")
        except Exception as e:
            print(f"Failed: {inp} -> {out_path} | {e}")


if __name__ == "__main__":
    main()
