#!/usr/bin/env python3
from __future__ import annotations

# Owner: kevinrunescape1997
# Purpose: Convert SVG/SVGZ to EPS/PDF/TIFF/PNG with transparent background when possible.

"""
svg_exporter.py

Convert SVG or SVGZ to one of: EPS, PDF, TIFF, PNG.

Backends:
- Preferred for transparency in PDF/EPS: CairoSVG (svg2pdf/svg2ps) + Ghostscript (PDF->EPS)
- Preferred for PNG (fast): Inkscape CLI (explicit transparent background)
- Fallbacks: Inkscape for vector when CairoSVG not installed; Pillow for TIFF from PNG

Transparency policy:
- PNG: transparent via Inkscape "--export-background-opacity=0"
- PDF: transparent via CairoSVG "background_color='transparent'" (preferred)
- EPS: generated via PDF -> eps2write; EPS has limited/no true transparency, viewers often show white backgrounds
- TIFF: maintain alpha when converting from PNG (RGBA) where supported by Pillow
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
    """Discover Inkscape once (cached)."""
    global _INKSCAPE_BIN
    if _INKSCAPE_BIN is not None:
        return _INKSCAPE_BIN
    _INKSCAPE_BIN = shutil.which("inkscape")
    return _INKSCAPE_BIN


def which_ghostscript() -> Optional[str]:
    """
    Find Ghostscript executable (cached).

    Order:
    1) System PATH (gs, gswin64c, gswin32c, gs.exe)
    2) Bundled location next to the packaged app: <exe_dir>/ghostscript/bin/<gs*>
    """
    global _GS_BIN
    if _GS_BIN is not None:
        return _GS_BIN

    # System PATH first
    for cmd in ("gs", "gswin64c", "gswin32c", "gs.exe"):
        p = shutil.which(cmd)
        if p:
            _GS_BIN = p
            return _GS_BIN

    # Bundled location (Windows builds or local packaging)
    try:
        base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
        gs_bin = base / "ghostscript" / "bin"
        candidates = [
            gs_bin / "gswin64c.exe",
            gs_bin / "gswin32c.exe",
            gs_bin / "gs.exe",
            gs_bin / "gs",  # *nix name (less likely bundled)
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


def _inkscape_export(in_svg: Path, fmt: str, out_path: Path, dpi: Optional[int], background: str) -> Tuple[bool, str]:
    """
    Use Inkscape CLI primarily for PNG (fast) with enforced transparent background.
    For PDF/EPS, Inkscape may not expose background opacity; prefer CairoSVG for transparency instead.
    """
    inkscape = which_inkscape()
    if not inkscape:
        return False, "Inkscape not found in PATH."
    ensure_parent_dir(out_path)

    base_cmd = [inkscape, str(in_svg), "--export-filename", str(out_path)]
    # Export type mapping supported directly by Inkscape
    type_map = {"png": "png", "pdf": "pdf", "eps": "eps"}
    if fmt in type_map:
        base_cmd.extend(["--export-type", type_map[fmt]])
        if fmt == "png":
            # Enforce transparent background for PNG
            if background.lower().strip() == "transparent":
                base_cmd.extend(["--export-background-opacity", "0"])
            else:
                # If a color is provided (e.g., "#ffffff"), set it with 100% opacity
                base_cmd.extend(["--export-background", background, "--export-background-opacity", "1"])
            if dpi:
                base_cmd.extend(["--export-dpi", str(int(dpi))])
        elif fmt in {"pdf", "eps"}:
            # Inkscape CLI does not expose a simple transparent background switch for vector outputs.
            # We keep the call for robustness when CairoSVG is unavailable.
            if dpi and fmt == "pdf":
                # DPI affects rasterized components inside PDF
                base_cmd.extend(["--export-dpi", str(int(dpi))])
        ok, out = _run(base_cmd)
        if ok:
            return True, f"Inkscape OK: {fmt}"
        return False, f"Inkscape failed: {out}"
    return False, "Inkscape cannot directly export this format."


def _cairosvg_export(in_svg: Path, fmt: str, out_path: Path, dpi: Optional[int], background: str) -> Tuple[bool, str]:
    """
    Use CairoSVG for PDF/PS/PNG with explicit background_color control.
    EPS is produced via PDF -> Ghostscript eps2write.
    """
    if cairosvg is None:
        return False, "CairoSVG not installed. pip install cairosvg"

    ensure_parent_dir(out_path)
    try:
        if fmt == "pdf":
            cairosvg.svg2pdf(url=str(in_svg), write_to=str(out_path), background_color=background)
            return True, "CairoSVG OK: pdf (transparent bg)"
        elif fmt == "png":
            kw = {"background_color": background}
            if dpi and dpi > 0:
                kw["dpi"] = int(dpi)
            cairosvg.svg2png(url=str(in_svg), write_to=str(out_path), **kw)
            return True, "CairoSVG OK: png (transparent bg)"
        elif fmt == "eps":
            # Generate PDF with transparent background, then EPS via Ghostscript
            with tempfile.TemporaryDirectory() as td:
                tmp_pdf = Path(td) / "tmp.pdf"
                cairosvg.svg2pdf(url=str(in_svg), write_to=str(tmp_pdf), background_color=background)
                ok, msg = _ghostscript_pdf_to_eps(tmp_pdf, out_path)
                if ok:
                    # Note: EPS transparency is limited; many viewers render white.
                    return True, "CairoSVG+Ghostscript OK: eps (transparency limited by EPS)"
                if "Ghostscript not found" in msg:
                    return False, (
                        "Ghostscript not found. On Linux/macOS, install ghostscript "
                        "(e.g., apt-get install ghostscript or brew install ghostscript) "
                        "or install Inkscape and use its native EPS export."
                    )
                # Fallback: PS and rename to .eps (not true EPS transparency)
                tmp_ps = Path(td) / "tmp.ps"
                cairosvg.svg2ps(url=str(in_svg), write_to=str(tmp_ps), background_color=background)
                shutil.copyfile(str(tmp_ps), str(out_path))
                return True, "CairoSVG PS fallback: wrote .eps (PS content; transparency limited)"
        elif fmt == "tiff":
            # Produce PNG with transparent background then convert via Pillow preserving alpha
            if Image is None:
                return False, "Pillow not installed. pip install pillow"
            with tempfile.TemporaryDirectory() as td:
                tmp_png = Path(td) / "tmp.png"
                kw = {"background_color": background}
                if dpi and dpi > 0:
                    kw["dpi"] = int(dpi)
                cairosvg.svg2png(url=str(in_svg), write_to=str(tmp_png), **kw)
                im = Image.open(str(tmp_png))
                # Save TIFF with alpha if present (Pillow supports RGBA TIFF)
                im.save(str(out_path), format="TIFF")
                return True, "CairoSVG+Pillow OK: tiff (alpha preserved where supported)"
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
    """
    Convert PNG to TIFF via Pillow preserving alpha channel where supported.
    """
    if Image is None:
        return False, "Pillow not installed. pip install pillow"
    ensure_parent_dir(tiff_out)
    try:
        im = Image.open(str(png_path))
        im.save(str(tiff_out), format="TIFF")
        return True, "Pillow OK: tiff"
    except Exception as e:
        return False, f"Pillow TIFF failed: {e}"


def convert_svg(in_svg: Path, fmt: str, out_path: Path, dpi: Optional[int] = None, background_color: str = "transparent") -> Tuple[bool, str]:
    """
    Convert a single input SVG/SVGZ to the desired output format.

    Parameters:
    - in_svg: Path to input .svg or .svgz
    - fmt: 'pdf' | 'png' | 'eps' | 'tiff'
    - out_path: Output file path
    - dpi: DPI for rasterization (PNG/TIFF; affects raster components for PDF)
    - background_color: CSS color string; default 'transparent'
    """
    fmt = fmt.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {fmt}"

    # Preferred transparency path for vector formats (pdf/eps): CairoSVG
    if fmt in {"pdf", "eps"}:
        ok, msg = _cairosvg_export(in_svg, fmt, out_path, dpi, background_color)
        if ok:
            return True, msg
        # Fallback to Inkscape when CairoSVG not available or failed
        ok2, msg2 = _inkscape_export(in_svg, fmt, out_path, dpi, background_color)
        return (ok2, msg2)

    # PNG: prefer Inkscape for speed with transparent background flag; fallback CairoSVG
    if fmt == "png":
        ok, msg = _inkscape_export(in_svg, fmt, out_path, dpi, background_color)
        if ok:
            return True, msg
        ok2, msg2 = _cairosvg_export(in_svg, fmt, out_path, dpi, background_color)
        return (ok2, msg2)

    # TIFF: prefer CairoSVG->PNG->Pillow to preserve alpha; fallback Inkscape->PNG->Pillow if needed
    if fmt == "tiff":
        if cairosvg is not None:
            ok, msg = _cairosvg_export(in_svg, "tiff", out_path, dpi, background_color)
            return (ok, msg)
        # Fallback: Inkscape -> PNG -> TIFF
        inkscape = which_inkscape()
        if inkscape and Image is not None:
            with tempfile.TemporaryDirectory() as td:
                tmp_png = Path(td) / "tmp.png"
                ok_png, msg_png = _inkscape_export(in_svg, "png", tmp_png, dpi, background_color)
                if ok_png:
                    ok_tif, msg_tif = _pillow_png_to_tiff(tmp_png, out_path)
                    if ok_tif:
                        return True, f"Inkscape+Pillow OK: tiff"
                return False, f"TIFF fallback failed: {msg_png}"
        return False, "No backend available for TIFF (install cairosvg or pillow)."

    return False, f"Unhandled format: {fmt}"


def _suffix_for(fmt: str) -> str:
    return {"pdf": ".pdf", "png": ".png", "eps": ".eps", "tiff": ".tiff"}[fmt]


def default_output_path(inp: Path, fmt: str) -> Path:
    return inp.with_suffix(_suffix_for(fmt))


def main():
    ap = argparse.ArgumentParser(
        description="Convert SVG/SVGZ to EPS/PDF/TIFF/PNG using CairoSVG (preferred for transparency) "
                    "or Inkscape/Pillow/Ghostscript fallbacks."
    )
    ap.add_argument("input", nargs="+", help="Input SVG/SVGZ file(s). Supports wildcards via shell.")
    ap.add_argument("--format", required=True, choices=sorted(SUPPORTED_FORMATS), help="Output format.")
    ap.add_argument("-o", "--output", help="Output path (only valid with a single input).")
    ap.add_argument("--dpi", type=int, default=None, help="Raster DPI for PNG/TIFF (default: backend default).")
    ap.add_argument(
        "--background-color",
        default="transparent",
        help="Background color for outputs (default: transparent). Examples: 'transparent', '#ffffff'."
    )
    args = ap.parse_args()

    inputs = [Path(p) for p in args.input]
    fmt = args.format.lower().strip()
    bg = args.background_color.strip().lower() if args.background_color else "transparent"

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
            ok, msg = convert_svg(inp, fmt, out_path, dpi=args.dpi, background_color=bg)
            if ok:
                size = out_path.stat().st_size if out_path.exists() else 0
                print(f"Wrote: {out_path} | bytes: {size:,} | {msg}")
            else:
                # Add friendly guidance when EPS fails and neither CairoSVG nor Ghostscript helped
                if fmt == "eps" and ("Ghostscript not found" in msg or "No backend" in msg):
                    msg += " | Tip: Install CairoSVG+Ghostscript or Inkscape to enable EPS export."
                print(f"Failed: {inp} -> {out_path} | {msg}")
        except Exception as e:
            print(f"Failed: {inp} -> {out_path} | {e}")


if __name__ == "__main__":
    main()
