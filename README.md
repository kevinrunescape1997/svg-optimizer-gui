# Vector Pixel Tools

A cross‑platform suite of tools for pixel‑art workflows built around SVGs, with a unified launcher:

- Vector Pixel Tools Launcher
- Bitmap SVG Converter
- SVG Pixel Optimizer
- SVG Exporter

This suite is ideal for rect‑based pixel art SVGs and vector export pipelines. It is not a general‑purpose SVG path optimizer.

---

## Repository layout (current files)

- `copilot-instructions.md` — coding standards and assistant protocol
- Launcher:
  - `vector_pixel_tools_launcher.py`
- Bitmap → SVG Converter:
  - `GUI_bitmap_converter.py` (GUI)
  - `bitmap_svg_converter.py` (CLI/core)
- SVG Pixel Optimizer:
  - `GUI_svg_optimizer.py` (GUI)
  - `pixel_svg_optimizer.py` (CLI/core)
- SVG Exporter:
  - `GUI_svg_exporter.py` (GUI)
  - `svg_exporter.py` (CLI/core)
- Documentation:
  - `README.md`

---

## Tools Overview

### 1) Bitmap → SVG Converter
- Files: `GUI_bitmap_converter.py` (GUI), `bitmap_svg_converter.py` (CLI/core)
- Converts many bitmap formats (PNG/JPG/GIF/BMP/TIFF/WebP/AVIF/HEIF/HEIC/JXL/FITS/DICOM/OpenEXR/HDR/RGBE/PFM/SGI RGB/DDS/KTX/KTX2/RAW camera formats…) into per‑pixel SVG using `<rect>` per pixel.
- Multi‑frame formats export all frames when supported (GIF/TIFF/WebP/AVIF/HEIF/JXL).
- Live progress with per‑row updates and overall percent/file counter overlay.

Requirements:
- pip install pillow
- Optional backends (broader format support):
  - pip install imageio numpy
  - pip install pillow-heif pillow-avif-plugin pillow-jxl-plugin
  - pip install pydicom
  - pip install astropy
  - pip install rawpy
- Optional for drag & drop:
  - pip install tkinterdnd2

CLI examples:
```bash
python bitmap_svg_converter.py input.png -o output.svg
python bitmap_svg_converter.py input.gif -o output.svg --frame 0
```

---

### 2) SVG Pixel‑Rect Optimizer
- Files: `GUI_svg_optimizer.py` (GUI), `pixel_svg_optimizer.py` (CLI/core)
- Computes final per‑pixel RGBA via source‑over compositing in DOM order.
- Emits merged rectangles (horizontal + optional vertical stacking) or connected `<path>` shapes for like‑colored pixels.
- Post‑processing can merge same‑color shapes, sort/minify attributes, and remove defaults.
- Streaming rect optimizer for very large SVGs (>200 MB) via `lxml` iterparse to reduce memory use.
- GUI automatically disables path mode for very large inputs and falls back to streaming rects for stability.

CLI examples:
```bash
python pixel_svg_optimizer.py input.svg
python pixel_svg_optimizer.py input.svg --paths --minify --svgz --zopfli
```

---

### 3) SVG to EPS/PDF/TIFF/PNG Exporter
- Files: `GUI_svg_exporter.py` (GUI), `svg_exporter.py` (CLI/core)
- Exports SVG/SVGZ into EPS, PDF, TIFF, or PNG.
- Backends:
  - Preferred: Inkscape CLI (fast, robust vector output for PDF/EPS; raster for PNG)
  - Fallbacks: CairoSVG (svg2pdf/svg2png/svg2ps), Ghostscript (PDF→EPS), Pillow (PNG→TIFF)
- DPI is supported for PNG/TIFF.
- GUI includes preserve‑folder‑structure option, custom naming/stem, live naming preview, and a percent/file counter overlay.

Requirements:
- Inkscape (preferred; install via OS packages)
- pip install cairosvg pillow
- Ghostscript for EPS fallback (install via OS packages)

CLI examples:
```bash
python svg_exporter.py input.svg --format pdf
python svg_exporter.py input.svgz --format eps
python svg_exporter.py input.svg --format png --dpi 300
python svg_exporter.py input.svg --format tiff -o out.tiff --dpi 300
```

---

## Pixel Tools Launcher

- File: `vector_pixel_tools_launcher.py`
- Single‑instance launcher window with buttons to open:
  - Bitmap → SVG Converter
  - SVG Pixel Optimizer
  - SVG Exporter
- Uses a small TCP server on `127.0.0.1:51262` to bring an existing launcher window to the front if a new instance is started.
- Styled with a dark theme consistent across the GUIs.

Run:
```bash
python vector_pixel_tools_launcher.py
```

---

## GUI Features (common)

- Drag & drop files/folders (optional via `tkinterdnd2`)
- Batch processing with progress bar and percent/file counter overlay (e.g., “42.3% • File 3/12”)
- Output folder selection with “Up” and native dialogs on Linux (Zenity/KDialog) when available
- Optional preservation of folder structure for outputs
- Custom name all + custom stem all, with live naming preview
- Unified dark theme and consistent scrollbars
- Summary dialog with OK/Failed counts and a log file link per run

---

## Platform Notes

- Windows/macOS/Linux supported
- Linux:
  - GUIs prefer Zenity/KDialog when available; otherwise standard Tk dialogs
  - Ensure Tk is available (e.g., `sudo apt-get install python3-tk`)
- Python: 3.10+

---

## Development

- Python 3.10+
- Tkinter (usually included with Python on Windows/macOS; install `python3-tk` on Linux)
- Optional: `tkinterdnd2` for drag & drop

Run any GUI:
```bash
python GUI_bitmap_converter.py
python GUI_svg_optimizer.py
python GUI_svg_exporter.py
```

Run the launcher:
```bash
python vector_pixel_tools_launcher.py
```
