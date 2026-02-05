# SVG Pixel-Rect Optimizer (GUI)

Cross-platform GUI tool for optimizing **pixel art stored as SVG `<rect>` elements**.
It reduces file size and rect count by merging adjacent pixels while preserving visual fidelity.

This project consists of:

* a **GUI frontend** (`svg_optimizer_gui.py`)
* a **core optimizer** (`svg_pixel_rect_optimizer.py`)
* simple executables for Windows, Mac, and Linux Ubuntu 22.04

---

## What This Tool Is For

This tool is designed for SVGs that represent **pixel art**, where each pixel (or group of pixels) is drawn using `<rect>` elements.

It is **not** a general-purpose SVG optimizer.

This tool **does not**:

* Simplify paths
* Optimize curves
* Reduce gradients
* Rewrite arbitrary SVGs

Typical use cases:

* Game sprites and tiles
* Pixel-perfect UI assets
* Exported SVG pixel art from editors

It is purpose-built for **rect-based pixel art SVGs**.

---

## Core Optimizer (`svg_pixel_rect_optimizer.py`)

* Safe to run **multiple times** on the same file
* Will not break SVGs that already contain merged rectangles
* Designed for deterministic, stable output

The optimizer performs the following steps:

* Normalizes `fill` and `opacity`
* Expands any `<rect>` with `width > 1` or `height > 1` into per-pixel coverage
  (prevents corruption when re-optimizing already-merged SVGs)
* Merges pixels:

  * **Horizontally** into runs
  * **Vertically** into stacks
  * **Connected Paths** merges connected runs and stacks into a filled path

* Emits a clean SVG and/or SVGZ:

  * One `<g>` element
  * `shape-rendering="crispEdges"` on the root
  * Minimal attributes per `<rect>`

---

## GUI Application (`svg_optimizer_gui.py`)

### Features

* Drag & drop SVG files or folders
* Batch processing
* Recursive folder scanning
* Optional preservation of folder structure
* Optional preservation of original file names
* Optional choice between merged paths or just runs and stacks
* Skips already-optimized files (`*_optimized*.svg`)
* Choice between SVG, SVG + SVGZ, or just SVGZ
* Progress bar + per-run log file

---

## Platform Notes

### Windows

* Uses Tk dialogs
* May show SmartScreen warnings (unsigned)

### macOS

* Uses native file dialogs
* May require “Open Anyway” for unsigned apps

### Linux

* GUI **prefers native system dialogs** Zenity/KDialog
* If neither is available, it **falls back to Tk’s file dialogs**.
* Native dialogs avoid Tk’s built-in “Directory:” dropdown UX issues

---

## Drag & Drop

* Supports dropping:

  * Individual SVG files
  * Entire folders (recursively scanned)
* Dropping a folder onto the **Output field** sets the output directory
* Drag & drop is enabled when `tkinterdnd2` is installed

---

## Toggles Explained

| Toggle                    | Description                                       |
| ------------------------- | ------------------------------------------------- |
| Recursive folder scan     | Scan subfolders when adding a directory           |
| Preserve folder structure | Mirror input directory structure in output        |
| Preserve file names       | Keep original file names (no `_optimized` suffix) |
| Skip `_optimized` files   | Prevent re-adding previously optimized SVGs       |

---

## Output

* Output directory is user-selectable
* Files are written as optimized SVGs
* A log file is written per run:

  ```
  svg_optimizer_log.txt
  ```

Log entries include:

* input path
* output path
* rect count
* byte size
* errors (if any)

---

## Requirements

* Python 3.10+
* Tkinter
* Optional (Linux):

  * `zenity` or `kdialog`
  * `tkinterdnd2`

### Run

```bash
python3 svg_optimizer_gui.py
```

---
