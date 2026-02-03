# SVG Pixel-Rect Optimizer (GUI)

Cross-platform GUI tool for optimizing **pixel art stored as SVG `<rect>` elements**.
It reduces file size and rect count by merging adjacent pixels while preserving visual fidelity. It is mainly ment to be used in conjunction with this tool: https://www.scalablepixels.com

scalablepixels.com is made by Roy Tettero (https://www.roytettero.com), so full credit to him for that. 

This project consists of:

* a **GUI frontend** (`svg_optimizer_gui.py`)
* a **core optimizer** (`svg_pixel_rect_optimizer.py`)

---

## What This Tool Is For

This tool is designed for SVGs that represent **pixel art**, where each pixel (or group of pixels) is drawn using `<rect>` elements.

It is **not** a general-purpose SVG optimizer.

Typical use cases:

* Game sprites and tiles
* Pixel-perfect UI assets
* Exported SVG pixel art from editors

---

## Core Optimizer (`svg_pixel_rect_optimizer.py`)

The optimizer performs the following steps:

* Normalizes `fill` and `opacity`
* Expands any `<rect>` with `width > 1` or `height > 1` into per-pixel coverage
  (prevents corruption when re-optimizing already-merged SVGs)
* Merges pixels:

  * **Horizontally** into runs
  * **Vertically** into stacks (optional)
* Emits a clean SVG:

  * One `<g>` element
  * `shape-rendering="crispEdges"` on the root
  * Minimal attributes per `<rect>`

### Important behavior

* Safe to run **multiple times** on the same file
* Will not “break” SVGs that already contain merged rectangles
* Designed for deterministic, stable output

---

## GUI Application (`svg_optimizer_gui.py`)

### Features

* Drag & drop SVG files or folders
* Batch processing
* Recursive folder scanning
* Optional preservation of folder structure
* Optional preservation of original file names
* Skips already-optimized files (`*_optimized*.svg`)
* Progress bar + per-run log file
* Visual hover states for:

  * Drag & drop area
  * Toggle buttons

---

## File & Folder Pickers (Important)

### Linux behavior

On Linux, the GUI **prefers native system dialogs**:

1. **Zenity** (GNOME / GTK)
2. **KDialog** (KDE)

If neither is available, it **falls back to Tk’s file dialogs**.

### Why this matters

* Native dialogs avoid Tk’s built-in “Directory:” dropdown UX issues
* Canceling a native dialog **does not** trigger a fallback dialog
* Fallback only occurs if no system picker is available

---

## Drag & Drop

* Supports dropping:

  * Individual SVG files
  * Entire folders (recursively scanned)
* Dropping a folder onto the **Output field** sets the output directory
* Drag & drop is enabled when `tkinterdnd2` is installed

If unavailable, the GUI still functions normally without drag & drop.

---

## Toggles Explained

| Toggle                    | Description                                       |
| ------------------------- | ------------------------------------------------- |
| Recursive folder scan     | Scan subfolders when adding a directory           |
| Preserve folder structure | Mirror input directory structure in output        |
| Preserve file names       | Keep original file names (no `_optimized` suffix) |
| Skip `_optimized` files   | Prevent re-adding previously optimized SVGs       |

All toggles include hover feedback and clear ON/OFF states.

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

## Running from Source

### Requirements

* Python 3.10+
* Tkinter
* Optional (Linux):

  * `zenity` or `kdialog`
  * `tkinterdnd2` (for drag & drop)

### Run

```bash
python3 svg_optimizer_gui.py
```

---

## Platform Notes

### Windows

* Uses Tk dialogs
* May show SmartScreen warnings (unsigned)

### macOS

* Uses native file dialogs
* May require “Open Anyway” for unsigned apps

### Linux

* Prefers Zenity/KDialog
* Falls back cleanly to Tk dialogs
* No duplicate dialog opening

---

## Not a Vector Optimizer

This tool **does not**:

* Simplify paths
* Optimize curves
* Reduce gradients
* Rewrite arbitrary SVGs

It is purpose-built for **rect-based pixel art SVGs**.

---

## License

MIT (or your chosen license)

---

## Summary

* Safe, repeatable optimization for pixel-rect SVGs
* Native dialogs where possible
* Clean GUI behavior (no duplicate pickers)
* Designed for batch workflows and asset pipelines

It is **not** a general-purpose SVG optimizer.

Typical use cases:

* Game sprites and tiles
* Pixel-perfect UI assets
* Exported SVG pixel art from editors

---

## Core Optimizer (`svg_pixel_rect_optimizer.py`)

The optimizer performs the following steps:

* Normalizes `fill` and `opacity`
* Expands any `<rect>` with `width > 1` or `height > 1` into per-pixel coverage
  (prevents corruption when re-optimizing already-merged SVGs)
* Merges pixels:

  * **Horizontally** into runs
  * **Vertically** into stacks (optional)
* Emits a clean SVG:

  * One `<g>` element
  * `shape-rendering="crispEdges"` on the root
  * Minimal attributes per `<rect>`

### Important behavior

* Safe to run **multiple times** on the same file
* Will not “break” SVGs that already contain merged rectangles
* Designed for deterministic, stable output

---

## GUI Application (`svg_optimizer_gui.py`)

### Features

* Drag & drop SVG files or folders
* Batch processing
* Recursive folder scanning
* Optional preservation of folder structure
* Optional preservation of original file names
* Skips already-optimized files (`*_optimized*.svg`)
* Progress bar + per-run log file
* Visual hover states for:

  * Drag & drop area
  * Toggle buttons

---

## File & Folder Pickers (Important)

### Linux behavior

On Linux, the GUI **prefers native system dialogs**:

1. **Zenity** (GNOME / GTK)
2. **KDialog** (KDE)

If neither is available, it **falls back to Tk’s file dialogs**.

### Why this matters

* Native dialogs avoid Tk’s built-in “Directory:” dropdown UX issues
* Canceling a native dialog **does not** trigger a fallback dialog
* Fallback only occurs if no system picker is available

---

## Drag & Drop

* Supports dropping:

  * Individual SVG files
  * Entire folders (recursively scanned)
* Dropping a folder onto the **Output field** sets the output directory
* Drag & drop is enabled when `tkinterdnd2` is installed

If unavailable, the GUI still functions normally without drag & drop.

---

## Toggles Explained

| Toggle                    | Description                                       |
| ------------------------- | ------------------------------------------------- |
| Recursive folder scan     | Scan subfolders when adding a directory           |
| Preserve folder structure | Mirror input directory structure in output        |
| Preserve file names       | Keep original file names (no `_optimized` suffix) |
| Skip `_optimized` files   | Prevent re-adding previously optimized SVGs       |

All toggles include hover feedback and clear ON/OFF states.

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

## Running from Source

### Requirements

* Python 3.10+
* Tkinter
* Optional (Linux):

  * `zenity` or `kdialog`
  * `tkinterdnd2` (for drag & drop)

### Run

```bash
python3 svg_optimizer_gui.py
```

---

## Platform Notes

### Windows

* Uses Tk dialogs
* May show SmartScreen warnings (unsigned)

### macOS

* Uses native file dialogs
* May require “Open Anyway” for unsigned apps

### Linux

* Prefers Zenity/KDialog
* Falls back cleanly to Tk dialogs
* No duplicate dialog opening

---

## Not a Vector Optimizer

This tool **does not**:

* Simplify paths
* Optimize curves
* Reduce gradients
* Rewrite arbitrary SVGs

It is purpose-built for **rect-based pixel art SVGs**.

---

## License

MIT

---

## Summary

* Safe, repeatable optimization for pixel-rect SVGs
* Native dialogs where possible
* Clean GUI behavior (no duplicate pickers)
* Designed for batch workflows and asset pipelines
