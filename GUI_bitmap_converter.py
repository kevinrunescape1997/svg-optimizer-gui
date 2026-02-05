#!/usr/bin/env python3
from __future__ import annotations

# Owner: kevinrunescape1997
# Purpose: GUI for bitmap → SVG conversion; preserves progress and UX. Converter batched writes handled in backend.

"""
GUI for converting bitmaps (PNG/JPEG/GIF/BMP/TIFF/WebP/AVIF/HEIF/HEIC/JXL/EXR/HDR/…)
to pixel-accurate SVG using rect-per-pixel.

Drag & drop supported (optional), batch processing, progress, and summary dialog.

Requirements:
- pip install pillow
- optional: pip install tkinterdnd2 (for drag & drop)

Optional format backends for broader support (install on demand):
- pip install imageio numpy
- pip install pillow-heif pillow-avif-plugin pillow-jxl-plugin
- pip install pydicom
- pip install astropy
- pip install rawpy

This GUI wraps functions from bitmap_svg_converter.py:
- open_image
- generate_svg_per_pixel

Files viewer resize changes:
- Replace ttk.Sizegrip in the files viewer with a custom vertical-only grip that resizes just the list area.
- Use a mirrored corner glyph “◣” so the icon faces the opposite way.
"""

import os
import threading
import socket
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image

try:
    from bitmap_svg_converter import (
        open_image,
        generate_svg_per_pixel,
    )
except Exception as e:
    raise RuntimeError(f"Failed to import bitmap_svg_converter. Make sure it is in PYTHONPATH. Error: {e}")

LAUNCHER_PORT = 51262

SMALL_MARGIN = 8
TOGGLE_ROW_SPACING = 12

PRIMARY_BG = "#0072B2"
PRIMARY_BG_ACTIVE = "#1f8ed6"
PRIMARY_FG = "#ffffff"

DARK_BG = "#1B1F23"
DARK_SURFACE = "#2D333B"
DARK_SURFACE_ALT = "#22272E"
DARK_TEXT = "#ADBAC7"
DARK_TEXT_MUTED = "#95A7B8"
DARK_BORDER = "#3A4149"
DARK_SCROLL_TROUGH = DARK_SURFACE

DROP_BG = "#3B4854"
DROP_BG_HOVER = "#465260"

SCROLLBAR_THICKNESS = 26

def _try_get_dnd():
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES  # type: ignore
        return TkinterDnD, DND_FILES
    except Exception:
        return None


def _norm_drop_path(raw: str) -> str:
    s = raw.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s.strip()


def system_pick_folder(title: str, parent_winid: int | None = None) -> str | None:
    if shutil.which("zenity"):
        args = ["zenity", "--file-selection", "--directory", f"--title={title}"]
        if parent_winid is not None:
            args.append(f"--attach={parent_winid}")
        p = subprocess.run(args, capture_output=True, text=True)
        return (p.stdout.strip() if p.returncode == 0 else "")
    if shutil.which("kdialog"):
        args = ["kdialog", "--getexistingdirectory", ".", f"--title={title}"]
        if parent_winid is not None:
            args.extend(["--attach", str(parent_winid)])
        p = subprocess.run(args, capture_output=True, text=True)
        return (p.stdout.strip() if p.returncode == 0 else "")
    return None


def system_pick_files(title: str, patterns: list[str], parent_winid: int | None = None) -> list[str] | None:
    if shutil.which("zenity"):
        filt = " ".join(patterns) if patterns else "*"
        args = [
            "zenity", "--file-selection", "--multiple", "--separator=\n",
            f"--title={title}", f"--file-filter={filt} | {filt}",
        ]
        if parent_winid is not None:
            args.append(f"--attach={parent_winid}")
        p = subprocess.run(args, capture_output=True, text=True)
        return ([s for s in p.stdout.splitlines() if s.strip()] if p.returncode == 0 else [])
    if shutil.which("kdialog"):
        filt = " ".join(patterns) if patterns else "*"
        args = ["kdialog", "--getopenfilename", ".", filt, "--multiple", "--separate-output", f"--title={title}"]
        if parent_winid is not None:
            args.extend(["--attach", str(parent_winid)])
        p = subprocess.run(args, capture_output=True, text=True)
        return ([s for s in p.stdout.splitlines() if s.strip()] if p.returncode == 0 else [])
    return None


def style_scrollbar(sb: tk.Scrollbar):
    """Apply dark theme colors to a classic tk.Scrollbar (best-effort across platforms)."""
    for opts in (
        dict(bg=DROP_BG, activebackground=DROP_BG_HOVER, troughcolor=DARK_SCROLL_TROUGH, highlightthickness=0, bd=0, relief="flat"),
        dict(bg=DROP_BG, activebackground=DROP_BG_HOVER, highlightthickness=0, bd=0, relief="flat"),
    ):
        try:
            sb.configure(**opts)
            break
        except Exception:
            pass
    for opt in ("arrowcolor",):
        try:
            sb.configure(**{opt: DARK_TEXT})
        except Exception:
            pass


# Expanded suffix list. Lower-case comparison is used.
BITMAP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp",
    ".tif", ".tiff", ".webp",
    ".avif", ".heif", ".heic", ".jxl",
    ".fits", ".fit", ".fts", ".dcm", ".exr", ".hdr", ".rgbe", ".pfm", ".sgi", ".rgb",
    ".dds", ".ktx", ".ktx2", ".astc", ".pvr",
    ".cr2", ".nef", ".arw", ".rw2", ".dng", ".orf", ".raf", ".sr2", ".pef", ".srw", ".rwl", ".nrw",
    ".3fr", ".fff", ".mef"
}

ANIMATED_SUFFIXES = {".gif", ".tif", ".tiff", ".webp", ".avif", ".heif", ".heic", ".jxl"}

def is_bitmap_file(p: Path) -> bool:
    return p.suffix.lower() in BITMAP_SUFFIXES


def find_bitmaps_in_folder(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    out: list[Path] = []
    for p in it:
        if not p.is_file():
            continue
        if is_bitmap_file(p):
            out.append(p)
    return sorted(out)


@dataclass
class JobResult:
    input_path: Path
    output_svg: Path
    ok: bool
    message: str


class ScrollableFrame(ttk.Frame):
    """Canvas-backed frame with vertical/horizontal scrolling."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg=DARK_BG, bd=0)
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview, width=SCROLLBAR_THICKNESS)
        self.hsb = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview, width=SCROLLBAR_THICKNESS)
        style_scrollbar(self.vsb)
        style_scrollbar(self.hsb)

        self.sizegrip = ttk.Sizegrip(self)
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")
        self.sizegrip.grid(row=1, column=1, sticky="se")

        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._clamp_view_if_no_scroll()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._win, width=max(event.width, self.inner.winfo_reqwidth()))
        self.canvas.coords(self._win, 0, 0)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._clamp_view_if_no_scroll()

    def _clamp_view_if_no_scroll(self):
        bbox = self.canvas.bbox("all")
        if not bbox:
            return
        content_w = bbox[2] - bbox[0]
        content_h = bbox[3] - bbox[1]
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        if content_h <= canvas_h:
            self.canvas.yview_moveto(0.0)
        if content_w <= canvas_w:
            self.canvas.xview_moveto(0.0)


class App:
    """GUI for bitmap → SVG conversion."""
    def _get_winid(self) -> int | None:
        try:
            self.root.update_idletasks()
            return int(self.root.winfo_id())
        except Exception:
            return None

    def _guard(self, fn):
        """Prevent opening new modals/widgets while one is already open."""
        def _wrapped(*args, **kwargs):
            if self._widget_open:
                return
            return fn(*args, **kwargs)
        return _wrapped

    def __init__(self, root: tk.Tk, dnd_files):
        self.root = root
        self.dnd_files = dnd_files

        self.root.title("Bitmap SVG Converter")
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1750, int(sw * 0.95))}x{min(1000, int(sh * 0.92))}")
        self.root.minsize(720, 480)

        self._setup_theme()
        self._setup_toggle_styles()
        self._setup_radio_toggle_styles()
        self._setup_progress_styles()
        self._setup_button_styles()
        self._setup_dark_theme()

        self.files: list[Path] = []

        # Settings
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "bitmap_svgs"))
        self.recursive = tk.BooleanVar(value=True)
        self.preserve_tree = tk.BooleanVar(value=True)

        self.rename_all = tk.BooleanVar(value=False)
        self.rename_base = tk.StringVar(value="")
        self.use_custom_stem = tk.BooleanVar(value=False)
        self.custom_stem = tk.StringVar(value="")

        # Status / progress
        self.status = tk.StringVar(value="Ready.")
        self.progress = tk.DoubleVar(value=0.0)
        self._pb_style_name = "Success.Horizontal.TProgressbar"
        self.pct_label: ttk.Label | None = None
        self.pct_places: int = 1  # number of decimal places in percent label (0..n)

        # Drag & drop panel colors
        self._drop_bg = DROP_BG
        self._drop_bg_hover = DROP_BG_HOVER

        # Modal lock
        self._widget_open = False

        # Control refs
        self.header_open_btn: ttk.Button | None = None
        self.run_btn: ttk.Button | None = None
        self.btn_add_files: ttk.Button | None = None
        self.btn_add_folder: ttk.Button | None = None
        self.btn_clear: ttk.Button | None = None
        self.btn_remove: ttk.Button | None = None
        self.btn_choose_out: ttk.Button | None = None
        self.btn_up: ttk.Button | None = None

        # Scrollbar refs and custom vertical grip for files viewer
        self.lb_vsb: tk.Scrollbar | None = None
        self.lb_hsb: tk.Scrollbar | None = None
        self.lb_grip: tk.Widget | None = None  # bottom-left vertical-only grip

        # Track current file index for display
        self._current_file_idx: int = 0

        # Vertical resize tracking for files viewer
        self._lb_resize_start_y: int = 0
        self._lb_row0_minsize: int = 0

        # Preview label
        self.naming_preview: ttk.Label | None = None

        self._build_ui()
        self._enable_dnd_if_available()
        self._bind_global_mousewheel()
        self._wire_preview_updates()
        self._bind_progress_percentage()
        self._update_naming_preview()

    def _setup_theme(self):
        try:
            style = ttk.Style(self.root)
            for t in ("clam", "alt", "default"):
                if t in style.theme_names():
                    style.theme_use(t)
                    break
        except Exception:
            pass

    def _setup_toggle_styles(self):
        self._tog_style = "OnOff.TCheckbutton"
        try:
            style = ttk.Style(self.root)
            style.layout(self._tog_style, [("Checkbutton.padding", {"sticky": "nswe", "children": [("Checkbutton.label", {"sticky": "nswe"})]})])
            style.configure(self._tog_style, padding=(10, 4))
            style.map(
                self._tog_style,
                background=[
                    ("active", "selected", "#00b980"),
                    ("active", "!selected", "#e07000"),
                    ("selected", "#009E73"),
                    ("!selected", "#D55E00"),
                ],
                foreground=[("selected", "#ffffff"), ("!selected", "#ffffff")],
                relief=[("active", "solid"), ("selected", "solid"), ("!selected", "solid")],
                borderwidth=[("selected", 1), ("!selected", 1)],
            )
        except Exception:
            pass

    def _setup_radio_toggle_styles(self):
        self._radio_style = "OnOff.TRadiobutton"
        try:
            style = ttk.Style(self.root)
            style.layout(self._radio_style, [("Radiobutton.padding", {"sticky": "nswe", "children": [("Radiobutton.label", {"sticky": "nswe"})]})])
            style.configure(self._radio_style, padding=(10, 4), indicatoron=False, borderwidth=1, relief="solid", foreground="#ffffff")
            style.map(
                self._radio_style,
                background=[
                    ("active", "selected", "#00b980"),
                    ("active", "!selected", "#e07000"),
                    ("selected", "#009E73"),
                    ("!selected", "#D55E00"),
                ],
                foreground=[("selected", "#ffffff"), ("!selected", "#ffffff")],
                relief=[("active", "solid"), ("selected", "solid"), ("!selected", "solid")],
            )
        except Exception:
            pass

    def _setup_progress_styles(self):
        style = ttk.Style(self.root)
        style.configure("Success.Horizontal.TProgressbar", troughcolor=DARK_SCROLL_TROUGH, background="#2ecc71", bordercolor=DARK_BORDER)
        style.configure("Fail.Horizontal.TProgressbar", troughcolor=DARK_SCROLL_TROUGH, background="#e74c3c", bordercolor=DARK_BORDER)

    def _setup_button_styles(self):
        try:
            style = ttk.Style(self.root)
            style.layout("Primary.TButton", [("Button.padding", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]})])
            style.configure("Primary.TButton", padding=(10, 4), foreground=PRIMARY_FG, background=PRIMARY_BG, borderwidth=1, relief="solid")
            style.map("Primary.TButton", background=[("active", PRIMARY_BG_ACTIVE), ("disabled", "#a0a0a0")], foreground=[("disabled", "#ffffff")], relief=[("pressed", "solid"), ("active", "solid")])
        except Exception:
            pass

    def _setup_dark_theme(self):
        try:
            self.root.configure(bg=DARK_BG)
            style = ttk.Style(self.root)
            style.configure("TFrame", background=DARK_BG, bordercolor=DARK_BORDER)
            style.configure("TLabelframe", background=DARK_BG, bordercolor=DARK_BORDER)
            style.configure("TLabelframe.Label", background=DARK_BG, foreground=DARK_TEXT)
            style.configure("TLabel", background=DARK_BG, foreground=DARK_TEXT, bordercolor=DARK_BORDER)
            style.configure("TButton", background=DARK_SURFACE, foreground=DARK_TEXT, borderwidth=1, relief="solid", bordercolor=DARK_BORDER)
            style.map("TButton", background=[("active", DARK_SURFACE_ALT), ("pressed", DARK_SURFACE_ALT)], foreground=[("disabled", DARK_TEXT_MUTED)])
            style.configure("TEntry", fieldbackground=DARK_SURFACE, foreground=DARK_TEXT, background=DARK_SURFACE, bordercolor=DARK_BORDER)
            style.configure("Readonly.TEntry", fieldbackground=DARK_SURFACE, foreground=DARK_TEXT, background=DARK_SURFACE, bordercolor=DARK_BORDER)
            style.configure("TSizegrip", background=DARK_BG, bordercolor=DARK_BORDER)
            style.map("TSizegrip", background=[("active", DARK_BG), ("pressed", DARK_BG)])
        except Exception:
            pass

    def _build_ui(self):
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)
        self._container = container

        header = ttk.Frame(container, padding=(10, 10, 10, 0))
        header.pack(fill="x")
        header.columnconfigure(1, weight=1)

        self.header_open_btn = ttk.Button(header, text="Open Pixel Tools Launcher", command=self._guard(self._open_tools_launcher))
        self.header_open_btn.grid(row=0, column=0, sticky="w", pady=(0, SMALL_MARGIN))

        self.pb = ttk.Progressbar(header, variable=self.progress, maximum=100.0, style=self._pb_style_name)
        self.pb.grid(row=0, column=1, sticky="ew", padx=(10, 10))
        # Percent overlay label centered on the progressbar, showing percent, current file counter
        self.pct_label = ttk.Label(header, text="0.0% • File 0/0", foreground="#ffffff")
        try:
            self.pct_label.place(in_=self.pb, relx=0.5, rely=0.5, anchor="center")
        except Exception:
            self.pct_label.grid(row=0, column=1, sticky="e", padx=(0, 10))

        self.run_btn = ttk.Button(header, text="Convert", command=self._guard(self.run), style="Primary.TButton")
        self.run_btn.grid(row=0, column=2, sticky="e", pady=(0, SMALL_MARGIN))

        status_hdr = ttk.Label(header, textvariable=self.status, anchor="center", justify="center")
        status_hdr.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, SMALL_MARGIN))

        sep_line = tk.Frame(container, height=2, bg=DARK_BORDER, bd=0, highlightthickness=0)
        sep_line.pack(fill="x")

        self.sc = ScrollableFrame(container)
        self.sc.pack(fill="both", expand=True)

        lb_wrap = ttk.Frame(self.sc.inner, padding=(10, 6, 10, 0))
        lb_wrap.pack(fill="both", expand=True, pady=(SMALL_MARGIN, SMALL_MARGIN))
        lb_wrap.rowconfigure(0, weight=1)
        lb_wrap.columnconfigure(1, weight=1)

        self.lb_vsb = tk.Scrollbar(lb_wrap, orient="vertical", width=SCROLLBAR_THICKNESS)
        style_scrollbar(self.lb_vsb)
        self.lb_vsb.grid(row=0, column=0, sticky="ns")

        self.listbox = tk.Listbox(
            lb_wrap,
            selectmode=tk.EXTENDED,
            bg=DARK_SURFACE,
            fg=DARK_TEXT,
            highlightthickness=0,
            bd=0,
            selectbackground=PRIMARY_BG_ACTIVE,
            selectforeground="#ffffff",
        )
        self.listbox.grid(row=0, column=1, sticky="nsew")

        self.lb_hsb = tk.Scrollbar(lb_wrap, orient="horizontal", width=SCROLLBAR_THICKNESS)
        style_scrollbar(self.lb_hsb)
        self.lb_hsb.grid(row=1, column=1, sticky="ew")

        # Bottom-left custom vertical-only grip for files viewer (mirrored icon)
        self.lb_grip = tk.Label(
            lb_wrap,
            text="◣",  # mirrored corner glyph for bottom-left
            bg=DARK_BG, fg=DARK_TEXT_MUTED,
            width=2,
            cursor="sb_v_double_arrow",
        )
        self.lb_grip.grid(row=1, column=0, sticky="sw")

        # Initialize vertical-only resizing for the list area
        self.root.update_idletasks()
        self._init_listbox_vertical_resizer(lb_wrap)

        self.listbox.configure(yscrollcommand=self.lb_vsb.set, xscrollcommand=self.lb_hsb.set)
        self.lb_vsb.configure(command=self.listbox.yview)
        self.lb_hsb.configure(command=self.listbox.xview)

        btns = ttk.Frame(self.sc.inner, padding=(10, 6, 10, 0))
        btns.pack(fill="x")
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        self.btn_clear = ttk.Button(btns, text="Clear All", command=self._guard(self.clear_list))
        self.btn_clear.grid(row=0, column=0, sticky="ew", pady=(0, SMALL_MARGIN))
        self.btn_remove = ttk.Button(btns, text="Remove Selected", command=self._guard(self.remove_selected))
        self.btn_remove.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=(0, SMALL_MARGIN))

        drop_wrap = ttk.Frame(self.sc.inner, padding=(10, 0, 10, 0))
        drop_wrap.pack(fill="x")
        self.drop_label = tk.Label(
            drop_wrap,
            text=("Drag and drop bitmap files or folders here.\n\n"
                  "Supported include: PNG, JPG/JPEG, GIF, BMP, TIFF/TIF, WebP,\n"
                  "AVIF, HEIF/HEIC, JXL, FITS, DICOM, OpenEXR, HDR/RGBE, PFM,\n"
                  "SGI RGB, DDS, KTX/KTX2, ASTC, PVR/RAW camera formats.\n\n"
                  "Tip: Drop a folder to add all bitmaps.\n"
                  "You can also drop a folder onto the Output folder field to set it."),
            justify="center", anchor="center", padx=12, pady=12,
            bd=0, relief="flat", bg=self._drop_bg, fg="#ffffff",
            highlightthickness=0, cursor="hand2",
        )
        self.drop_label.pack(fill="x", pady=(0, SMALL_MARGIN))
        self.drop_label.bind("<Enter>", lambda _e: self._set_drop_hover(True))
        self.drop_label.bind("<Leave>", lambda _e: self._set_drop_hover(False))

        add_wrap = ttk.Frame(self.sc.inner, padding=(10, 0, 10, 10))
        add_wrap.pack(fill="x")
        add_wrap.columnconfigure(0, weight=1)
        add_wrap.columnconfigure(1, weight=1)
        self.btn_add_files = ttk.Button(add_wrap, text="Add Bitmap Files", command=self._guard(self.add_files))
        self.btn_add_files.grid(row=0, column=0, sticky="ew", pady=(0, SMALL_MARGIN))
        self.btn_add_folder = ttk.Button(add_wrap, text="Add Folder", command=self._guard(self.add_folder))
        self.btn_add_folder.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=(0, SMALL_MARGIN))

        out = ttk.Frame(self.sc.inner, padding=(10, 0, 10, 10))
        out.pack(fill="x")
        out.columnconfigure(1, weight=1)

        ttk.Label(out, text="Output folder:").grid(row=0, column=0, sticky="w")
        self.out_entry = ttk.Entry(out, textvariable=self.output_dir)
        self.out_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.btn_up = ttk.Button(out, text="Up", command=self._guard(self.output_dir_up))
        self.btn_up.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        self.btn_choose_out = ttk.Button(out, text="Choose", command=self._guard(self.choose_output_dir))
        self.btn_choose_out.grid(row=0, column=3, sticky="ns", padx=(8, 0))

        opt_wrap = ttk.Frame(self.sc.inner, padding=(10, 6, 10, 10))
        opt_wrap.pack(fill="x", pady=(0, 10))
        ttk.Label(opt_wrap, text="Output options").pack(anchor="w")

        preserve_wrap = ttk.Frame(opt_wrap)
        preserve_wrap.pack(fill="x", pady=(TOGGLE_ROW_SPACING, 0))
        ttk.Checkbutton(preserve_wrap, text="Preserve folder structure", variable=self.preserve_tree, style=self._tog_style).pack(side="left")

        ren_wrap = ttk.Frame(opt_wrap)
        ren_wrap.pack(fill="x", pady=(TOGGLE_ROW_SPACING, 0))
        ttk.Checkbutton(ren_wrap, text="Custom name all", variable=self.rename_all, style=self._tog_style).pack(side="left")
        self.rename_entry = ttk.Entry(ren_wrap, textvariable=self.rename_base)
        self.rename_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

        stem_wrap = ttk.Frame(opt_wrap)
        stem_wrap.pack(fill="x", pady=(TOGGLE_ROW_SPACING, 0))
        ttk.Checkbutton(stem_wrap, text="Custom stem all", variable=self.use_custom_stem, style=self._tog_style).pack(side="left")
        self.stem_entry = ttk.Entry(stem_wrap, textvariable=self.custom_stem)
        self.stem_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.naming_preview = ttk.Label(opt_wrap, text="", foreground=DARK_TEXT_MUTED, padding=(0, 4))
        self.naming_preview.pack(fill="x")

    def _init_listbox_vertical_resizer(self, lb_wrap: ttk.Frame):
        """Initialize vertical-only resizing of the list area via a custom grip."""
        initial = max(180, self.listbox.winfo_reqheight())
        lb_wrap.rowconfigure(0, minsize=initial)
        self._lb_row0_minsize = initial

        if self.lb_grip is not None:
            self.lb_grip.bind("<Button-1>", lambda e: self._start_lb_resize(e))
            self.lb_grip.bind("<B1-Motion>", lambda e, wrap=lb_wrap: self._perform_lb_resize(e, wrap))
            self.lb_grip.bind("<ButtonRelease-1>", lambda e, wrap=lb_wrap: self._end_lb_resize(e, wrap))

    def _start_lb_resize(self, event):
        self._lb_resize_start_y = event.y_root

    def _perform_lb_resize(self, event, lb_wrap: ttk.Frame):
        dy = event.y_root - self._lb_resize_start_y
        newsize = max(120, self._lb_row0_minsize + dy)
        try:
            lb_wrap.rowconfigure(0, minsize=int(newsize))
        except Exception:
            pass

    def _end_lb_resize(self, event, lb_wrap: ttk.Frame):
        try:
            current_h = max(120, self.listbox.winfo_height())
            self._lb_row0_minsize = current_h
        except Exception:
            pass

    def _raise_existing_launcher(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", LAUNCHER_PORT), timeout=0.5) as s:
                s.sendall(b"RAISE\n")
            return True
        except Exception:
            return False

    def _open_tools_launcher(self):
        if self._raise_existing_launcher():
            return
        try:
            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable], close_fds=True)
            else:
                script = Path(__file__).resolve().parent / "vector_pixel_tools_launcher.py"
                if not script.exists():
                    messagebox.showerror("Launcher not found", f"Could not find:\n{script}")
                    return
                subprocess.Popen([sys.executable, str(script)], close_fds=True)
        except Exception as e:
            messagebox.showerror("Open failed", f"Could not open tools launcher:\n{e}")

    def _set_drop_hover(self, on: bool) -> None:
        if hasattr(self, "drop_label"):
            self.drop_label.configure(bg=self._drop_bg_hover if on else self._drop_bg, relief="flat")

    def _bind_global_mousewheel(self):
        self.root.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_global_mousewheel_linux, add="+")
        self.root.bind_all("<Button-5>", self._on_global_mousewheel_linux, add="+")
        self.listbox.bind("<Enter>", lambda _e: self.listbox.focus_set())

    def _on_global_mousewheel(self, event):
        steps = int(-1 * (event.delta / 120)) if event.delta else 0
        target = self.listbox if self._event_over_widget(self.listbox, event) else self.sc.canvas
        try:
            target.yview_scroll(steps, "units")
        except Exception:
            pass

    def _on_global_mousewheel_linux(self, event):
        steps = -1 if getattr(event, "num", None) == 4 else 1
        target = self.listbox if self._event_over_widget(self.listbox, event) else self.sc.canvas
        try:
            target.yview_scroll(steps, "units")
        except Exception:
            pass

    def _event_over_widget(self, widget: tk.Widget, event) -> bool:
        try:
            x0, y0 = widget.winfo_rootx(), widget.winfo_rooty()
            x1, y1 = x0 + widget.winfo_width(), y0 + widget.winfo_height()
            return (x0 <= event.x_root <= x1) and (y0 <= event.y_root <= y1)
        except Exception:
            return False

    def _enable_dnd_if_available(self):
        if self.dnd_files is None:
            self.status.set("Ready. (Drag & drop disabled: install tkinterdnd2 to enable.)")
            return

        for widget in (self.listbox, self.drop_label, self.out_entry):
            try:
                widget.drop_target_register(self.dnd_files)
            except Exception:
                pass

        for widget, handler in ((self.listbox, self._on_drop_inputs), (self.drop_label, self._on_drop_inputs), (self.out_entry, self._on_drop_output_dir)):
            try:
                widget.dnd_bind("<<Drop>>", handler)
            except Exception:
                pass

        self.status.set("Ready.")

    def _drop_paths(self, event) -> list[Path]:
        raw_list = self.root.tk.splitlist(event.data)
        paths: list[Path] = []
        for raw in raw_list:
            s = _norm_drop_path(str(raw))
            if s:
                paths.append(Path(s))
        return paths

    def _on_drop_inputs(self, event):
        self._handle_input_paths(self._drop_paths(event))
        return "break"

    def _on_drop_output_dir(self, event):
        for p in self._drop_paths(event):
            p = p.expanduser()
            if p.is_dir():
                self.output_dir.set(str(p.resolve()))
                self.status.set("Output folder set via drag & drop.")
                break
        return "break"

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for p in self.files:
            self.listbox.insert(tk.END, str(p))
        self.status.set(f"{len(self.files)} file(s) queued.")
        self._update_naming_preview()
        self._update_pct_label()

    def _add_paths(self, paths: list[Path]):
        existing = set(self.files)
        added = 0

        for p in paths:
            try:
                p = p.expanduser().resolve()
            except Exception:
                p = p.expanduser()

            if not is_bitmap_file(p):
                continue
            if not p.is_file():
                continue
            if p in existing:
                continue

            self.files.append(p)
            existing.add(p)
            added += 1

        if added:
            self.files.sort()
        self._refresh_listbox()

    def _handle_input_paths(self, paths: list[Path]):
        files: list[Path] = []
        folders: list[Path] = []

        for p in paths:
            p = p.expanduser()
            if p.is_dir():
                folders.append(p)
            elif p.is_file():
                files.append(p)

        for folder in folders:
            imgs = find_bitmaps_in_folder(folder, recursive=self.recursive.get())
            self._add_paths(imgs)

        if files:
            self._add_paths(files)

    def _set_controls_enabled(self, enabled: bool):
        widgets = [
            self.header_open_btn,
            self.run_btn,
            self.btn_add_files,
            self.btn_add_folder,
            self.btn_clear,
            self.btn_remove,
            self.btn_choose_out,
            self.btn_up,
        ]
        state = "normal" if enabled else "disabled"
        for w in widgets:
            try:
                if w is not None:
                    w.configure(state=state)
            except Exception:
                pass

    def _with_modal_lock(self, fn):
        """Disable actionable controls while a picker or modal is open; prevent opening additional modals."""
        if self._widget_open:
            return None
        self._widget_open = True
        self._set_controls_enabled(False)
        try:
            return fn()
        finally:
            self._widget_open = False
            self._set_controls_enabled(True)

    def add_files(self):
        def do_pick():
            patterns = [
                "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp",
                "*.tif", "*.tiff", "*.webp",
                "*.avif", "*.heif", "*.heic", "*.jxl",
                "*.fits", "*.fit", "*.fts", "*.dcm", "*.exr", "*.hdr", "*.rgbe", "*.pfm", "*.sgi", "*.rgb",
                "*.dds", "*.ktx", "*.ktx2", "*.astc", "*.pvr",
                "*.cr2", "*.nef", "*.arw", "*.rw2", "*.dng", "*.orf", "*.raf", "*.sr2", "*.pef", "*.srw", "*.rwl", "*.nrw",
                "*.3fr", "*.fff", "*.mef",
            ]
            picked = system_pick_files("Select bitmap files", patterns=patterns, parent_winid=self._get_winid())
            if picked is None:
                picked = list(
                    filedialog.askopenfilenames(
                        title="Select bitmap files",
                        filetypes=[
                            ("Bitmap images", " ".join(patterns)),
                            ("PNG", "*.png"),
                            ("JPEG", "*.jpg *.jpeg"),
                            ("GIF", "*.gif"),
                            ("BMP", "*.bmp"),
                            ("TIFF", "*.tif *.tiff"),
                            ("WebP", "*.webp"),
                            ("AVIF", "*.avif"),
                            ("HEIF/HEIC", "*.heif *.heic"),
                            ("JPEG XL", "*.jxl"),
                            ("FITS", "*.fits *.fit *.fts"),
                            ("DICOM", "*.dcm"),
                            ("OpenEXR", "*.exr"),
                            ("HDR/RGBE", "*.hdr *.rgbe"),
                            ("PFM", "*.pfm"),
                            ("SGI RGB", "*.sgi *.rgb"),
                            ("DDS", "*.dds"),
                            ("KTX/KTX2", "*.ktx *.ktx2"),
                            ("ASTC", "*.astc"),
                            ("PVR (PVRTC)", "*.pvr"),
                            ("RAW camera", "*.cr2 *.nef *.arw *.rw2 *.dng *.orf *.raf *.sr2 *.pef *.srw *.rwl *.nrw *.3fr *.fff *.mef"),
                            ("All files", "*.*"),
                        ],
                        parent=self.root,
                    )
                )
            elif picked == []:
                return None
            return [Path(f) for f in picked] if picked else None

        res = self._with_modal_lock(do_pick)
        if res:
            self._add_paths(res)

    def add_folder(self):
        def do_pick():
            folder = system_pick_folder("Select a folder containing bitmaps", parent_winid=self._get_winid())
            if folder is None:
                folder = filedialog.askdirectory(title="Select a folder containing bitmaps", parent=self.root)
            elif folder == "":
                return None
            return folder or None

        folder = self._with_modal_lock(do_pick)
        if not folder:
            return

        imgs = find_bitmaps_in_folder(Path(folder), recursive=self.recursive.get())
        if not imgs:
            messagebox.showinfo("No bitmaps found", "No supported bitmap files were found in the selected folder.", parent=self.root)
            return
        self._add_paths(imgs)

    def choose_output_dir(self):
        def do_pick():
            folder = system_pick_folder("Choose output folder", parent_winid=self._get_winid())
            if folder is None:
                folder = filedialog.askdirectory(title="Choose output folder", parent=self.root)
            elif folder == "":
                return None
            return folder or None

        folder = self._with_modal_lock(do_pick)
        if folder:
            self.output_dir.set(str(Path(folder)))
            self.status.set("Output folder set.")

    def output_dir_up(self):
        try:
            cur = Path(self.output_dir.get()).expanduser().resolve()
            parent = cur.parent if cur.parent != cur else cur
            self.output_dir.set(str(parent))
            self.status.set("Output folder moved up one level.")
        except Exception:
            pass

    def clear_list(self):
        self.files = []
        self._refresh_listbox()

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        sel_set = set(sel)
        self.files = [p for i, p in enumerate(self.files) if i not in sel_set]
        self._refresh_listbox()

    def _compute_output_stem(self, inp: Path, file_index_zero_based: int) -> str:
        """
        Compute output stem based on rename/stem settings.

        Behavior:
        - If "Custom name all" is enabled AND a base name is provided, use that base,
          optionally appending the custom stem, and add the sequential _00000 index.
        - Otherwise, use the file's original stem, optionally appending the custom stem.
        """
        cust = self.custom_stem.get().strip()
        if self.rename_all.get() and self.rename_base.get().strip():
            base = self.rename_base.get().strip()
            stem = f"{base}_{cust}" if (self.use_custom_stem.get() and cust) else base
            return f"{stem}_{file_index_zero_based:05d}"
        stem = inp.stem
        return f"{stem}_{cust}" if (self.use_custom_stem.get() and cust) else stem

    def _update_naming_preview(self) -> None:
        if not hasattr(self, "naming_preview"):
            return
        lines: list[str] = []
        lines.append("Multi-frame formats (GIF/TIFF/WebP/AVIF/HEIF/JXL) append a per-frame suffix: _frame_00000, _frame_00001, etc.")
        if not self.files:
            lines.append("Preview: No files queued.")
        else:
            to_show = min(3, len(self.files))
            for i in range(to_show):
                p = self.files[i]
                stem = self._compute_output_stem(p, i)
                ext = p.suffix.lower()
                try:
                    frames = 1
                    with Image.open(str(p)) as im_info:
                        frames = max(1, getattr(im_info, "n_frames", 1))
                    if frames > 1 or ext in ANIMATED_SUFFIXES:
                        lines.append(f"Preview: {p.name} -> {stem}_frame_00000.svg, {stem}_frame_00001.svg, …")
                    else:
                        lines.append(f"Preview: {p.name} -> {stem}.svg")
                except Exception:
                    if ext in ANIMATED_SUFFIXES:
                        lines.append(f"Preview: {p.name} -> {stem}_frame_00000.svg, {stem}_frame_00001.svg, …")
                    else:
                        lines.append(f"Preview: {p.name} -> {stem}.svg")
        self.naming_preview.configure(text="\n".join(lines))

    def _wire_preview_updates(self):
        for var in (self.rename_all, self.rename_base, self.use_custom_stem, self.custom_stem, self.preserve_tree):
            try:
                var.trace_add("write", lambda *_: self._update_naming_preview())
            except Exception:
                pass

    def _update_pct_label(self, value: float | None = None) -> None:
        if not hasattr(self, "pct_label") or self.pct_label is None:
            return
        try:
            val = float(self.progress.get()) if value is None else float(value)
            val = max(0.0, min(100.0, val))
            total_files = len(self.files)
            cur_idx = self._current_file_idx if total_files > 0 else 0
            cur_idx = max(0, min(cur_idx, total_files))
            self.pct_label.configure(text=f"{val:.{self.pct_places}f}% • File {cur_idx}/{total_files}")
        except Exception:
            pass

    def _bind_progress_percentage(self) -> None:
        try:
            self.progress.trace_add("write", lambda *_: self._update_pct_label())
        except Exception:
            pass
        self._update_pct_label(0.0)

    def run(self):
        if not self.files:
            messagebox.showwarning("Nothing to do", "Add at least one bitmap file.", parent=self.root)
            return
        out_dir = Path(self.output_dir.get()).expanduser()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Output folder error", f"Cannot create output folder:\n{e}", parent=self.root)
            return
        self.run_btn.configure(state="disabled")
        self.progress.set(0.0)
        self._set_progress_style(success=True)
        self.status.set("Converting...")

        threading.Thread(target=self._run_worker, args=(out_dir,), daemon=True).start()

    def _run_worker(self, out_dir: Path):
        preserve_tree = self.preserve_tree.get()

        common_root = None
        if preserve_tree:
            try:
                common_root = Path(os.path.commonpath([str(p.parent) for p in self.files]))
            except Exception:
                common_root = None

        results: list[JobResult] = []
        total_files = len(self.files)

        for idx1, inp in enumerate(self.files, start=1):
            # Update current file index for overlay
            self._current_file_idx = idx1
            self.root.after(0, self._update_pct_label)

            file_index = idx1 - 1
            stem_base = self._compute_output_stem(inp, file_index)

            frames = 1
            try:
                with Image.open(str(inp)) as im_info:
                    frames = max(1, getattr(im_info, "n_frames", 1))
            except Exception:
                frames = 1

            # Base and span for this file within overall progress
            base_pct_for_prev_files = (file_index / max(1, total_files)) * 100.0
            per_file_span = 100.0 / max(1, total_files)

            try:
                for frame_idx in range(frames):
                    stem = stem_base + (f"_frame_{frame_idx:05d}" if frames > 1 else "")
                    out_svg = out_dir / (stem + ".svg")
                    if preserve_tree and common_root is not None:
                        try:
                            rel_parent = inp.parent.relative_to(common_root)
                            out_svg = out_dir / rel_parent / out_svg.name
                        except Exception:
                            pass

                    out_svg.parent.mkdir(parents=True, exist_ok=True)
                    im = open_image(str(inp), frame_index=frame_idx)
                    svg_id = stem  # use final stem as SVG id

                    # Per-row callback to advance progress smoothly and update percent label
                    def _row_cb(rows_done: int, total_rows: int) -> None:
                        try:
                            inner = (frame_idx + (rows_done / max(1, total_rows))) / max(1, frames)
                            pct = base_pct_for_prev_files + inner * per_file_span
                            self.root.after(0, self.progress.set, pct)
                            self.root.after(0, self._update_pct_label, pct)
                        except Exception:
                            pass

                    generate_svg_per_pixel(im, str(out_svg), svg_id, scale=1, progress_cb=_row_cb)
                    msg = "OK | mode=per-pixel"
                    results.append(JobResult(inp, out_svg, True, msg))

            except Exception as e:
                out_svg_fail = out_dir / (stem_base + ".svg")
                results.append(JobResult(inp, out_svg_fail, False, str(e)))
                self.root.after(0, self._set_progress_style, False)

            # Snap to end of this file’s span and refresh overlay text/state
            pct_done = (idx1 / max(1, total_files)) * 100.0
            self.root.after(0, self.progress.set, pct_done)
            self.root.after(0, self._update_pct_label, pct_done)
            self.root.after(0, self.status.set, f"Processing {idx1}/{total_files}: {inp.name}")

        self.root.after(0, self._finish, results, out_dir)

    def _set_progress_style(self, success: bool):
        self._pb_style_name = "Success.Horizontal.TProgressbar" if success else "Fail.Horizontal.TProgressbar"
        try:
            self.pb.configure(style=self._pb_style_name)
        except Exception:
            pass

    def _finish(self, results: list[JobResult], out_dir: Path):
        ok = sum(1 for r in results if r.ok)
        fail = len(results) - ok

        log_path = out_dir / "bitmap_to_svg_log.txt"
        lines = []
        for r in results:
            lines.append(f"{'OK  ' if r.ok else 'FAIL'} | {r.input_path} -> {r.output_svg} | {r.message}")

        try:
            log_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

        self.run_btn.configure(state="normal")
        self.progress.set(100.0)
        self._set_progress_style(success=(fail == 0))
        self.status.set(f"Done. OK: {ok}, Failed: {fail}. Log: {log_path}")
        # Reset overlay to final counts
        self._current_file_idx = len(self.files)
        self._update_pct_label(100.0)

        self._show_completion_dialog(results, out_dir, log_path)

    def _show_completion_dialog(self, results: list[JobResult], out_dir: Path, log_path: Path):
        dlg = tk.Toplevel(self.root)
        dlg.title("Conversion Summary")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("+%d+%d" % (self.root.winfo_rootx() + 40, self.root.winfo_rooty() + 40))

        self._widget_open = True
        self._set_controls_enabled(False)

        def on_close():
            try:
                dlg.grab_release()
            except Exception:
                pass
            try:
                dlg.destroy()
            except Exception:
                pass
            self._widget_open = False
            self._set_controls_enabled(True)

        dlg.protocol("WM_DELETE_WINDOW", on_close)

        main = ttk.Frame(dlg, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        dlg.rowconfigure(0, weight=1)
        dlg.columnconfigure(0, weight=1)

        ok = sum(1 for r in results if r.ok)
        fail = len(results) - ok
        hdr = ttk.Label(main, text=f"Completed | OK: {ok} | Failed: {fail}", font=("TkDefaultFont", 11, "bold"))
        hdr.grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(main, text="Output folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        out_entry = ttk.Entry(main)
        out_entry.insert(0, str(out_dir))
        out_entry.configure(state="readonly")
        out_entry.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(main, text="Open", command=lambda: self._open_path(out_dir)).grid(row=1, column=2, sticky="e", pady=(0, SMALL_MARGIN))

        ttk.Label(main, text="Log file:").grid(row=2, column=0, sticky="w")
        log_entry = ttk.Entry(main)
        log_entry.insert(0, str(log_path))
        log_entry.configure(state="readonly")
        log_entry.grid(row=2, column=1, sticky="ew")
        ttk.Button(main, text="Open", command=lambda: self._open_path(log_path)).grid(row=2, column=2, sticky="e", pady=(0, SMALL_MARGIN))

        main.columnconfigure(1, weight=1)

        details = ttk.Frame(main)
        details.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        main.rowconfigure(3, weight=1)

        txt = tk.Text(details, wrap="word", height=12, bg=DARK_SURFACE, fg=DARK_TEXT, insertbackground=DARK_TEXT, highlightthickness=0, bd=0)
        txt_scroll = tk.Scrollbar(details, orient="vertical", command=txt.yview, width=SCROLLBAR_THICKNESS)
        style_scrollbar(txt_scroll)
        txt.configure(yscrollcommand=txt_scroll.set)
        txt.grid(row=0, column=0, sticky="nsew")
        txt_scroll.grid(row=0, column=1, sticky="ns")
        details.rowconfigure(0, weight=1)
        details.columnconfigure(0, weight=1)

        txt.insert("1.0", "\n".join([
            f"{'✓' if r.ok else '✗'} {r.input_path.name} -> {r.output_svg.name} | {r.message}"
            for r in results
        ]))
        txt.configure(state="disabled")

        txt.bind("<MouseWheel>", lambda e: txt.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        txt.bind("<Button-4>", lambda e: txt.yview_scroll(-1, "units"))
        txt.bind("<Button-5>", lambda e: txt.yview_scroll(1, "units"))

        btns = ttk.Frame(main)
        btns.grid(row=4, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Close", command=on_close).pack(side="right", pady=(0, SMALL_MARGIN))

    def _open_path(self, p: Path):
        try:
            if sys.platform == "win32":
                os.startfile(str(p))  # type: ignore
            elif sys.platform == "darwin":
                subprocess.run(["open", str(p)], check=False)
            else:
                subprocess.run(["xdg-open", str(p)], check=False)
        except Exception:
            messagebox.showerror("Open failed", f"Could not open:\n{p}", parent=self.root)


def main():
    dnd = _try_get_dnd()
    if dnd is None:
        root = tk.Tk()
        App(root, None)
    else:
        TkinterDnD, DND_FILES = dnd
        root = TkinterDnD.Tk()
        App(root, DND_FILES)
    root.mainloop()


if __name__ == "__main__":
    main()
