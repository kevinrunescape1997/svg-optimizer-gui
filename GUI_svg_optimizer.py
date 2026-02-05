#!/usr/bin/env python3
from __future__ import annotations

# Owner: kevinrunescape1997
# Purpose: GUI for SVG optimization; preserves streaming and progress behavior.

import os
import shutil
import subprocess
import threading
import socket
from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pixel_svg_optimizer import (
    optimize_svg_rects_bytes,
    write_svgz,
    optimize_svg_paths_bytes,
    optimize_svg_rects_stream,           # Streaming rect optimizer
    write_svgz_stream_from_svg,          # Streaming SVGZ writer
    LARGE_BYTES,                         # Threshold: 200 MB
)

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


def is_optimized_output(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(".svg") and "_optimized" in p.stem.lower()


def find_svgs_in_folder(folder: Path, recursive: bool, skip_outputs: bool) -> list[Path]:
    it = folder.rglob("*.svg") if recursive else folder.glob("*.svg")
    out: list[Path] = []
    for p in it:
        if not p.is_file():
            continue
        if skip_outputs and is_optimized_output(p):
            continue
        out.append(p)
    return sorted(out)


@dataclass
class JobResult:
    input_path: Path
    output_svg: Path
    output_svgz: Path | None
    ok: bool
    message: str


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


class ScrollableFrame(ttk.Frame):
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
    """SVG optimization GUI."""

    def _get_winid(self) -> int | None:
        try:
            self.root.update_idletasks()
            return int(self.root.winfo_id())
        except Exception:
            return None

    def _guard(self, fn):
        def _wrapped(*args, **kwargs):
            if self._widget_open:
                return
            return fn(*args, **kwargs)
        return _wrapped

    def __init__(self, root: tk.Tk, dnd_files):
        self.root = root
        self.dnd_files = dnd_files

        self.root.title("SVG Pixel Optimizer")
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

        self.output_dir = tk.StringVar(value=str(Path.cwd() / "optimized_svgs"))
        self.recursive = tk.BooleanVar(value=True)
        self.preserve_tree = tk.BooleanVar(value=True)
        self.rename_all = tk.BooleanVar(value=False)
        self.rename_base = tk.StringVar(value="")
        self.use_custom_stem = tk.BooleanVar(value=False)
        self.custom_stem = tk.StringVar(value="")
        self.skip_outputs = tk.BooleanVar(value=True)
        self.output_mode = tk.StringVar(value="svg")
        self.use_paths = tk.BooleanVar(value=True)
        self.minify = tk.BooleanVar(value=True)

        self.status = tk.StringVar(value="Ready.")
        self.progress = tk.DoubleVar(value=0.0)
        self._pb_style_name = "Success.Horizontal.TProgressbar"
        self.pct_label: ttk.Label | None = None
        self.pct_places: int = 1

        self._current_file_idx: int = 0

        self._drop_bg = DROP_BG
        self._drop_bg_hover = DROP_BG_HOVER

        self._widget_open = False

        self.header_open_btn: ttk.Button | None = None
        self.run_btn: ttk.Button | None = None
        self.btn_add_files: ttk.Button | None = None
        self.btn_add_folder: ttk.Button | None = None
        self.btn_clear: ttk.Button | None = None
        self.btn_remove: ttk.Button | None = None
        self.btn_choose_out: ttk.Button | None = None
        self.btn_up: ttk.Button | None = None

        self.lb_vsb: tk.Scrollbar | None = None
        self.lb_hsb: tk.Scrollbar | None = None
        self.lb_grip: tk.Widget | None = None

        self._lb_resize_start_y: int = 0
        self._lb_row0_minsize: int = 0

        self.cb_use_paths: ttk.Checkbutton | None = None

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
        self.pct_label = ttk.Label(header, text="0.0% • File 0/0", foreground="#ffffff")
        try:
            self.pct_label.place(in_=self.pb, relx=0.5, rely=0.5, anchor="center")
        except Exception:
            self.pct_label.grid(row=0, column=1, sticky="e", padx=(0, 10))

        self.run_btn = ttk.Button(header, text="Optimize", command=self._guard(self.run), style="Primary.TButton")
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

        self.lb_grip = tk.Label(
            lb_wrap,
            text="◣",
            bg=DARK_BG, fg=DARK_TEXT_MUTED,
            width=2,
            cursor="sb_v_double_arrow",
        )
        self.lb_grip.grid(row=1, column=0, sticky="sw")

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
            text=("Drag and drop SVG files or folders here.\n\n\n\n\n\n"
                  "Tip: Drop a folder to add all SVGs.\n"
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
        self.btn_add_files = ttk.Button(add_wrap, text="Add SVG Files", command=self._guard(self.add_files))
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

        fmt_wrap = ttk.Frame(self.sc.inner, padding=(10, 6, 10, 10))
        fmt_wrap.pack(fill="x", pady=(0, 10))
        ttk.Label(fmt_wrap, text="Output options").pack(anchor="w")

        fmt = ttk.Frame(fmt_wrap)
        fmt.pack(fill="x", pady=(TOGGLE_ROW_SPACING, 0))
        self.cb_use_paths = ttk.Checkbutton(fmt_wrap, text="Merge touching like-pixels into connected paths", variable=self.use_paths, style="OnOff.TCheckbutton")
        ttk.Radiobutton(fmt, text="Optimized SVG (.svg)", value="svg", variable=self.output_mode, style="OnOff.TRadiobutton").pack(side="left")
        ttk.Radiobutton(fmt, text="Optimized SVGZ (.svgz)", value="svgz_only", variable=self.output_mode, style="OnOff.TRadiobutton").pack(side="left", padx=(8, 0))
        ttk.Radiobutton(fmt, text="Optimized SVG (.svg) + SVGZ (.svgz)", value="svgz", variable=self.output_mode, style="OnOff.TRadiobutton").pack(side="left", padx=(8, 0))

        self.cb_use_paths.pack(anchor="w", pady=(TOGGLE_ROW_SPACING, 0))
        ttk.Checkbutton(fmt_wrap, text="Post-process to merge same-color shapes and minify", variable=self.minify, style="OnOff.TCheckbutton").pack(anchor="w", pady=(TOGGLE_ROW_SPACING, 0))

        preserve_wrap = ttk.Frame(fmt_wrap)
        preserve_wrap.pack(fill="x", pady=(TOGGLE_ROW_SPACING, 0))
        ttk.Checkbutton(preserve_wrap, text="Preserve folder structure", variable=self.preserve_tree, style="OnOff.TCheckbutton").pack(side="left")

        ren_wrap = ttk.Frame(fmt_wrap)
        ren_wrap.pack(fill="x", pady=(TOGGLE_ROW_SPACING, 0))
        ttk.Checkbutton(ren_wrap, text="Custom name all", variable=self.rename_all, style="OnOff.TCheckbutton").pack(side="left")
        self.rename_entry = ttk.Entry(ren_wrap, textvariable=self.rename_base)
        self.rename_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

        stem_wrap = ttk.Frame(fmt_wrap)
        stem_wrap.pack(fill="x", pady=(TOGGLE_ROW_SPACING, 0))
        ttk.Checkbutton(stem_wrap, text="Custom stem all", variable=self.use_custom_stem, style="OnOff.TCheckbutton").pack(side="left")
        self.stem_entry = ttk.Entry(stem_wrap, textvariable=self.custom_stem)
        self.stem_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.naming_preview = ttk.Label(fmt_wrap, text="", foreground=DARK_TEXT_MUTED, padding=(0, 4))
        self.naming_preview.pack(fill="x")

    def _init_listbox_vertical_resizer(self, lb_wrap: ttk.Frame):
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

            if p.suffix.lower() != ".svg":
                continue
            if not p.is_file():
                continue
            if p in existing:
                continue
            if self.skip_outputs.get() and is_optimized_output(p):
                continue

            self.files.append(p)
            existing.add(p)
            added += 1

        if added:
            self.files.sort()
        self._refresh_listbox()
        self._enforce_large_file_policy()

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
            svgs = find_svgs_in_folder(folder, recursive=self.recursive.get(), skip_outputs=self.skip_outputs.get())
            self._add_paths(svgs)

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
            self.cb_use_paths,
        ]
        state = "normal" if enabled else "disabled"
        for w in widgets:
            try:
                if w is not None:
                    w.configure(state=state)
            except Exception:
                pass

    def _with_modal_lock(self, fn):
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
            picked = system_pick_files("Select SVG files", patterns=["*.svg"], parent_winid=self._get_winid())
            if picked is None:
                picked = list(
                    filedialog.askopenfilenames(
                        title="Select SVG files",
                        filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
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
            folder = system_pick_folder("Select a folder containing SVGs", parent_winid=self._get_winid())
            if folder is None:
                folder = filedialog.askdirectory(title="Select a folder containing SVGs", parent=self.root)
            elif folder == "":
                return None
            return folder or None

        folder = self._with_modal_lock(do_pick)
        if not folder:
            return

        svgs = find_svgs_in_folder(Path(folder), recursive=self.recursive.get(), skip_outputs=self.skip_outputs.get())
        if not svgs:
            messagebox.showinfo("No SVGs found", "No .svg files were found in the selected folder.")
            return
        self._add_paths(svgs)

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
        self._enforce_large_file_policy()

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        sel_set = set(sel)
        self.files = [p for i, p in enumerate(self.files) if i not in sel_set]
        self._refresh_listbox()
        self._enforce_large_file_policy()

    def _compute_output_stem(self, inp: Path, file_index_zero_based: int) -> str:
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

        mode = self.output_mode.get()
        total_files = len(self.files)
        if not self.files:
            lines.append("Preview: No files queued.")
        else:
            to_show = min(3, total_files)
            for i in range(to_show):
                p = self.files[i]
                stem = self._compute_output_stem(p, i)
                if mode == "svg":
                    names = [stem + ".svg"]
                elif mode == "svgz":
                    names = [stem + ".svg", stem + ".svgz"]
                elif mode == "svgz_only":
                    names = [stem + ".svgz"]
                else:
                    names = [stem + ".svg"]
                lines.append(f"Preview: {p.name} -> {', '.join(names)}")

        self.naming_preview.configure(text="\n".join(lines))

    def _wire_preview_updates(self):
        for var in (self.rename_all, self.rename_base, self.use_custom_stem, self.custom_stem, self.output_mode, self.preserve_tree, self.minify, self.use_paths):
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
            cur = max(0, min(total_files, self._current_file_idx))
            self.pct_label.configure(text=f"{val:.{self.pct_places}f}% • File {cur}/{total_files}")
        except Exception:
            pass

    def _bind_progress_percentage(self) -> None:
        try:
            self.progress.trace_add("write", lambda *_: self._update_pct_label())
        except Exception:
            pass
        self._update_pct_label(0.0)

    def _enforce_large_file_policy(self) -> None:
        any_large = False
        for p in self.files:
            try:
                if p.stat().st_size >= LARGE_BYTES:
                    any_large = True
                    break
            except Exception:
                continue

        try:
            if any_large:
                self.use_paths.set(False)
                if self.cb_use_paths is not None:
                    self.cb_use_paths.configure(state="disabled")
                self.status.set("Large file detected (>200 MB): switched to rects mode (paths disabled).")
            else:
                if self.cb_use_paths is not None:
                    self.cb_use_paths.configure(state="normal")
        except Exception:
            pass

    def run(self):
        if not self.files:
            messagebox.showwarning("Nothing to do", "Add at least one SVG file.")
            return

        out_dir = Path(self.output_dir.get()).expanduser()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Output folder error", f"Cannot create output folder:\n{e}")
            return

        self.run_btn.configure(state="disabled")
        self.progress.set(0.0)
        self._set_progress_style(success=True)
        self.status.set("Running...")

        threading.Thread(target=self._run_worker, args=(out_dir,), daemon=True).start()

    def _run_worker(self, out_dir: Path):
        vertical = True
        preserve_tree = self.preserve_tree.get()
        mode = self.output_mode.get()
        write_svg_flag = mode in ("svg", "svgz")
        write_svgz_flag = mode in ("svgz", "svgz_only")
        use_paths = self.use_paths.get()
        minify = self.minify.get()
        use_zopfli = True

        common_root = None
        if preserve_tree:
            try:
                common_root = Path(os.path.commonpath([str(p.parent) for p in self.files]))
            except Exception:
                common_root = None

        results: list[JobResult] = []
        total = len(self.files)

        for idx1, inp in enumerate(self.files, start=1):
            self._current_file_idx = idx1
            file_index = idx1 - 1
            stem = self._compute_output_stem(inp, file_index)
            out_svg = (out_dir / (stem + ".svg"))
            if preserve_tree and common_root is not None:
                try:
                    rel_parent = inp.parent.relative_to(common_root)
                    out_svg = out_dir / rel_parent / out_svg.name
                except Exception:
                    pass

            out_svgz: Path | None = None

            try:
                self.root.after(0, self.pb.configure, {"mode": "determinate"})
            except Exception:
                pass

            try:
                out_svg.parent.mkdir(parents=True, exist_ok=True)

                is_large = False
                try:
                    is_large = inp.stat().st_size >= LARGE_BYTES
                except Exception:
                    is_large = False

                base_pct = ((idx1 - 1) / max(1, total)) * 100.0

                def _update_overall(pct_file_stage: float, stage_weight: float = 100.0):
                    overall = base_pct + (pct_file_stage * stage_weight) / max(1, total)
                    self.root.after(0, self.progress.set, overall)
                    self.root.after(0, self._update_pct_label, overall)

                self.root.after(0, self.status.set, f"Processing {idx1}/{total}: {inp.name}")

                if is_large:
                    parse_weight = 95.0
                    gzip_weight = 5.0

                    def parse_cb(pct):
                        _update_overall(pct, stage_weight=parse_weight)

                    optimize_svg_rects_stream(inp, out_svg, minify=minify, progress_cb=parse_cb)

                    msg = "OK | rects-stream"

                    if write_svgz_flag:
                        out_svgz = out_svg.with_suffix(out_svg.suffix + "z")

                        def gz_cb(pct):
                            _update_overall(pct, stage_weight=gzip_weight)

                        write_svgz_stream_from_svg(out_svg, out_svgz, compresslevel=9, progress_cb=gz_cb)
                        msg += " | svgz=(stream gzip)"
                    else:
                        _update_overall(100.0, stage_weight=parse_weight)

                else:
                    parse_weight = 95.0
                    gzip_weight = 5.0

                    def parse_cb(pct):
                        _update_overall(pct, stage_weight=parse_weight)

                    if use_paths:
                        svg_bytes, path_count = optimize_svg_paths_bytes(inp, minify=minify, progress_cb=parse_cb)
                        msg = f"OK | paths={path_count:,}"
                        if write_svg_flag:
                            out_svg.write_bytes(svg_bytes)
                            msg += f" | svg={len(svg_bytes):,} bytes"
                    else:
                        svg_bytes, rect_count = optimize_svg_rects_bytes(inp, vertical_merge=vertical, minify=minify, progress_cb=parse_cb)
                        msg = f"OK | rects={rect_count:,}"
                        if write_svg_flag:
                            out_svg.write_bytes(svg_bytes)
                            msg += f" | svg={len(svg_bytes):,} bytes"

                    if write_svgz_flag:
                        out_svgz = out_svg.with_suffix(out_svg.suffix + "z")
                        raw = out_svg.read_bytes()
                        bytes_svgz = write_svgz(raw, out_svgz, compresslevel=9, use_zopfli=use_zopfli)
                        msg += f" | svgz={bytes_svgz:,} bytes"
                        _update_overall(100.0, stage_weight=parse_weight + gzip_weight)
                    else:
                        _update_overall(100.0, stage_weight=parse_weight)

                results.append(JobResult(inp, out_svg, out_svgz, True, msg))

            except Exception as e:
                results.append(JobResult(inp, out_svg, None, False, str(e)))
                self.root.after(0, self._set_progress_style, False)

            try:
                pct_done = (idx1 / max(1, total)) * 100.0
                self.root.after(0, self.progress.set, pct_done)
                self.root.after(0, self._update_pct_label, pct_done)
            except Exception:
                pass

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

        log_path = out_dir / "svg_optimizer_log.txt"
        lines = []
        for r in results:
            out2 = f" | {r.output_svgz}" if r.output_svgz else ""
            lines.append(f"{'OK  ' if r.ok else 'FAIL'} | {r.input_path} -> {r.output_svg}{out2} | {r.message}")

        try:
            log_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

        self.run_btn.configure(state="normal")
        self.progress.set(100.0)
        self._set_progress_style(success=(fail == 0))
        self.status.set(f"Done. OK: {ok}, Failed: {fail}. Log: {log_path}")

        self._show_completion_dialog(results, out_dir, log_path)

    def _show_completion_dialog(self, results: list[JobResult], out_dir: Path, log_path: Path):
        dlg = tk.Toplevel(self.root)
        dlg.title("Optimization Summary")
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
            f"{'✓' if r.ok else '✗'} {r.input_path.name} -> {r.output_svg.name}"
            f"{(' (svgz)') if r.output_svgz else ''} | {r.message}"
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
            messagebox.showerror("Open failed", f"Could not open:\n{p}")
        

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
