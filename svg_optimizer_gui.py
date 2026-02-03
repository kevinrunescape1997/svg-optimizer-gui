#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from svg_pixel_rect_optimizer import optimize_svg_rects


# -----------------------------
# Helpers
# -----------------------------

def is_optimized_output(p: Path) -> bool:
    name = p.name.lower()
    if not name.endswith(".svg"):
        return False
    return "_optimized" in p.stem.lower()


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
    output_path: Path
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


# -----------------------------
# System-native pickers (Linux)
# -----------------------------
# Contract:
#   - return None   => system picker tool not available (caller may fallback to Tk)
#   - return ""     => user cancelled (caller should NOT fallback; just return)
#   - return value  => selection

def system_pick_folder(title: str) -> str | None:
    if shutil.which("zenity"):
        p = subprocess.run(
            ["zenity", "--file-selection", "--directory", f"--title={title}"],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            return ""  # cancelled
        return p.stdout.strip()  # can be "" if something odd, treat as cancelled by caller

    if shutil.which("kdialog"):
        p = subprocess.run(
            ["kdialog", "--getexistingdirectory", ".", f"--title={title}"],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            return ""  # cancelled
        return p.stdout.strip()

    return None  # unavailable


# Contract for files:
#   - return None      => unavailable (caller may fallback)
#   - return []        => cancelled (caller should NOT fallback)
#   - return [..paths] => selection

def system_pick_files(title: str, patterns: list[str]) -> list[str] | None:
    if shutil.which("zenity"):
        # Example filter: "*.svg | *.svg"
        filt = " ".join(patterns) if patterns else "*"
        p = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--multiple",
                "--separator=\n",
                f"--title={title}",
                f"--file-filter={filt} | {filt}",
            ],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            return []  # cancelled
        files = [s for s in p.stdout.splitlines() if s.strip()]
        return files

    if shutil.which("kdialog"):
        filt = " ".join(patterns) if patterns else "*"
        p = subprocess.run(
            [
                "kdialog",
                "--getopenfilename",
                ".",
                filt,
                "--multiple",
                "--separate-output",
                f"--title={title}",
            ],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            return []  # cancelled
        files = [s for s in p.stdout.splitlines() if s.strip()]
        return files

    return None  # unavailable


# -----------------------------
# Scrollable container
# -----------------------------

class ScrollableFrame(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0)

        self.vsb = tk.Scrollbar(self, orient="vertical", width=16, command=self.canvas.yview)
        self.hsb = tk.Scrollbar(self, orient="horizontal", width=16, command=self.canvas.xview)
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

        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

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

    def _on_mousewheel(self, event):
        if event.state & 0x0001:
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        delta = -1 if getattr(event, "num", None) == 4 else 1
        self.canvas.yview_scroll(delta, "units")

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind("<MouseWheel>")
        self.canvas.unbind("<Button-4>")
        self.canvas.unbind("<Button-5>")


# -----------------------------
# App
# -----------------------------

class App:
    def __init__(self, root: tk.Tk, dnd_files):
        self.root = root
        self.dnd_files = dnd_files

        self.root.title("SVG Pixel-Rect Optimizer")
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = min(1750, int(sw * 0.95))
        h = min(870, int(sh * 0.90))
        self.root.geometry(f"{w}x{h}")

        self._setup_theme()
        self._setup_toggle_styles()

        self.files: list[Path] = []

        self.output_dir = tk.StringVar(value=str(Path.cwd() / "optimized_svgs"))
        self.recursive = tk.BooleanVar(value=True)
        self.preserve_tree = tk.BooleanVar(value=True)
        self.preserve_names = tk.BooleanVar(value=True)
        self.skip_outputs = tk.BooleanVar(value=True)

        self.status = tk.StringVar(value="Ready.")
        self.progress = tk.DoubleVar(value=0.0)

        self._drop_bg = "#5e5e5e"
        self._drop_bg_hover = "#707070"

        self._build_ui()
        self._enable_dnd_if_available()

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
            style.layout(
                self._tog_style,
                [("Checkbutton.padding", {"sticky": "nswe", "children": [
                    ("Checkbutton.label", {"sticky": "nswe"})
                ]})]
            )
            style.configure(self._tog_style, padding=(10, 4))
            style.map(
                self._tog_style,
                background=[
                    ("active", "selected", "#00b980"),
                    ("active", "!selected", "#e07000"),
                    ("selected", "#009E73"),
                    ("!selected", "#D55E00"),
                ],
                foreground=[
                    ("selected", "#ffffff"),
                    ("!selected", "#ffffff"),
                ],
                relief=[
                    ("active", "solid"),
                    ("selected", "solid"),
                    ("!selected", "solid"),
                ],
                borderwidth=[
                    ("selected", 1),
                    ("!selected", 1),
                ],
            )
        except Exception:
            pass


    def _build_ui(self):
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self.sc = ScrollableFrame(container)
        self.sc.pack(fill="both", expand=True)

        top = ttk.Frame(self.sc.inner, padding=10)
        top.pack(fill="x")
        ttk.Button(top, text="Add SVG Files…", command=self.add_files).pack(side="left")
        ttk.Button(top, text="Add Folder…", command=self.add_folder).pack(side="left", padx=(8, 0))

        mid = ttk.Frame(self.sc.inner, padding=(10, 0, 10, 10))
        mid.pack(fill="both", expand=True)

        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)

        self.drop_label = tk.Label(
            left,
            text=(
                "Drop SVG files or folders here.\n\n"
                "Tip: Drop a folder to add all SVGs.\n"
                "You can also drop a folder onto the Output field to set it."
            ),
            justify="center",
            anchor="center",
            padx=12,
            pady=12,
            bd=2,
            relief="groove",
            bg=self._drop_bg,
            fg="#ffffff",
            cursor="hand2",
        )
        self.drop_label.pack(fill="x", pady=(6, 0))

        self.drop_label.bind("<Enter>", lambda _e: self._set_drop_hover(True))
        self.drop_label.bind("<Leave>", lambda _e: self._set_drop_hover(False))

        lb_wrap = ttk.Frame(left)
        lb_wrap.pack(fill="both", expand=True, pady=(6, 0))

        SCROLLBAR_SIZE = 16
        lb_wrap.rowconfigure(0, weight=1)
        lb_wrap.columnconfigure(1, weight=1)

        lb_vsb = tk.Scrollbar(lb_wrap, orient="vertical", width=SCROLLBAR_SIZE)
        lb_vsb.grid(row=0, column=0, sticky="ns")

        self.listbox = tk.Listbox(lb_wrap, selectmode=tk.EXTENDED)
        self.listbox.grid(row=0, column=1, sticky="nsew")

        lb_hsb = tk.Scrollbar(lb_wrap, orient="horizontal", width=SCROLLBAR_SIZE)
        lb_hsb.grid(row=1, column=1, sticky="ew")

        ttk.Frame(lb_wrap, width=SCROLLBAR_SIZE, height=SCROLLBAR_SIZE).grid(row=1, column=0)

        self.listbox.configure(yscrollcommand=lb_vsb.set, xscrollcommand=lb_hsb.set)
        lb_vsb.configure(command=self.listbox.yview)
        lb_hsb.configure(command=self.listbox.xview)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Remove Selected", command=self.remove_selected).pack(side="left")
        ttk.Button(btns, text="Clear Entire List", command=self.clear_list).pack(side="left", padx=(8, 0))

        out = ttk.Frame(self.sc.inner, padding=(10, 0, 10, 10))
        out.pack(fill="x")

        ttk.Label(out, text="Output folder:").pack(side="left")
        self.out_entry = ttk.Entry(out, textvariable=self.output_dir)
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Button(out, text="Choose…", command=self.choose_output_dir).pack(side="left", padx=(8, 0))
        ttk.Button(out, text="Up", command=self.output_dir_up).pack(side="left", padx=(8, 0))

        toggles = ttk.Frame(self.sc.inner, padding=(10, 0, 10, 10))
        toggles.pack(fill="x")

        ttk.Checkbutton(toggles, text="Recursive folder scan", variable=self.recursive,
                        style=self._tog_style).pack(side="left")
        ttk.Checkbutton(toggles, text="Preserve folder structure", variable=self.preserve_tree,
                        style=self._tog_style).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(toggles, text="Preserve file names", variable=self.preserve_names,
                        style=self._tog_style).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(toggles, text="Skip *_optimized*.svg in searches", variable=self.skip_outputs,
                        style=self._tog_style).pack(side="left", padx=(8, 0))

        ttk.Label(self.sc.inner, textvariable=self.status, padding=(10, 6, 10, 6)).pack(fill="x")

        bottom = ttk.Frame(self.sc.inner, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")

        self.run_btn = ttk.Button(bottom, text="Optimize", command=self.run)
        self.run_btn.pack(side="left")

        self.pb = ttk.Progressbar(bottom, variable=self.progress, maximum=100.0)
        self.pb.pack(side="left", fill="x", expand=True, padx=(10, 0))
    def _set_drop_hover(self, on: bool) -> None:
        if not hasattr(self, "drop_label"):
            return
        self.drop_label.configure(
            bg=self._drop_bg_hover if on else self._drop_bg,
            relief="ridge" if on else "groove",
        )

    # -----------------------------
    # Drag & drop
    # -----------------------------

    def _enable_dnd_if_available(self):
        if self.dnd_files is None:
            self.status.set("Ready. (Drag & drop disabled: install tkinterdnd2 to enable.)")
            return

        for widget in (self.listbox, self.drop_label, self.out_entry):
            try:
                widget.drop_target_register(self.dnd_files)
            except Exception:
                pass

        try:
            self.listbox.dnd_bind("<<Drop>>", self._on_drop_inputs)
        except Exception:
            pass
        try:
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop_inputs)
        except Exception:
            pass
        try:
            self.out_entry.dnd_bind("<<Drop>>", self._on_drop_output_dir)
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

    # -----------------------------
    # File list management
    # -----------------------------

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for p in self.files:
            self.listbox.insert(tk.END, str(p))
        self.status.set(f"{len(self.files)} file(s) queued.")

    def _add_paths(self, paths: list[Path]):
        added = 0
        existing = set(self.files)

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

    def _handle_input_paths(self, paths: list[Path]):
        files: list[Path] = []
        folders: list[Path] = []

        for p in paths:
            p = p.expanduser()
            if p.is_dir():
                folders.append(p)
            elif p.is_file():
                files.append(p)

        if folders:
            for folder in folders:
                svgs = find_svgs_in_folder(folder, recursive=self.recursive.get(), skip_outputs=self.skip_outputs.get())
                self._add_paths(svgs)

        if files:
            self._add_paths(files)

    # -----------------------------
    # Pickers (prefer system, fallback to Tk only when unavailable)
    # -----------------------------

    def add_files(self):
        picked = system_pick_files("Select SVG files", patterns=["*.svg"])
        if picked is None:
            # System picker unavailable -> fallback
            picked = list(filedialog.askopenfilenames(
                title="Select SVG files",
                filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
            ))
        elif picked == []:
            # User cancelled system picker -> do nothing
            return

        if picked:
            self._add_paths([Path(f) for f in picked])

    def add_folder(self):
        folder = system_pick_folder("Select a folder containing SVGs")
        if folder is None:
            folder = filedialog.askdirectory(title="Select a folder containing SVGs")
        elif folder == "":
            return  # cancelled

        if not folder:
            return

        svgs = find_svgs_in_folder(Path(folder), recursive=self.recursive.get(), skip_outputs=self.skip_outputs.get())
        if not svgs:
            messagebox.showinfo("No SVGs found", "No .svg files were found in the selected folder.")
            return
        self._add_paths(svgs)

    def choose_output_dir(self):
        folder = system_pick_folder("Choose output folder")
        if folder is None:
            folder = filedialog.askdirectory(title="Choose output folder")
        elif folder == "":
            return  # cancelled

        if folder:
            self.output_dir.set(str(Path(folder)))
            self.status.set("Output folder set.")

    # -----------------------------
    # Output dir
    # -----------------------------

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

    # -----------------------------
    # Processing
    # -----------------------------

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
        self.status.set("Running...")

        threading.Thread(target=self._run_worker, args=(out_dir,), daemon=True).start()

    def _run_worker(self, out_dir: Path):
        vertical = True
        preserve_tree = self.preserve_tree.get()

        common_root = None
        if preserve_tree:
            try:
                common_root = Path(os.path.commonpath([str(p.parent) for p in self.files]))
            except Exception:
                common_root = None

        results: list[JobResult] = []
        total = len(self.files)

        for idx, inp in enumerate(self.files, start=1):
            try:
                if self.preserve_names.get():
                    out_name = inp.name
                else:
                    out_name = inp.stem + "_optimized" + inp.suffix

                if preserve_tree and common_root is not None:
                    rel_parent = inp.parent.relative_to(common_root)
                    out_path = out_dir / rel_parent / out_name
                else:
                    out_path = out_dir / out_name

                out_path.parent.mkdir(parents=True, exist_ok=True)

                rect_count, bytes_out = optimize_svg_rects(inp, out_path, vertical_merge=vertical)
                results.append(JobResult(inp, out_path, True, f"OK | rects={rect_count:,} | bytes={bytes_out:,}"))
            except Exception as e:
                results.append(JobResult(inp, out_dir / (inp.stem + "_FAILED.svg"), False, str(e)))

            pct = (idx / total) * 100.0
            self.root.after(0, self.progress.set, pct)
            self.root.after(0, self.status.set, f"Processing {idx}/{total}: {inp.name}")

        self.root.after(0, self._finish, results, out_dir)

    def _finish(self, results: list[JobResult], out_dir: Path):
        ok = sum(1 for r in results if r.ok)
        fail = len(results) - ok

        log_path = out_dir / "svg_optimizer_log.txt"
        lines = [
            f"{'OK  ' if r.ok else 'FAIL'} | {r.input_path} -> {r.output_path} | {r.message}"
            for r in results
        ]
        try:
            log_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

        self.run_btn.configure(state="normal")
        self.progress.set(100.0)

        self.status.set(f"Done. OK: {ok}, Failed: {fail}. Log: {log_path}")
        if fail:
            messagebox.showwarning("Completed with errors", f"OK: {ok}\nFailed: {fail}\n\nSee log:\n{log_path}")
        else:
            messagebox.showinfo("Completed", f"Processed {ok} file(s).\n\nOutput:\n{out_dir}\n\nLog:\n{log_path}")


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

