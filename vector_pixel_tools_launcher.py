#!/usr/bin/env python3
from __future__ import annotations

"""
Simple launcher GUI with three buttons:
- Open Bitmap → SVG Converter interface
- Open SVG Pixelart Optimizer interface
- Open SVG → EPS/PDF/TIFF/PNG Exporter interface

Starts each tool in its own process for isolation and simplicity.
Place this file alongside:
- GUI_bitmap_converter.py
- GUI_svg_optimizer.py
- GUI_svg_exporter.py

Single-instance behavior:
- The launcher listens on 127.0.0.1:51262 for 'RAISE' messages.
- If another instance tries to start, it sends 'RAISE' to the running instance and exits.
- The main optimizer and converter GUIs also send 'RAISE' before attempting to start a new launcher.

Styled to match the dark theme of the GUIs with consistent button styles.
"""

import subprocess
import sys
import socket
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

LAUNCHER_PORT = 51262

# Dark theme palette (matching GUIs)
DARK_BG = "#1B1F23"
DARK_SURFACE = "#2D333B"
DARK_SURFACE_ALT = "#22272E"
DARK_TEXT = "#ADBAC7"
DARK_BORDER = "#3A4149"
SMALL_MARGIN = 8


def _script_path(name: str) -> Path:
    here = Path(__file__).resolve().parent
    return (here / name).resolve()


def _run_script(script: Path):
    if not script.exists():
        messagebox.showerror("Not found", f"Could not find:\n{script}")
        return False
    try:
        subprocess.Popen([sys.executable, str(script)], close_fds=True)
        return True
    except Exception as e:
        messagebox.showerror("Launch failed", f"Failed to launch:\n{script}\n\n{e}")
        return False


def _bring_to_front(root: tk.Tk):
    try:
        root.deiconify()
        root.lift()
        root.focus_force()
        root.attributes("-topmost", True)
        root.after(200, lambda: root.attributes("-topmost", False))
    except Exception:
        pass


def _start_raise_server(root: tk.Tk):
    """Start a small TCP server to receive 'RAISE' and bring the window to front."""
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", LAUNCHER_PORT))
        srv.listen(5)
    except OSError:
        # Another instance likely owns the port; send RAISE to it and exit.
        try:
            with socket.create_connection(("127.0.0.1", LAUNCHER_PORT), timeout=0.5) as s:
                s.sendall(b"RAISE\n")
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        sys.exit(0)
        return

    def serve():
        while True:
            try:
                conn, _addr = srv.accept()
            except Exception:
                break
            try:
                data = conn.recv(64)
                if data and data.strip().upper().startswith(b"RAISE"):
                    root.after(0, _bring_to_front, root)
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    t = threading.Thread(target=serve, daemon=True)
    t.start()


def _setup_theme(root: tk.Tk):
    try:
        style = ttk.Style(root)
        for t in ("clam", "alt", "default"):
            if t in style.theme_names():
                style.theme_use(t)
                break
        # Dark styling
        root.configure(bg=DARK_BG)
        style.configure("TFrame", background=DARK_BG, bordercolor=DARK_BORDER)
        style.configure("TLabel", background=DARK_BG, foreground=DARK_TEXT, bordercolor=DARK_BORDER)
        style.configure(
            "TButton",
            background=DARK_SURFACE,
            foreground=DARK_TEXT,
            borderwidth=1,
            relief="solid",
            bordercolor=DARK_BORDER,
            padding=(10, 6),
        )
        style.map(
            "TButton",
            background=[("active", DARK_SURFACE_ALT), ("pressed", DARK_SURFACE_ALT)],
            foreground=[("disabled", "#8a97a6")],
        )
    except Exception:
        pass


def main():
    root = tk.Tk()
    root.title("Pixel Tools Launcher")
    root.geometry("600x200")
    _setup_theme(root)

    # Start single-instance raise server (or exit if another is already running)
    _start_raise_server(root)

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)

    btns = ttk.Frame(frame)
    btns.pack(fill="x")

    def launch_and_close(script_name: str):
        ok = _run_script(_script_path(script_name))
        try:
            root.destroy()
        except Exception:
            pass

    ttk.Button(
        btns,
        text="Open Bitmap SVG Converter",
        command=lambda: launch_and_close("GUI_bitmap_converter.py"),
    ).pack(fill="x", pady=(0, SMALL_MARGIN))

    ttk.Button(
        btns,
        text="Open SVG Pixel Optimizer",
        command=lambda: launch_and_close("GUI_svg_optimizer.py"),
    ).pack(fill="x", pady=(0, SMALL_MARGIN))

    ttk.Button(
        btns,
        text="Open SVG Exporter",
        command=lambda: launch_and_close("GUI_svg_exporter.py"),
    ).pack(fill="x")

    root.mainloop()


if __name__ == "__main__":
    main()
