#!/usr/bin/env python3
from __future__ import annotations

# Owner: kevinrunescape1997
# Purpose: Launcher GUI for bitmap converter, SVG optimizer, and exporter. Single-instance raise; robust spawns.

"""
Launcher GUI with three buttons:
- Open Bitmap → SVG Converter
- Open SVG Pixel Optimizer
- Open SVG → EPS/PDF/TIFF/PNG Exporter

Starts each tool in its own process:
- Frozen (PyInstaller/AppImage): spawns the same executable with a --run=<tool> flag
- Unfrozen (dev): runs the corresponding .py script with the system Python

Single-instance behavior:
- The launcher listens on 127.0.0.1:51262 for 'RAISE' messages.
- If another instance tries to start, it sends 'RAISE' to the running instance and exits.
- The GUIs also send 'RAISE' before attempting to start a new launcher.
"""

import os
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

# Map tool aliases to script filenames (used in dev/unfrozen mode)
TOOL_SCRIPT_MAP = {
    "bitmap": "GUI_bitmap_converter.py",
    "optimizer": "GUI_svg_optimizer.py",
    "exporter": "GUI_svg_exporter.py",
}


def _script_path(name: str) -> Path:
    here = Path(__file__).resolve().parent
    return (here / name).resolve()


def _run_tool(tool: str) -> bool:
    """
    Start the requested tool in a new process.

    - In frozen mode (PyInstaller/AppImage), spawn the SAME executable with --run=<tool>.
    - In dev/unfrozen mode, run the corresponding .py script with the system Python.
    """
    try:
        if getattr(sys, "frozen", False):
            exe = None
            appimage = os.environ.get("APPIMAGE")
            if sys.platform.startswith("linux") and appimage:
                exe = appimage
            else:
                exe = sys.argv[0] or sys.executable
            close_fds = (sys.platform != "win32")
            subprocess.Popen([exe, f"--run={tool}"], close_fds=close_fds)
            return True
        else:
            script_name = TOOL_SCRIPT_MAP.get(tool)
            if not script_name:
                messagebox.showerror("Unknown tool", f"Unknown tool: {tool}")
                return False
            script = _script_path(script_name)
            if not script.exists():
                messagebox.showerror("Not found", f"Could not find:\n{script}")
                return False
            subprocess.Popen([sys.executable, str(script)], close_fds=True)
            return True
    except Exception as e:
        messagebox.showerror("Launch failed", f"Failed to launch '{tool}':\n{e}")
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


def _dispatch_run_flag() -> bool:
    """
    If invoked with --run=<tool> or --run <tool>, import the selected GUI and run it in this process.
    Returns True if a tool was dispatched, False otherwise.
    """
    args = sys.argv[1:]
    mode = None
    for i, arg in enumerate(args):
        if arg.startswith("--run="):
            mode = arg.split("=", 1)[1].strip()
            break
        if arg == "--run" and i + 1 < len(args):
            mode = args[i + 1].strip()
            break
    if not mode:
        return False

    try:
        if mode == "bitmap":
            from GUI_bitmap_converter import main as run
            run()
            return True
        elif mode == "optimizer":
            from GUI_svg_optimizer import main as run
            run()
            return True
        elif mode == "exporter":
            from GUI_svg_exporter import main as run
            run()
            return True
        else:
            messagebox.showerror("Unknown tool", f"Unknown tool: {mode}")
            return False
    except Exception as e:
        messagebox.showerror("Launch failed", f"Failed to start '{mode}':\n{e}")
        return False


def main():
    if _dispatch_run_flag():
        return

    root = tk.Tk()
    root.title("Pixel Tools Launcher")
    root.geometry("600x200")
    _setup_theme(root)

    _start_raise_server(root)

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)

    btns = ttk.Frame(frame)
    btns.pack(fill="x")

    def launch_and_close(tool: str):
        ok = _run_tool(tool)
        if ok:
            try:
                root.after(250, root.destroy)
            except Exception:
                try:
                    root.destroy()
                except Exception:
                    pass

    ttk.Button(
        btns,
        text="Open Bitmap SVG Converter",
        command=lambda: launch_and_close("bitmap"),
    ).pack(fill="x", pady=(0, SMALL_MARGIN))

    ttk.Button(
        btns,
        text="Open SVG Pixel Optimizer",
        command=lambda: launch_and_close("optimizer"),
    ).pack(fill="x", pady=(0, SMALL_MARGIN))

    ttk.Button(
        btns,
        text="Open SVG Exporter",
        command=lambda: launch_and_close("exporter"),
    ).pack(fill="x")

    root.mainloop()


if __name__ == "__main__":
    main()
