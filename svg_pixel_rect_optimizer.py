#!/usr/bin/env python3
"""
svg_pixel_rect_optimizer.py

Optimize "pixel art as SVG rects" by:
- normalizing style/opacity
- merging adjacent same-color pixels:
  - horizontal runs
  - + optional vertical stacking

Optional connected path mode:
- --paths: emit one <path> per connected region (4-connected adjacency) of same final RGBA.
  This replaces <rect> outputs with compact axis-aligned path outlines. Width/height>1 input rects
  are expanded to pixels before union to preserve images.

Note: Final colours are computed by true source-over alpha compositing in DOM order.
This preserves visual detail when semi-transparent layers overlap.

CLI usage:
  python svg_pixel_rect_optimizer.py input.svg
  python svg_pixel_rect_optimizer.py input.svg -o output.svg
  python svg_pixel_rect_optimizer.py input.svg --no-vertical
  python svg_pixel_rect_optimizer.py input.svg --paths
  python svg_pixel_rect_optimizer.py input.svg --svgz
  python svg_pixel_rect_optimizer.py input.svg --svgz-only

Batch:
  python svg_pixel_rect_optimizer.py *.svg
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict, deque
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple

SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}

# -----------------------------
# Style parsing helpers
# -----------------------------

def parse_style(style: str | None) -> dict[str, str]:
    d: dict[str, str] = {}
    if not style:
        return d
    for part in style.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def norm_opacity(op_str: str | None) -> float:
    """Opacity in SVG is 0..1. Some exporters write 0..255; normalize."""
    if not op_str:
        return 1.0
    try:
        op = float(op_str)
    except ValueError:
        return 1.0
    if op > 1.0:
        op = op / 255.0
    if op < 0.0:
        op = 0.0
    if op > 1.0:
        op = 1.0
    return op


def fmt_opacity(op: float) -> str:
    s = f"{op:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _as_int(attr: str | None, default: int) -> int:
    if attr is None or attr == "":
        return default
    try:
        return int(float(attr))
    except Exception:
        return default


def _parse_rgb(fill: str | None) -> Tuple[float, float, float]:
    """
    Parse fill colour to normalized RGB (0..1 per channel).
    Supports #RGB, #RRGGBB, rgb(r,g,b), and a few common names.
    """
    if not fill or fill.lower() == "none":
        return (0.0, 0.0, 0.0)
    s = fill.strip().lower()
    if s.startswith("#"):
        s = s[1:]
        if len(s) == 3:
            r = int(s[0] * 2, 16)
            g = int(s[1] * 2, 16)
            b = int(s[2] * 2, 16)
        elif len(s) == 6:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
        else:
            r = g = b = 0
        return (r / 255.0, g / 255.0, b / 255.0)
    if s.startswith("rgb(") and s.endswith(")"):
        try:
            inner = s[4:-1]
            parts = inner.split(",")
            r = int(parts[0].strip())
            g = int(parts[1].strip())
            b = int(parts[2].strip())
            return (max(0, min(255, r)) / 255.0,
                    max(0, min(255, g)) / 255.0,
                    max(0, min(255, b)) / 255.0)
        except Exception:
            return (0.0, 0.0, 0.0)
    # minimal named colours commonly seen
    NAMED = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
    }
    return NAMED.get(s, (0.0, 0.0, 0.0))


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r = max(0, min(255, int(round(rgb[0] * 255))))
    g = max(0, min(255, int(round(rgb[1] * 255))))
    b = max(0, min(255, int(round(rgb[2] * 255))))
    return f"#{r:02x}{g:02x}{b:02x}"


# -----------------------------
# Types
# -----------------------------

Point = Tuple[int, int]
RGBA = Tuple[float, float, float, float]  # normalized 0..1
StyleKey = Tuple[str, float]  # (hex_rgb_lower, opacity)

# -----------------------------
# Final per-pixel RGBA (paint-order aware, with alpha compositing)
# -----------------------------

def _collect_final_rgba_pixels(svg_in: Path) -> tuple[Dict[StyleKey, Set[Point]], dict[str, str]]:
    """
    Parse input SVG, expand all rects to unit pixels, and composite colours in DOM order
    using source-over alpha. This preserves semi-transparent layering.
    Returns: (style_pixels, out_root_attrs)
      style_pixels[(hex_rgb, opacity)] = set of (x, y) pixels that end up with that final RGBA.
    Assumes a transparent background.
    """
    tree = ET.parse(str(svg_in))
    root = tree.getroot()

    rects = root.findall(".//svg:rect", NS)
    if not rects:
        raise ValueError("No <rect> elements found. This script expects pixel-rect SVGs.")

    # Per-pixel final premultiplied RGBA (we store unpremultiplied RGB with alpha)
    pix_rgba: Dict[Point, RGBA] = {}

    for r in rects:  # DOM paint order
        st = parse_style(r.get("style"))
        fill_raw = st.get("fill", r.get("fill"))
        if not fill_raw or str(fill_raw).strip().lower() == "none":
            continue

        # combined opacity = opacity * fill-opacity (both default to 1)
        op = norm_opacity(st.get("opacity", r.get("opacity")))
        fop = norm_opacity(st.get("fill-opacity", r.get("fill-opacity")))
        a_src = op * fop
        if a_src <= 0.0:
            continue

        rgb_src = _parse_rgb(str(fill_raw))
        # Expand rect to pixels
        x0 = _as_int(r.get("x"), 0)
        y0 = _as_int(r.get("y"), 0)
        w0 = _as_int(r.get("width"), 1)
        h0 = _as_int(r.get("height"), 1)
        if w0 <= 0:
            w0 = 1
        if h0 <= 0:
            h0 = 1

        for yy in range(y0, y0 + h0):
            for xx in range(x0, x0 + w0):
                dr, dg, db, da = pix_rgba.get((xx, yy), (0.0, 0.0, 0.0, 0.0))
                # Source-over compositing:
                out_a = a_src + da * (1.0 - a_src)
                if out_a <= 1e-12:
                    pix_rgba[(xx, yy)] = (0.0, 0.0, 0.0, 0.0)
                    continue
                # Work in unpremultiplied form; compute resulting colour
                out_r = (rgb_src[0] * a_src + dr * da * (1.0 - a_src)) / out_a
                out_g = (rgb_src[1] * a_src + dg * da * (1.0 - a_src)) / out_a
                out_b = (rgb_src[2] * a_src + db * da * (1.0 - a_src)) / out_a
                pix_rgba[(xx, yy)] = (out_r, out_g, out_b, out_a)

    # Group pixels by final RGBA (quantized to 8-bit channels)
    style_pixels: Dict[StyleKey, Set[Point]] = defaultdict(set)
    for pt, (r, g, b, a) in pix_rgba.items():
        if a <= 0.0:
            continue
        hex_rgb = _rgb_to_hex((r, g, b)).lower()
        # round alpha to 4 decimals for stable keys
        a_key = round(a, 6)
        style_pixels[(hex_rgb, a_key)].add(pt)

    out_attrs: dict[str, str] = {}
    for k in ("width", "height", "viewBox", "preserveAspectRatio"):
        v = root.get(k)
        if v:
            out_attrs[k] = v
    out_attrs["shape-rendering"] = "crispEdges"
    return style_pixels, out_attrs

# -----------------------------
# Rect mode: build merges from final RGBA map
# -----------------------------

def _build_rect_list(svg_in: Path, vertical_merge: bool = True) -> tuple[list[tuple[int, int, int, int, tuple[str, float]]], dict[str, str]]:
    """
    Returns: (rect_list_sorted, out_root_attrs)
    rect_list_sorted items: (x, y, w, h, (fill_hex, opacity))
    """
    style_pixels, out_attrs = _collect_final_rgba_pixels(svg_in)

    # rows[y][stylekey] -> list of x's
    rows: dict[int, dict[tuple[str, float], list[int]]] = defaultdict(lambda: defaultdict(list))
    for stylekey, pts in style_pixels.items():
        for (x, y) in pts:
            rows[y][stylekey].append(x)

    # 1) Horizontal merges per row/style
    merged_h: list[tuple[int, int, int, int, tuple[str, float]]] = []
    for y, style_map in rows.items():
        for stylekey, xs in style_map.items():
            if not xs:
                continue
            xs = sorted(xs)
            start = prev = xs[0]
            for x in xs[1:]:
                if x == prev + 1:
                    prev = x
                else:
                    merged_h.append((start, y, prev - start + 1, 1, stylekey))
                    start = prev = x
            merged_h.append((start, y, prev - start + 1, 1, stylekey))

    # 2) Optional vertical stacking: same x+width+style across consecutive rows
    if vertical_merge:
        cols: dict[tuple[int, int, tuple[str, float]], list[int]] = defaultdict(list)
        for x, y, w, _h, stylekey in merged_h:
            cols[(x, w, stylekey)].append(y)

        rect_list: list[tuple[int, int, int, int, tuple[str, float]]] = []
        for (x, w, stylekey), ys in cols.items():
            ys = sorted(ys)
            start = prev = ys[0]
            for y in ys[1:]:
                if y == prev + 1:
                    prev = y
                else:
                    rect_list.append((x, start, w, prev - start + 1, stylekey))
                    start = prev = y
            rect_list.append((x, start, w, prev - start + 1, stylekey))
    else:
        rect_list = merged_h

    # Deterministic order; no overlaps remain, so order is visually irrelevant
    rect_list_sorted = sorted(rect_list, key=lambda t: (t[1], t[0], t[2], t[3]))
    return rect_list_sorted, out_attrs


def optimize_svg_rects_bytes(svg_in: Path, vertical_merge: bool = True) -> tuple[bytes, int]:
    """Return optimized SVG bytes and rect count, without writing to disk."""
    rect_list_sorted, out_attrs = _build_rect_list(svg_in, vertical_merge=vertical_merge)

    ET.register_namespace("", SVG_NS)
    new_root = ET.Element(f"{{{SVG_NS}}}svg", out_attrs)
    g = ET.SubElement(new_root, f"{{{SVG_NS}}}g")

    for x, y, w, h, (fill, op) in rect_list_sorted:
        r_attrs = {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "fill": fill}
        if abs(op - 1.0) > 1e-6:
            r_attrs["opacity"] = fmt_opacity(op)
        ET.SubElement(g, f"{{{SVG_NS}}}rect", r_attrs)

    data = ET.tostring(new_root, encoding="utf-8", xml_declaration=True)
    return data, len(rect_list_sorted)

# -----------------------------
# Connected path mode (same final RGBA map)
# -----------------------------

PointT = Tuple[int, int]
Edge = Tuple[PointT, PointT]

def _connected_components(pixels: Set[PointT]) -> List[Set[PointT]]:
    """Split a pixel set into 4-connected components."""
    comps: List[Set[PointT]] = []
    unvisited = set(pixels)

    while unvisited:
        start = min(unvisited, key=lambda p: (p[1], p[0]))  # deterministic
        comp: Set[PointT] = set()
        q: deque[PointT] = deque([start])
        unvisited.remove(start)

        while q:
            x, y = q.popleft()
            comp.add((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in unvisited:
                    unvisited.remove((nx, ny))
                    q.append((nx, ny))
        comps.append(comp)

    return comps


def _norm_edge(a: PointT, b: PointT) -> Edge:
    """Store undirected edge with endpoints in a deterministic order."""
    return (a, b) if a <= b else (b, a)


def _component_edges(comp: Set[PointT]) -> Set[Edge]:
    """
    Build the set of outer boundary edges for a component by adding all cell edges
    and removing interior shared edges (count==2).
    """
    edge_counts: Dict[Edge, int] = defaultdict(int)

    for (x, y) in comp:
        e_top = _norm_edge((x, y), (x + 1, y))
        e_right = _norm_edge((x + 1, y), (x + 1, y + 1))
        e_bottom = _norm_edge((x + 1, y + 1), (x, y + 1))
        e_left = _norm_edge((x, y + 1), (x, y))
        for e in (e_top, e_right, e_bottom, e_left):
            edge_counts[e] += 1

    boundary_edges: Set[Edge] = {e for e, c in edge_counts.items() if c == 1}
    return boundary_edges


def _edges_to_cycles(edges: Set[Edge]) -> List[List[PointT]]:
    """
    Convert a set of undirected boundary edges into closed cycles of vertices.
    Cycles are lists of points; the last point equals the first.
    """
    adj: Dict[PointT, List[PointT]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for v in adj:
        adj[v].sort()

    visited: Set[Edge] = set()
    cycles: List[List[PointT]] = []

    for start in sorted(adj.keys(), key=lambda p: (p[1], p[0])):  # deterministic
        for nb in adj[start]:
            e = _norm_edge(start, nb)
            if e in visited:
                continue
            path: List[PointT] = [start, nb]
            visited.add(e)
            prev, curr = start, nb

            while True:
                nxt = None
                for cand in adj[curr]:
                    if cand == prev:
                        continue
                    ec = _norm_edge(curr, cand)
                    if ec not in visited:
                        nxt = cand
                        break
                if nxt is None:
                    nxt = path[0]
                path.append(nxt)
                visited.add(_norm_edge(curr, nxt))
                if nxt == path[0]:
                    break
                prev, curr = curr, nxt

            cycles.append(path)
    return cycles


def _compress_corners(points: List[PointT]) -> List[PointT]:
    """Keep only turning points to emit H/V commands; ensure path closes."""
    if not points:
        return points
    pts = points[:-1]
    if len(pts) <= 2:
        return points
    corners: List[PointT] = [pts[0]]
    def dir(a: PointT, b: PointT) -> int:
        return 0 if a[1] == b[1] else 1
    d_prev = dir(pts[0], pts[1])
    for i in range(1, len(pts) - 1):
        d_next = dir(pts[i], pts[i + 1])
        if d_next != d_prev:
            corners.append(pts[i])
        d_prev = d_next
    corners.append(pts[-1])
    corners.append(corners[0])
    return corners


def _corners_to_path_d(corners: List[PointT]) -> str:
    """Emit a compact axis-aligned path using M/H/V/Z with integer coordinates."""
    if not corners or len(corners) < 2:
        return ""
    parts: List[str] = []
    x0, y0 = corners[0]
    parts.append(f"M {x0} {y0}")
    for i in range(1, len(corners)):
        x, y = corners[i]
        _, py = corners[i - 1]
        if py == y:
            parts.append(f"H {x}")
        else:
            parts.append(f"V {y}")
    parts.append("Z")
    return " ".join(parts)


def optimize_svg_paths_bytes(svg_in: Path) -> tuple[bytes, int]:
    """
    Return path-based optimized SVG bytes and path count, without writing to disk.
    Emits one <path> per connected component per final (fill, opacity), using even-odd fill rule.
    Paint-order effects are baked into the composited RGBA.
    """
    style_pixels, out_attrs = _collect_final_rgba_pixels(svg_in)

    ET.register_namespace("", SVG_NS)
    new_root = ET.Element(f"{{{SVG_NS}}}svg", out_attrs)
    # You can keep fill-rule on the group or set it per path; per-path is clearer.
    g = ET.SubElement(new_root, f"{{{SVG_NS}}}g")

    path_count = 0

    for (fill_hex, op) in sorted(style_pixels.keys(), key=lambda k: (k[0], k[1])):
        pixels = style_pixels[(fill_hex, op)]
        if not pixels:
            continue

        # One path per 4-connected component
        comps = _connected_components(pixels)
        for comp in comps:
            edges = _component_edges(comp)
            cycles = _edges_to_cycles(edges)

            # Build one path 'd' with multiple subpaths (outer + holes)
            d_parts: List[str] = []
            for cyc in cycles:
                corners = _compress_corners(cyc)
                d_sub = _corners_to_path_d(corners)
                if d_sub:
                    d_parts.append(d_sub)

            if not d_parts:
                continue

            attrs = {"d": " ".join(d_parts), "fill": fill_hex, "fill-rule": "evenodd"}
            if abs(op - 1.0) > 1e-6:
                attrs["opacity"] = fmt_opacity(op)

            ET.SubElement(g, f"{{{SVG_NS}}}path", attrs)
            path_count += 1

    data = ET.tostring(new_root, encoding="utf-8", xml_declaration=True)
    return data, path_count

# -----------------------------
# IO helpers
# -----------------------------

def optimize_svg_rects(svg_in: Path, svg_out: Path, vertical_merge: bool = True) -> tuple[int, int]:
    """Write optimized SVG to disk. Returns: (rect_count_out, bytes_out)."""
    data, rect_count = optimize_svg_rects_bytes(svg_in, vertical_merge=vertical_merge)
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    svg_out.write_bytes(data)
    return rect_count, len(data)


def write_svgz(svg_bytes: bytes, svgz_out: Path, compresslevel: int = 9) -> int:
    """Write gzipped SVG (.svgz). Returns bytes written."""
    svgz_out.parent.mkdir(parents=True, exist_ok=True)
    level = max(1, min(9, int(compresslevel)))
    with gzip.open(svgz_out, "wb", compresslevel=level) as f:
        f.write(svg_bytes)
    return svgz_out.stat().st_size


def default_output_path(inp: Path, vertical_merge: bool) -> Path:
    suffix = "_optimized_hv" if vertical_merge else "_optimized_h"
    return inp.with_name(inp.stem + suffix + inp.suffix)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge/optimize pixel-rect SVGs.")
    ap.add_argument("input", nargs="+", help="Input SVG file(s). Supports wildcards via shell.")
    ap.add_argument("-o", "--output", help="Output path (only valid with a single input).")
    ap.add_argument("--paths", action="store_true", help="Emit connected <path> shapes per final RGBA (replaces <rect> output).")
    ap.add_argument("--no-vertical", action="store_true", help="Disable vertical stacking merge (rect mode only).")
    ap.add_argument("--svgz", action="store_true", help="Also write a .svgz (gzipped) alongside the .svg output.")
    ap.add_argument("--svgz-only", action="store_true", help="Write only .svgz output (implies --svgz).")
    ap.add_argument("--svgz-level", type=int, default=9, help="GZip level for .svgz (1-9, default 9).")
    args = ap.parse_args()

    vertical = not args.no_vertical
    inputs = [Path(p) for p in args.input]

    if args.output and len(inputs) != 1:
        raise SystemExit("Error: -o/--output can only be used with a single input file.")

    for inp in inputs:
        if not inp.exists():
            print(f"Skip (not found): {inp}")
            continue

        out_svg = Path(args.output) if args.output else (
            default_output_path(inp, vertical) if not args.paths
            else inp.with_name(inp.stem + "_optimized_paths" + inp.suffix)
        )

        try:
            out_svg.parent.mkdir(parents=True, exist_ok=True)

            if args.paths:
                svg_bytes, path_count = optimize_svg_paths_bytes(inp)
                if not args.svgz_only:
                    out_svg.write_bytes(svg_bytes)
                    print(f"Wrote: {out_svg} | paths: {path_count:,} | bytes: {len(svg_bytes):,}")
                if args.svgz or args.svgz_only:
                    svgz_out = out_svg.with_suffix(out_svg.suffix + "z")
                    bytes_svgz = write_svgz(svg_bytes, svgz_out, compresslevel=args.svgz_level)
                    print(f"Wrote: {svgz_out} | bytes: {bytes_svgz:,}")
            else:
                svg_bytes, rect_count = optimize_svg_rects_bytes(inp, vertical_merge=vertical)
                if not args.svgz_only:
                    out_svg.write_bytes(svg_bytes)
                    print(f"Wrote: {out_svg} | rects: {rect_count:,} | bytes: {len(svg_bytes):,}")
                if args.svgz or args.svgz_only:
                    svgz_out = out_svg.with_suffix(out_svg.suffix + "z")
                    bytes_svgz = write_svgz(svg_bytes, svgz_out, compresslevel=args.svgz_level)
                    print(f"Wrote: {svgz_out} | bytes: {bytes_svgz:,}")

        except Exception as e:
            print(f"Failed: {inp} | {e}")


if __name__ == "__main__":
    main()
