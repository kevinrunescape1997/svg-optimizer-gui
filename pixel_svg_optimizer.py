#!/usr/bin/env python3
# Owner: kevinrunescape1997
# Purpose: Optimize pixel-art SVGs with compositing, rect merging, path emission, and streaming support.
# Notes: Batches write operations where safe to reduce I/O overhead without changing behavior.

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict, deque
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple, Optional, Callable

# Prefer lxml for streaming on huge files and tolerant parsing
try:
    from lxml import etree as LET  # type: ignore
    HAVE_LXML = True
except Exception:
    import xml.etree.ElementTree as LET  # type: ignore
    HAVE_LXML = False

SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}

# Large-file threshold (bytes)
LARGE_BYTES = 200 * 1024 * 1024  # 200 MB


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
    rr, gg, bb = f"{r:02x}", f"{g:02x}", f"{b:02x}"
    if rr[0] == rr[1] and gg[0] == gg[1] and bb[0] == bb[1]:
        return f"#{rr[0]}{gg[0]}{bb[0]}"
    return f"#{rr}{gg}{bb}"


def _short_hex(hex_str: str) -> str:
    s = hex_str.strip().lower()
    if s.startswith("#") and len(s) == 7:
        h = s[1:]
        if h[0] == h[1] and h[2] == h[3] and h[4] == h[5]:
            return f"#{h[0]}{h[2]}{h[4]}"
    return hex_str


Point = Tuple[int, int]
RGBA = Tuple[float, float, float, float]
StyleKey = Tuple[str, float]


def _sorted_attribs(attrs: dict[str, str], prefer_order: List[str] | None = None) -> dict[str, str]:
    if not attrs:
        return {}
    prefer_order = prefer_order or []
    prefer_rank = {k: i for i, k in enumerate(prefer_order)}

    def key_sort(k: str) -> tuple[int, int, str]:
        pref = prefer_rank.get(k, len(prefer_order))
        d_last = 1 if k == "d" else 0
        return (pref, d_last, k)

    ordered: dict[str, str] = {}
    for k in sorted(attrs.keys(), key=key_sort):
        ordered[k] = attrs[k]
    return ordered


def _collect_final_rgba_pixels(svg_in: Path) -> tuple[Dict[StyleKey, Set[Point]], dict[str, str]]:
    """
    Non-streaming collector. Uses lxml tolerant parse (recover=True) when available.
    """
    if HAVE_LXML:
        parser = LET.XMLParser(recover=True, huge_tree=True)
        tree = LET.parse(str(svg_in), parser=parser)
        root = tree.getroot()
        rects = root.findall(".//svg:rect", namespaces=NS)
    else:
        tree = ET.parse(str(svg_in))
        root = tree.getroot()
        rects = root.findall(".//svg:rect", NS)

    if not rects:
        raise ValueError("No <rect> elements found. This script expects pixel-rect SVGs.")

    pix_rgba: Dict[Point, RGBA] = {}

    for r in rects:
        st = parse_style(r.get("style"))
        fill_raw = st.get("fill", r.get("fill"))
        if not fill_raw or str(fill_raw).strip().lower() == "none":
            continue

        op = norm_opacity(st.get("opacity", r.get("opacity")))
        fop = norm_opacity(st.get("fill-opacity", r.get("fill-opacity")))
        a_src = op * fop
        if a_src <= 0.0:
            continue

        rgb_src = _parse_rgb(str(fill_raw))
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
                out_a = a_src + da * (1.0 - a_src)
                if out_a <= 1e-12:
                    pix_rgba[(xx, yy)] = (0.0, 0.0, 0.0, 0.0)
                    continue
                out_r = (rgb_src[0] * a_src + dr * da * (1.0 - a_src)) / out_a
                out_g = (rgb_src[1] * a_src + dg * da * (1.0 - a_src)) / out_a
                out_b = (rgb_src[2] * a_src + db * da * (1.0 - a_src)) / out_a
                pix_rgba[(xx, yy)] = (out_r, out_g, out_b, out_a)

    style_pixels: Dict[StyleKey, Set[Point]] = defaultdict(set)
    for pt, (r, g, b, a) in pix_rgba.items():
        if a <= 0.0:
            continue
        hex_rgb = _rgb_to_hex((r, g, b)).lower()
        style_pixels[(hex_rgb, round(a, 6))].add(pt)

    out_attrs: dict[str, str] = {}
    for k in ("width", "height", "viewBox", "preserveAspectRatio"):
        v = root.get(k)
        if v:
            out_attrs[k] = v
    out_attrs["shape-rendering"] = "crispEdges"
    return style_pixels, out_attrs


def _collect_final_rgba_pixels_stream(svg_in: Path, progress_cb: Optional[Callable[[float], None]] = None, report_every_bytes: int = 4 * 1024 * 1024) -> tuple[Dict[StyleKey, Set[Point]], dict[str, str]]:
    ns_svg = SVG_NS
    rect_tag = f"{{{ns_svg}}}rect"
    svg_tag = f"{{{ns_svg}}}svg"

    class ProgressFile:
        def __init__(self, path: Path, total_bytes: int, cb: Optional[Callable[[float], None]], report_every: int):
            self._f = open(path, "rb")
            self._total = max(1, int(total_bytes))
            self._cb = cb
            self._report_every = max(1024, int(report_every))
            self._read = 0
            self._next = self._report_every

        def read(self, size: int = -1) -> bytes:
            data = self._f.read(size)
            self._read += len(data)
            if self._cb and self._read >= self._next:
                try:
                    self._cb((self._read / self._total) * 100.0)
                except Exception:
                    pass
                self._next += self._report_every
            return data

        def __getattr__(self, name: str):
            return getattr(self._f, name)

        def close(self) -> None:
            try:
                self._f.close()
            except Exception:
                pass

    fileobj = None
    try:
        try:
            total = svg_in.stat().st_size
        except Exception:
            total = 0
        fileobj = ProgressFile(svg_in, total, cb=progress_cb, report_every=report_every_bytes)

        if HAVE_LXML:
            context = LET.iterparse(fileobj, events=("start", "end"), recover=True, huge_tree=True)
        else:
            context = LET.iterparse(fileobj, events=("start", "end"))

        root_attrs: dict[str, str] = {}
        pix_rgba: Dict[Point, RGBA] = {}

        for event, elem in context:
            if event == "start" and elem.tag == svg_tag:
                for k in ("width", "height", "viewBox", "preserveAspectRatio"):
                    v = elem.attrib.get(k)
                    if v:
                        root_attrs[k] = v
                break

        for event, elem in context:
            if event == "end" and elem.tag == rect_tag:
                st = elem.attrib.get("style", "")
                fill = elem.attrib.get("fill")
                opacity = elem.attrib.get("opacity")
                fill_opacity = elem.attrib.get("fill-opacity")

                if st:
                    for part in st.split(";"):
                        part = part.strip()
                        if not part or ":" not in part:
                            continue
                        k, v = part.split(":", 1)
                        k = k.strip().lower()
                        v = v.strip()
                        if k == "fill":
                            fill = v
                        elif k == "opacity":
                            opacity = v
                        elif k == "fill-opacity":
                            fill_opacity = v

                if not fill or str(fill).strip().lower() == "none":
                    if HAVE_LXML:
                        try:
                            elem.clear()
                            while hasattr(elem, "getprevious") and elem.getprevious() is not None:
                                del elem.getparent()[0]
                        except Exception:
                            pass
                    continue

                op = norm_opacity(opacity)
                fop = norm_opacity(fill_opacity)
                a_src = op * fop
                if a_src <= 0.0:
                    if HAVE_LXML:
                        try:
                            elem.clear()
                            while hasattr(elem, "getprevious") and elem.getprevious() is not None:
                                del elem.getparent()[0]
                        except Exception:
                            pass
                    continue

                rgb_src = _parse_rgb(str(fill))
                x0 = _as_int(elem.attrib.get("x"), 0)
                y0 = _as_int(elem.attrib.get("y"), 0)
                w0 = _as_int(elem.attrib.get("width"), 1)
                h0 = _as_int(elem.attrib.get("height"), 1)
                if w0 <= 0:
                    w0 = 1
                if h0 <= 0:
                    h0 = 1

                for yy in range(y0, y0 + h0):
                    for xx in range(x0, x0 + w0):
                        dr, dg, db, da = pix_rgba.get((xx, yy), (0.0, 0.0, 0.0, 0.0))
                        out_a = a_src + da * (1.0 - a_src)
                        if out_a <= 1e-12:
                            pix_rgba[(xx, yy)] = (0.0, 0.0, 0.0, 0.0)
                            continue
                        out_r = (rgb_src[0] * a_src + dr * da * (1.0 - a_src)) / out_a
                        out_g = (rgb_src[1] * a_src + dg * da * (1.0 - a_src)) / out_a
                        out_b = (rgb_src[2] * a_src + db * da * (1.0 - a_src)) / out_a
                        pix_rgba[(xx, yy)] = (out_r, out_g, out_b, out_a)

                if HAVE_LXML:
                    try:
                        elem.clear()
                        while hasattr(elem, "getprevious") and elem.getprevious() is not None:
                            del elem.getparent()[0]
                    except Exception:
                        pass

        style_pixels: Dict[StyleKey, Set[Point]] = defaultdict(set)
        for pt, (r, g, b, a) in pix_rgba.items():
            if a <= 0.0:
                continue
            hex_rgb = _rgb_to_hex((r, g, b)).lower()
            style_pixels[(hex_rgb, round(a, 6))].add(pt)

        out_attrs: dict[str, str] = {}
        for k in ("width", "height", "viewBox", "preserveAspectRatio"):
            v = root_attrs.get(k)
            if v:
                out_attrs[k] = v
        out_attrs["shape-rendering"] = "crispEdges"
        if progress_cb:
            try:
                progress_cb(100.0)
            except Exception:
                pass
        return style_pixels, out_attrs
    finally:
        try:
            if fileobj is not None:
                fileobj.close()
        except Exception:
            pass


def _build_rect_list(svg_in: Path, vertical_merge: bool = True) -> tuple[list[tuple[int, int, int, int, tuple[str, float]]], dict[str, str]]:
    style_pixels, out_attrs = _collect_final_rgba_pixels(svg_in)

    rows: dict[int, dict[tuple[str, float], list[int]]] = defaultdict(lambda: defaultdict(list))
    for stylekey, pts in style_pixels.items():
        for (x, y) in pts:
            rows[y][stylekey].append(x)

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

    rect_list_sorted = sorted(rect_list, key=lambda t: (t[1], t[0], t[2], t[3]))
    return rect_list_sorted, out_attrs


def _build_rect_list_progress(svg_in: Path, vertical_merge: bool = True, progress_cb: Optional[Callable[[float], None]] = None) -> tuple[list[tuple[int, int, int, int, tuple[str, float]]], dict[str, str]]:
    style_pixels, out_attrs = _collect_final_rgba_pixels_stream(svg_in, progress_cb=progress_cb)

    rows: dict[int, dict[tuple[str, float], list[int]]] = defaultdict(lambda: defaultdict(list))
    for stylekey, pts in style_pixels.items():
        for (x, y) in pts:
            rows[y][stylekey].append(x)

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

    rect_list_sorted = sorted(rect_list, key=lambda t: (t[1], t[0], t[2], t[3]))
    if progress_cb:
        try:
            progress_cb(100.0)
        except Exception:
            pass
    return rect_list_sorted, out_attrs


def optimize_svg_rects_bytes(svg_in: Path, vertical_merge: bool = True, minify: bool = True, progress_cb: Optional[Callable[[float], None]] = None) -> tuple[bytes, int]:
    if progress_cb:
        rect_list_sorted, out_attrs = _build_rect_list_progress(svg_in, vertical_merge=vertical_merge, progress_cb=progress_cb)
    else:
        rect_list_sorted, out_attrs = _build_rect_list(svg_in, vertical_merge=vertical_merge)

    if minify and out_attrs.get("preserveAspectRatio", "").strip() == "xMidYMid meet":
        out_attrs.pop("preserveAspectRatio", None)

    ET.register_namespace("", SVG_NS)
    new_root = ET.Element(f"{{{SVG_NS}}}svg", _sorted_attribs(out_attrs, prefer_order=["width", "height", "viewBox", "preserveAspectRatio", "shape-rendering"]))
    g = ET.SubElement(new_root, f"{{{SVG_NS}}}g")

    for x, y, w, h, (fill, op) in rect_list_sorted:
        r_attrs = {"x": str(x), "y": str(y), "width": str(w), "height": str(h)}
        r_attrs["fill"] = fill
        if abs(op - 1.0) > 1e-6:
            r_attrs["opacity"] = fmt_opacity(op)
        ET.SubElement(g, f"{{{SVG_NS}}}rect", _sorted_attribs(r_attrs, prefer_order=["x", "y", "width", "height", "fill", "opacity"]))

    if minify:
        _postprocess_minify(new_root, mode="rects", have_holes=False)

    data = ET.tostring(new_root, encoding="utf-8", xml_declaration=not minify)
    if progress_cb:
        try:
            progress_cb(100.0)
        except Exception:
            pass
    return data, len(rect_list_sorted)


PointT = Tuple[int, int]
Edge = Tuple[PointT, PointT]

def _connected_components(pixels: Set[PointT]) -> List[Set[PointT]]:
    comps: List[Set[PointT]] = []
    unvisited = set(pixels)
    while unvisited:
        start = min(unvisited, key=lambda p: (p[1], p[0]))
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
    return (a, b) if a <= b else (b, a)


def _component_edges(comp: Set[PointT]) -> Set[Edge]:
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
    adj: Dict[PointT, List[PointT]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for v in adj:
        adj[v].sort()

    visited: Set[Edge] = set()
    cycles: List[List[PointT]] = []

    for start in sorted(adj.keys(), key=lambda p: (p[1], p[0])):
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
    if not points:
        return points

    dedup: List[PointT] = []
    for p in points:
        if not dedup or p != dedup[-1]:
            dedup.append(p)

    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup = dedup[:-1]
    if len(dedup) <= 2:
        return dedup + ([dedup[0]] if dedup else [])

    no_rev: List[PointT] = [dedup[0]]
    for i in range(1, len(dedup) - 1):
        if dedup[i - 1] == dedup[i + 1]:
            continue
        no_rev.append(dedup[i])
    no_rev.append(dedup[-1])

    corners: List[PointT] = [no_rev[0]]
    for i in range(1, len(no_rev) - 1):
        a, b, c = no_rev[i - 1], no_rev[i], no_rev[i + 1]
        if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
            continue
        corners.append(b)
    corners.append(no_rev[-1])
    corners.append(corners[0])
    return corners


def _relative_steps_from_corners(corners: List[PointT]) -> str:
    last_axis: str | None = None
    last_value: int = 0
    parts: List[str] = []
    for i in range(1, len(corners)):
        x, y = corners[i]
        px, py = corners[i - 1]
        dx = x - px
        dy = y - py
        if dy == 0 and dx != 0:
            if last_axis == "h":
                last_value += dx
                parts[-1] = f"h{last_value}"
            else:
                parts.append(f"h{dx}")
                last_axis = "h"
                last_value = dx
        elif dx == 0 and dy != 0:
            if last_axis == "v":
                last_value += dy
                parts[-1] = f"v{last_value}"
            else:
                parts.append(f"v{dy}")
                last_axis = "v"
                last_value = dy
        else:
            parts.append(f"l{dx} {dy}")
            last_axis = None
            last_value = 0
    parts.append("z")
    return " ".join(parts)


def _component_path_parts_from_cycles(cycles: List[List[PointT]]) -> tuple[PointT, PointT, str]:
    first_start: PointT | None = None
    last_start: PointT | None = None
    prev_start: PointT | None = None
    body_parts: List[str] = []

    for cyc in cycles:
        corners = _compress_corners(cyc)
        if not corners:
            continue

        start = corners[0]
        if first_start is None:
            first_start = start
        else:
            dx = start[0] - (prev_start[0] if prev_start else 0)
            dy = start[1] - (prev_start[1] if prev_start else 0)
            if dx != 0 or dy != 0:
                body_parts.append(f"m{dx} {dy}")

        body_parts.append(_relative_steps_from_corners(corners))
        prev_start = start
        last_start = start

    if first_start is None:
        return (0, 0), (0, 0), ""
    return first_start, last_start, " ".join(body_parts).strip()


def optimize_svg_paths_bytes(svg_in: Path, minify: bool = True, progress_cb: Optional[Callable[[float], None]] = None) -> tuple[bytes, int]:
    if progress_cb:
        style_pixels, out_attrs = _collect_final_rgba_pixels_stream(svg_in, progress_cb=progress_cb)
    else:
        style_pixels, out_attrs = _collect_final_rgba_pixels(svg_in)

    if minify and out_attrs.get("preserveAspectRatio", "").strip() == "xMidYMid meet":
        out_attrs.pop("preserveAspectRatio", None)

    ET.register_namespace("", SVG_NS)
    new_root = ET.Element(f"{{{SVG_NS}}}svg", _sorted_attribs(out_attrs, prefer_order=["width", "height", "viewBox", "preserveAspectRatio", "shape-rendering"]))
    g = ET.SubElement(new_root, f"{{{SVG_NS}}}g")

    path_count = 0
    have_holes_global = False

    for (fill_hex, op) in sorted(style_pixels.keys(), key=lambda k: (k[0], k[1])):
        pixels = style_pixels[(fill_hex, op)]
        if not pixels:
            continue

        comps = _connected_components(pixels)
        for comp in comps:
            edges = _component_edges(comp)
            cycles = _edges_to_cycles(edges)
            has_holes = len(cycles) > 1
            have_holes_global = have_holes_global or has_holes

            first_start, last_start, body = _component_path_parts_from_cycles(cycles)
            if first_start == (0, 0) and not body:
                continue

            d_full = f"M {first_start[0]} {first_start[1]}{(' ' + body) if body else ''}"

            attrs = {"d": d_full, "fill": fill_hex}
            if not minify and has_holes:
                attrs["fill-rule"] = "evenodd"
            if abs(op - 1.0) > 1e-6:
                attrs["opacity"] = fmt_opacity(op)

            if minify:
                attrs["data-first"] = f"{first_start[0]},{first_start[1]}"
                attrs["data-last"] = f"{last_start[0]},{last_start[1]}"
                attrs["data-body"] = body
                attrs["data-hole"] = "1" if has_holes else "0"

            ET.SubElement(g, f"{{{SVG_NS}}}path", _sorted_attribs(attrs, prefer_order=["d", "fill", "opacity", "fill-rule"]))
            path_count += 1

    if minify:
        _postprocess_minify(new_root, mode="paths", have_holes=have_holes_global)

    data = ET.tostring(new_root, encoding="utf-8", xml_declaration=not minify)
    if progress_cb:
        try:
            progress_cb(100.0)
        except Exception:
            pass
    return data, path_count


def _minify_path_d(d: str) -> str:
    if not d:
        return d
    d_min = " ".join(d.split())
    for cmd in ("H", "V", "M", "L", "h", "v", "m", "l"):
        d_min = d_min.replace(f"{cmd} ", f"{cmd}")
    for cmd in ("M", "H", "V", "L", "Z", "m", "h", "v", "l", "z"):
        d_min = d_min.replace(f" {cmd}", cmd)
    d_min = d_min.replace(" Z", "Z").replace("Z ", "Z")
    d_min = d_min.replace(" z", "z").replace("z ", "z")
    return d_min


def _collapse_outer_group_if_empty(root: ET.Element) -> None:
    g = None
    for child in list(root):
        if child.tag == f"{{{SVG_NS}}}g":
            g = child
            break
    if g is None or g.attrib:
        return
    idx = list(root).index(g)
    for ch in list(g):
        root.insert(idx, ch)
        idx += 1
    root.remove(g)


def _sort_all_attributes(root: ET.Element) -> None:
    for el in root.iter():
        if el.attrib:
            items = sorted(el.attrib.items())
            el.attrib.clear()
            for k, v in items:
                el.attrib[k] = v


def _postprocess_minify(root: ET.Element, mode: str, have_holes: bool = False) -> None:
    pa = root.attrib.get("preserveAspectRatio", "")
    if pa.strip() == "xMidYMid meet":
        root.attrib.pop("preserveAspectRatio", None)

    sr = root.attrib.get("shape-rendering")
    if sr != "crispEdges":
        root.attrib["shape-rendering"] = "crispEdges"

    g = None
    for child in list(root):
        if child.tag == f"{{{SVG_NS}}}g":
            g = child
            break
    if g is None:
        _sort_all_attributes(root)
        return

    if mode == "paths":
        by_style: Dict[Tuple[str, str], List[Tuple[Tuple[int, int], Tuple[int, int], str]]] = defaultdict(list)
        style_has_holes: Dict[Tuple[str, str], bool] = defaultdict(bool)

        paths = [p for p in list(g) if p.tag == f"{{{SVG_NS}}}path"]

        for p in paths:
            fill = p.attrib.get("fill", "")
            op = p.attrib.get("opacity", "")
            first_s = p.attrib.get("data-first")
            last_s = p.attrib.get("data-last")
            body = p.attrib.get("data-body", "")
            hole_flag = p.attrib.get("data-hole", "0") == "1"

            p.attrib.pop("fill-rule", None)
            p.attrib.pop("data-first", None)
            p.attrib.pop("data-last", None)
            p.attrib.pop("data-body", None)
            p.attrib.pop("data-hole", None)

            if first_s and last_s:
                fx, fy = (int(v) for v in first_s.split(",", 1))
                lx, ly = (int(v) for v in last_s.split(",", 1))
                body_min = _minify_path_d(body) if body else ""
                by_style[(fill, op)].append(((fx, fy), (lx, ly), body_min))
                style_has_holes[(fill, op)] = style_has_holes[(fill, op)] or hole_flag
            else:
                d_raw = p.attrib.get("d", "")
                if d_raw:
                    by_style[(fill, op)].append(((0, 0), (0, 0), _minify_path_d(d_raw)))

        for p in paths:
            g.remove(p)

        styles_with_holes = sum(1 for k in by_style.keys() if style_has_holes.get(k, False))
        use_group_evenodd = styles_with_holes > 1

        if use_group_evenodd:
            g.attrib["fill-rule"] = "evenodd"
        else:
            g.attrib.pop("fill-rule", None)

        for (fill, op), entries in sorted(by_style.items(), key=lambda k: (k[0][0], k[0][1])):
            if not entries:
                continue
            short_fill = _short_hex(fill) if fill else fill

            merged_parts: List[str] = []
            cur_start: Tuple[int, int] | None = None

            for (first_start, last_start, body) in entries:
                fx, fy = first_start
                lx, ly = last_start
                if cur_start is None:
                    merged_parts.append(f"M {fx} {fy}")
                else:
                    merged_parts.append(f"m{fx - cur_start[0]} {fy - cur_start[1]}")
                if body:
                    merged_parts.append(body)
                cur_start = (lx, ly)

            merged_d = _minify_path_d("".join(merged_parts))
            attrs = {"d": merged_d, "fill": short_fill}
            if op and op not in ("", "1", "1.0"):
                attrs["opacity"] = op
            if not use_group_evenodd and style_has_holes.get((fill, op), False):
                attrs["fill-rule"] = "evenodd"

            ET.SubElement(g, f"{{{SVG_NS}}}path", attrs)

        if not use_group_evenodd and not g.attrib:
            _collapse_outer_group_if_empty(root)

    elif mode == "rects":
        rects = [r for r in list(g) if r.tag == f"{{{SVG_NS}}}rect"]
        by_style: Dict[Tuple[str, str], List[ET.Element]] = defaultdict(list)
        for r in rects:
            fill = r.attrib.get("fill", "")
            op = r.attrib.get("opacity", "")
            by_style[(fill, op)].append(r)

        for r in rects:
            g.remove(r)

        for (fill, op), elems in sorted(by_style.items(), key=lambda k: (k[0][0], k[0][1])):
            short_fill = _short_hex(fill) if fill else fill
            sub_attrs: Dict[str, str] = {"fill": short_fill} if short_fill else {}
            if op and op not in ("", "1", "1.0"):
                sub_attrs["opacity"] = op
            sub = ET.SubElement(g, f"{{{SVG_NS}}}g", sub_attrs)
            for r in elems:
                r.attrib.pop("fill", None)
                r.attrib.pop("opacity", None)
                sub.append(r)

        if not g.attrib:
            _collapse_outer_group_if_empty(root)

    def _remove_empty_groups_rec(el: ET.Element) -> None:
        for ch in list(el):
            _remove_empty_groups_rec(ch)
        for ch in list(el):
            if ch.tag == f"{{{SVG_NS}}}g" and not ch.attrib and len(list(ch)) == 0:
                el.remove(ch)

    _remove_empty_groups_rec(root)
    _sort_all_attributes(root)


def optimize_svg_rects(svg_in: Path, svg_out: Path, vertical_merge: bool = True, minify: bool = True) -> tuple[int, int]:
    data, rect_count = optimize_svg_rects_bytes(svg_in, vertical_merge=vertical_merge, minify=minify)
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    svg_out.write_bytes(data)
    return rect_count, len(data)


def write_svgz(svg_bytes: bytes, svgz_out: Path, compresslevel: int = 9, use_zopfli: bool = False) -> int:
    svgz_out.parent.mkdir(parents=True, exist_ok=True)
    if use_zopfli:
        try:
            import zopfli.gzip
            with open(svgz_out, "wb") as f:
                f.write(zopfli.gzip.compress(svg_bytes))
            return svgz_out.stat().st_size
        except Exception:
            pass
    level = max(1, min(9, int(compresslevel)))
    with gzip.open(svgz_out, "wb", compresslevel=level) as f:
        f.write(svg_bytes)
    return svgz_out.stat().st_size


def default_output_path(inp: Path, vertical_merge: bool) -> Path:
    suffix = "_optimized_hv" if vertical_merge else "_optimized_h"
    return inp.with_name(inp.stem + suffix + inp.suffix)


class ProgressFile:
    """
    Wrap a file object to report read progress via a callback.
    lxml.etree.iterparse will call .read(); we count bytes and report periodically.
    """
    def __init__(self, path: Path, total_bytes: int, cb: Optional[Callable[[float], None]] = None, report_every_bytes: int = 4 * 1024 * 1024):
        self._f = open(path, "rb")
        self._total = max(1, int(total_bytes))
        self._cb = cb
        self._report_every = max(1024, int(report_every_bytes))
        self._read = 0
        self._next = self._report_every

    def read(self, size: int = -1) -> bytes:
        data = self._f.read(size)
        self._read += len(data)
        if self._cb and self._read >= self._next:
            try:
                pct = (self._read / self._total) * 100.0
                self._cb(pct)
            except Exception:
                pass
            self._next += self._report_every
        return data

    def __getattr__(self, name: str):
        return getattr(self._f, name)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def _iterparse_rects(svg_in: Path, progress_cb: Optional[Callable[[float], None]] = None, report_every_bytes: int = 4 * 1024 * 1024):
    """
    Yield rects as (x:int, y:int, w:int, h:int, fill:str, op:str).
    First yield is ("__root__", root_attrs_dict).
    If progress_cb is provided, progress is bytes-based on input reads.
    """
    ns_svg = SVG_NS
    rect_tag = f"{{{ns_svg}}}rect"
    svg_tag = f"{{{ns_svg}}}svg"

    fileobj = None
    context = None
    try:
        if progress_cb:
            try:
                total = svg_in.stat().st_size
            except Exception:
                total = 0
            fileobj = ProgressFile(svg_in, total, cb=progress_cb, report_every_bytes=report_every_bytes)
            if HAVE_LXML:
                context = LET.iterparse(fileobj, events=("start", "end"), recover=True, huge_tree=True)
            else:
                context = LET.iterparse(fileobj, events=("start", "end"))
        else:
            if HAVE_LXML:
                context = LET.iterparse(str(svg_in), events=("start", "end"), recover=True, huge_tree=True)
            else:
                context = LET.iterparse(str(svg_in), events=("start", "end"))

        root_attrs: dict[str, str] = {}

        for event, elem in context:
            if event == "start" and elem.tag == svg_tag:
                for k in ("width", "height", "viewBox", "preserveAspectRatio"):
                    v = elem.attrib.get(k)
                    if v:
                        root_attrs[k] = v
                yield ("__root__", root_attrs)
                break

        for event, elem in context:
            if event == "end" and elem.tag == rect_tag:
                st = elem.attrib.get("style", "")
                fill = elem.attrib.get("fill", "")
                op = elem.attrib.get("opacity", "")

                if st:
                    for part in st.split(";"):
                        part = part.strip()
                        if not part or ":" not in part:
                            continue
                        k, v = part.split(":", 1)
                        k = k.strip().lower()
                        v = v.strip()
                        if k == "fill":
                            fill = v
                        elif k == "opacity":
                            op = v

                def _as_int_or(default: int, s: Optional[str]):
                    if not s:
                        return default
                    try:
                        return int(float(s))
                    except Exception:
                        return default

                x = _as_int_or(0, elem.attrib.get("x"))
                y = _as_int_or(0, elem.attrib.get("y"))
                w = _as_int_or(1, elem.attrib.get("width"))
                h = _as_int_or(1, elem.attrib.get("height"))

                try:
                    of = float(op) if op else 1.0
                    if of > 1.0:
                        op = fmt_opacity(of / 255.0)
                    else:
                        op = fmt_opacity(of)
                except Exception:
                    op = ""

                yield (x, y, w, h, fill, op)

                if HAVE_LXML:
                    try:
                        elem.clear()
                        while hasattr(elem, "getprevious") and elem.getprevious() is not None:
                            del elem.getparent()[0]
                    except Exception:
                        pass
    finally:
        try:
            if fileobj is not None:
                fileobj.close()
        except Exception:
            pass


def optimize_svg_rects_stream(svg_in: Path, svg_out: Path, minify: bool = True, progress_cb: Optional[Callable[[float], None]] = None, report_every_bytes: int = 4 * 1024 * 1024) -> None:
    """
    Streaming rect optimizer:
    - Merges contiguous runs per row per (fill,opacity)
    - Stacks identical (x,w,style) runs vertically across successive rows
    - Writes directly to svg_out
    If progress_cb is provided, it is bytes-based on input reads.
    """
    svg_out.parent.mkdir(parents=True, exist_ok=True)

    it = _iterparse_rects(svg_in, progress_cb=progress_cb, report_every_bytes=report_every_bytes)
    first = next(it, None)
    if not first or first[0] != "__root__":
        raise ValueError("Invalid SVG: missing <svg> root")

    root_attrs = first[1]
    out_attrs: dict[str, str] = {}
    for k in ("width", "height", "viewBox"):
        v = root_attrs.get(k)
        if v:
            out_attrs[k] = v
    pa = root_attrs.get("preserveAspectRatio", "")
    if not minify or pa.strip() != "xMidYMid meet":
        if pa:
            out_attrs["preserveAspectRatio"] = pa
    out_attrs["shape-rendering"] = "crispEdges"

    def _sorted_attrib_line(attrs: dict[str, str]) -> str:
        ordered = _sorted_attribs(attrs, prefer_order=["width", "height", "viewBox", "preserveAspectRatio", "shape-rendering"])
        parts = [f'{k}="{v}"' for k, v in ordered.items()]
        return " ".join(parts)

    with open(svg_out, "w", encoding="utf-8") as f:
        f_write = f.write
        if not minify:
            f_write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
        f_write(f'<svg xmlns="{SVG_NS}" {_sorted_attrib_line(out_attrs)}>')
        f_write('<g>')

        active: dict[tuple[int, int, str, str], list[int]] = {}
        current_y: Optional[int] = None
        row_state: dict[tuple[str, str], tuple[int, int]] = {}
        row_runs: list[tuple[int, int, str, str]] = []

        def flush_row(y_done: int):
            for (fill, op), (sx, px) in list(row_state.items()):
                w = (px - sx + 1)
                row_runs.append((sx, w, fill, op))
            row_state.clear()

            row_keys = {(x, w, fill, op) for (x, w, fill, op) in row_runs}

            out_lines: List[str] = []

            for key in list(active.keys()):
                if key not in row_keys:
                    sy, ly = active[key]
                    x, w, fill, op = key
                    h = ly - sy + 1
                    if op and op not in ("", "1", "1.0"):
                        out_lines.append(f'<rect x="{x}" y="{sy}" width="{w}" height="{h}" fill="{fill}" opacity="{op}"></rect>')
                    else:
                        out_lines.append(f'<rect x="{x}" y="{sy}" width="{w}" height="{h}" fill="{fill}"></rect>')
                    del active[key]

            for (x, w, fill, op) in row_runs:
                key = (x, w, fill, op)
                if key in active:
                    sy, ly = active[key]
                    if y_done == ly + 1:
                        active[key][1] = y_done
                    else:
                        h = ly - sy + 1
                        if op and op not in ("", "1", "1.0"):
                            out_lines.append(f'<rect x="{x}" y="{sy}" width="{w}" height="{h}" fill="{fill}" opacity="{op}"></rect>')
                        else:
                            out_lines.append(f'<rect x="{x}" y="{sy}" width="{w}" height="{h}" fill="{fill}"></rect>')
                        active[key] = [y_done, y_done]
                else:
                    active[key] = [y_done, y_done]

            row_runs.clear()
            if out_lines:
                f_write("".join(out_lines))

        for item in it:
            if not isinstance(item[0], int):
                continue
            x, y, w, h, fill, op = item

            if current_y is None:
                current_y = y
            elif y != current_y:
                flush_row(current_y)
                current_y = y

            for yy in range(y, y + h):
                if current_y != yy:
                    flush_row(current_y)
                    current_y = yy
                for s_run_x in range(x, x + w):
                    key = (fill, op)
                    if key in row_state:
                        sx, px = row_state[key]
                        if s_run_x == px + 1:
                            row_state[key] = (sx, s_run_x)
                        else:
                            rw = (px - sx + 1)
                            row_runs.append((sx, rw, fill, op))
                            row_state[key] = (s_run_x, s_run_x)
                    else:
                        row_state[key] = (s_run_x, s_run_x)

        if current_y is not None:
            flush_row(current_y)

        out_lines_end: List[str] = []
        for key, (sy, ly) in list(active.items()):
            x, w, fill, op = key
            h = ly - sy + 1
            if op and op not in ("", "1", "1.0"):
                out_lines_end.append(f'<rect x="{x}" y="{sy}" width="{w}" height="{h}" fill="{fill}" opacity="{op}"></rect>')
            else:
                out_lines_end.append(f'<rect x="{x}" y="{sy}" width="{w}" height="{h}" fill="{fill}"></rect>')
        if out_lines_end:
            f_write("".join(out_lines_end))

        f_write('</g></svg>')

    try:
        if progress_cb:
            progress_cb(100.0)
    except Exception:
        pass


def write_svgz_stream_from_svg(svg_in: Path, svgz_out: Path, compresslevel: int = 9, progress_cb: Optional[Callable[[float], None]] = None) -> None:
    """
    Stream-copy an SVG file into SVGZ without holding bytes in memory.
    If progress_cb is provided, it is called with 0..100 based on bytes copied.
    """
    svgz_out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        total = max(1, int(svg_in.stat().st_size))
    except Exception:
        total = 0
    copied = 0
    chunk_size = 4 * 1024 * 1024

    with open(svg_in, "rb") as fin, gzip.open(svgz_out, "wb", compresslevel=max(1, min(9, compresslevel))) as fout:
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            fout.write(chunk)
            copied += len(chunk)
            if progress_cb and total > 0:
                try:
                    progress_cb(min(100.0, (copied / total) * 100.0))
                except Exception:
                    pass
    if progress_cb:
        try:
            progress_cb(100.0)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge/optimize pixel-rect SVGs.")
    ap.add_argument("input", nargs="+", help="Input SVG file(s). Supports wildcards via shell.")
    ap.add_argument("-o", "--output", help="Output path (only valid with a single input).")
    ap.add_argument("--paths", action="store_true", help="Emit connected <path> shapes per final RGBA (replaces <rect> output).")
    ap.add_argument("--no-vertical", action="store_true", help="Disable vertical stacking merge (rect mode only).")
    ap.add_argument("--svgz", action="store_true", help="Also write a .svgz (gzipped) alongside the .svg output.")
    ap.add_argument("--svgz-only", action="store_true", help="Write only .svgz output (implies --svgz).")
    ap.add_argument("--svgz-level", type=int, default=9, help="GZip level for .svgz (1-9, default 9).")
    ap.add_argument("--zopfli", action="store_true", help="Use Zopfli to produce smaller .svgz (requires python-zopfli).")
    ap.add_argument("--minify", action="store_true", help="Post-process output: merge same-color shapes, minify attributes, drop XML header, and remove default preserveAspectRatio.")
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

            size = 0
            try:
                size = inp.stat().st_size
            except Exception:
                size = 0

            if size >= LARGE_BYTES:
                optimize_svg_rects_stream(inp, out_svg, minify=args.minify)
                print(f"Wrote (stream): {out_svg} | mode=rects-stream | lxml={'yes' if HAVE_LXML else 'no'}")
                if args.svgz or args.svgz_only:
                    svgz_out = out_svg.with_suffix(out_svg.suffix + "z")
                    write_svgz_stream_from_svg(out_svg, svgz_out, compresslevel=args.svgz_level)
                    print(f"Wrote: {svgz_out} | mode=stream-gzip")
            else:
                if args.paths:
                    svg_bytes, path_count = optimize_svg_paths_bytes(inp, minify=args.minify)
                    if not args.svgz_only:
                        out_svg.write_bytes(svg_bytes)
                        print(f"Wrote: {out_svg} | paths: {path_count:,} | bytes: {len(svg_bytes):,}")
                    if args.svgz or args.svgz_only:
                        svgz_out = out_svg.with_suffix(out_svg.suffix + "z")
                        bytes_svgz = write_svgz(svg_bytes, svgz_out, compresslevel=args.svgz_level, use_zopfli=args.zopfli)
                        print(f"Wrote: {svgz_out} | bytes: {bytes_svgz:,}")
                else:
                    svg_bytes, rect_count = optimize_svg_rects_bytes(inp, vertical_merge=vertical, minify=args.minify)
                    if not args.svgz_only:
                        out_svg.write_bytes(svg_bytes)
                        print(f"Wrote: {out_svg} | rects: {rect_count:,} | bytes: {len(svg_bytes):,}")
                    if args.svgz or args.svgz_only:
                        svgz_out = out_svg.with_suffix(out_svg.suffix + "z")
                        bytes_svgz = write_svgz(svg_bytes, svgz_out, compresslevel=args.svgz_level, use_zopfli=args.zopfli)
                        print(f"Wrote: {svgz_out} | bytes: {bytes_svgz:,}")

        except Exception as e:
            print(f"Failed: {inp} | {e}")


if __name__ == "__main__":
    main()
