# utils_visualization.py
# Centralized visualization utilities:
# - real-time HUD sparkline (render_sparkline_strip)
# - ROM result panel (compose_result_panel + show_and_save_result_panel)
# - gallery compositor (compose_gallery)
# - ROM-only line drawer for angle visualization (draw_rom_lines)
# - small internal helpers for image/text layout

import os
import time
import math
from collections import deque

import numpy as np
import cv2
import mmcv
from pathlib import Path
from typing import List, Optional, Tuple

# ---------- internal helpers (kept private) ----------

def _stack_h(images, gap=8, bg_color=(24, 24, 24)):
    """Stack images horizontally with a gap; pad heights as needed."""
    imgs = [img.copy() for img in images if img is not None]
    if not imgs:
        return None
    hmax = max(im.shape[0] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[0] < hmax:
            pad = np.full((hmax - im.shape[0], im.shape[1], 3), bg_color, dtype=np.uint8)
            im = np.vstack([im, pad])
        padded.append(im)
    gaps = [np.full((hmax, gap, 3), bg_color, dtype=np.uint8) for _ in range(len(padded) - 1)]
    out = []
    for i, im in enumerate(padded):
        out.append(im)
        if i < len(gaps):
            out.append(gaps[i])
    return np.hstack(out)

def _text(img, txt, org, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def _load_img(path, fallback_shape=None):
    if path and os.path.isfile(path):
        im = mmcv.imread(path)
        if im is not None and im.ndim == 3:
            return im
    if fallback_shape is None:
        fallback_shape = (360, 640, 3)
    return np.full(fallback_shape, (40, 40, 40), dtype=np.uint8)

def _crop_bottom(img, crop_px: int):
    """Crop N pixels from the bottom (no-op if <=0)."""
    if img is None or crop_px <= 0:
        return img
    h = img.shape[0]
    cp = min(max(int(crop_px), 0), max(h - 1, 0))
    return img[:h - cp, :, :]

def _resize_keep_aspect(img, target_w):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

# ---------- NEW: ROM-only line drawing ----------

def _resolve_point_2d_with_scores(keypoints_xy, scores, idx, kpt_thr):
    """
    Return (x,y) if score ok, else None.
    idx may be int or [int,int] for midpoint. For list longer than 2, use mean of all valid.
    """
    if keypoints_xy is None or scores is None:
        return None
    K = int(len(keypoints_xy))
    def _one(i):
        try:
            j = int(i)
            if j < 0 or j >= K:
                return None
            if float(scores[j]) < float(kpt_thr):
                return None
            p = np.asarray(keypoints_xy[j], dtype=float)[:2]
            if not np.all(np.isfinite(p)):
                return None
            return p
        except Exception:
            return None

    if isinstance(idx, (list, tuple)):
        pts = [p for p in (_one(i) for i in idx) if p is not None]
        if not pts:
            return None
        p = np.mean(np.vstack(pts), axis=0)
        return (int(round(p[0])), int(round(p[1])))
    p = _one(idx)
    if p is None:
        return None
    return (int(round(p[0])), int(round(p[1])))

def _normalize_vecpairs(preset):
    """
    Normalize get_vectors_for_preset output to a list of vector-pairs:
      [ [[P0,P1],[Q0,Q1]], ... ]
    Accepts single pair or left/right pair.
    """
    if preset is None:
        return []
    # if it looks like [[p0,p1],[q0,q1]]
    if len(preset) == 2 and not isinstance(preset[0][0], (list, tuple)):
        return [preset]
    # else assume already a list of pairs (e.g., left/right)
    return list(preset)

def draw_rom_lines(frame_bgr, keypoints_xy, scores, preset, kpt_thr=0.3, thickness=2,
                   color_vec1=(0, 255, 0), color_vec2=(255, 255, 0)):
    """
    Draw only the ROM vectors used for angle computation on a BGR frame.
    - frame_bgr: np.ndarray (BGR). Modified and also returned.
    - keypoints_xy: (K,2)
    - scores: (K,)
    - preset: e.g. [[5,7], [[5,6],[11,12]]] or [ [[5,7],[[5,6],[11,12]]], [[...] , [...]] ] for L/R
    """
    if frame_bgr is None:
        return frame_bgr
    pairs = _normalize_vecpairs(preset)
    if not pairs:
        return frame_bgr

    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        vec1, vec2 = pair
        # vec1 = [P0,P1]; vec2 = [Q0,Q1]; entries may themselves be lists for midpoints
        B = _resolve_point_2d_with_scores(keypoints_xy, scores, vec1[0], kpt_thr)
        A = _resolve_point_2d_with_scores(keypoints_xy, scores, vec1[1], kpt_thr)
        C0 = _resolve_point_2d_with_scores(keypoints_xy, scores, vec2[0], kpt_thr)
        C1 = _resolve_point_2d_with_scores(keypoints_xy, scores, vec2[1], kpt_thr)

        if B is not None and A is not None:
            cv2.line(frame_bgr, B, A, color_vec1, int(thickness), cv2.LINE_AA)
        if C0 is not None and C1 is not None:
            cv2.line(frame_bgr, C0, C1, color_vec2, int(thickness), cv2.LINE_AA)

    return frame_bgr

# ---------- HUD: real-time sparkline ----------

def render_sparkline_strip(width, height, series, t_now, window_sec, label_text=None, gap_sec=0.4):
    """
    Draw a small history strip for the last `window_sec` seconds of angle values.
    - series: list[(timestamp, value)]
    - returns BGR image of size (height, width, 3)
    """
    strip = np.zeros((height, width, 3), dtype=np.uint8)

    if window_sec <= 0 or len(series) < 2:
        if label_text:
            cv2.putText(strip, label_text, (10, int(0.65 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return strip

    t0 = t_now - float(window_sec)
    filtered = [(max(t0, t), a) for (t, a) in series if (t >= t0 and np.isfinite(a))]
    if len(filtered) < 2:
        if label_text:
            cv2.putText(strip, label_text, (10, int(0.65 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return strip

    ts = [t for (t, _) in filtered]
    ys = [a for (_, a) in filtered]

    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    if (not np.isfinite(y_min)) or (not np.isfinite(y_max)) or abs(y_max - y_min) < 1e-9:
        y_min, y_max = (0.0, 1.0) if (not np.isfinite(y_min) or not np.isfinite(y_max)) else (y_min - 1.0, y_max + 1.0)

    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (32, 32, 32), thickness=-1)
    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (180, 180, 180), thickness=1)

    def map_x(t):
        tx = (t - t0) / max(window_sec, 1e-6)
        tx = 0.0 if not np.isfinite(tx) else min(max(tx, 0.0), 1.0)
        return int(round(tx * (width - 1)))

    def map_y(y):
        ty = (y - y_min) / max((y_max - y_min), 1e-6)
        ty = 0.0 if not np.isfinite(ty) else min(max(ty, 0.0), 1.0)
        return int(round((height - 1) - ty * (height - 1)))

    segments = []
    seg = [(map_x(ts[0]), map_y(ys[0]))]
    for i in range(1, len(ts)):
        if (ts[i] - ts[i - 1]) > gap_sec:
            if len(seg) >= 2:
                segments.append(np.array(seg, dtype=np.int32))
            seg = []
        seg.append((map_x(ts[i]), map_y(ys[i])))
    if len(seg) >= 2:
        segments.append(np.array(seg, dtype=np.int32))

    for poly in segments:
        cv2.polylines(strip, [poly], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.putText(strip, f"[{y_min:.1f}, {y_max:.1f}]", (10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    last_y = ys[-1]
    if np.isfinite(last_y):
        cv2.putText(strip, f"{last_y:.1f} deg", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if label_text:
        txt_w = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(strip, label_text, (width - txt_w - 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return strip

def render_text_strip(
    width: int,
    height: int,
    lines,
    bg_mode: str = "dark",
    left_pad: int = 12,
    line_gap: int = 8,
    font_scale: float = 0.8,
    thickness: int = 2,
):
    """
    Draw a text-only strip (BGR) of size (height, width).
    lines: list[str] or list[Tuple[str, Tuple[int,int,int]]]
           If tuple, second item is BGR color for that line.
    """
    if bg_mode not in ("dark", "light"):
        bg_mode = "dark"

    bg = (18, 18, 18) if bg_mode == "dark" else (245, 245, 245)
    fg_default = (255, 255, 255) if bg_mode == "dark" else (32, 32, 32)
    strip = np.full((height, width, 3), bg, dtype=np.uint8)

    (w0, h0), base0 = cv2.getTextSize("Hg", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    line_h = h0 + base0 + 2
    total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap
    y0 = max(line_h, (height - total_h) // 2 + h0)

    for i, item in enumerate(lines):
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            text = str(item[0])
            color = tuple(item[1]) if len(item) >= 2 else fg_default
        else:
            text = str(item)
            color = fg_default
        y = y0 + i * (line_h + line_gap)
        if y > height - 4:
            break
        cv2.putText(strip, text, (left_pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return strip

# ---------- Result panel (finalized summary) ----------

def compose_result_panel(test_name, status, session_stamp, trial_dir,
                         min_path=None, min_deg=None,
                         max_path=None, max_deg=None,
                         rom_deg=None,
                         crop_bottom_px: int = 0):
    """
    Build a composite image panel:
      [Title]
      [ MIN card | MAX card ]
      [Footer: ROM only (big)]
    Returns: np.ndarray (BGR)

    crop_bottom_px: crop HUD/sparkline from the bottom of each MIN/MAX image.
    """
    title_h = 64
    footer_h = 48
    pad = 16
    card_w = 640
    card_h = 360

    min_img = _load_img(min_path, (card_h, card_w, 3))
    max_img = _load_img(max_path, (card_h, card_w, 3))

    if crop_bottom_px > 0:
        min_img = _crop_bottom(min_img, crop_bottom_px)
        max_img = _crop_bottom(max_img, crop_bottom_px)

    min_card = min_img.copy()
    max_card = max_img.copy()
    min_txt = "--.-" if min_deg is None else f"{min_deg:.1f}"
    max_txt = "--.-" if max_deg is None else f"{max_deg:.1f}"
    _text(min_card, f"MIN: {min_txt} deg", (12, 28), 0.9, (0, 255, 255), 2)
    _text(max_card, f"MAX: {max_txt} deg", (12, 28), 0.9, (0, 255, 255), 2)

    row = _stack_h([min_card, max_card], gap=pad)
    if row is None:
        row = np.full((card_h, card_w * 2 + pad, 3), (18, 18, 18), dtype=np.uint8)

    W = row.shape[1]
    H = title_h + row.shape[0] + footer_h + pad * 2
    panel = np.full((H, W, 3), (18, 18, 18), dtype=np.uint8)

    title = f"{test_name} - {status.upper()}"
    _text(panel, title, (16, int(title_h * 0.7)), 1.2, (255, 255, 255), 2)

    panel[title_h:title_h + row.shape[0], 0:W, :] = row

    rom_txt = "--.-" if rom_deg is None else f"{rom_deg:.1f}"
    footer = f"ROM: {rom_txt} deg"
    _text(panel, footer, (16, title_h + row.shape[0] + int(footer_h * 0.8)), 1.1, (255, 255, 255), 2)

    return panel

# ---------- Gallery (collage of recent panels) ----------

def _compose_gallery(panel_imgs, cols=1, cell_w=560, pad=16, bg=(18, 18, 18)):
    """
    Build a collage from a list of panel images (BGR).
    - cols: desired columns (>=1)
    - cell_w: target width per cell
    """
    if not panel_imgs:
        return np.full((360, 640, 3), bg, dtype=np.uint8)

    cells = [_resize_keep_aspect(im, cell_w) for im in panel_imgs]
    n = len(cells)
    cols = max(1, int(cols))
    rows = int(math.ceil(n / cols))

    row_heights, row_widths = [], []
    for r in range(rows):
        used = min(cols, n - r * cols)
        hmax = max(cells[r * cols + c].shape[0] for c in range(used))
        wsum = sum(cells[r * cols + c].shape[1] for c in range(used)) + pad * (used - 1 if used > 1 else 0)
        row_heights.append(hmax)
        row_widths.append(wsum)

    total_h = sum(row_heights) + pad * (rows - 1 if rows > 1 else 0)
    total_w = max(row_widths)

    canvas = np.full((total_h, total_w, 3), bg, dtype=np.uint8)

    y = 0
    for r in range(rows):
        used = min(cols, n - r * cols)
        x = 0
        for c in range(used):
            cell = cells[r * cols + c]
            h, w = cell.shape[:2]
            y_off = (row_heights[r] - h) // 2
            canvas[y + y_off:y + y_off + h, x:x + w, :] = cell
            x += w + (pad if c < used - 1 else 0)
        y += row_heights[r] + (pad if r < rows - 1 else 0)

    return canvas

# ---------- Show & Save (single / gallery) ----------

def show_and_save_result_panel(args, state, result: dict):

    """
    Compose and save ROM_PANEL.png into the trial dir.
    If --show and --show-result:
      - single: one window showing the latest panel
      - gallery: grid of the last N panels
    """

    if getattr(args, 'debug', False):
        return

    test_name = result.get('test_name', getattr(args, 'rom_test', 'unknown'))
    status = result.get('status', 'ok')
    session_stamp = getattr(state, 'session_stamp', None) or time.strftime("%Y%m%d_%H%M%S")
    trial_dir = getattr(state, 'rom_save_dir', None) or os.path.join(getattr(args, 'output_root', '.') or '.', "unknown_trial")

    text_h = int(getattr(args, 'text_strip_height', 0) or 0)
    plot_sec = float(getattr(args, 'plot_seconds', 0.0) or 0.0)
    plot_h = int(getattr(args, 'plot_height', 0) or 0)
    crop_bottom_px = text_h + (plot_h if plot_sec > 0 else 0)

    panel = compose_result_panel(
        test_name=test_name,
        status=status,
        session_stamp=session_stamp,
        trial_dir=trial_dir,
        min_path=result.get('min_image'),
        min_deg=result.get('min_deg'),
        max_path=result.get('max_image'),
        max_deg=result.get('max_deg'),
        rom_deg=result.get('rom_deg'),
        crop_bottom_px=crop_bottom_px
    )

    panel_path = os.path.join(trial_dir, "ROM_PANEL.png")
    mmcv.imwrite(panel, panel_path)
    print(f"[ROM] Panel saved: {panel_path}")

    if not (getattr(args, 'show', False) and getattr(args, 'show_result', False)):
        return panel_path

    mode = getattr(args, 'panel_window', 'single')

    if mode == 'single':
        win = f"ROM Result - {test_name}"
        prev_win = getattr(state, 'result_window_name', None)
        if prev_win and prev_win != win:
            try:
                cv2.destroyWindow(prev_win)
            except Exception:
                pass
        state.result_window_name = win

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        target_w = min(1280, panel.shape[1])
        scale = target_w / float(panel.shape[1])
        target_h = int(panel.shape[0] * scale)
        cv2.resizeWindow(win, target_w, target_h)
        cv2.imshow(win, panel)
        return panel_path

    if not hasattr(state, 'gallery_paths') or state.gallery_paths is None:
        state.gallery_paths = deque(maxlen=max(1, int(getattr(args, 'gallery_size', 4))))
    else:
        desired_len = max(1, int(getattr(args, 'gallery_size', 4)))
        if state.gallery_paths.maxlen != desired_len:
            items = list(state.gallery_paths)
            state.gallery_paths = deque(items[-desired_len:], maxlen=desired_len)

    state.gallery_paths.append(panel_path)

    paths = list(state.gallery_paths)
    imgs = [ _load_img(p) for p in paths ]
    cols = max(1, int(getattr(args, 'gallery_cols', 2)))
    cell_w = max(240, int(getattr(args, 'gallery_cell_width', 560)))
    gallery = _compose_gallery(imgs, cols=cols, cell_w=cell_w, pad=16, bg=(18, 18, 18))

    win = "ROM Gallery"
    prev_win = getattr(state, 'result_window_name', None)
    if prev_win and prev_win != win:
        try:
            cv2.destroyWindow(prev_win)
        except Exception:
            pass
    state.result_window_name = win

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    target_w = min(1400, gallery.shape[1])
    scale = target_w / float(gallery.shape[1])
    target_h = int(gallery.shape[0] * scale)
    cv2.resizeWindow(win, target_w, target_h)
    cv2.imshow(win, gallery)

    return panel_path

# ---------- Simple compare panel (side-by-side) + saver ----------

def compose_compare_panel(imgA: np.ndarray,
                          imgB: np.ndarray,
                          header_lines: Optional[List[str]] = None,
                          footer_lines: Optional[List[str]] = None,
                          left_title: Optional[str] = None,
                          right_title: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Create a side-by-side BGR panel with a header and optional footer.
    Returns BGR np.ndarray or None if inputs invalid.
    """
    if imgA is None or imgB is None:
        return None

    def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        scale = target_h / float(img.shape[0])
        w = int(round(img.shape[1] * scale))
        return cv2.resize(img, (w, target_h), interpolation=cv2.INTER_AREA)

    # match heights
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    target_h = max(hA, hB)
    A = _resize_to_height(imgA, target_h)
    B = _resize_to_height(imgB, target_h)

    # optional per-side titles strip
    title_h = 0
    if left_title or right_title:
        title_h = 36
        title_strip = np.zeros((title_h, A.shape[1] + B.shape[1], 3), dtype=np.uint8)
        if left_title:
            cv2.putText(title_strip, str(left_title), (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        if right_title:
            # right title anchored roughly to the center of right half
            x0 = A.shape[1] + 12
            cv2.putText(title_strip, str(right_title), (x0, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    # header strip
    header_h = 0
    if header_lines:
        header_h = 48
        header = np.zeros((header_h, A.shape[1] + B.shape[1], 3), dtype=np.uint8)
        y0 = 30
        for i, line in enumerate(header_lines):
            cv2.putText(header, str(line), (12, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        header = None

    # footer strip
    footer_h = 0
    if footer_lines:
        footer_h = 36
        footer = np.zeros((footer_h, A.shape[1] + B.shape[1], 3), dtype=np.uint8)
        y0 = 24
        for i, line in enumerate(footer_lines):
            cv2.putText(footer, str(line), (12, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        footer = None

    # compose center panel
    center = cv2.hconcat([A, B])

    # stack everything
    parts = []
    if header_h > 0:
        parts.append(header)
    if title_h > 0:
        parts.append(title_strip)
    parts.append(center)
    if footer_h > 0:
        parts.append(footer)

    panel = cv2.vconcat(parts) if len(parts) > 1 else center
    return panel


def save_and_maybe_show_compare(panel: np.ndarray,
                                out_dir: Path,
                                stem: str,
                                args,
                                window_title: Optional[str] = None) -> Path:
    """
    Save compare panel to {out_dir}/{stem}.png.
    If args.show is True and not suppressed elsewhere, show in a resizable window.
    Returns saved Path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / f"{stem}.png"
    if panel is not None:
        cv2.imwrite(str(panel_path), panel)

    # Optional UI display
    if getattr(args, 'show', False):
        win = window_title or "Compare"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            target_w = min(1400, panel.shape[1])
            scale = target_w / float(panel.shape[1])
            target_h = int(panel.shape[0] * scale)
            cv2.resizeWindow(win, target_w, target_h)
            cv2.imshow(win, panel)
            # Non-blocking 1 ms to allow window paint; caller decides further waits
            cv2.waitKey(1)
        except Exception:
            # headless or display error; ignore
            pass

    return panel_path
