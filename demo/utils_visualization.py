# utils_visualization.py
# Centralized visualization utilities:
# - real-time HUD sparkline (render_sparkline_strip)
# - ROM result panel (compose_result_panel + show_and_save_result_panel)
# - gallery compositor (compose_gallery)
# - small internal helpers for image/text layout

import os
import time
import math
from collections import deque

import numpy as np
import cv2
import mmcv

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

    # Measure text height/baseline, compute comfortable line height
    (w0, h0), base0 = cv2.getTextSize("Hg", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    line_h = h0 + base0 + 2
    total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap

    # Center block vertically with top padding
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
    footer_h = 48  # taller to allow bigger ROM text
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
    # Use ASCII " deg" to avoid glyph issues
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

    # Title (ASCII hyphen to avoid '???')
    title = f"{test_name} - {status.upper()}"
    _text(panel, title, (16, int(title_h * 0.7)), 1.2, (255, 255, 255), 2)

    # Place row
    panel[title_h:title_h + row.shape[0], 0:W, :] = row

    # Footer: ROM only, larger text
    rom_txt = "--.-" if rom_deg is None else f"{rom_deg:.1f}"
    footer = f"ROM: {rom_txt} deg"
    _text(panel, footer, (16, title_h + row.shape[0] + int(footer_h * 0.8)), 1.1, (255, 255, 255), 2)

    return panel

# ---------- Gallery (collage of recent panels) ----------

def _compose_gallery(panel_imgs, cols=1, cell_w=560, pad=16, bg=(18, 18, 18)):
    """
    Build a collage from a list of panel images (BGR).
    - cols: desired columns (>=1). Canvas width adapts to the actual #items in each row,
            so there is no extra black area when the last row is partially filled.
    - cell_w: target width per cell (images are resized preserving aspect).
    """
    if not panel_imgs:
        return np.full((360, 640, 3), bg, dtype=np.uint8)

    # Resize all to cell_w keeping aspect
    cells = [_resize_keep_aspect(im, cell_w) for im in panel_imgs]
    n = len(cells)
    cols = max(1, int(cols))
    rows = int(math.ceil(n / cols))

    # Per-row max height & actual row widths (no padding at row end)
    row_heights, row_widths = [], []
    for r in range(rows):
        used = min(cols, n - r * cols)               # how many items in this row
        hmax = max(cells[r * cols + c].shape[0] for c in range(used))
        wsum = sum(cells[r * cols + c].shape[1] for c in range(used)) + pad * (used - 1 if used > 1 else 0)
        row_heights.append(hmax)
        row_widths.append(wsum)

    total_h = sum(row_heights) + pad * (rows - 1 if rows > 1 else 0)
    total_w = max(row_widths)                        # fit to the widest actual row (no extra black)

    canvas = np.full((total_h, total_w, 3), bg, dtype=np.uint8)

    # Blit cells row by row, spacing with pad but no trailing gap
    y = 0
    for r in range(rows):
        used = min(cols, n - r * cols)
        x = 0
        for c in range(used):
            cell = cells[r * cols + c]
            h, w = cell.shape[:2]
            y_off = (row_heights[r] - h) // 2        # vertical centering within the row slot
            canvas[y + y_off:y + y_off + h, x:x + w, :] = cell
            x += w + (pad if c < used - 1 else 0)    # no pad after last in row
        y += row_heights[r] + (pad if r < rows - 1 else 0)

    return canvas

# ---------- Show & Save (single / gallery) ----------

def show_and_save_result_panel(args, state, result: dict):
    """
    Compose and save ROM_PANEL.png into the trial dir.
    If --show and --show-result:
      - single: one window showing the latest panel
      - gallery: one window showing a grid of the last N panels
    """
    test_name = result.get('test_name', getattr(args, 'rom_test', 'unknown'))
    status = result.get('status', 'ok')
    session_stamp = getattr(state, 'session_stamp', None) or time.strftime("%Y%m%d_%H%M%S")
    trial_dir = getattr(state, 'rom_save_dir', None) or os.path.join(getattr(args, 'output_root', '.') or '.', "unknown_trial")

    # Crop off the HUD (text strip + optional sparkline) from saved MIN/MAX frames
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

    # UI
    if not (getattr(args, 'show', False) and getattr(args, 'show_result', False)):
        return panel_path

    mode = getattr(args, 'panel_window', 'single')

    if mode == 'single':
        win = f"ROM Result - {test_name}"
        # close previous panel window if any (regardless of previous mode)
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

    # ----- gallery mode -----
    # Keep deque of recent panel paths on state
    if not hasattr(state, 'gallery_paths') or state.gallery_paths is None:
        state.gallery_paths = deque(maxlen=max(1, int(getattr(args, 'gallery_size', 4))))
    else:
        # Update maxlen if changed
        desired_len = max(1, int(getattr(args, 'gallery_size', 4)))
        if state.gallery_paths.maxlen != desired_len:
            # rebuild with new maxlen
            items = list(state.gallery_paths)
            state.gallery_paths = deque(items[-desired_len:], maxlen=desired_len)

    state.gallery_paths.append(panel_path)

    # Load last N panels and compose
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
