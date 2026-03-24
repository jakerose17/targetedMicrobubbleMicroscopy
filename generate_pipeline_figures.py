#!/usr/bin/env python3
"""
Generate step-by-step pipeline visualization figures for both test videos.

Produces annotated overlays showing each detection and tracking stage,
saved as publication-quality PNGs.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from bubble_core import BubbleTracker, DEFAULT_CONFIG, MAGNIFICATION_MAP

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "pipeline_figures"
OUT_DIR.mkdir(exist_ok=True)

VIDEOS = {
    "sparse": ("testvid_lessDense_10x.mp4", "10x"),
    "dense":  ("testvid_dense_35x.mp4",     "35x"),
}

# Use a nice frame from the middle of each video
SAMPLE_FRAMES = {"sparse": 200, "dense": 200}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def read_frame(cap, fi):
    """Read a specific frame from a video capture."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {fi}")
    return frame


def compute_median_background(cap, n_frames, n_samples=50):
    """Compute temporal median background."""
    idxs = np.linspace(0, n_frames - 1, min(n_samples, n_frames), dtype=int)
    samples = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return np.median(np.array(samples), axis=0).astype(np.uint8)


def gray_to_rgb(gray):
    """Convert grayscale to RGB for matplotlib display."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def add_scalebar(ax, px_per_mm, length_um=100, y_frac=0.93, x_frac=0.05,
                 color="white"):
    """Add a scale bar to an axis."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w_px = xlim[1] - xlim[0]
    h_px = ylim[0] - ylim[1]  # ylim is inverted for images

    bar_px = length_um / 1000.0 * px_per_mm
    x0 = xlim[0] + x_frac * w_px
    y0 = ylim[1] + y_frac * h_px

    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0 + bar_px / 2, y0 - h_px * 0.02, f"{length_um} \u00b5m",
            ha="center", va="bottom", color=color, fontsize=7, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: PIPELINE OVERVIEW (6-panel)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_pipeline_overview(cap, n_frames, fi, label, mag, px_per_mm):
    """
    6-panel figure showing each detection step:
    1. Raw frame
    2. Median background
    3. Background subtraction (bg - frame)
    4. Thresholded binary mask
    5. Morphologically cleaned mask
    6. Final detections overlaid on raw frame
    """
    cfg = DEFAULT_CONFIG.copy()
    frame_bgr = read_frame(cap, fi)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg = compute_median_background(cap, n_frames)

    # Step-by-step detection
    diff = cv2.subtract(bg, gray)

    _, raw_mask = cv2.threshold(diff, cfg["bg_sub_threshold"], 255,
                                cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg["morph_kernel_size"], cfg["morph_kernel_size"]))
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Full detection with filtering
    tracker = BubbleTracker()
    dets = tracker._detect_frame(gray, bg, kernel, cfg)

    # Draw detections on frame
    overlay = frame_bgr.copy()
    for d in dets:
        r = max(int(d["radius"]), 2)
        cx, cy = int(d["x"]), int(d["y"])
        # Color by contrast: green = high contrast (in-focus), red = low
        norm_c = min(1.0, d["contrast"] / 40.0)
        color = (0, int(255 * norm_c), int(255 * (1 - norm_c)))
        cv2.circle(overlay, (cx, cy), r + 2, color, 1)

    # Build figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Detection Pipeline — {label.capitalize()} ({mag}), Frame {fi}",
                 fontsize=16, fontweight="bold", y=0.98)

    panels = [
        (gray_to_rgb(gray), "1. Raw Frame (Grayscale)"),
        (gray_to_rgb(bg), "2. Median Background"),
        (gray_to_rgb(diff), f"3. Background Subtraction\n(bg \u2212 frame)"),
        (gray_to_rgb(raw_mask), f"4. Threshold Mask\n(diff > {cfg['bg_sub_threshold']})"),
        (gray_to_rgb(clean_mask), "5. Morphological Cleanup\n(open + close)"),
        (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
         f"6. Final Detections ({len(dets)} bubbles)\n"
         f"(contrast > {cfg['min_contrast']}, circ > {cfg['min_circularity']:.2f})"),
    ]

    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis("off")

    # Add scale bar to last panel
    add_scalebar(axes[1, 2], px_per_mm,
                 length_um=100 if mag == "35x" else 500)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / f"pipeline_overview_{label}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: CONTRAST FILTERING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def fig_contrast_filtering(cap, n_frames, fi, label, mag, px_per_mm):
    """
    Show how contrast filtering separates in-focus from out-of-focus bubbles.
    3-panel: all contours colored by contrast, rejected vs accepted, histogram.
    """
    cfg = DEFAULT_CONFIG.copy()
    frame_bgr = read_frame(cap, fi)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg = compute_median_background(cap, n_frames)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Get ALL contours (before contrast/circularity filtering)
    diff = cv2.subtract(bg, gray)
    _, mask = cv2.threshold(diff, cfg["bg_sub_threshold"], 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg["min_blob_area_px"] or area > cfg["max_blob_area_px"]:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0

        x, y, w, h = cv2.boundingRect(c)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roi_mask, [c - np.array([x, y])], -1, 255, -1)
        mean_int = cv2.mean(gray[y:y+h, x:x+w], mask=roi_mask)[0]
        mean_bg = cv2.mean(bg[y:y+h, x:x+w], mask=roi_mask)[0]
        contrast = mean_bg - mean_int

        all_blobs.append({
            "x": cx, "y": cy, "area": area,
            "radius": np.sqrt(area / np.pi),
            "contrast": contrast, "circularity": circ,
            "accepted": contrast >= cfg["min_contrast"] and circ >= cfg["min_circularity"],
        })

    contrasts = [b["contrast"] for b in all_blobs]
    accepted = [b for b in all_blobs if b["accepted"]]
    rejected = [b for b in all_blobs if not b["accepted"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Contrast & Circularity Filtering — {label.capitalize()} ({mag}), Frame {fi}",
                 fontsize=14, fontweight="bold")

    # Panel 1: All blobs colored by contrast
    ax = axes[0]
    ax.imshow(gray_to_rgb(gray))
    if contrasts:
        norm = Normalize(vmin=0, vmax=max(contrasts))
        cmap = plt.cm.RdYlGn
        for b in all_blobs:
            color = cmap(norm(b["contrast"]))
            circle = Circle((b["x"], b["y"]), max(b["radius"], 2),
                           fill=False, edgecolor=color, lw=0.8, alpha=0.8)
            ax.add_patch(circle)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Contrast (bg \u2212 bubble)", fontsize=9)
    ax.set_title(f"All Contours ({len(all_blobs)})\nColored by Contrast", fontsize=11)
    ax.axis("off")

    # Panel 2: Accepted vs rejected
    ax = axes[1]
    ax.imshow(gray_to_rgb(gray))
    for b in rejected:
        circle = Circle((b["x"], b["y"]), max(b["radius"], 2),
                       fill=False, edgecolor="red", lw=0.6, alpha=0.5)
        ax.add_patch(circle)
    for b in accepted:
        circle = Circle((b["x"], b["y"]), max(b["radius"], 2),
                       fill=False, edgecolor="lime", lw=0.8, alpha=0.8)
        ax.add_patch(circle)
    ax.set_title(f"Accepted ({len(accepted)}, green) vs "
                 f"Rejected ({len(rejected)}, red)", fontsize=11)
    ax.axis("off")

    # Panel 3: Contrast histogram
    ax = axes[2]
    if contrasts:
        ax.hist(contrasts, bins=50, color="steelblue", edgecolor="white",
                alpha=0.8, label="All contours")
        ax.axvline(cfg["min_contrast"], color="red", lw=2, ls="--",
                   label=f"Threshold = {cfg['min_contrast']}")
        # Shade rejected region
        ax.axvspan(0, cfg["min_contrast"], alpha=0.15, color="red",
                   label="Rejected (low contrast)")
    ax.set_xlabel("Contrast (bg \u2212 bubble intensity)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Contrast Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / f"contrast_filtering_{label}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: CIRCULARITY FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def fig_circularity_filtering(cap, n_frames, fi, label, mag, px_per_mm):
    """
    Scatter of circularity vs contrast, showing the filtering regions.
    """
    cfg = DEFAULT_CONFIG.copy()
    frame_bgr = read_frame(cap, fi)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg = compute_median_background(cap, n_frames)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    diff = cv2.subtract(bg, gray)
    _, mask = cv2.threshold(diff, cfg["bg_sub_threshold"], 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circs, cons, areas, accepted_flags = [], [], [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg["min_blob_area_px"] or area > cfg["max_blob_area_px"]:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0

        x, y, w, h = cv2.boundingRect(c)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roi_mask, [c - np.array([x, y])], -1, 255, -1)
        mean_int = cv2.mean(gray[y:y+h, x:x+w], mask=roi_mask)[0]
        mean_bg = cv2.mean(bg[y:y+h, x:x+w], mask=roi_mask)[0]
        contrast = mean_bg - mean_int

        circs.append(circ)
        cons.append(contrast)
        areas.append(area)
        accepted_flags.append(
            contrast >= cfg["min_contrast"] and circ >= cfg["min_circularity"])

    circs = np.array(circs)
    cons = np.array(cons)
    areas = np.array(areas)
    accepted_flags = np.array(accepted_flags)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"Detection Feature Space — {label.capitalize()} ({mag}), Frame {fi}",
                 fontsize=14, fontweight="bold")

    # Rejected points
    rej = ~accepted_flags
    if rej.any():
        ax.scatter(circs[rej], cons[rej], s=np.sqrt(areas[rej]) * 2,
                  c="lightcoral", alpha=0.4, edgecolors="none",
                  label=f"Rejected ({rej.sum()})")
    # Accepted points
    if accepted_flags.any():
        sc = ax.scatter(circs[accepted_flags], cons[accepted_flags],
                       s=np.sqrt(areas[accepted_flags]) * 2,
                       c=cons[accepted_flags], cmap="viridis", alpha=0.7,
                       edgecolors="black", linewidth=0.3,
                       label=f"Accepted ({accepted_flags.sum()})")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Contrast", fontsize=10)

    # Draw threshold lines
    ax.axvline(cfg["min_circularity"], color="red", lw=2, ls="--",
               label=f"Min circularity = {cfg['min_circularity']}")
    ax.axhline(cfg["min_contrast"], color="orange", lw=2, ls="--",
               label=f"Min contrast = {cfg['min_contrast']}")

    # Shade rejected region
    ax.axvspan(0, cfg["min_circularity"], alpha=0.08, color="red")
    ax.axhspan(0, cfg["min_contrast"], alpha=0.08, color="orange")

    ax.set_xlabel("Circularity (4\u03c0A/P\u00b2)", fontsize=11)
    ax.set_ylabel("Contrast (bg \u2212 bubble)", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / f"feature_space_{label}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: LINKING VISUALIZATION (5 consecutive frames)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_linking(cap, n_frames, fi_start, label, mag, px_per_mm):
    """
    Show 5 consecutive frames with track linking arrows connecting
    matched detections between frames.
    """
    cfg = DEFAULT_CONFIG.copy()
    bg = compute_median_background(cap, n_frames)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tracker = BubbleTracker()

    n_show = 5
    frames_gray = []
    frames_dets = []
    for i in range(n_show):
        fi = fi_start + i
        frame_bgr = read_frame(cap, fi)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dets = tracker._detect_frame(gray, bg, kernel, cfg)
        frames_gray.append(gray)
        frames_dets.append(dets)

    # Run simple linking across these frames
    max_dist = cfg["max_link_distance_px"]
    alpha = cfg["velocity_alpha"]
    area_w = cfg["area_cost_weight"]

    active = []
    all_tracks = {}  # id -> list of (frame_offset, x, y, color)
    next_id = 0

    # Assign distinct colors
    np.random.seed(42)
    track_colors = {}

    for frame_i, dets in enumerate(frames_dets):
        # Expire
        still = [t for t in active if frame_i - t["lf"] <= 2]
        active = still

        if not dets:
            continue

        matched = set()
        if active:
            det_pos = np.array([[d["x"], d["y"]] for d in dets])
            tree = cKDTree(det_pos)
            cost = np.full((len(active), len(dets)), 1e6)
            for ti, t in enumerate(active):
                dt = frame_i - t["lf"]
                px = t["lx"] + t["vx"] * dt
                py = t["ly"] + t["vy"] * dt
                cands = tree.query_ball_point([px, py], max_dist)
                for di in cands:
                    d = dets[di]
                    dist = np.hypot(px - d["x"], py - d["y"])
                    area_ratio = max(t["la"], d["area"]) / max(min(t["la"], d["area"]), 1)
                    cost[ti, di] = dist + area_w * (area_ratio - 1.0)

            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] >= 1e5:
                    continue
                t = active[r]
                d = dets[c]
                dt = frame_i - t["lf"]
                if dt > 0:
                    nvx = (d["x"] - t["lx"]) / dt
                    nvy = (d["y"] - t["ly"]) / dt
                    t["vx"] = alpha * nvx + (1 - alpha) * t["vx"]
                    t["vy"] = alpha * nvy + (1 - alpha) * t["vy"]
                t["lf"] = frame_i
                t["lx"] = d["x"]
                t["ly"] = d["y"]
                t["la"] = d["area"]
                all_tracks[t["id"]].append((frame_i, d["x"], d["y"]))
                matched.add(c)

        for di in range(len(dets)):
            if di not in matched:
                d = dets[di]
                tid = next_id
                next_id += 1
                active.append({
                    "id": tid, "lf": frame_i,
                    "lx": d["x"], "ly": d["y"],
                    "vx": 0.0, "vy": 0.0, "la": d["area"],
                })
                all_tracks[tid] = [(frame_i, d["x"], d["y"])]
                track_colors[tid] = plt.cm.tab20(tid % 20)

    # Filter to tracks that span multiple frames
    multi_tracks = {k: v for k, v in all_tracks.items() if len(v) >= 3}

    # For dense video, pick a zoomed-in region to make it readable
    if label == "dense":
        # Zoom to a 400x300 region
        cx, cy = 960, 540
        crop_w, crop_h = 400, 300
        x1, y1 = cx - crop_w // 2, cy - crop_h // 2
        x2, y2 = x1 + crop_w, y1 + crop_h
    else:
        x1, y1 = 0, 0
        x2, y2 = frames_gray[0].shape[1], frames_gray[0].shape[0]

    fig, axes = plt.subplots(1, n_show, figsize=(20, 4.5))
    fig.suptitle(f"Frame-to-Frame Linking — {label.capitalize()} ({mag}), "
                 f"Frames {fi_start}\u2013{fi_start + n_show - 1}"
                 + (f"\n(zoomed to {crop_w}\u00d7{crop_h}px region)" if label == "dense" else ""),
                 fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes):
        crop = frames_gray[i][y1:y2, x1:x2]
        ax.imshow(gray_to_rgb(crop), aspect="equal")

        # Draw detection circles and linking arrows
        for tid, pts in multi_tracks.items():
            color = track_colors[tid]
            # Find points in this frame and next
            for p in pts:
                if p[0] == i:
                    px, py = p[1] - x1, p[2] - y1
                    if 0 <= px <= (x2-x1) and 0 <= py <= (y2-y1):
                        circle = Circle((px, py), 4, fill=False,
                                       edgecolor=color, lw=1.2)
                        ax.add_patch(circle)

            # Draw arrow from this frame to next if both exist
            this_pts = [p for p in pts if p[0] == i]
            next_pts = [p for p in pts if p[0] == i + 1]
            if this_pts and next_pts:
                p0 = this_pts[0]
                p1 = next_pts[0]
                sx, sy = p0[1] - x1, p0[2] - y1
                ex, ey = p1[1] - x1, p1[2] - y1
                if (0 <= sx <= (x2-x1) and 0 <= sy <= (y2-y1) and
                    0 <= ex <= (x2-x1) and 0 <= ey <= (y2-y1)):
                    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                               arrowprops=dict(arrowstyle="->", color=color,
                                              lw=1.0, alpha=0.7))

        ax.set_title(f"Frame {fi_start + i}", fontsize=10)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = OUT_DIR / f"linking_{label}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: DETECTION STABILITY OVER TIME
# ═══════════════════════════════════════════════════════════════════════════════

def fig_detection_stability(cap, n_frames, label, mag, px_per_mm):
    """
    Line plot showing detection count per frame over the full video.
    Demonstrates temporal stability of the detection approach.
    """
    cfg = DEFAULT_CONFIG.copy()
    bg = compute_median_background(cap, n_frames)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tracker = BubbleTracker()
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample every 3rd frame for speed
    step = 3
    frame_nums = []
    det_counts = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fi in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = tracker._detect_frame(gray, bg, kernel, cfg)
        frame_nums.append(fi)
        det_counts.append(len(dets))

    times = np.array(frame_nums) / fps
    counts = np.array(det_counts)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, counts, color="steelblue", lw=1, alpha=0.8)
    ax.fill_between(times, counts, alpha=0.2, color="steelblue")
    ax.axhline(counts.mean(), color="red", ls="--", lw=1.5,
               label=f"Mean = {counts.mean():.0f}")
    ax.axhline(counts.mean() + counts.std(), color="orange", ls=":", lw=1,
               label=f"\u00b1\u03c3 = {counts.std():.1f} ({counts.std()/counts.mean()*100:.1f}%)")
    ax.axhline(counts.mean() - counts.std(), color="orange", ls=":", lw=1)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Detections per Frame", fontsize=11)
    ax.set_title(f"Detection Stability — {label.capitalize()} ({mag})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, counts.max() * 1.1)

    fig.tight_layout()
    path = OUT_DIR / f"detection_stability_{label}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: ZOOMED COMPARISON (raw vs detected)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_zoomed_comparison(cap, n_frames, fi, label, mag, px_per_mm):
    """
    Side-by-side zoomed crop showing raw frame vs detected bubbles.
    For dense video this really shows the detection quality.
    """
    cfg = DEFAULT_CONFIG.copy()
    frame_bgr = read_frame(cap, fi)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg = compute_median_background(cap, n_frames)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tracker = BubbleTracker()
    dets = tracker._detect_frame(gray, bg, kernel, cfg)

    # Pick a zoom region
    if label == "dense":
        crops = [
            ("Center", 860, 440, 200, 200),
            ("Corner", 100, 100, 200, 200),
        ]
    else:
        crops = [
            ("Full width detail", 400, 350, 600, 300),
        ]

    n_crops = len(crops)
    fig, axes = plt.subplots(n_crops, 2, figsize=(12, 6 * n_crops))
    if n_crops == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(f"Zoomed Detection Detail — {label.capitalize()} ({mag}), Frame {fi}",
                 fontsize=14, fontweight="bold")

    for row, (region_name, cx, cy, cw, ch) in enumerate(crops):
        x1, y1 = cx, cy
        x2, y2 = cx + cw, cy + ch

        # Raw
        axes[row, 0].imshow(gray_to_rgb(gray[y1:y2, x1:x2]))
        axes[row, 0].set_title(f"Raw Frame — {region_name}", fontsize=11)
        axes[row, 0].axis("off")

        # Overlay
        crop_overlay = cv2.cvtColor(gray[y1:y2, x1:x2], cv2.COLOR_GRAY2RGB)
        crop_dets = [d for d in dets
                     if x1 <= d["x"] <= x2 and y1 <= d["y"] <= y2]

        axes[row, 1].imshow(crop_overlay)
        for d in crop_dets:
            r = max(d["radius"], 1.5)
            norm_c = min(1.0, d["contrast"] / 40.0)
            color = plt.cm.RdYlGn(norm_c)
            circle = Circle((d["x"] - x1, d["y"] - y1), r + 1,
                           fill=False, edgecolor=color, lw=1.2)
            axes[row, 1].add_patch(circle)
        axes[row, 1].set_title(
            f"Detected ({len(crop_dets)} bubbles) — {region_name}",
            fontsize=11)
        axes[row, 1].axis("off")

        # Add scale bar
        add_scalebar(axes[row, 1], px_per_mm,
                     length_um=50 if mag == "35x" else 200,
                     color="yellow")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / f"zoomed_comparison_{label}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: VELOCITY SPIKE COMPARISON (before/after median filter)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_velocity_smoothing(results, label, mag, px_per_mm):
    """
    Show velocity profile of a sample track before and after median filtering.
    """
    fps = results["fps"]
    moving = results["moving_tracks"]
    if not moving:
        return

    # Pick a long track
    longest = max(moving, key=lambda t: len(t["points"]))
    pts = longest["points"]
    if len(pts) < 20:
        return

    times, raw_vels = [], []
    for j in range(1, len(pts)):
        t_ms = (pts[j][0] + pts[j-1][0]) / 2.0 / fps * 1000
        dx = (pts[j][1] - pts[j-1][1]) / px_per_mm
        dy = (pts[j][2] - pts[j-1][2]) / px_per_mm
        dt = (pts[j][0] - pts[j-1][0]) / fps
        if dt > 0:
            times.append(t_ms)
            raw_vels.append(np.hypot(dx, dy) / dt * 1000)

    # Median filter
    from bubble_core import _median_filter_velocities
    smoothed = _median_filter_velocities(raw_vels, window=5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Velocity Median Filtering — {label.capitalize()} ({mag}), "
                 f"Track {longest['id']} ({len(pts)} points)",
                 fontsize=13, fontweight="bold")

    ax1.plot(times, raw_vels, color="steelblue", lw=0.8, alpha=0.8)
    ax1.set_ylabel("Velocity (\u00b5m/s)", fontsize=10)
    ax1.set_title("Raw Instantaneous Velocity", fontsize=11)
    ax1.grid(True, alpha=0.3)
    if raw_vels:
        med = np.median(raw_vels)
        ax1.axhline(med, color="red", ls="--", lw=1,
                    label=f"Median = {med:.1f}")
        ax1.legend(fontsize=9)

    ax2.plot(times, smoothed, color="forestgreen", lw=0.8, alpha=0.8)
    ax2.set_xlabel("Time (ms)", fontsize=10)
    ax2.set_ylabel("Velocity (\u00b5m/s)", fontsize=10)
    ax2.set_title("Median-Filtered (window=5)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    if smoothed:
        med = np.median(smoothed)
        ax2.axhline(med, color="red", ls="--", lw=1,
                    label=f"Median = {med:.1f}")
        ax2.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / f"velocity_smoothing_{label}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    for label, (video_name, mag) in VIDEOS.items():
        video_path = BASE_DIR / video_name
        px_per_mm = MAGNIFICATION_MAP[mag]
        fi = SAMPLE_FRAMES[label]

        print(f"\n{'='*60}")
        print(f"  Generating figures for: {label} ({mag})")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("\n  [1/6] Pipeline overview...")
        fig_pipeline_overview(cap, n_frames, fi, label, mag, px_per_mm)

        print("  [2/6] Contrast filtering...")
        fig_contrast_filtering(cap, n_frames, fi, label, mag, px_per_mm)

        print("  [3/6] Feature space (circularity vs contrast)...")
        fig_circularity_filtering(cap, n_frames, fi, label, mag, px_per_mm)

        print("  [4/6] Frame-to-frame linking...")
        fig_linking(cap, n_frames, fi, label, mag, px_per_mm)

        print("  [5/6] Detection stability...")
        fig_detection_stability(cap, n_frames, label, mag, px_per_mm)

        print("  [6/6] Zoomed comparison...")
        fig_zoomed_comparison(cap, n_frames, fi, label, mag, px_per_mm)

        cap.release()

        # Velocity smoothing needs full tracking results
        print("  [bonus] Running full tracker for velocity figure...")
        tracker = BubbleTracker()
        results = tracker.process_video(video_path)
        fig_velocity_smoothing(results, label, mag, px_per_mm)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
