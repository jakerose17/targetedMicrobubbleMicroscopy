"""
bubble_core.py — Microbubble detection, tracking, and export engine.

No GUI dependencies. Can be imported independently for scripting or testing.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAGNIFICATION_MAP = {
    "10x": 1000.0,   # pixels per mm
    "20x": 2000.0,
    "35x": 3500.0,
    "50x": 5000.0,
}

DEFAULT_CONFIG = {
    "bg_subtract_threshold": 10,
    "min_blob_area_px": 8,
    "max_blob_area_px": 50000,
    "morph_kernel_size": 3,
    "max_link_distance_px": 50,
    "max_frame_skip": 5,
    "min_track_length": 5,
    "min_displacement_px": 5.0,
    "gaussian_blur_ksize": 3,
}

TRACK_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#e6beff", "#ffe119", "#ffd8b1",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BubbleTracker:
    """Detects and tracks microbubbles across video frames."""

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def build_background(self, frames_gray):
        """Build median background model from grayscale frame stack."""
        stack = np.array(frames_gray, dtype=np.float32)
        return np.median(stack, axis=0).astype(np.uint8)

    def detect_blobs(self, frame_gray, background):
        """Detect moving blobs by background subtraction + morphological cleanup."""
        cfg = self.config
        diff = cv2.absdiff(frame_gray, background)
        k_blur = cfg["gaussian_blur_ksize"]
        if k_blur > 1:
            diff = cv2.GaussianBlur(diff, (k_blur, k_blur), 0)
        _, mask = cv2.threshold(diff, cfg["bg_subtract_threshold"], 255, cv2.THRESH_BINARY)

        k = cfg["morph_kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg["min_blob_area_px"] or area > cfg["max_blob_area_px"]:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            radius = np.sqrt(area / np.pi)
            detections.append({"x": cx, "y": cy, "area": area, "radius": radius})
        return detections

    def link_tracks(self, all_detections, n_frames):
        """
        Link per-frame detections into coherent tracks using Hungarian assignment.
        Returns list of track dicts with 'id' and 'points'.
        """
        cfg = self.config
        max_dist = cfg["max_link_distance_px"]
        max_skip = cfg["max_frame_skip"]

        active = []
        finished = []
        next_id = 0

        for fi in range(n_frames):
            dets = all_detections[fi]

            # Expire stale tracks
            still_active = []
            for t in active:
                if fi - t["last_frame"] > max_skip:
                    finished.append(t)
                else:
                    still_active.append(t)
            active = still_active

            if not dets:
                continue

            if not active:
                for d in dets:
                    active.append({
                        "id": next_id,
                        "points": [(fi, d["x"], d["y"], d["area"], d["radius"])],
                        "last_frame": fi, "last_x": d["x"], "last_y": d["y"],
                    })
                    next_id += 1
                continue

            # Hungarian assignment
            n_t = len(active)
            n_d = len(dets)
            cost = np.full((n_t, n_d), 1e6)
            for ti, t in enumerate(active):
                for di, d in enumerate(dets):
                    dist = np.hypot(t["last_x"] - d["x"], t["last_y"] - d["y"])
                    if dist <= max_dist:
                        cost[ti, di] = dist

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_dets = set()
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] < 1e5:
                    d = dets[ci]
                    active[ri]["points"].append((fi, d["x"], d["y"], d["area"], d["radius"]))
                    active[ri]["last_frame"] = fi
                    active[ri]["last_x"] = d["x"]
                    active[ri]["last_y"] = d["y"]
                    matched_dets.add(ci)

            for di, d in enumerate(dets):
                if di not in matched_dets:
                    active.append({
                        "id": next_id,
                        "points": [(fi, d["x"], d["y"], d["area"], d["radius"])],
                        "last_frame": fi, "last_x": d["x"], "last_y": d["y"],
                    })
                    next_id += 1

        finished.extend(active)
        return finished

    def classify_tracks(self, tracks, fps):
        """Separate moving tracks from static noise bubbles."""
        cfg = self.config
        moving, static = [], []
        for t in tracks:
            pts = t["points"]
            if len(pts) < cfg["min_track_length"]:
                continue
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            displacement = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
            if displacement >= cfg["min_displacement_px"]:
                moving.append(t)
            else:
                static.append(t)
        return moving, static

    def process_video(self, video_path, progress_cb=None):
        """Full pipeline: read → background → detect → link → classify."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if progress_cb:
            progress_cb("Reading frames...", 0, n_frames * 2 + 10)

        frames_gray = []
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if progress_cb and i % 20 == 0:
                progress_cb("Reading frames...", i, n_frames * 2 + 10)
        cap.release()
        actual = len(frames_gray)
        if actual < 3:
            raise ValueError("Video has fewer than 3 readable frames.")

        if progress_cb:
            progress_cb("Building background...", actual, n_frames * 2 + 10)
        background = self.build_background(frames_gray)

        all_detections = []
        for i, fg in enumerate(frames_gray):
            all_detections.append(self.detect_blobs(fg, background))
            if progress_cb and i % 20 == 0:
                progress_cb("Detecting bubbles...", actual + i, n_frames * 2 + 10)

        if progress_cb:
            progress_cb("Linking tracks...", n_frames * 2, n_frames * 2 + 10)
        tracks = self.link_tracks(all_detections, actual)
        moving, static = self.classify_tracks(tracks, fps)

        if progress_cb:
            progress_cb("Done!", n_frames * 2 + 10, n_frames * 2 + 10)

        return {
            "video_path": str(video_path),
            "fps": fps, "n_frames": actual,
            "width": width, "height": height,
            "background": background,
            "first_frame": frames_gray[0],
            "moving_tracks": moving,
            "static_tracks": static,
            "all_detections": all_detections,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def tracks_to_csv(tracks, fps, px_per_mm, filepath):
    """Export moving tracks to CSV with physical units."""
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "track_id", "frame", "time_ms",
            "x_px", "y_px", "x_mm", "y_mm",
            "area_px2", "radius_px", "radius_um",
            "cumulative_dist_mm", "inst_velocity_mm_per_s"
        ])
        for track in tracks:
            tid = track["id"]
            pts = track["points"]
            cum = 0.0
            for j, (fr, x, y, area, rad) in enumerate(pts):
                t_ms = fr / fps * 1000.0
                x_mm = x / px_per_mm
                y_mm = y / px_per_mm
                rad_um = rad / px_per_mm * 1000.0
                vel = 0.0
                if j > 0:
                    p = pts[j-1]
                    dx = (x - p[1]) / px_per_mm
                    dy = (y - p[2]) / px_per_mm
                    seg = np.hypot(dx, dy)
                    cum += seg
                    dt = (fr - p[0]) / fps
                    vel = seg / dt if dt > 0 else 0.0
                w.writerow([
                    tid, fr, f"{t_ms:.3f}",
                    f"{x:.2f}", f"{y:.2f}", f"{x_mm:.5f}", f"{y_mm:.5f}",
                    f"{area:.1f}", f"{rad:.2f}", f"{rad_um:.2f}",
                    f"{cum:.5f}", f"{vel:.4f}"
                ])


def tracks_to_json(tracks, fps, px_per_mm, filepath):
    """Export moving tracks to JSON with physical units."""
    data = {"px_per_mm": px_per_mm, "fps": fps, "tracks": []}
    for track in tracks:
        td = {"id": track["id"], "points": []}
        cum = 0.0
        for j, (fr, x, y, area, rad) in enumerate(track["points"]):
            if j > 0:
                p = track["points"][j-1]
                cum += np.hypot((x-p[1])/px_per_mm, (y-p[2])/px_per_mm)
            td["points"].append({
                "frame": int(fr), "time_ms": round(fr/fps*1000, 3),
                "x_px": round(x, 2), "y_px": round(y, 2),
                "x_mm": round(x/px_per_mm, 5), "y_mm": round(y/px_per_mm, 5),
                "area_px2": round(area, 1),
                "radius_um": round(rad/px_per_mm*1000, 2),
                "cumulative_dist_mm": round(cum, 5),
            })
        data["tracks"].append(td)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def generate_summary(results, px_per_mm, mag_label):
    """Generate text summary of tracking results."""
    fps = results["fps"]
    moving = results["moving_tracks"]
    lines = [
        f"{'═'*48}",
        f"  Microbubble Tracking Summary",
        f"{'═'*48}",
        f"  Video:          {Path(results['video_path']).name}",
        f"  Resolution:     {results['width']} × {results['height']} px",
        f"  Frames:         {results['n_frames']}",
        f"  FPS:            {fps:.2f}",
        f"  Duration:       {results['n_frames']/fps:.3f} s",
        f"  Magnification:  {mag_label} ({px_per_mm:.0f} px/mm)",
        f"",
        f"  Moving tracks:  {len(moving)}",
        f"  Static (noise): {len(results['static_tracks'])} (filtered out)",
        f"{'─'*48}",
    ]
    for track in moving:
        pts = track["points"]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        frames_span = pts[-1][0] - pts[0][0]
        dur_ms = frames_span / fps * 1000

        path_len = 0
        vels = []
        for j in range(1, len(pts)):
            dx = (pts[j][1] - pts[j-1][1]) / px_per_mm
            dy = (pts[j][2] - pts[j-1][2]) / px_per_mm
            seg = np.hypot(dx, dy)
            path_len += seg
            dt = (pts[j][0] - pts[j-1][0]) / fps
            if dt > 0:
                vels.append(seg / dt)

        disp = np.hypot((xs[-1]-xs[0])/px_per_mm, (ys[-1]-ys[0])/px_per_mm)
        mean_r = np.mean([p[4] for p in pts]) / px_per_mm * 1000

        lines.append(f"")
        lines.append(f"  Track {track['id']}")
        lines.append(f"    Detections:     {len(pts)}")
        lines.append(f"    Duration:       {dur_ms:.1f} ms")
        lines.append(f"    Path length:    {path_len*1000:.1f} µm")
        lines.append(f"    Displacement:   {disp*1000:.1f} µm")
        lines.append(f"    Mean radius:    {mean_r:.1f} µm")
        if vels:
            lines.append(f"    Mean velocity:  {np.mean(vels)*1000:.1f} µm/s")
            lines.append(f"    Max velocity:   {np.max(vels)*1000:.1f} µm/s")

    lines.append(f"\n{'═'*48}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tracks_on_image(results, px_per_mm, fig=None):
    """Bubble trajectories overlaid on background image."""
    if fig is None:
        fig = Figure(figsize=(12, 3), dpi=100)
    fig.clear()
    ax = fig.add_subplot(111)

    bg = results["background"]
    h, w = bg.shape
    ext = [0, w/px_per_mm, h/px_per_mm, 0]
    ax.imshow(bg, cmap="gray", extent=ext, aspect="equal")

    for i, track in enumerate(results["moving_tracks"]):
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        pts = track["points"]
        xs = [p[1]/px_per_mm for p in pts]
        ys = [p[2]/px_per_mm for p in pts]
        ax.plot(xs, ys, color=color, linewidth=1.5, label=f"Track {track['id']}")
        ax.plot(xs[0], ys[0], "o", color=color, ms=5, mec="white", mew=0.5)
        ax.plot(xs[-1], ys[-1], "s", color=color, ms=5, mec="white", mew=0.5)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Bubble Trajectories")
    if len(results["moving_tracks"]) <= 10:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def plot_velocity_profiles(results, px_per_mm, fig=None):
    """Instantaneous velocity vs time for each moving track."""
    if fig is None:
        fig = Figure(figsize=(8, 4), dpi=100)
    fig.clear()
    ax = fig.add_subplot(111)
    fps = results["fps"]

    for i, track in enumerate(results["moving_tracks"]):
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        pts = track["points"]
        times, vels = [], []
        for j in range(1, len(pts)):
            t_ms = (pts[j][0] + pts[j-1][0]) / 2.0 / fps * 1000
            dx = (pts[j][1] - pts[j-1][1]) / px_per_mm
            dy = (pts[j][2] - pts[j-1][2]) / px_per_mm
            dt = (pts[j][0] - pts[j-1][0]) / fps
            if dt > 0:
                times.append(t_ms)
                vels.append(np.hypot(dx, dy) / dt * 1000)
        ax.plot(times, vels, color=color, lw=1, alpha=0.8, label=f"Track {track['id']}")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity (µm/s)")
    ax.set_title("Instantaneous Velocity")
    if len(results["moving_tracks"]) <= 10:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_displacement_vs_time(results, px_per_mm, fig=None):
    """Cumulative displacement vs time for each moving track."""
    if fig is None:
        fig = Figure(figsize=(8, 4), dpi=100)
    fig.clear()
    ax = fig.add_subplot(111)
    fps = results["fps"]

    for i, track in enumerate(results["moving_tracks"]):
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        pts = track["points"]
        times = [pts[0][0] / fps * 1000]
        cum = [0.0]
        for j in range(1, len(pts)):
            times.append(pts[j][0] / fps * 1000)
            dx = (pts[j][1] - pts[j-1][1]) / px_per_mm * 1000
            dy = (pts[j][2] - pts[j-1][2]) / px_per_mm * 1000
            cum.append(cum[-1] + np.hypot(dx, dy))
        ax.plot(times, cum, color=color, lw=1.2, label=f"Track {track['id']}")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cumulative Displacement (µm)")
    ax.set_title("Cumulative Displacement")
    if len(results["moving_tracks"]) <= 10:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
