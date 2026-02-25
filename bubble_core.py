"""
bubble_core.py — Microbubble detection, tracking, and export engine (v2).

Streaming architecture with MOG2 + frame-diff gated detection,
velocity-predicted Hungarian linking, and automatic track merging.

No GUI dependencies. Can be imported for scripting or batch processing.
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAGNIFICATION_MAP = {
    "10x": 1000.0,
    "20x": 2000.0,
    "35x": 3500.0,
    "50x": 5000.0,
}

DEFAULT_CONFIG = {
    # Background model
    "mog2_history":         500,     # MOG2 history length (frames)
    "mog2_var_threshold":   16,      # MOG2 variance threshold
    "mog2_learning_rate":   0.01,    # MOG2 learning rate during training
    # Frame-diff gating (rejects static objects)
    "frame_diff_threshold": 4,       # intensity threshold on |frame_n - frame_n-1|
    "frame_diff_dilate":    2,       # dilation iterations on frame-diff mask
    # Blob filtering
    "min_blob_area_px":     5,       # minimum contour area
    "max_blob_area_px":     50000,   # maximum contour area
    "morph_kernel_size":    3,       # morphological cleanup kernel
    "gaussian_blur_ksize":  3,       # Gaussian blur on diff image
    # Linking
    "max_link_distance_px": 60,      # max distance for frame-to-frame linking
    "max_frame_skip":       10,      # max frames a track can skip
    "velocity_alpha":       0.3,     # EMA smoothing for velocity estimate
    # Track classification
    "min_track_length":     5,       # minimum detections to keep a track
    "min_displacement_px":  8.0,     # minimum net displacement to be "moving"
    # Merging
    "merge_max_gap_frames": 20,      # max frame gap for merging fragments
    "merge_max_distance_px": 60,     # max spatial distance for merging
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
    """
    Two-pass streaming tracker for microbubble microscopy.

    Pass 1: Train MOG2 background model (streaming, constant memory).
    Pass 2: Detect + link + merge in a single streaming pass.
    """

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def process_video(self, video_path, progress_cb=None):
        """
        Full pipeline. Returns results dict with tracks and metadata.
        Only ~3 frames are held in memory at a time.
        """
        cfg = self.config
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_steps = n_frames * 2 + 20  # for progress

        # ── PASS 1: Train MOG2 background model ──
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=min(cfg["mog2_history"], n_frames),
            varThreshold=cfg["mog2_var_threshold"],
            detectShadows=False,
        )

        if progress_cb:
            progress_cb("Training background model...", 0, total_steps)

        first_frame = None
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_sub.apply(gray, learningRate=cfg["mog2_learning_rate"])
            if first_frame is None:
                first_frame = gray.copy()
            if progress_cb and i % 30 == 0:
                progress_cb("Training background model...", i, total_steps)

        actual_frames = i + 1 if ret else i
        if actual_frames < 3:
            cap.release()
            raise ValueError("Video has fewer than 3 readable frames.")

        background = bg_sub.getBackgroundImage()
        if background is None:
            background = first_frame

        # ── PASS 2: Detect + Link (streaming) ──
        if progress_cb:
            progress_cb("Detecting & linking...", n_frames, total_steps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg["morph_kernel_size"], cfg["morph_kernel_size"]))
        blur_k = cfg["gaussian_blur_ksize"]

        active_tracks = []
        finished_tracks = []
        next_id = 0
        prev_gray = None
        all_detections = []  # stored for video player overlay

        for fi in range(actual_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # MOG2 foreground
            fg_mask = bg_sub.apply(gray, learningRate=0)

            # Frame-diff gating: require recent motion
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                if blur_k > 1:
                    diff = cv2.GaussianBlur(diff, (blur_k, blur_k), 0)
                _, diff_mask = cv2.threshold(
                    diff, cfg["frame_diff_threshold"], 255, cv2.THRESH_BINARY)
                diff_mask = cv2.dilate(
                    diff_mask, kernel, iterations=cfg["frame_diff_dilate"])
                combined = cv2.bitwise_and(fg_mask, diff_mask)
            else:
                combined = fg_mask

            # Morphological cleanup
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            # Contour detection
            contours, _ = cv2.findContours(
                combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            dets = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < cfg["min_blob_area_px"] or area > cfg["max_blob_area_px"]:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                dets.append({
                    "x": M["m10"] / M["m00"],
                    "y": M["m01"] / M["m00"],
                    "area": area,
                    "radius": np.sqrt(area / np.pi),
                })

            all_detections.append(dets)

            # ── Link detections to tracks ──
            active_tracks, finished_tracks, next_id = self._link_frame(
                fi, dets, active_tracks, finished_tracks, next_id)

            prev_gray = gray

            if progress_cb and fi % 30 == 0:
                progress_cb("Detecting & linking...", n_frames + fi, total_steps)

        cap.release()

        # Flush remaining active tracks
        finished_tracks.extend(active_tracks)

        if progress_cb:
            progress_cb("Merging track fragments...", n_frames * 2, total_steps)

        # ── Merge fragmented tracks ──
        tracks = self._merge_tracks(finished_tracks)

        # ── Classify ──
        moving, static = self._classify_tracks(tracks)

        if progress_cb:
            progress_cb("Done!", total_steps, total_steps)

        return {
            "video_path": str(video_path),
            "fps": fps,
            "n_frames": actual_frames,
            "width": width,
            "height": height,
            "background": background,
            "first_frame": first_frame,
            "moving_tracks": moving,
            "static_tracks": static,
            "all_detections": all_detections,
        }

    def _link_frame(self, fi, dets, active, finished, next_id):
        """Link detections in frame fi to active tracks."""
        cfg = self.config
        max_dist = cfg["max_link_distance_px"]
        max_skip = cfg["max_frame_skip"]

        # Expire stale tracks
        still_active = []
        for t in active:
            if fi - t["last_frame"] > max_skip:
                finished.append(t)
            else:
                still_active.append(t)
        active = still_active

        if not dets:
            return active, finished, next_id

        if not active:
            for d in dets:
                active.append(self._new_track(next_id, fi, d))
                next_id += 1
            return active, finished, next_id

        # Build cost matrix using velocity-predicted positions
        n_t, n_d = len(active), len(dets)
        cost = np.full((n_t, n_d), 1e6)

        for ti, t in enumerate(active):
            dt = fi - t["last_frame"]
            pred_x = t["last_x"] + t["vx"] * dt
            pred_y = t["last_y"] + t["vy"] * dt
            for di, d in enumerate(dets):
                dist = np.hypot(pred_x - d["x"], pred_y - d["y"])
                if dist <= max_dist:
                    cost[ti, di] = dist

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_dets = set()
        alpha = cfg["velocity_alpha"]
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= 1e5:
                continue
            d = dets[ci]
            t = active[ri]
            dt = fi - t["last_frame"]
            if dt > 0:
                nvx = (d["x"] - t["last_x"]) / dt
                nvy = (d["y"] - t["last_y"]) / dt
                t["vx"] = alpha * nvx + (1 - alpha) * t["vx"]
                t["vy"] = alpha * nvy + (1 - alpha) * t["vy"]
            t["points"].append((fi, d["x"], d["y"], d["area"], d["radius"]))
            t["last_frame"] = fi
            t["last_x"] = d["x"]
            t["last_y"] = d["y"]
            matched_dets.add(ci)

        for di, d in enumerate(dets):
            if di not in matched_dets:
                active.append(self._new_track(next_id, fi, d))
                next_id += 1

        return active, finished, next_id

    @staticmethod
    def _new_track(tid, frame, det):
        return {
            "id": tid,
            "points": [(frame, det["x"], det["y"], det["area"], det["radius"])],
            "last_frame": frame,
            "last_x": det["x"], "last_y": det["y"],
            "vx": 0.0, "vy": 0.0,
        }

    def _merge_tracks(self, tracks):
        """Merge temporally adjacent, spatially compatible track fragments."""
        cfg = self.config
        max_gap = cfg["merge_max_gap_frames"]
        max_dist = cfg["merge_max_distance_px"]

        for t in tracks:
            t["start_frame"] = t["points"][0][0]
            t["end_frame"] = t["points"][-1][0]
        tracks.sort(key=lambda t: t["start_frame"])

        changed = True
        while changed:
            changed = False
            for i in range(len(tracks)):
                if tracks[i] is None:
                    continue
                ti = tracks[i]
                # End velocity estimate
                if len(ti["points"]) >= 2:
                    p1, p2 = ti["points"][-2], ti["points"][-1]
                    df = p2[0] - p1[0]
                    evx = (p2[1] - p1[1]) / df if df > 0 else 0
                    evy = (p2[2] - p1[2]) / df if df > 0 else 0
                else:
                    evx = evy = 0

                for j in range(i + 1, len(tracks)):
                    if tracks[j] is None:
                        continue
                    tj = tracks[j]
                    gap = tj["start_frame"] - ti["end_frame"]
                    if gap < 1 or gap > max_gap:
                        continue

                    pred_x = ti["points"][-1][1] + evx * gap
                    pred_y = ti["points"][-1][2] + evy * gap
                    sx = tj["points"][0][1]
                    sy = tj["points"][0][2]

                    if np.hypot(pred_x - sx, pred_y - sy) < max_dist:
                        ti["points"].extend(tj["points"])
                        ti["end_frame"] = tj["end_frame"]
                        tracks[j] = None
                        changed = True
                        break

        return [t for t in tracks if t is not None]

    def _classify_tracks(self, tracks):
        """Separate moving tracks from static noise."""
        cfg = self.config
        moving, static = [], []
        for t in tracks:
            pts = t["points"]
            if len(pts) < cfg["min_track_length"]:
                continue
            disp = np.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
            if disp >= cfg["min_displacement_px"]:
                moving.append(t)
            else:
                static.append(t)
        return moving, static

    def delete_track(self, results, track_id):
        """Remove a track by ID from moving_tracks."""
        results["moving_tracks"] = [
            t for t in results["moving_tracks"] if t["id"] != track_id]

    def merge_tracks_manual(self, results, id_a, id_b):
        """Manually merge two tracks by ID."""
        tracks = results["moving_tracks"]
        ta = next((t for t in tracks if t["id"] == id_a), None)
        tb = next((t for t in tracks if t["id"] == id_b), None)
        if ta and tb:
            ta["points"] = sorted(ta["points"] + tb["points"], key=lambda p: p[0])
            ta["start_frame"] = ta["points"][0][0]
            ta["end_frame"] = ta["points"][-1][0]
            results["moving_tracks"] = [t for t in tracks if t["id"] != id_b]


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def tracks_to_csv(tracks, fps, px_per_mm, filepath):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "track_id", "frame", "time_ms",
            "x_px", "y_px", "x_mm", "y_mm",
            "area_px2", "radius_px", "radius_um",
            "cumulative_dist_mm", "inst_velocity_mm_per_s",
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
                    p = pts[j - 1]
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
                    f"{cum:.5f}", f"{vel:.4f}",
                ])


def tracks_to_json(tracks, fps, px_per_mm, filepath):
    data = {"px_per_mm": px_per_mm, "fps": fps, "tracks": []}
    for track in tracks:
        td = {"id": track["id"], "points": []}
        cum = 0.0
        for j, (fr, x, y, area, rad) in enumerate(track["points"]):
            if j > 0:
                p = track["points"][j - 1]
                cum += np.hypot((x - p[1]) / px_per_mm, (y - p[2]) / px_per_mm)
            td["points"].append({
                "frame": int(fr), "time_ms": round(fr / fps * 1000, 3),
                "x_px": round(x, 2), "y_px": round(y, 2),
                "x_mm": round(x / px_per_mm, 5), "y_mm": round(y / px_per_mm, 5),
                "area_px2": round(area, 1),
                "radius_um": round(rad / px_per_mm * 1000, 2),
                "cumulative_dist_mm": round(cum, 5),
            })
        data["tracks"].append(td)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def generate_summary(results, px_per_mm, mag_label):
    fps = results["fps"]
    moving = results["moving_tracks"]
    lines = [
        "=" * 50,
        "  Microbubble Tracking Summary",
        "=" * 50,
        f"  Video:          {Path(results['video_path']).name}",
        f"  Resolution:     {results['width']} x {results['height']} px",
        f"  Frames:         {results['n_frames']}",
        f"  FPS:            {fps:.2f}",
        f"  Duration:       {results['n_frames'] / fps:.3f} s",
        f"  Magnification:  {mag_label} ({px_per_mm:.0f} px/mm)",
        "",
        f"  Moving tracks:  {len(moving)}",
        f"  Static (noise): {len(results['static_tracks'])} (filtered)",
        "-" * 50,
    ]
    for track in moving:
        pts = track["points"]
        dur_ms = (pts[-1][0] - pts[0][0]) / fps * 1000
        path_len = 0
        vels = []
        for j in range(1, len(pts)):
            dx = (pts[j][1] - pts[j - 1][1]) / px_per_mm
            dy = (pts[j][2] - pts[j - 1][2]) / px_per_mm
            seg = np.hypot(dx, dy)
            path_len += seg
            dt = (pts[j][0] - pts[j - 1][0]) / fps
            if dt > 0:
                vels.append(seg / dt)

        disp = np.hypot(
            (pts[-1][1] - pts[0][1]) / px_per_mm,
            (pts[-1][2] - pts[0][2]) / px_per_mm)
        mean_r = np.mean([p[4] for p in pts]) / px_per_mm * 1000

        lines.append("")
        lines.append(f"  Track {track['id']}")
        lines.append(f"    Detections:     {len(pts)}")
        lines.append(f"    Duration:       {dur_ms:.1f} ms")
        lines.append(f"    Path length:    {path_len * 1000:.1f} um")
        lines.append(f"    Displacement:   {disp * 1000:.1f} um")
        lines.append(f"    Mean radius:    {mean_r:.1f} um")
        if vels:
            lines.append(f"    Mean velocity:  {np.mean(vels) * 1000:.1f} um/s")
            lines.append(f"    Max velocity:   {np.max(vels) * 1000:.1f} um/s")

    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tracks_on_image(results, px_per_mm, fig=None):
    if fig is None:
        fig = Figure(figsize=(12, 3), dpi=100)
    fig.clear()
    ax = fig.add_subplot(111)
    bg = results["background"]
    h, w = bg.shape[:2]
    ext = [0, w / px_per_mm, h / px_per_mm, 0]
    ax.imshow(bg, cmap="gray", extent=ext, aspect="equal")

    for i, track in enumerate(results["moving_tracks"]):
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        pts = track["points"]
        xs = [p[1] / px_per_mm for p in pts]
        ys = [p[2] / px_per_mm for p in pts]
        ax.plot(xs, ys, color=color, lw=1.5, label=f"Track {track['id']}")
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
            t_ms = (pts[j][0] + pts[j - 1][0]) / 2.0 / fps * 1000
            dx = (pts[j][1] - pts[j - 1][1]) / px_per_mm
            dy = (pts[j][2] - pts[j - 1][2]) / px_per_mm
            dt = (pts[j][0] - pts[j - 1][0]) / fps
            if dt > 0:
                times.append(t_ms)
                vels.append(np.hypot(dx, dy) / dt * 1000)
        ax.plot(times, vels, color=color, lw=1, alpha=0.8,
                label=f"Track {track['id']}")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity (um/s)")
    ax.set_title("Instantaneous Velocity")
    if len(results["moving_tracks"]) <= 10:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_displacement_vs_time(results, px_per_mm, fig=None):
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
            dx = (pts[j][1] - pts[j - 1][1]) / px_per_mm * 1000
            dy = (pts[j][2] - pts[j - 1][2]) / px_per_mm * 1000
            cum.append(cum[-1] + np.hypot(dx, dy))
        ax.plot(times, cum, color=color, lw=1.2, label=f"Track {track['id']}")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cumulative Displacement (um)")
    ax.set_title("Cumulative Displacement")
    if len(results["moving_tracks"]) <= 10:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO FRAME READER (for player)
# ═══════════════════════════════════════════════════════════════════════════════

class VideoFrameReader:
    """Random-access frame reader for the video player."""

    def __init__(self, video_path):
        self.path = str(video_path)
        self.cap = cv2.VideoCapture(self.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cache = {}
        self._cache_max = 200

    def read_frame(self, index):
        """Read a single frame by index. Returns BGR numpy array or None."""
        if index in self._cache:
            return self._cache[index]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if ret:
            if len(self._cache) >= self._cache_max:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[index] = frame
            return frame
        return None

    def release(self):
        self.cap.release()

    def __del__(self):
        try:
            self.cap.release()
        except Exception:
            pass
