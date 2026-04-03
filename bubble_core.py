"""
bubble_core.py — Microbubble detection, tracking, and export engine (v5).

v5 — Template-matching tracker replaces detect-and-link.

Tracks bubbles the way a human eye does:
  1. Selective initialization — only high-confidence detections start tracks
  2. Visual following — each bubble carries a small image template that is
     matched frame-by-frame via normalized cross-correlation (NCC)
  3. Inertia enforcement — velocity prediction centers the search window;
     physically impossible jumps are rejected
  4. Post-hoc validation — tracks are split at acceleration spikes that
     indicate mistracking

Detection (for track spawning only):
  - Temporal median background model
  - Background-subtracted dark-pixel detection (bg - frame > threshold)
  - Contrast + circularity filtering

Tracking:
  - Per-track appearance template (patch_size x patch_size pixels)
  - NCC template matching in a small search window around predicted position
  - Slow template adaptation to handle gradual focus/appearance changes
  - Periodic re-detection to spawn tracks for newly arriving bubbles

Post-processing:
  - Acceleration-based track splitting (inertia validation)
  - Velocity-extrapolated fragment merging
  - Median velocity filter for export

No GUI dependencies. Can be imported for scripting or batch processing.
"""

import cv2
import numpy as np
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
    "bg_n_samples":         50,      # frames sampled for median background
    # Detection (used for track spawning)
    "bg_sub_threshold":     10,      # min (background - frame) to detect dark bubble
    "min_contrast":         8,       # min mean contrast vs background inside contour
    "min_circularity":      0.20,    # reject non-circular blobs (0-1)
    "min_blob_area_px":     5,       # minimum contour area
    "max_blob_area_px":     5000,    # maximum contour area
    "morph_kernel_size":    3,       # morphological cleanup kernel
    # Template tracking
    "patch_size":           21,      # appearance template size (pixels, odd)
    "search_margin_px":     12,      # search window around predicted position
    "min_ncc":              0.7,     # minimum template match score to continue track
    "template_adapt_rate":  0.05,    # how fast template updates (0=frozen, 1=instant)
    "spawn_interval":       10,      # re-detect for new bubbles every N frames
    "spawn_min_distance_px": 15,     # don't spawn near existing tracks
    "max_frame_skip":       3,       # max consecutive lost frames before termination
    "velocity_alpha":       0.4,     # EMA smoothing for velocity estimate
    # Track validation
    "max_acceleration_px":  3.0,     # split tracks at acceleration above this (px/f^2)
    # Track classification
    "min_track_length":     5,       # minimum detections to keep a track
    "min_displacement_px":  8.0,     # minimum net displacement to be "moving"
    # Merging
    "merge_max_gap_frames": 8,       # max frame gap for merging fragments
    "merge_max_distance_px": 20,     # max spatial distance for merging
    # Velocity smoothing
    "velocity_median_window": 5,     # median filter window for velocity output
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
    Template-matching tracker for microbubble microscopy.

    Pass 1 (fast): Build temporal median background from sampled frames.
    Pass 2 (streaming): Template-track bubbles frame by frame.

    Each bubble carries a small image patch (template) that is matched
    in each new frame using normalized cross-correlation. Tracks are
    initialized from high-confidence detections and validated post-hoc
    using acceleration constraints (inertia).
    """

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def process_video(self, video_path, progress_cb=None):
        """Full pipeline. Returns results dict with tracks and metadata."""
        cfg = self.config
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_steps = n_frames + 30

        # ── Pass 1: Build median background ──
        if progress_cb:
            progress_cb("Computing background...", 0, total_steps)

        background = self._compute_background(cap, n_frames, cfg["bg_n_samples"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_bgr = cap.read()
        first_frame = (cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
                       if ret else background.copy())

        # ── Pass 2: Template tracking ──
        if progress_cb:
            progress_cb("Tracking bubbles...", 5, total_steps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg["morph_kernel_size"], cfg["morph_kernel_size"]))

        active_tracks = []
        finished_tracks = []
        next_id = 0
        all_detections = []

        half = cfg["patch_size"] // 2
        spawn_interval = cfg["spawn_interval"]

        actual_frames = 0
        for fi in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            actual_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Spawn new tracks periodically
            if fi % spawn_interval == 0:
                dets = self._detect_frame(gray, background, kernel, cfg)
                next_id = self._spawn_tracks(
                    dets, gray, fi, active_tracks, next_id, cfg)
            else:
                dets = []

            # Template-match all active tracks
            still_active = []
            for t in active_tracks:
                if t["points"][-1][0] == fi:
                    # Just spawned this frame, skip tracking
                    still_active.append(t)
                    continue

                matched = self._track_template(t, gray, fi, cfg)
                if matched:
                    t["lost_count"] = 0
                    still_active.append(t)
                else:
                    t["lost_count"] = t.get("lost_count", 0) + 1
                    if t["lost_count"] < cfg["max_frame_skip"]:
                        still_active.append(t)
                    else:
                        finished_tracks.append(t)

            active_tracks = still_active

            # Synthesize detection list for GUI compatibility
            frame_dets = []
            for t in active_tracks:
                if t["points"][-1][0] == fi:
                    pt = t["points"][-1]
                    frame_dets.append({
                        "x": pt[1], "y": pt[2],
                        "area": t["init_area"],
                        "radius": t["init_radius"],
                        "intensity": 0,
                        "contrast": t.get("init_contrast", 0),
                    })
            all_detections.append(frame_dets)

            if progress_cb and fi % 30 == 0:
                progress_cb("Tracking bubbles...", 5 + fi, total_steps)

        cap.release()

        if actual_frames < 3:
            raise ValueError("Video has fewer than 3 readable frames.")

        # Flush remaining active tracks
        finished_tracks.extend(active_tracks)

        if progress_cb:
            progress_cb("Validating tracks...", 5 + actual_frames, total_steps)

        # ── Validate tracks (split at acceleration spikes) ──
        tracks = self._validate_tracks(finished_tracks, cfg)

        # ── Merge fragmented tracks ──
        tracks = self._merge_tracks(tracks, cfg)

        # ── Classify ──
        moving, static = self._classify_tracks(tracks, cfg)

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

    # ── Background computation ──────────────────────────────────────────────

    @staticmethod
    def _compute_background(cap, n_frames, n_samples):
        """Compute temporal median background from evenly sampled frames."""
        sample_idxs = np.linspace(0, n_frames - 1,
                                  min(n_samples, n_frames), dtype=int)
        samples = []
        for fi in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if not samples:
            raise ValueError("Could not read any frames for background.")

        return np.median(np.array(samples), axis=0).astype(np.uint8)

    # ── Detection (for track spawning) ─────────────────────────────────────

    @staticmethod
    def _detect_frame(gray, background, kernel, cfg):
        """
        Detect dark bubbles via background subtraction.

        Used only for spawning new tracks, not for frame-to-frame linking.
        """
        bg_thresh = cfg["bg_sub_threshold"]
        min_area = cfg["min_blob_area_px"]
        max_area = cfg["max_blob_area_px"]
        min_contrast = cfg["min_contrast"]
        min_circ = cfg["min_circularity"]

        diff = cv2.subtract(background, gray)
        _, mask = cv2.threshold(diff, bg_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            perim = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
            if circ < min_circ:
                continue

            x, y, w, h = cv2.boundingRect(c)
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(roi_mask, [c - np.array([x, y])], -1, 255, -1)

            mean_int = cv2.mean(gray[y:y+h, x:x+w], mask=roi_mask)[0]
            mean_bg = cv2.mean(background[y:y+h, x:x+w], mask=roi_mask)[0]
            contrast = mean_bg - mean_int

            if contrast < min_contrast:
                continue

            dets.append({
                "x": cx,
                "y": cy,
                "area": area,
                "radius": np.sqrt(area / np.pi),
                "intensity": mean_int,
                "contrast": contrast,
            })

        return dets

    # ── Track spawning ─────────────────────────────────────────────────────

    @staticmethod
    def _spawn_tracks(dets, gray, fi, active_tracks, next_id, cfg):
        """Spawn new tracks from detections not near existing tracks."""
        half = cfg["patch_size"] // 2
        min_dist = cfg["spawn_min_distance_px"]
        h_img, w_img = gray.shape

        active_positions = [(t["x"], t["y"]) for t in active_tracks]

        for d in dets:
            cx, cy = d["x"], d["y"]

            # Skip if too close to an existing track
            too_close = False
            for ax, ay in active_positions:
                if abs(cx - ax) < min_dist and abs(cy - ay) < min_dist:
                    if np.hypot(cx - ax, cy - ay) < min_dist:
                        too_close = True
                        break
            if too_close:
                continue

            ix, iy = int(round(cx)), int(round(cy))
            if (iy - half < 0 or iy + half + 1 > h_img
                    or ix - half < 0 or ix + half + 1 > w_img):
                continue

            template = gray[iy-half:iy+half+1, ix-half:ix+half+1].astype(
                np.float32)

            active_tracks.append({
                "id": next_id,
                "points": [(fi, cx, cy, d["area"], d["radius"], 1.0)],
                "x": cx,
                "y": cy,
                "vx": 0.0,
                "vy": 0.0,
                "template": template,
                "init_area": d["area"],
                "init_radius": d["radius"],
                "init_contrast": d["contrast"],
                "lost_count": 0,
            })
            active_positions.append((cx, cy))
            next_id += 1

        return next_id

    # ── Template matching per track ────────────────────────────────────────

    @staticmethod
    def _track_template(track, gray, fi, cfg):
        """
        Match a track's template in the current frame.

        Returns True if match found, False otherwise.
        Updates track position, velocity, template, and appends point.
        """
        half = cfg["patch_size"] // 2
        margin = cfg["search_margin_px"]
        min_ncc = cfg["min_ncc"]
        alpha = cfg["velocity_alpha"]
        adapt = cfg["template_adapt_rate"]
        h_img, w_img = gray.shape

        # Predict position using velocity
        px = track["x"] + track["vx"]
        py = track["y"] + track["vy"]
        ipx, ipy = int(round(px)), int(round(py))

        # Define search region
        sy1 = max(0, ipy - half - margin)
        sy2 = min(h_img, ipy + half + 1 + margin)
        sx1 = max(0, ipx - half - margin)
        sx2 = min(w_img, ipx + half + 1 + margin)

        patch_size = cfg["patch_size"]
        if sy2 - sy1 < patch_size or sx2 - sx1 < patch_size:
            return False

        search = gray[sy1:sy2, sx1:sx2].astype(np.float32)

        result = cv2.matchTemplate(search, track["template"],
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < min_ncc:
            return False

        # Update position
        new_x = float(sx1 + max_loc[0] + half)
        new_y = float(sy1 + max_loc[1] + half)

        # Update velocity with EMA
        nvx = new_x - track["x"]
        nvy = new_y - track["y"]
        track["vx"] = alpha * nvx + (1 - alpha) * track["vx"]
        track["vy"] = alpha * nvy + (1 - alpha) * track["vy"]

        track["x"] = new_x
        track["y"] = new_y
        track["points"].append(
            (fi, new_x, new_y, track["init_area"], track["init_radius"],
             max_val))

        # Slowly adapt template
        ix, iy = int(round(new_x)), int(round(new_y))
        if (iy - half >= 0 and iy + half + 1 <= h_img
                and ix - half >= 0 and ix + half + 1 <= w_img):
            new_patch = gray[iy-half:iy+half+1, ix-half:ix+half+1].astype(
                np.float32)
            track["template"] = ((1 - adapt) * track["template"]
                                 + adapt * new_patch)

        return True

    # ── Track validation (inertia check) ───────────────────────────────────

    @staticmethod
    def _validate_tracks(tracks, cfg):
        """
        Split tracks at acceleration spikes that indicate mistracking.

        Bubbles maintain inertia — sudden jumps in velocity direction or
        magnitude are physically impossible and indicate the tracker locked
        onto a different bubble.
        """
        max_accel = cfg["max_acceleration_px"]
        if max_accel <= 0:
            return tracks

        validated = []
        for t in tracks:
            pts = t["points"]
            if len(pts) < 3:
                validated.append(t)
                continue

            # Find split points where acceleration exceeds threshold
            split_indices = []
            for i in range(2, len(pts)):
                # Only check consecutive frames
                if pts[i][0] != pts[i-1][0] + 1 or pts[i-1][0] != pts[i-2][0] + 1:
                    continue
                vx1 = pts[i-1][1] - pts[i-2][1]
                vy1 = pts[i-1][2] - pts[i-2][2]
                vx2 = pts[i][1] - pts[i-1][1]
                vy2 = pts[i][2] - pts[i-1][2]
                accel = np.hypot(vx2 - vx1, vy2 - vy1)
                if accel > max_accel:
                    split_indices.append(i)

            if not split_indices:
                validated.append(t)
                continue

            # Split into fragments
            boundaries = [0] + split_indices + [len(pts)]
            for j in range(len(boundaries) - 1):
                fragment_pts = pts[boundaries[j]:boundaries[j+1]]
                if len(fragment_pts) >= 2:
                    validated.append({
                        "id": t["id"],
                        "points": fragment_pts,
                    })

        return validated

    # ── Track fragment merging ──────────────────────────────────────────────

    @staticmethod
    def _merge_tracks(tracks, cfg):
        """
        Merge temporally adjacent, spatially compatible track fragments.

        Uses velocity extrapolation to predict where a track's continuation
        should start, then matches nearby fragment starts.
        """
        max_gap = cfg["merge_max_gap_frames"]
        max_dist = cfg["merge_max_distance_px"]

        if not tracks:
            return []

        for t in tracks:
            t["start_frame"] = t["points"][0][0]
            t["end_frame"] = t["points"][-1][0]

        tracks.sort(key=lambda t: t["end_frame"])
        consumed = set()
        n = len(tracks)

        sf_index = {}
        for idx, t in enumerate(tracks):
            sf = t["start_frame"]
            if sf not in sf_index:
                sf_index[sf] = []
            sf_index[sf].append(idx)

        for i in range(n):
            if i in consumed:
                continue
            ti = tracks[i]

            if len(ti["points"]) >= 2:
                p1, p2 = ti["points"][-2], ti["points"][-1]
                df = p2[0] - p1[0]
                evx = (p2[1] - p1[1]) / df if df > 0 else 0
                evy = (p2[2] - p1[2]) / df if df > 0 else 0
            else:
                evx = evy = 0

            best_j = None
            best_score = max_dist

            for gap in range(1, max_gap + 1):
                target = ti["end_frame"] + gap
                if target not in sf_index:
                    continue
                pred_x = ti["points"][-1][1] + evx * gap
                pred_y = ti["points"][-1][2] + evy * gap
                for j in sf_index[target]:
                    if j == i or j in consumed:
                        continue
                    tj = tracks[j]
                    dist = np.hypot(pred_x - tj["points"][0][1],
                                    pred_y - tj["points"][0][2])
                    if dist < best_score:
                        best_score = dist
                        best_j = j

            if best_j is not None:
                tj = tracks[best_j]
                ti["points"].extend(tj["points"])
                ti["points"].sort(key=lambda p: p[0])
                ti["end_frame"] = ti["points"][-1][0]
                consumed.add(best_j)
                sf = tj["start_frame"]
                if sf in sf_index:
                    sf_index[sf] = [
                        x for x in sf_index[sf] if x != best_j]

        return [tracks[i] for i in range(n) if i not in consumed]

    # ── Classification ──────────────────────────────────────────────────────

    @staticmethod
    def _classify_tracks(tracks, cfg):
        """Separate moving tracks from static noise."""
        min_len = cfg["min_track_length"]
        min_disp = cfg["min_displacement_px"]
        moving, static = [], []
        for t in tracks:
            pts = t["points"]
            if len(pts) < min_len:
                continue
            disp = np.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
            if disp >= min_disp:
                moving.append(t)
            else:
                static.append(t)
        return moving, static

    # ── Manual editing ──────────────────────────────────────────────────────

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
            ta["points"] = sorted(
                ta["points"] + tb["points"], key=lambda p: p[0])
            ta["start_frame"] = ta["points"][0][0]
            ta["end_frame"] = ta["points"][-1][0]
            results["moving_tracks"] = [
                t for t in tracks if t["id"] != id_b]


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _median_filter_velocities(raw_vels, window=5):
    """Apply median filter to a list of velocities to remove spikes."""
    if len(raw_vels) < window:
        return raw_vels[:]
    hw = window // 2
    smoothed = []
    for i in range(len(raw_vels)):
        start = max(0, i - hw)
        end = min(len(raw_vels), i + hw + 1)
        smoothed.append(float(np.median(raw_vels[start:end])))
    return smoothed


def tracks_to_csv(tracks, fps, px_per_mm, filepath, velocity_window=5):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "track_id", "frame", "time_ms",
            "x_px", "y_px", "x_mm", "y_mm",
            "area_px2", "radius_px", "radius_um",
            "cumulative_dist_mm", "inst_velocity_mm_per_s",
            "ncc_score",
        ])
        for track in tracks:
            tid = track["id"]
            pts = track["points"]

            raw_vels = []
            for j in range(len(pts)):
                if j == 0:
                    raw_vels.append(0.0)
                else:
                    p = pts[j - 1]
                    dx = (pts[j][1] - p[1]) / px_per_mm
                    dy = (pts[j][2] - p[2]) / px_per_mm
                    seg = np.hypot(dx, dy)
                    dt = (pts[j][0] - p[0]) / fps
                    raw_vels.append(seg / dt if dt > 0 else 0.0)

            if len(raw_vels) > 1:
                smoothed = [0.0] + _median_filter_velocities(
                    raw_vels[1:], velocity_window)
            else:
                smoothed = raw_vels

            cum = 0.0
            for j, pt in enumerate(pts):
                fr, x, y, area, rad = pt[0], pt[1], pt[2], pt[3], pt[4]
                ncc = pt[5] if len(pt) > 5 else 1.0
                t_ms = fr / fps * 1000.0
                x_mm = x / px_per_mm
                y_mm = y / px_per_mm
                rad_um = rad / px_per_mm * 1000.0
                if j > 0:
                    p = pts[j - 1]
                    dx = (x - p[1]) / px_per_mm
                    dy = (y - p[2]) / px_per_mm
                    cum += np.hypot(dx, dy)
                w.writerow([
                    tid, fr, f"{t_ms:.3f}",
                    f"{x:.2f}", f"{y:.2f}", f"{x_mm:.5f}", f"{y_mm:.5f}",
                    f"{area:.1f}", f"{rad:.2f}", f"{rad_um:.2f}",
                    f"{cum:.5f}", f"{smoothed[j]:.4f}",
                    f"{ncc:.3f}",
                ])


def tracks_to_json(tracks, fps, px_per_mm, filepath, velocity_window=5):
    data = {"px_per_mm": px_per_mm, "fps": fps, "tracks": []}
    for track in tracks:
        td = {"id": track["id"], "points": []}
        cum = 0.0
        for j, pt in enumerate(track["points"]):
            fr, x, y, area, rad = pt[0], pt[1], pt[2], pt[3], pt[4]
            ncc = pt[5] if len(pt) > 5 else 1.0
            if j > 0:
                p = track["points"][j - 1]
                cum += np.hypot(
                    (x - p[1]) / px_per_mm, (y - p[2]) / px_per_mm)
            td["points"].append({
                "frame": int(fr),
                "time_ms": round(fr / fps * 1000, 3),
                "x_px": round(x, 2), "y_px": round(y, 2),
                "x_mm": round(x / px_per_mm, 5),
                "y_mm": round(y / px_per_mm, 5),
                "area_px2": round(area, 1),
                "radius_um": round(rad / px_per_mm * 1000, 2),
                "cumulative_dist_mm": round(cum, 5),
                "ncc_score": round(ncc, 3),
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

        if vels:
            smoothed = _median_filter_velocities(vels)
        else:
            smoothed = []

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
        if smoothed:
            lines.append(
                f"    Mean velocity:  {np.mean(smoothed) * 1000:.1f} um/s")
            lines.append(
                f"    Max velocity:   {np.max(smoothed) * 1000:.1f} um/s")

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
    vel_window = 5

    for i, track in enumerate(results["moving_tracks"]):
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        pts = track["points"]
        times, raw_vels = [], []
        for j in range(1, len(pts)):
            t_ms = (pts[j][0] + pts[j - 1][0]) / 2.0 / fps * 1000
            dx = (pts[j][1] - pts[j - 1][1]) / px_per_mm
            dy = (pts[j][2] - pts[j - 1][2]) / px_per_mm
            dt = (pts[j][0] - pts[j - 1][0]) / fps
            if dt > 0:
                times.append(t_ms)
                raw_vels.append(np.hypot(dx, dy) / dt * 1000)

        if raw_vels:
            smoothed = _median_filter_velocities(raw_vels, vel_window)
            ax.plot(times, smoothed, color=color, lw=1, alpha=0.8,
                    label=f"Track {track['id']}")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity (um/s)")
    ax.set_title("Instantaneous Velocity (median-filtered)")
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
        ax.plot(times, cum, color=color, lw=1.2,
                label=f"Track {track['id']}")

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
