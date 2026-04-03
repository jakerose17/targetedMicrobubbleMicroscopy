#!/usr/bin/env python3
"""
MicroBubble Tracker — PyQt6 GUI Application (v2)

Features:
  - Drag-and-drop video loading (1 to many)
  - Per-video magnification (10x/20x/35x/50x) with calibrated pixel scaling
  - Median-background dark-bubble detection with contrast filtering
  - Velocity-predicted Hungarian linking + automatic track merging
  - Video player with track overlays, play/pause, scrub
  - Track selection, deletion, and manual merging
  - Export: CSV, JSON, publication-quality plots

Usage:
    python microbubble_tracker.py
    python microbubble_tracker.py video1.mp4 video2.mp4

Dependencies:
    pip install PyQt6 opencv-python numpy scipy matplotlib
"""

import sys
import os
import threading
from pathlib import Path
from functools import partial

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QListWidgetItem, QTabWidget, QTextEdit,
    QToolBar, QComboBox, QLabel, QProgressBar, QFileDialog,
    QMessageBox, QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QPushButton, QGroupBox, QAbstractItemView, QSizePolicy, QSlider,
    QCheckBox, QSpinBox,
)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QObject, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QAction, QPainter, QColor, QPen

import numpy as np
import cv2

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from bubble_core import (
    BubbleTracker, MAGNIFICATION_MAP, DEFAULT_CONFIG, TRACK_COLORS,
    tracks_to_csv, tracks_to_json, generate_summary,
    plot_tracks_on_image, plot_velocity_profiles, plot_displacement_vs_time,
    VideoFrameReader,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class TrackingSignals(QObject):
    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(int, bool, str)
    all_done = pyqtSignal()


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

class VideoEntry:
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        self.magnification = "10x"
        self.results = None
        self.status = "Pending"
        self.reader = None  # VideoFrameReader, created on demand


# ═══════════════════════════════════════════════════════════════════════════════
# DRAG-AND-DROP LIST
# ═══════════════════════════════════════════════════════════════════════════════

class VideoListWidget(QListWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        font = QFont("Menlo", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class MplCanvas(QWidget):
    def __init__(self, figsize=(10, 4), parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def redraw(self):
        self.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO PLAYER WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class VideoPlayerWidget(QWidget):
    """Frame-by-frame video player with track overlays."""

    track_selected = pyqtSignal(int)   # emits track_id when clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_entry = None
        self.reader = None
        self.current_frame = 0
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(100)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("background: #1a1a1a; border-radius: 4px;")
        layout.addWidget(self.image_label, 1)

        # Controls row
        ctrl = QHBoxLayout()

        self.btn_prev = QPushButton("◀◀")
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        ctrl.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(50)
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.btn_play)

        self.btn_next = QPushButton("▶▶")
        self.btn_next.setFixedWidth(40)
        self.btn_next.clicked.connect(lambda: self._step(1))
        ctrl.addWidget(self.btn_next)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._slider_changed)
        ctrl.addWidget(self.slider, 1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFixedWidth(120)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl.addWidget(self.frame_label)

        self.time_label = QLabel("0.000 s")
        self.time_label.setFixedWidth(80)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl.addWidget(self.time_label)

        layout.addLayout(ctrl)

        # Overlay options row
        opts = QHBoxLayout()

        self.chk_tracks = QCheckBox("Show tracks")
        self.chk_tracks.setChecked(True)
        self.chk_tracks.toggled.connect(self._refresh_display)
        opts.addWidget(self.chk_tracks)

        self.chk_detections = QCheckBox("Show detections")
        self.chk_detections.setChecked(True)
        self.chk_detections.toggled.connect(self._refresh_display)
        opts.addWidget(self.chk_detections)

        self.chk_trail = QCheckBox("Show trail")
        self.chk_trail.setChecked(True)
        self.chk_trail.toggled.connect(self._refresh_display)
        opts.addWidget(self.chk_trail)

        opts.addWidget(QLabel("Trail length:"))
        self.trail_spin = QSpinBox()
        self.trail_spin.setRange(5, 500)
        self.trail_spin.setValue(50)
        self.trail_spin.valueChanged.connect(self._refresh_display)
        opts.addWidget(self.trail_spin)

        opts.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.1x", "0.25x", "0.5x", "1x", "2x", "5x"])
        self.speed_combo.setCurrentText("0.5x")
        self.speed_combo.currentTextChanged.connect(self._speed_changed)
        opts.addWidget(self.speed_combo)

        opts.addStretch()
        layout.addLayout(opts)

    def set_video(self, video_entry):
        """Load a video entry for playback."""
        self.stop()
        self.video_entry = video_entry
        if video_entry and video_entry.path.exists():
            if self.reader:
                self.reader.release()
            self.reader = VideoFrameReader(video_entry.path)
            self.slider.setMaximum(max(0, self.reader.n_frames - 1))
            self.current_frame = 0
            self.slider.setValue(0)
            self._refresh_display()
        else:
            self.reader = None
            self.image_label.clear()
            self.frame_label.setText("0 / 0")
            self.time_label.setText("0.000 s")

    def _toggle_play(self):
        if self.playing:
            self.stop()
        else:
            self.play()

    def play(self):
        if not self.reader:
            return
        self.playing = True
        self.btn_play.setText("⏸")
        speed = float(self.speed_combo.currentText().replace("x", ""))
        interval = max(1, int(1000.0 / self.reader.fps / speed))
        self.timer.start(interval)

    def stop(self):
        self.playing = False
        self.btn_play.setText("▶")
        self.timer.stop()

    def _advance_frame(self):
        if not self.reader:
            return
        nf = self.current_frame + 1
        if nf >= self.reader.n_frames:
            self.stop()
            return
        self.current_frame = nf
        self.slider.blockSignals(True)
        self.slider.setValue(nf)
        self.slider.blockSignals(False)
        self._refresh_display()

    def _step(self, delta):
        if not self.reader:
            return
        nf = max(0, min(self.reader.n_frames - 1, self.current_frame + delta))
        self.current_frame = nf
        self.slider.blockSignals(True)
        self.slider.setValue(nf)
        self.slider.blockSignals(False)
        self._refresh_display()

    def _slider_changed(self, value):
        self.current_frame = value
        self._refresh_display()

    def _speed_changed(self, text):
        if self.playing:
            self.stop()
            self.play()

    def _refresh_display(self, *_args):
        if not self.reader:
            return
        frame_bgr = self.reader.read_frame(self.current_frame)
        if frame_bgr is None:
            return

        # Draw overlays
        display = frame_bgr.copy()
        ve = self.video_entry

        if ve and ve.results:
            fi = self.current_frame
            tracks = ve.results["moving_tracks"]

            # Draw track trails
            if self.chk_trail.isChecked() and self.chk_tracks.isChecked():
                trail_len = self.trail_spin.value()
                for ti, track in enumerate(tracks):
                    color_hex = TRACK_COLORS[ti % len(TRACK_COLORS)]
                    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                    pts = track["points"]
                    trail_pts = [(int(p[1]), int(p[2]))
                                 for p in pts
                                 if fi - trail_len <= p[0] <= fi]
                    if len(trail_pts) >= 2:
                        for j in range(1, len(trail_pts)):
                            alpha = j / len(trail_pts)
                            col = (int(b * alpha), int(g * alpha), int(r * alpha))
                            cv2.line(display, trail_pts[j - 1], trail_pts[j], col, 2,
                                     cv2.LINE_AA)

            # Draw current-frame detections
            if self.chk_detections.isChecked():
                for ti, track in enumerate(tracks):
                    color_hex = TRACK_COLORS[ti % len(TRACK_COLORS)]
                    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                    for p in track["points"]:
                        if p[0] == fi:
                            cx, cy, rad = int(p[1]), int(p[2]), max(3, int(p[4]))
                            cv2.circle(display, (cx, cy), rad, (b, g, r), 2, cv2.LINE_AA)
                            # Label
                            cv2.putText(display, f"T{track['id']}",
                                        (cx + rad + 2, cy - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (b, g, r), 1, cv2.LINE_AA)

        # Convert to QPixmap and display
        h, w = display.shape[:2]
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        # Scale to fit label
        label_size = self.image_label.size()
        pixmap = QPixmap.fromImage(qimg).scaled(
            label_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        # Update labels
        fps = self.reader.fps if self.reader.fps > 0 else 60
        self.frame_label.setText(f"{self.current_frame} / {self.reader.n_frames - 1}")
        self.time_label.setText(f"{self.current_frame / fps:.3f} s")


# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════════════════════

class SettingsDialog(QDialog):
    PARAMS = [
        # Background model
        ("bg_n_samples",           "Background sample frames",               int),
        # Detection (track spawning)
        ("bg_sub_threshold",       "Background subtract threshold",          int),
        ("min_contrast",           "Min contrast vs background",             int),
        ("min_circularity",        "Min circularity (0-1)",                  float),
        ("min_blob_area_px",       "Min blob area (px^2)",                   int),
        ("max_blob_area_px",       "Max blob area (px^2)",                   int),
        ("morph_kernel_size",      "Morphology kernel (px)",                 int),
        # Template tracking
        ("patch_size",             "Template patch size (px, odd)",          int),
        ("search_margin_px",       "Search margin around prediction (px)",   int),
        ("min_ncc",                "Min template match score (0-1)",         float),
        ("template_adapt_rate",    "Template adaptation rate (0-1)",         float),
        ("spawn_interval",         "Spawn new tracks every N frames",        int),
        ("spawn_min_distance_px",  "Min distance to spawn near track (px)", int),
        ("max_frame_skip",         "Max lost frames before termination",     int),
        ("velocity_alpha",         "Velocity EMA alpha",                     float),
        # Track validation
        ("max_acceleration_px",    "Max acceleration (px/frame^2)",          float),
        # Track classification
        ("min_track_length",       "Min track length (detections)",          int),
        ("min_displacement_px",    "Min displacement for 'moving' (px)",     float),
        # Merging
        ("merge_max_gap_frames",   "Merge max gap (frames)",                 int),
        ("merge_max_distance_px",  "Merge max distance (px)",                int),
        # Velocity smoothing
        ("velocity_median_window", "Velocity median filter window",          int),
    ]

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tracking Settings")
        self.setMinimumWidth(460)
        self.config = config
        self.edits = {}

        layout = QVBoxLayout(self)
        form = QFormLayout()
        for key, label, dtype in self.PARAMS:
            edit = QLineEdit(str(config[key]))
            edit.setFixedWidth(80)
            form.addRow(label, edit)
            self.edits[key] = (edit, dtype)
        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        layout.addLayout(btn_layout)

    def _reset(self):
        for key, (edit, _) in self.edits.items():
            edit.setText(str(DEFAULT_CONFIG[key]))

    def _validate_and_accept(self):
        for key, (edit, dtype) in self.edits.items():
            val = edit.text().strip()
            try:
                dtype(val)
            except ValueError:
                QMessageBox.warning(self, "Invalid", f"Bad value for '{key}'")
                return
        for key, (edit, dtype) in self.edits.items():
            val = edit.text().strip()
            self.config[key] = dtype(val)
        self.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroBubble Tracker")
        self.resize(1400, 900)
        self.setMinimumSize(960, 640)

        self.videos = []
        self.config = {**DEFAULT_CONFIG}
        self.signals = TrackingSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_video_finished)
        self.signals.all_done.connect(self._on_all_done)
        self._worker_running = False

        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

    # ─── TOOLBAR ──────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        self.act_add = QAction("+ Add Videos", self)
        self.act_add.triggered.connect(self._add_videos)
        tb.addAction(self.act_add)

        self.act_remove = QAction("Remove", self)
        self.act_remove.triggered.connect(self._remove_selected)
        tb.addAction(self.act_remove)

        tb.addSeparator()
        tb.addWidget(QLabel("  Mag: "))
        self.mag_combo = QComboBox()
        self.mag_combo.addItems(list(MAGNIFICATION_MAP.keys()))
        self.mag_combo.setCurrentText("10x")
        self.mag_combo.currentTextChanged.connect(self._mag_changed)
        self.mag_combo.setFixedWidth(75)
        tb.addWidget(self.mag_combo)

        tb.addSeparator()
        self.act_settings = QAction("Settings", self)
        self.act_settings.triggered.connect(self._open_settings)
        tb.addAction(self.act_settings)

        tb.addSeparator()
        self.act_track_all = QAction("Track All", self)
        self.act_track_all.triggered.connect(self._run_all)
        tb.addAction(self.act_track_all)

        self.act_track_sel = QAction("Track Selected", self)
        self.act_track_sel.triggered.connect(self._run_selected)
        tb.addAction(self.act_track_sel)

        tb.addSeparator()
        self.act_csv = QAction("Export CSV", self)
        self.act_csv.triggered.connect(lambda: self._export("csv"))
        tb.addAction(self.act_csv)

        self.act_json = QAction("Export JSON", self)
        self.act_json.triggered.connect(lambda: self._export("json"))
        tb.addAction(self.act_json)

        self.act_plots = QAction("Export Plots", self)
        self.act_plots.triggered.connect(self._export_plots)
        tb.addAction(self.act_plots)

        tb.addSeparator()
        self.act_del_track = QAction("Delete Track", self)
        self.act_del_track.triggered.connect(self._delete_selected_track)
        tb.addAction(self.act_del_track)

        self.act_merge_tracks = QAction("Merge Tracks", self)
        self.act_merge_tracks.triggered.connect(self._merge_selected_tracks)
        tb.addAction(self.act_merge_tracks)

    # ─── CENTRAL ──────────────────────────────────────────────────────────

    def _build_central(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Left: video list + track list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        lbl_v = QLabel("Videos")
        lbl_v.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        left_layout.addWidget(lbl_v)

        self.video_list = VideoListWidget()
        self.video_list.files_dropped.connect(self._on_files_dropped)
        self.video_list.currentRowChanged.connect(self._on_video_select)
        left_layout.addWidget(self.video_list, 2)

        self.drop_hint = QLabel("Drag & drop videos here\nor click '+ Add Videos'")
        self.drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_hint.setStyleSheet("color: #999; font-size: 12px; padding: 20px;")
        left_layout.addWidget(self.drop_hint)

        lbl_t = QLabel("Tracks")
        lbl_t.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        left_layout.addWidget(lbl_t)

        self.track_list = QListWidget()
        self.track_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        font_t = QFont("Menlo", 10)
        font_t.setStyleHint(QFont.StyleHint.Monospace)
        self.track_list.setFont(font_t)
        left_layout.addWidget(self.track_list, 1)

        splitter.addWidget(left)

        # Right: tabs
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        # Tab: Video Player
        self.player = VideoPlayerWidget()
        self.tabs.addTab(self.player, "Video Player")

        # Tab: Trajectories
        self.mpl_traj = MplCanvas(figsize=(12, 3))
        self.tabs.addTab(self.mpl_traj, "Trajectories")

        # Tab: Velocity
        self.mpl_vel = MplCanvas(figsize=(10, 4))
        self.tabs.addTab(self.mpl_vel, "Velocity")

        # Tab: Displacement
        self.mpl_disp = MplCanvas(figsize=(10, 4))
        self.tabs.addTab(self.mpl_disp, "Displacement")

        # Tab: Summary
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Menlo", 11))
        self.tabs.addTab(self.summary_text, "Summary")

        splitter.setSizes([280, 1120])

    # ─── STATUS BAR ──────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = self.statusBar()
        self.status_label = QLabel("Ready")
        sb.addWidget(self.status_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        sb.addPermanentWidget(self.progress_bar)

    # ─── VIDEO MANAGEMENT ────────────────────────────────────────────────

    def _add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v);;All (*)")
        self._add_paths([Path(f) for f in files])

    def _on_files_dropped(self, paths):
        self._add_paths(paths)

    def _add_paths(self, paths):
        existing = {v.path for v in self.videos}
        for p in paths:
            if p not in existing:
                entry = VideoEntry(p)
                entry.magnification = self.mag_combo.currentText()
                self.videos.append(entry)
        self._refresh_video_list()

    def _remove_selected(self):
        row = self.video_list.currentRow()
        if row >= 0:
            v = self.videos.pop(row)
            if v.reader:
                v.reader.release()
            self._refresh_video_list()
            self._clear_all()

    def _refresh_video_list(self):
        self.video_list.blockSignals(True)
        current = self.video_list.currentRow()
        self.video_list.clear()
        icons = {"Pending": "○", "Running": "◉", "Done": "●", "Error": "✗"}
        for v in self.videos:
            ic = icons.get(v.status, "?")
            self.video_list.addItem(f" {ic}  {v.name}  [{v.magnification}]  {v.status}")
        if 0 <= current < len(self.videos):
            self.video_list.setCurrentRow(current)
        self.video_list.blockSignals(False)
        self.drop_hint.setVisible(len(self.videos) == 0)

    def _on_video_select(self, row):
        if 0 <= row < len(self.videos):
            v = self.videos[row]
            self.mag_combo.blockSignals(True)
            self.mag_combo.setCurrentText(v.magnification)
            self.mag_combo.blockSignals(False)
            self.player.set_video(v)
            if v.results:
                self._update_plots(v)
                self._refresh_track_list(v)
            else:
                self._clear_all()

    def _mag_changed(self, text):
        row = self.video_list.currentRow()
        if 0 <= row < len(self.videos):
            self.videos[row].magnification = text
            self._refresh_video_list()
            self.video_list.setCurrentRow(row)
            if self.videos[row].results:
                self._update_plots(self.videos[row])

    def _open_settings(self):
        SettingsDialog(self.config, self).exec()

    # ─── TRACKING ────────────────────────────────────────────────────────

    def _run_all(self):
        targets = [i for i, v in enumerate(self.videos) if v.status != "Running"]
        if targets:
            self._run_tracking(targets)

    def _run_selected(self):
        row = self.video_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "Info", "Select a video first.")
            return
        self._run_tracking([row])

    def _run_tracking(self, indices):
        if self._worker_running:
            return
        self._worker_running = True
        self._set_tracking_enabled(False)

        def worker():
            for idx in indices:
                video = self.videos[idx]
                video.status = "Running"
                self.signals.progress.emit("Starting...", 0, 100)
                try:
                    tracker = BubbleTracker(self.config)
                    video.results = tracker.process_video(
                        video.path,
                        lambda msg, c, t: self.signals.progress.emit(msg, c, t))
                    video.status = "Done"
                    self.signals.finished.emit(idx, True, "")
                except Exception as e:
                    video.status = "Error"
                    video.results = None
                    self.signals.finished.emit(idx, False, str(e))
            self.signals.all_done.emit()

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, msg, cur, total):
        self.status_label.setText(msg)
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(cur)
        self._refresh_video_list()

    def _on_video_finished(self, idx, success, err):
        self._refresh_video_list()
        if not success:
            QMessageBox.critical(self, "Error", f"{self.videos[idx].name}:\n{err}")

    def _on_all_done(self):
        self._worker_running = False
        self._set_tracking_enabled(True)
        self.status_label.setText("Tracking complete")
        self.progress_bar.setValue(0)
        for i in range(len(self.videos) - 1, -1, -1):
            if self.videos[i].results:
                self.video_list.setCurrentRow(i)
                self._update_plots(self.videos[i])
                self._refresh_track_list(self.videos[i])
                self.player.set_video(self.videos[i])
                self.tabs.setCurrentIndex(0)  # switch to player
                break

    def _set_tracking_enabled(self, enabled):
        self.act_track_all.setEnabled(enabled)
        self.act_track_sel.setEnabled(enabled)

    # ─── TRACK LIST + EDITING ────────────────────────────────────────────

    def _refresh_track_list(self, video):
        self.track_list.clear()
        if not video or not video.results:
            return
        fps = video.results["fps"]
        px = MAGNIFICATION_MAP[video.magnification]
        for i, t in enumerate(video.results["moving_tracks"]):
            pts = t["points"]
            dur = (pts[-1][0] - pts[0][0]) / fps * 1000
            disp = np.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) / px * 1000
            color_hex = TRACK_COLORS[i % len(TRACK_COLORS)]
            item = QListWidgetItem(
                f"Track {t['id']:3d} | {len(pts):4d} pts | {dur:7.0f} ms | {disp:6.0f} um")
            item.setData(Qt.ItemDataRole.UserRole, t["id"])
            item.setForeground(QColor(color_hex))
            self.track_list.addItem(item)

    def _get_current_video(self):
        row = self.video_list.currentRow()
        if 0 <= row < len(self.videos):
            return self.videos[row]
        return None

    def _delete_selected_track(self):
        v = self._get_current_video()
        if not v or not v.results:
            return
        items = self.track_list.selectedItems()
        if not items:
            QMessageBox.information(self, "Info", "Select track(s) in the track list first.")
            return
        tracker = BubbleTracker(self.config)
        for item in items:
            tid = item.data(Qt.ItemDataRole.UserRole)
            tracker.delete_track(v.results, tid)
        self._update_plots(v)
        self._refresh_track_list(v)
        self.player._refresh_display()
        self.status_label.setText(f"Deleted {len(items)} track(s)")

    def _merge_selected_tracks(self):
        v = self._get_current_video()
        if not v or not v.results:
            return
        items = self.track_list.selectedItems()
        if len(items) < 2:
            QMessageBox.information(self, "Info", "Select 2+ tracks in the track list to merge.")
            return
        ids = [item.data(Qt.ItemDataRole.UserRole) for item in items]
        tracker = BubbleTracker(self.config)
        base_id = ids[0]
        for other_id in ids[1:]:
            tracker.merge_tracks_manual(v.results, base_id, other_id)
        self._update_plots(v)
        self._refresh_track_list(v)
        self.player._refresh_display()
        self.status_label.setText(f"Merged {len(ids)} tracks into Track {base_id}")

    # ─── PLOTS ───────────────────────────────────────────────────────────

    def _update_plots(self, video):
        if not video.results:
            return
        px = MAGNIFICATION_MAP[video.magnification]

        plot_tracks_on_image(video.results, px, fig=self.mpl_traj.figure)
        self.mpl_traj.redraw()

        plot_velocity_profiles(video.results, px, fig=self.mpl_vel.figure)
        self.mpl_vel.redraw()

        plot_displacement_vs_time(video.results, px, fig=self.mpl_disp.figure)
        self.mpl_disp.redraw()

        summary = generate_summary(video.results, px, video.magnification)
        self.summary_text.setPlainText(summary)

    def _clear_all(self):
        for mpl in (self.mpl_traj, self.mpl_vel, self.mpl_disp):
            mpl.figure.clear()
            mpl.redraw()
        self.summary_text.clear()
        self.track_list.clear()

    # ─── EXPORT ──────────────────────────────────────────────────────────

    def _get_export_video(self):
        v = self._get_current_video()
        if not v or not v.results:
            QMessageBox.information(self, "No data", "Track a video first.")
            return None
        return v

    def _export(self, fmt):
        v = self._get_export_video()
        if not v:
            return
        px = MAGNIFICATION_MAP[v.magnification]
        moving = v.results["moving_tracks"]
        if fmt == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", f"{v.path.stem}_tracks.csv", "CSV (*.csv)")
            if path:
                tracks_to_csv(moving, v.results["fps"], px, path)
                self.status_label.setText(f"CSV exported: {Path(path).name}")
        elif fmt == "json":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export JSON", f"{v.path.stem}_tracks.json", "JSON (*.json)")
            if path:
                tracks_to_json(moving, v.results["fps"], px, path)
                self.status_label.setText(f"JSON exported: {Path(path).name}")

    def _export_plots(self):
        v = self._get_export_video()
        if not v:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not folder:
            return
        stem = v.path.stem
        px = MAGNIFICATION_MAP[v.magnification]
        for name, fn in [("trajectories", plot_tracks_on_image),
                         ("velocity", plot_velocity_profiles),
                         ("displacement", plot_displacement_vs_time)]:
            fig = fn(v.results, px)
            fig.savefig(os.path.join(folder, f"{stem}_{name}.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        summary = generate_summary(v.results, px, v.magnification)
        with open(os.path.join(folder, f"{stem}_summary.txt"), "w") as f:
            f.write(summary)
        self.status_label.setText(f"Exported to {folder}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MicroBubble Tracker")
    app.setStyleSheet("""
        QMainWindow { background: #f5f5f5; }
        QToolBar { spacing: 6px; padding: 4px; background: #e8e8e8;
                   border-bottom: 1px solid #ccc; }
        QToolBar QLabel { font-size: 13px; }
        QToolBar QPushButton, QToolBar QToolButton {
            padding: 4px 10px; border-radius: 4px; border: 1px solid #bbb;
            background: #fff; font-size: 12px; }
        QToolBar QPushButton:hover, QToolBar QToolButton:hover { background: #e0e0e0; }
        QTabWidget::pane { border: 1px solid #ccc; border-radius: 4px; background: white; }
        QTabBar::tab { padding: 6px 16px; margin-right: 2px; border: 1px solid #ccc;
                       border-bottom: none; border-radius: 4px 4px 0 0; background: #eee; }
        QTabBar::tab:selected { background: white; font-weight: bold; }
        QListWidget { border: 1px solid #ccc; border-radius: 4px; background: white; }
        QListWidget::item { padding: 4px; }
        QListWidget::item:selected { background: #0078d4; color: white; }
        QTextEdit { border: 1px solid #ccc; border-radius: 4px; }
        QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; }
        QProgressBar::chunk { background: #0078d4; border-radius: 3px; }
        QStatusBar { background: #e8e8e8; }
        QSlider::groove:horizontal { height: 6px; background: #ccc; border-radius: 3px; }
        QSlider::handle:horizontal { background: #0078d4; width: 14px; margin: -4px 0;
                                     border-radius: 7px; }
    """)

    window = MainWindow()
    if len(sys.argv) > 1:
        paths = [Path(a) for a in sys.argv[1:] if Path(a).exists()]
        if paths:
            window._add_paths(paths)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
