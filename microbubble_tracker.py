#!/usr/bin/env python3
"""
MicroBubble Tracker - GUI Application

Tkinter-based GUI for high-speed microscopy microbubble path quantification.
Drag-and-drop videos, set magnification, track bubbles, export calibrated data.

Usage:
    python microbubble_tracker.py
    python microbubble_tracker.py video1.mp4 video2.mp4

Dependencies:
    pip install opencv-python numpy scipy matplotlib
    (tkinter ships with most Python installs)
    (optional: pip install tkinterdnd2   for drag-and-drop)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
from pathlib import Path

from bubble_core import (
    BubbleTracker, MAGNIFICATION_MAP, DEFAULT_CONFIG, TRACK_COLORS,
    tracks_to_csv, tracks_to_json, generate_summary,
    plot_tracks_on_image, plot_velocity_profiles, plot_displacement_vs_time,
)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class VideoEntry:
    """Represents a single video in the processing queue."""
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        self.magnification = "10x"
        self.results = None
        self.status = "Pending"


class MicrobubbleTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MicroBubble Tracker")
        self.root.geometry("1320x850")
        self.root.minsize(960, 640)

        self.videos = []
        self.selected_video = None
        self.config = {**DEFAULT_CONFIG}
        self.tracker = BubbleTracker(self.config)

        self._build_ui()
        self._setup_dnd()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Header.TLabel", font=("Helvetica", 11, "bold"))
        style.configure("Status.TLabel", font=("Helvetica", 9))
        style.configure("Accent.TButton", font=("Helvetica", 10, "bold"))

        # TOOLBAR
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="+ Add Videos", command=self._add_videos).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Remove", command=self._remove_selected).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(toolbar, text="Magnification:").pack(side="left", padx=(4, 2))
        self.mag_var = tk.StringVar(value="10x")
        mag_cb = ttk.Combobox(toolbar, textvariable=self.mag_var,
                              values=list(MAGNIFICATION_MAP.keys()), width=6, state="readonly")
        mag_cb.pack(side="left", padx=2)
        mag_cb.bind("<<ComboboxSelected>>", self._mag_changed)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="Settings", command=self._open_settings).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)

        self.run_all_btn = ttk.Button(toolbar, text="Track All", command=self._run_all, style="Accent.TButton")
        self.run_all_btn.pack(side="left", padx=4)
        self.run_sel_btn = ttk.Button(toolbar, text="Track Selected", command=self._run_selected)
        self.run_sel_btn.pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="Export CSV", command=lambda: self._export("csv")).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Export JSON", command=lambda: self._export("json")).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Export Plots", command=self._export_plots).pack(side="left", padx=2)

        # MAIN PANED LAYOUT
        main_pw = ttk.PanedWindow(self.root, orient="horizontal")
        main_pw.pack(fill="both", expand=True, padx=4, pady=4)

        # Left panel: video list
        left = ttk.LabelFrame(main_pw, text="Videos", padding=4)
        main_pw.add(left, weight=1)

        self.video_list = tk.Listbox(left, selectmode="browse", font=("Consolas", 10))
        vs = ttk.Scrollbar(left, orient="vertical", command=self.video_list.yview)
        self.video_list.configure(yscrollcommand=vs.set)
        self.video_list.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        self.video_list.bind("<<ListboxSelect>>", self._on_video_select)

        self.drop_label = ttk.Label(left,
            text="Drag & drop videos here\nor click '+ Add Videos'",
            font=("Helvetica", 10), anchor="center", justify="center", foreground="#888")

        # Right panel: results notebook
        right = ttk.Frame(main_pw, padding=4)
        main_pw.add(right, weight=4)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        # Tab: Trajectories
        self.tab_traj = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_traj, text=" Trajectories ")
        self.fig_traj = Figure(figsize=(12, 3), dpi=100)
        self.canvas_traj = FigureCanvasTkAgg(self.fig_traj, self.tab_traj)
        NavigationToolbar2Tk(self.canvas_traj, self.tab_traj)
        self.canvas_traj.get_tk_widget().pack(fill="both", expand=True)

        # Tab: Velocity
        self.tab_vel = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_vel, text=" Velocity ")
        self.fig_vel = Figure(figsize=(8, 4), dpi=100)
        self.canvas_vel = FigureCanvasTkAgg(self.fig_vel, self.tab_vel)
        NavigationToolbar2Tk(self.canvas_vel, self.tab_vel)
        self.canvas_vel.get_tk_widget().pack(fill="both", expand=True)

        # Tab: Displacement
        self.tab_disp = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_disp, text=" Displacement ")
        self.fig_disp = Figure(figsize=(8, 4), dpi=100)
        self.canvas_disp = FigureCanvasTkAgg(self.fig_disp, self.tab_disp)
        NavigationToolbar2Tk(self.canvas_disp, self.tab_disp)
        self.canvas_disp.get_tk_widget().pack(fill="both", expand=True)

        # Tab: Summary
        self.tab_sum = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sum, text=" Summary ")
        self.summary_text = tk.Text(self.tab_sum, font=("Consolas", 10), wrap="word", state="disabled")
        ss = ttk.Scrollbar(self.tab_sum, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=ss.set)
        self.summary_text.pack(side="left", fill="both", expand=True)
        ss.pack(side="right", fill="y")

        # STATUS BAR
        status = ttk.Frame(self.root, padding=(6, 2))
        status.pack(fill="x")
        self.status_lbl = ttk.Label(status, text="Ready", style="Status.TLabel")
        self.status_lbl.pack(side="left")
        self.progress = ttk.Progressbar(status, mode="determinate", length=300)
        self.progress.pack(side="right", padx=4)

        self._update_drop_label()

    def _setup_dnd(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.video_list.drop_target_register(DND_FILES)
            self.video_list.dnd_bind("<<Drop>>", self._on_drop)
        except ImportError:
            pass

    def _on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        for f in files:
            p = Path(f)
            if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"):
                self._add_video_entry(p)
        self._refresh_list()

    # VIDEO MANAGEMENT

    def _add_videos(self):
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"), ("All", "*.*")])
        for f in files:
            self._add_video_entry(Path(f))
        self._refresh_list()

    def _add_video_entry(self, path):
        for v in self.videos:
            if v.path == path:
                return
        entry = VideoEntry(path)
        entry.magnification = self.mag_var.get()
        self.videos.append(entry)

    def _remove_selected(self):
        sel = self.video_list.curselection()
        if sel:
            self.videos.pop(sel[0])
            self._refresh_list()
            self.selected_video = None
            self._clear_plots()

    def _refresh_list(self):
        self.video_list.delete(0, "end")
        icons = {"Pending": "o", "Running": "*", "Done": "+", "Error": "!"}
        for v in self.videos:
            ic = icons.get(v.status, "?")
            self.video_list.insert("end", f" [{ic}]  {v.name}  [{v.magnification}]  {v.status}")
        self._update_drop_label()

    def _update_drop_label(self):
        if not self.videos:
            self.drop_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.drop_label.place_forget()

    def _on_video_select(self, event):
        sel = self.video_list.curselection()
        if not sel:
            return
        self.selected_video = self.videos[sel[0]]
        self.mag_var.set(self.selected_video.magnification)
        if self.selected_video.results:
            self._update_plots(self.selected_video)

    def _mag_changed(self, event=None):
        sel = self.video_list.curselection()
        if sel:
            self.videos[sel[0]].magnification = self.mag_var.get()
            self._refresh_list()
            self.video_list.selection_set(sel[0])
            if self.videos[sel[0]].results:
                self._update_plots(self.videos[sel[0]])
        else:
            for v in self.videos:
                if v.status == "Pending":
                    v.magnification = self.mag_var.get()
            self._refresh_list()

    # SETTINGS DIALOG

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Tracking Settings")
        win.geometry("440x500")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        fr = ttk.Frame(win, padding=16)
        fr.pack(fill="both", expand=True)
        ttk.Label(fr, text="Tracking Parameters", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, pady=(0, 12), sticky="w")

        params = [
            ("bg_subtract_threshold", "Background subtract threshold", int),
            ("min_blob_area_px", "Min blob area (px^2)", int),
            ("max_blob_area_px", "Max blob area (px^2)", int),
            ("morph_kernel_size", "Morphology kernel size (px)", int),
            ("gaussian_blur_ksize", "Gaussian blur kernel (px)", int),
            ("max_link_distance_px", "Max linking distance (px)", int),
            ("max_frame_skip", "Max frame gap for linking", int),
            ("min_track_length", "Min track length (detections)", int),
            ("min_displacement_px", "Min displacement for 'moving' (px)", float),
        ]
        entries = {}
        for i, (key, label, dtype) in enumerate(params, 1):
            ttk.Label(fr, text=label).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(self.config[key]))
            ttk.Entry(fr, textvariable=var, width=10).grid(row=i, column=1, sticky="e", padx=(8,0), pady=2)
            entries[key] = (var, dtype)

        def apply():
            for key, (var, dtype) in entries.items():
                try:
                    self.config[key] = dtype(var.get())
                except ValueError:
                    messagebox.showerror("Invalid", f"Bad value for {key}")
                    return
            self.tracker = BubbleTracker(self.config)
            win.destroy()

        def reset():
            for key, (var, _) in entries.items():
                var.set(str(DEFAULT_CONFIG[key]))

        bf = ttk.Frame(fr)
        bf.grid(row=len(params)+1, column=0, columnspan=2, pady=(16, 0))
        ttk.Button(bf, text="Reset Defaults", command=reset).pack(side="left", padx=4)
        ttk.Button(bf, text="Apply", command=apply).pack(side="left", padx=4)

    # TRACKING EXECUTION

    def _run_all(self):
        targets = [v for v in self.videos if v.status != "Running"]
        if targets:
            self._run_tracking(targets)

    def _run_selected(self):
        sel = self.video_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select a video first.")
            return
        self._run_tracking([self.videos[sel[0]]])

    def _run_tracking(self, targets):
        self.run_all_btn.configure(state="disabled")
        self.run_sel_btn.configure(state="disabled")

        def worker():
            for video in targets:
                video.status = "Running"
                self.root.after(0, self._refresh_list)

                def pcb(msg, cur, total):
                    self.root.after(0, lambda m=msg, c=cur, t=total: self._prog(m, c, t))

                try:
                    tracker = BubbleTracker(self.config)
                    video.results = tracker.process_video(video.path, pcb)
                    video.status = "Done"
                except Exception as e:
                    video.status = "Error"
                    video.results = None
                    self.root.after(0, lambda err=str(e): messagebox.showerror("Error", err))
                self.root.after(0, self._refresh_list)

            self.root.after(0, self._tracking_done, targets)

        threading.Thread(target=worker, daemon=True).start()

    def _tracking_done(self, targets):
        self.run_all_btn.configure(state="normal")
        self.run_sel_btn.configure(state="normal")
        self.status_lbl.configure(text="Tracking complete")
        self.progress["value"] = 0

        done = [v for v in targets if v.results]
        if done:
            v = done[-1]
            idx = self.videos.index(v)
            self.video_list.selection_clear(0, "end")
            self.video_list.selection_set(idx)
            self.selected_video = v
            self._update_plots(v)

    def _prog(self, msg, cur, total):
        self.status_lbl.configure(text=msg)
        if total > 0:
            self.progress["maximum"] = total
            self.progress["value"] = cur

    # PLOT UPDATES

    def _update_plots(self, video):
        if not video.results:
            return
        px = MAGNIFICATION_MAP[video.magnification]

        plot_tracks_on_image(video.results, px, fig=self.fig_traj)
        self.canvas_traj.draw()

        plot_velocity_profiles(video.results, px, fig=self.fig_vel)
        self.canvas_vel.draw()

        plot_displacement_vs_time(video.results, px, fig=self.fig_disp)
        self.canvas_disp.draw()

        summary = generate_summary(video.results, px, video.magnification)
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", summary)
        self.summary_text.configure(state="disabled")

    def _clear_plots(self):
        for fig, canvas in [(self.fig_traj, self.canvas_traj),
                            (self.fig_vel, self.canvas_vel),
                            (self.fig_disp, self.canvas_disp)]:
            fig.clear()
            canvas.draw()
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.configure(state="disabled")

    # EXPORT

    def _export(self, fmt):
        if not self.selected_video or not self.selected_video.results:
            messagebox.showinfo("No data", "Track a video first.")
            return
        v = self.selected_video
        px = MAGNIFICATION_MAP[v.magnification]
        moving = v.results["moving_tracks"]

        if fmt == "csv":
            path = filedialog.asksaveasfilename(
                defaultextension=".csv", initialfile=f"{v.path.stem}_tracks.csv",
                filetypes=[("CSV", "*.csv")])
            if path:
                tracks_to_csv(moving, v.results["fps"], px, path)
                self.status_lbl.configure(text=f"CSV exported: {Path(path).name}")
        elif fmt == "json":
            path = filedialog.asksaveasfilename(
                defaultextension=".json", initialfile=f"{v.path.stem}_tracks.json",
                filetypes=[("JSON", "*.json")])
            if path:
                tracks_to_json(moving, v.results["fps"], px, path)
                self.status_lbl.configure(text=f"JSON exported: {Path(path).name}")

    def _export_plots(self):
        if not self.selected_video or not self.selected_video.results:
            messagebox.showinfo("No data", "Track a video first.")
            return
        folder = filedialog.askdirectory(title="Select export folder")
        if not folder:
            return
        v = self.selected_video
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

        self.status_lbl.configure(text=f"Plots exported to {folder}")


def main():
    root = tk.Tk()
    app = MicrobubbleTrackerApp(root)

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = Path(arg)
            if p.exists():
                app._add_video_entry(p)
        app._refresh_list()

    root.mainloop()


if __name__ == "__main__":
    main()
