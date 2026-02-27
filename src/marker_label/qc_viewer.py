"""
PyVista-based 3D viewer for labeled C3D/CSV. Playback and rotate view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyvista as pv
except ImportError as e:
    raise ImportError('Install PyVista: pip install "marker-label[qc]" or pip install pyvista') from e


def load_data(path: str) -> tuple[np.ndarray, list[str], float]:
    """Load C3D or labeled CSV; return (points n_frames,n_markers,3), labels, rate."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".c3d":
        from .io import load_c3d
        data = load_c3d(str(path))
        return data["points"], data["labels"], data.get("rate") or 0.0
    if suffix == ".csv":
        from .inspect_quality import load_labeled_csv
        return load_labeled_csv(str(path))
    raise ValueError(f"Unsupported format: {suffix}. Use .c3d or .csv.")


def run_viewer(
    path: str,
    *,
    point_size: float = 12.0,
    playback_speed: float = 1.0,
    background: str = "white",
    segment_color: str = "darkblue",
) -> None:
    """
    Open PyVista window: 3D markers and body segments (sticks), play/stop/back/forward.

    Segments are defined in marker_label.segments: each segment is an ordered list
    of marker names; consecutive pairs are drawn as lines (see segments.SEGMENTS).

    Parameters
    ----------
    path : path to labeled .c3d or labeled .csv
    point_size : size of marker spheres (default 12)
    playback_speed : 1.0 = real-time by rate; scale for slower/faster
    background : 'white' or 'black'
    segment_color : color of segment lines (default 'darkblue')
    """
    from .segments import segment_lines_for_frame

    points, labels, rate = load_data(path)
    n_frames, n_markers, _ = points.shape
    if n_frames == 0 or n_markers == 0:
        raise ValueError("No data to display.")
    # Replace NaN with 0 for display (markers)
    pts_display = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    # Frame interval in ms for timer
    if rate > 0 and playback_speed > 0:
        dt_ms = max(10, int(1000.0 / (rate * playback_speed)))
    else:
        dt_ms = 50
    # Build initial point cloud (first frame)
    first = pts_display[0]
    cloud = pv.PolyData(first)
    cloud["valid"] = np.isfinite(points[0]).all(axis=1).astype(np.float32)
    plotter = pv.Plotter(title=str(Path(path).name))
    plotter.set_background(background)
    plotter.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=point_size,
        scalars="valid",
        cmap="coolwarm",
        show_scalar_bar=False,
    )
    # Segment lines (from segments.SEGMENTS): build for frame 0
    line_pts, line_cells = segment_lines_for_frame(points[0], labels)
    if line_pts.size > 0:
        line_mesh = pv.PolyData(line_pts, lines=line_cells)
        plotter.add_mesh(line_mesh, color=segment_color, line_width=2, name="segment_lines")
    plotter.add_text(f"Frame 0 / {n_frames}  (rate: {rate:.1f} Hz)", font_size=10, name="frame_text")

    # Shared state: current frame index, playing flag, optional slider widget
    frame_idx = [0]
    playing = [False]
    slider_widget: Any = [None]  # ref so we can set value from update_display

    def set_frame(f: int) -> None:
        f = max(0, min(n_frames - 1, f))
        frame_idx[0] = f
        cloud.points = pts_display[f]
        cloud["valid"] = np.isfinite(points[f]).all(axis=1).astype(np.float32)
        # Update segment lines for this frame (use raw points so missing = no edge)
        line_pts, line_cells = segment_lines_for_frame(points[f], labels)
        try:
            plotter.remove_actor("segment_lines")
        except Exception:
            pass
        if line_pts.size > 0:
            line_mesh = pv.PolyData(line_pts, lines=line_cells)
            plotter.add_mesh(line_mesh, color=segment_color, line_width=2, name="segment_lines")
        plotter.add_text(f"Frame {f} / {n_frames}  (rate: {rate:.1f} Hz)", font_size=10, name="frame_text")
        if slider_widget[0] is not None:
            try:
                slider_widget[0].GetRepresentation().SetValue(f)
            except Exception:
                pass
        plotter.update()

    def on_timer(_step: int) -> None:
        if not playing[0]:
            return
        next_f = (frame_idx[0] + 1) % n_frames
        set_frame(next_f)

    def on_slider(value: float) -> None:
        f = int(round(value))
        if f != frame_idx[0]:
            set_frame(f)

    def on_play(checked: bool) -> None:
        playing[0] = bool(checked)

    def on_stop(_checked: bool) -> None:
        playing[0] = False

    def on_backward(_checked: bool) -> None:
        set_frame(max(0, frame_idx[0] - 1))
        playing[0] = False

    def on_forward(_checked: bool) -> None:
        set_frame(min(n_frames - 1, frame_idx[0] + 1))
        playing[0] = False

    # Frame slider (top): scrub through frames
    rng = (0, max(0, n_frames - 1))
    if rng[1] >= rng[0]:
        sw = plotter.add_slider_widget(
            on_slider,
            rng=rng,
            value=0,
            title="Frame",
            pointa=(0.25, 0.92),
            pointb=(0.75, 0.92),
        )
        slider_widget[0] = sw

    # Buttons: Play, Stop, Backward, Forward (bottom-left row)
    btn_y = 50
    btn_size = 42
    x = 15
    plotter.add_checkbox_button_widget(
        on_play,
        value=False,
        position=(x, btn_y),
        size=btn_size,
        border_size=4,
        color_on="green",
        color_off="grey",
        background_color="white",
    )
    plotter.add_text("Play", position=(x, btn_y + btn_size + 2), font_size=9, name="label_play")
    x += btn_size + 8
    plotter.add_checkbox_button_widget(
        on_stop,
        value=False,
        position=(x, btn_y),
        size=btn_size,
        border_size=4,
        color_on="red",
        color_off="grey",
        background_color="white",
    )
    plotter.add_text("Stop", position=(x, btn_y + btn_size + 2), font_size=9, name="label_stop")
    x += btn_size + 8
    plotter.add_checkbox_button_widget(
        on_backward,
        value=False,
        position=(x, btn_y),
        size=btn_size,
        border_size=4,
        color_on="blue",
        color_off="grey",
        background_color="white",
    )
    plotter.add_text("<<", position=(x + 8, btn_y + btn_size + 2), font_size=9, name="label_back")
    x += btn_size + 8
    plotter.add_checkbox_button_widget(
        on_forward,
        value=False,
        position=(x, btn_y),
        size=btn_size,
        border_size=4,
        color_on="blue",
        color_off="grey",
        background_color="white",
    )
    plotter.add_text(">>", position=(x + 8, btn_y + btn_size + 2), font_size=9, name="label_fwd")

    plotter.add_timer_event(max_steps=10**9, duration=dt_ms, callback=on_timer)
    plotter.show()


def main() -> None:
    """CLI entry point for 3D viewer."""
    import argparse
    parser = argparse.ArgumentParser(description="3D viewer for labeled C3D/CSV with segments (PyVista).")
    parser.add_argument("file", help="Labeled .c3d or labeled .csv file")
    parser.add_argument("--point-size", type=float, default=12.0, help="Marker size (default 12)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default 1)")
    parser.add_argument("--background", choices=("white", "black"), default="white", help="Background color")
    parser.add_argument("--segment-color", default="darkblue", help="Color of segment lines (default darkblue)")
    args = parser.parse_args()
    run_viewer(
        args.file,
        point_size=args.point_size,
        playback_speed=args.speed,
        background=args.background,
        segment_color=args.segment_color,
    )
