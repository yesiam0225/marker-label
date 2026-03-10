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


def load_data(path: str, scale_factor: float = 1.0) -> tuple[np.ndarray, list[str], float]:
    """Load C3D or labeled CSV; return (points n_frames,n_markers,3), labels, rate. Optionally scale coordinates by scale_factor (e.g. 1000 if file is in m)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".c3d":
        from .io import load_c3d
        data = load_c3d(str(path), scale_factor=scale_factor)
        return data["points"], data["labels"], data.get("rate") or 0.0
    if suffix == ".csv":
        from .inspect_quality import load_labeled_csv
        points, labels, rate = load_labeled_csv(str(path))
        if scale_factor != 1.0:
            points = points * scale_factor
        return points, labels, rate
    raise ValueError(f"Unsupported format: {suffix}. Use .c3d or .csv.")


def run_viewer(
    path: str,
    *,
    point_size: float = 12.0,
    playback_speed: float = 1.0,
    background: str = "white",
    segment_color: str = "darkblue",
    scale_factor: float = 1.0,
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
    scale_factor : multiply coordinates by this (e.g. 1000 if file is in meters and you want to display as mm).
    """
    from .segments import segment_lines_for_frame_by_segment, SEGMENT_COLORS, SEGMENTS

    points, labels, rate = load_data(path, scale_factor=scale_factor)
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
    # Segment lines per segment (different color per segment)
    def add_segment_meshes(pts: np.ndarray) -> None:
        for seg_name, line_pts, line_cells in segment_lines_for_frame_by_segment(pts, labels):
            if line_pts.size == 0:
                continue
            color = SEGMENT_COLORS.get(seg_name, segment_color)
            mesh = pv.PolyData(line_pts, lines=line_cells)
            plotter.add_mesh(mesh, color=color, line_width=2, name=f"segment_{seg_name}")

    def remove_segment_meshes() -> None:
        for seg_name in SEGMENTS:
            try:
                plotter.remove_actor(f"segment_{seg_name}")
            except Exception:
                pass

    add_segment_meshes(points[0])
    # Obstacle labels: find OBSTACLE_L / OBSTACLE_R indices; display as OBS1, OBS2
    label_stripped = [str(lab).strip() for lab in labels]
    obstacle_indices = []
    obstacle_display_names = []  # OBS1, OBS2 for viewer
    for name in ("OBSTACLE_L", "OBSTACLE_R"):
        try:
            i = next(ix for ix, s in enumerate(label_stripped) if str(s).strip().upper() == name.upper())
            obstacle_indices.append(i)
            obstacle_display_names.append("OBS1" if name == "OBSTACLE_L" else "OBS2")
        except StopIteration:
            pass
    # Head labels: LFHD, RFHD, LBHD, RBHD
    head_names_order = ("LFHD", "RFHD", "LBHD", "RBHD")
    head_indices = []
    head_names = []
    for name in head_names_order:
        try:
            i = next(ix for ix, s in enumerate(label_stripped) if str(s).strip().upper() == name.upper())
            head_indices.append(i)
            head_names.append(labels[i].strip() if i < len(labels) else name)
        except StopIteration:
            pass
    # C7 and shoulders: C7, LSHO, RSHO
    c7_shoulder_order = ("C7", "LSHO", "RSHO")
    c7_shoulder_indices = []
    c7_shoulder_names = []
    for name in c7_shoulder_order:
        try:
            i = next(ix for ix, s in enumerate(label_stripped) if str(s).strip().upper() == name.upper())
            c7_shoulder_indices.append(i)
            c7_shoulder_names.append(labels[i].strip() if i < len(labels) else name)
        except StopIteration:
            pass
    # CLAV and RBAK
    clav_rbak_order = ("CLAV", "RBAK")
    clav_rbak_indices = []
    clav_rbak_names = []
    for name in clav_rbak_order:
        try:
            i = next(ix for ix, s in enumerate(label_stripped) if str(s).strip().upper() == name.upper())
            clav_rbak_indices.append(i)
            clav_rbak_names.append(labels[i].strip() if i < len(labels) else name)
        except StopIteration:
            pass
    if background == "white":
        obs_text_color, obs_shape_color = "black", "lightgrey"
        head_text_color, head_shape_color = "darkblue", "lavender"
        c7_shoulder_text_color, c7_shoulder_shape_color = "darkgreen", "honeydew"
        clav_rbak_text_color, clav_rbak_shape_color = "saddlebrown", "wheat"
    else:
        obs_text_color, obs_shape_color = "white", "dimgrey"
        head_text_color, head_shape_color = "lightblue", "dimgrey"
        c7_shoulder_text_color, c7_shoulder_shape_color = "lightgreen", "dimgrey"
        clav_rbak_text_color, clav_rbak_shape_color = "wheat", "dimgrey"

    def add_obstacle_labels(f: int) -> None:
        try:
            plotter.remove_actor("obstacle_labels")
        except Exception:
            pass
        if not obstacle_indices:
            return
        pts_f = pts_display[f]
        obs_pts = pts_f[obstacle_indices]
        obs_cloud = pv.PolyData(obs_pts)
        obs_cloud["names"] = np.array(obstacle_display_names, dtype="U")
        plotter.add_point_labels(
            obs_cloud,
            "names",
            font_size=16,
            show_points=False,
            text_color=obs_text_color,
            shape_color=obs_shape_color,
            shape_opacity=0.85,
            always_visible=True,
            name="obstacle_labels",
        )

    def add_head_labels(f: int) -> None:
        try:
            plotter.remove_actor("head_labels")
        except Exception:
            pass
        if not head_indices:
            return
        pts_f = pts_display[f]
        head_pts = pts_f[head_indices]
        head_cloud = pv.PolyData(head_pts)
        head_cloud["names"] = np.array(head_names, dtype="U")
        plotter.add_point_labels(
            head_cloud,
            "names",
            font_size=14,
            show_points=False,
            text_color=head_text_color,
            shape_color=head_shape_color,
            shape_opacity=0.85,
            always_visible=True,
            name="head_labels",
        )

    def add_c7_shoulder_labels(f: int) -> None:
        try:
            plotter.remove_actor("c7_shoulder_labels")
        except Exception:
            pass
        if not c7_shoulder_indices:
            return
        pts_f = pts_display[f]
        c7_pts = pts_f[c7_shoulder_indices]
        c7_cloud = pv.PolyData(c7_pts)
        c7_cloud["names"] = np.array(c7_shoulder_names, dtype="U")
        plotter.add_point_labels(
            c7_cloud,
            "names",
            font_size=14,
            show_points=False,
            text_color=c7_shoulder_text_color,
            shape_color=c7_shoulder_shape_color,
            shape_opacity=0.85,
            always_visible=True,
            name="c7_shoulder_labels",
        )

    def add_clav_rbak_labels(f: int) -> None:
        try:
            plotter.remove_actor("clav_rbak_labels")
        except Exception:
            pass
        if not clav_rbak_indices:
            return
        pts_f = pts_display[f]
        clav_pts = pts_f[clav_rbak_indices]
        clav_cloud = pv.PolyData(clav_pts)
        clav_cloud["names"] = np.array(clav_rbak_names, dtype="U")
        plotter.add_point_labels(
            clav_cloud,
            "names",
            font_size=14,
            show_points=False,
            text_color=clav_rbak_text_color,
            shape_color=clav_rbak_shape_color,
            shape_opacity=0.85,
            always_visible=True,
            name="clav_rbak_labels",
        )

    add_obstacle_labels(0)
    add_head_labels(0)
    add_c7_shoulder_labels(0)
    add_clav_rbak_labels(0)
    plotter.add_text(f"Frame 0 / {n_frames}  (rate: {rate:.1f} Hz)", font_size=12, name="frame_text")

    # Shared state: current frame index, playing flag, optional slider widget
    frame_idx = [0]
    playing = [False]
    slider_widget: Any = [None]  # ref so we can set value from update_display

    def set_frame(f: int) -> None:
        f = max(0, min(n_frames - 1, f))
        frame_idx[0] = f
        cloud.points = pts_display[f]
        cloud["valid"] = np.isfinite(points[f]).all(axis=1).astype(np.float32)
        # Update segment lines for this frame (per-segment colors)
        remove_segment_meshes()
        add_segment_meshes(points[f])
        add_obstacle_labels(f)
        add_head_labels(f)
        add_c7_shoulder_labels(f)
        add_clav_rbak_labels(f)
        plotter.add_text(f"Frame {f} / {n_frames}  (rate: {rate:.1f} Hz)", font_size=12, name="frame_text")
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


def run_compare_viewer(
    static_path: str,
    dynamic_path: str,
    *,
    dynamic_frame: int = 0,
    point_size: float = 10.0,
    background: str = "white",
    static_facing_axis: str | None = None,
    dynamic_facing_axis: str | None = None,
) -> None:
    """
    Show static template (labeled) and dynamic trial (one frame, scrubbable) in the same 3D scene.

    Use this to see how template positions relate to dynamic points (orientation, translation)
    and to reason about why matching might fail. Static = template mean positions (one per label).
    Dynamic = one frame at a time; scrub or play to change frame.

    Parameters
    ----------
    static_path : path to labeled static C3D
    dynamic_path : path to dynamic C3D (can be unlabeled)
    dynamic_frame : initial dynamic frame to show (0)
    point_size : marker size
    background : 'white' or 'black'
    static_facing_axis, dynamic_facing_axis : if both set, rotate dynamic to align with static (same as pipeline)
    """
    from .io import load_c3d
    from .body_labeling import build_template_from_static

    static = load_c3d(static_path)
    dynamic = load_c3d(dynamic_path)
    points_s = static["points"]
    labels_s = static["labels"]
    points_d = dynamic["points"]
    rate = dynamic.get("rate") or 0.0
    n_frames_d, n_pts_d, _ = points_d.shape
    if not labels_s or points_s.size == 0:
        raise ValueError("Static C3D must have labels and points.")
    # Template from static (lab frame)
    template, _, _ = build_template_from_static(points_s, labels_s, use_pelvis_frame=False)
    labels_list = [l for l in labels_s if np.isfinite(template.get(l, np.nan)).all()]
    if not labels_list:
        raise ValueError("No valid template positions from static.")
    template_pts = np.array([template[l] for l in labels_list])
    # Dynamic: optional rotation to align with static
    pts_d = points_d.copy()
    if static_facing_axis and dynamic_facing_axis:
        from .pipeline import rotation_for_facing_axes
        R = rotation_for_facing_axes(static_facing_axis, dynamic_facing_axis)
        if R is not None:
            pts_d = pts_d @ R.T
    pts_display = np.nan_to_num(pts_d, nan=0.0, posinf=0.0, neginf=0.0)

    plotter = pv.Plotter(title="Static (template) vs Dynamic")
    plotter.set_background(background)
    # Static template: green, with labels
    cloud_static = pv.PolyData(template_pts)
    cloud_static["labels"] = np.array(labels_list, dtype="U")
    plotter.add_mesh(cloud_static, color="green", point_size=point_size, render_points_as_spheres=True)
    _text_color = "black" if background == "white" else "white"
    _shape_color = "lightgrey" if background == "white" else "dimgrey"
    plotter.add_point_labels(
        cloud_static, "labels", font_size=8, show_points=False,
        text_color=_text_color, shape_color=_shape_color, name="static_labels",
    )
    # Dynamic: red, one frame
    frame_idx = [min(max(0, dynamic_frame), n_frames_d - 1) if n_frames_d else 0]
    cloud_dynamic = pv.PolyData(pts_display[frame_idx[0]])
    plotter.add_mesh(cloud_dynamic, color="red", point_size=point_size, render_points_as_spheres=True, name="dynamic_cloud")
    plotter.add_text(
        f"Static (green, labeled) vs Dynamic frame {frame_idx[0]}/{n_frames_d} (red)",
        font_size=9,
        name="frame_text",
    )
    slider_widget: Any = [None]

    def set_frame(f: int) -> None:
        f = max(0, min(n_frames_d - 1, f))
        frame_idx[0] = f
        cloud_dynamic.points = pts_display[f]
        plotter.add_text(
            f"Static (green, labeled) vs Dynamic frame {f}/{n_frames_d} (red)",
            font_size=9,
            name="frame_text",
        )
        if slider_widget[0] is not None:
            try:
                slider_widget[0].GetRepresentation().SetValue(f)
            except Exception:
                pass
        plotter.update()

    if n_frames_d > 1:
        rng = (0, n_frames_d - 1)
        sw = plotter.add_slider_widget(
            lambda v: set_frame(int(round(v))),
            rng=rng,
            value=float(frame_idx[0]),
            title="Dynamic frame",
            pointa=(0.25, 0.92),
            pointb=(0.75, 0.92),
        )
        slider_widget[0] = sw
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
    parser.add_argument("--scale", type=float, default=1.0, metavar="FACTOR", help="Multiply coordinates by FACTOR (e.g. 1000 if file is in meters; default 1)")
    args = parser.parse_args()
    run_viewer(
        args.file,
        point_size=args.point_size,
        playback_speed=args.speed,
        background=args.background,
        segment_color=args.segment_color,
        scale_factor=args.scale,
    )


def main_compare() -> None:
    """CLI: compare static (template) and dynamic in one 3D scene."""
    import argparse
    parser = argparse.ArgumentParser(
        description="View static template (green, labeled) and dynamic trial (red) in same 3D scene to inspect alignment.",
    )
    parser.add_argument("static", help="Path to labeled static C3D")
    parser.add_argument("dynamic", help="Path to dynamic C3D")
    parser.add_argument("--frame", type=int, default=0, help="Initial dynamic frame (default 0)")
    parser.add_argument("--point-size", type=float, default=10.0, help="Marker size (default 10)")
    parser.add_argument("--background", choices=("white", "black"), default="white", help="Background color")
    parser.add_argument("--static-facing", choices=("x", "-x", "y", "-y"), default=None, help="Lab axis subject faces in static")
    parser.add_argument("--dynamic-facing", choices=("x", "-x", "y", "-y"), default=None, help="Lab axis subject faces in dynamic")
    args = parser.parse_args()
    run_compare_viewer(
        args.static,
        args.dynamic,
        dynamic_frame=args.frame,
        point_size=args.point_size,
        background=args.background,
        static_facing_axis=args.static_facing,
        dynamic_facing_axis=args.dynamic_facing,
    )
