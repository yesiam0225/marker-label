"""
PyVista-based 3D viewer for labeled C3D/CSV. Playback and rotate view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import pyvista as pv
except ImportError as e:
    raise ImportError('Install PyVista: pip install "marker-label[qc]" or pip install pyvista') from e


def _frame_points_y_clipped(
    pts_f: np.ndarray,
    y_min_mm: float | None,
    y_max_mm: float | None,
) -> np.ndarray:
    """
    Return a copy of one frame (n_markers, 3) with markers outside the lab-Y band set to NaN.

    Used so spheres and segment sticks omit out-of-range markers without shifting indices.
    """
    p = pts_f.copy()
    if y_min_mm is not None:
        bad = np.isfinite(p[:, 1]) & (p[:, 1] < float(y_min_mm))
        p[bad] = np.nan
    if y_max_mm is not None:
        bad = np.isfinite(p[:, 1]) & (p[:, 1] > float(y_max_mm))
        p[bad] = np.nan
    return p


def _finite_rows(pts: np.ndarray) -> np.ndarray:
    """Boolean mask (n_markers,) — row has finite x, y, z."""
    return np.isfinite(pts).all(axis=1)


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
    label_font_size: float = 18.0,
    y_clip_min_mm: float | None = None,
    y_clip_max_mm: float | None = None,
) -> None:
    """
    Open PyVista window: 3D markers and body segments (sticks), play/stop/back/forward.
    Zoom: use the "Zoom +" / "Zoom -" buttons or keys '+'/=' (zoom in) and '-' (zoom out).

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
    label_font_size : font size for point labels (default 18). Use --font-size in CLI to override.
    y_clip_min_mm : if set, hide markers (and segment endpoints) with lab Y below this (mm, after scale_factor).
    y_clip_max_mm : if set, hide markers with lab Y above this (mm, after scale_factor).
    """
    from .segments import segment_lines_for_frame_by_segment, SEGMENT_COLORS, SEGMENTS

    points, labels, rate = load_data(path, scale_factor=scale_factor)
    n_frames, n_markers, _ = points.shape
    if n_frames == 0 or n_markers == 0:
        raise ValueError("No data to display.")
    # Optional best-frame file (e.g. written by export_pre_clav_for_viewer)
    best_frame_1based: int | None = None
    bestframe_path = Path(path).with_suffix(Path(path).suffix + ".bestframe")
    if bestframe_path.exists():
        try:
            best_frame_1based = int(bestframe_path.read_text().strip().split()[0])
        except (ValueError, IndexError, OSError):
            pass
    use_y_clip = y_clip_min_mm is not None or y_clip_max_mm is not None
    # Full array for NaN handling when not clipping; Y-clip path uses NaN to drop spheres/segments
    pts_display = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    def frame_for_segments(f: int) -> np.ndarray:
        raw = points[f]
        if use_y_clip:
            return _frame_points_y_clipped(raw, y_clip_min_mm, y_clip_max_mm)
        return raw

    def frame_for_cloud(f: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions (n,3), valid scalar (n,)) for marker cloud."""
        raw = points[f]
        if use_y_clip:
            clipped = _frame_points_y_clipped(raw, y_clip_min_mm, y_clip_max_mm)
            vis = _finite_rows(raw) & _finite_rows(clipped)
            pos = np.where(vis[:, np.newaxis], np.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            return pos, vis.astype(np.float32)
        pos = pts_display[f]
        valid = np.isfinite(points[f]).all(axis=1).astype(np.float32)
        return pos, valid

    # Frame interval in ms for timer
    if rate > 0 and playback_speed > 0:
        dt_ms = max(10, int(1000.0 / (rate * playback_speed)))
    else:
        dt_ms = 50
    # Build initial point cloud (first frame)
    first_pos, first_valid = frame_for_cloud(0)
    cloud = pv.PolyData(first_pos)
    cloud["valid"] = first_valid
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

    add_segment_meshes(frame_for_segments(0))
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
    # STRN, T10, LUPA, RUPA, LELB, RELB
    strn_t10_arm_order = ("STRN", "T10", "LUPA", "RUPA", "LELB", "RELB")
    strn_t10_arm_indices = []
    strn_t10_arm_names = []
    for name in strn_t10_arm_order:
        try:
            i = next(ix for ix, s in enumerate(label_stripped) if str(s).strip().upper() == name.upper())
            strn_t10_arm_indices.append(i)
            strn_t10_arm_names.append(labels[i].strip() if i < len(labels) else name)
        except StopIteration:
            pass
    # Pelvis + arm/hand: LASI, RASI, LPSI, RPSI, LFRM, LWRA, LWRB, LFIN, RFRM, RWRA, RWRB, RFIN
    pelvis_arm12_order = ("LASI", "RASI", "LPSI", "RPSI", "LFRM", "LWRA", "LWRB", "LFIN", "RFRM", "RWRA", "RWRB", "RFIN")
    pelvis_arm12_indices = []
    pelvis_arm12_names = []
    for name in pelvis_arm12_order:
        try:
            i = next(ix for ix, s in enumerate(label_stripped) if str(s).strip().upper() == name.upper())
            pelvis_arm12_indices.append(i)
            pelvis_arm12_names.append(labels[i].strip() if i < len(labels) else name)
        except StopIteration:
            pass
    leg_foot12_order = ("LTHI", "RTHI", "LKNE", "RKNE", "LTIB", "RTIB", "LANK", "RANK", "LHEE", "LTOE", "RHEE", "RTOE")
    leg_foot12_indices = []
    leg_foot12_names = []
    for name in leg_foot12_order:
        try:
            i = next(ix for ix, s in enumerate(label_stripped) if str(s).strip().upper() == name.upper())
            leg_foot12_indices.append(i)
            leg_foot12_names.append(labels[i].strip() if i < len(labels) else name)
        except StopIteration:
            pass
    # Points not in any anatomical group (e.g. unlabeled / pre-CLAV indices "0", "1", "2", ...)
    known_indices = set(obstacle_indices + head_indices + c7_shoulder_indices + clav_rbak_indices
                        + strn_t10_arm_indices + pelvis_arm12_indices + leg_foot12_indices)
    other_indices = [i for i in range(len(labels)) if i not in known_indices]
    other_names = [labels[i].strip() if i < len(labels) else str(i) for i in other_indices]
    if background == "white":
        obs_text_color, obs_shape_color = "black", "lightgrey"
        head_text_color, head_shape_color = "darkblue", "lavender"
        c7_shoulder_text_color, c7_shoulder_shape_color = "darkgreen", "honeydew"
        clav_rbak_text_color, clav_rbak_shape_color = "saddlebrown", "wheat"
        strn_t10_arm_text_color, strn_t10_arm_shape_color = "darkcyan", "azure"
        pelvis_arm12_text_color, pelvis_arm12_shape_color = "purple", "plum"
        leg_foot12_text_color, leg_foot12_shape_color = "darkgreen", "honeydew"
    else:
        obs_text_color, obs_shape_color = "white", "dimgrey"
        head_text_color, head_shape_color = "lightblue", "dimgrey"
        c7_shoulder_text_color, c7_shoulder_shape_color = "lightgreen", "dimgrey"
        clav_rbak_text_color, clav_rbak_shape_color = "wheat", "dimgrey"
        strn_t10_arm_text_color, strn_t10_arm_shape_color = "cyan", "dimgrey"
        pelvis_arm12_text_color, pelvis_arm12_shape_color = "magenta", "dimgrey"
        leg_foot12_text_color, leg_foot12_shape_color = "lightgreen", "dimgrey"

    def _label_frame_points(f: int) -> np.ndarray:
        if use_y_clip:
            return _frame_points_y_clipped(points[f], y_clip_min_mm, y_clip_max_mm)
        return pts_display[f]

    def _finite_label_cloud(pts_block: np.ndarray, names_seq: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Drop rows with non-finite coordinates (e.g. Y-clipped or missing)."""
        names_arr = np.asarray(list(names_seq), dtype="U")
        m = np.isfinite(pts_block).all(axis=1)
        if not np.any(m):
            return np.empty((0, 3), dtype=np.float64), np.array([], dtype="U")
        return pts_block[m], names_arr[m]

    def add_obstacle_labels(f: int) -> None:
        try:
            plotter.remove_actor("obstacle_labels")
        except Exception:
            pass
        if not obstacle_indices:
            return
        pts_f = _label_frame_points(f)
        obs_pts, obs_nm = _finite_label_cloud(pts_f[obstacle_indices], obstacle_display_names)
        if obs_pts.shape[0] == 0:
            return
        obs_cloud = pv.PolyData(obs_pts)
        obs_cloud["names"] = obs_nm
        plotter.add_point_labels(
            obs_cloud,
            "names",
            font_size=int(label_font_size + 2),
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
        pts_f = _label_frame_points(f)
        head_pts, head_nm = _finite_label_cloud(pts_f[head_indices], head_names)
        if head_pts.shape[0] == 0:
            return
        head_cloud = pv.PolyData(head_pts)
        head_cloud["names"] = head_nm
        plotter.add_point_labels(
            head_cloud,
            "names",
            font_size=int(label_font_size),
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
        pts_f = _label_frame_points(f)
        c7_pts, c7_nm = _finite_label_cloud(pts_f[c7_shoulder_indices], c7_shoulder_names)
        if c7_pts.shape[0] == 0:
            return
        c7_cloud = pv.PolyData(c7_pts)
        c7_cloud["names"] = c7_nm
        plotter.add_point_labels(
            c7_cloud,
            "names",
            font_size=int(label_font_size),
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
        pts_f = _label_frame_points(f)
        clav_pts, clav_nm = _finite_label_cloud(pts_f[clav_rbak_indices], clav_rbak_names)
        if clav_pts.shape[0] == 0:
            return
        clav_cloud = pv.PolyData(clav_pts)
        clav_cloud["names"] = clav_nm
        plotter.add_point_labels(
            clav_cloud,
            "names",
            font_size=int(label_font_size),
            show_points=False,
            text_color=clav_rbak_text_color,
            shape_color=clav_rbak_shape_color,
            shape_opacity=0.85,
            always_visible=True,
            name="clav_rbak_labels",
        )

    def add_strn_t10_arm_labels(f: int) -> None:
        try:
            plotter.remove_actor("strn_t10_arm_labels")
        except Exception:
            pass
        if not strn_t10_arm_indices:
            return
        pts_f = _label_frame_points(f)
        arm_pts, arm_nm = _finite_label_cloud(pts_f[strn_t10_arm_indices], strn_t10_arm_names)
        if arm_pts.shape[0] == 0:
            return
        arm_cloud = pv.PolyData(arm_pts)
        arm_cloud["names"] = arm_nm
        plotter.add_point_labels(
            arm_cloud,
            "names",
            font_size=int(label_font_size),
            show_points=False,
            text_color=strn_t10_arm_text_color,
            shape_color=strn_t10_arm_shape_color,
            shape_opacity=0.85,
            always_visible=True,
            name="strn_t10_arm_labels",
        )

    def add_pelvis_arm12_labels(f: int) -> None:
        try:
            plotter.remove_actor("pelvis_arm12_labels")
        except Exception:
            pass
        if not pelvis_arm12_indices:
            return
        pts_f = _label_frame_points(f)
        pa_pts, pa_nm = _finite_label_cloud(pts_f[pelvis_arm12_indices], pelvis_arm12_names)
        if pa_pts.shape[0] == 0:
            return
        pa_cloud = pv.PolyData(pa_pts)
        pa_cloud["names"] = pa_nm
        plotter.add_point_labels(
            pa_cloud,
            "names",
            font_size=int(label_font_size - 2),
            show_points=False,
            text_color=pelvis_arm12_text_color,
            shape_color=pelvis_arm12_shape_color,
            shape_opacity=0.85,
            always_visible=True,
            name="pelvis_arm12_labels",
        )

    def add_leg_foot12_labels(f: int) -> None:
        try:
            plotter.remove_actor("leg_foot12_labels")
        except Exception:
            pass
        if not leg_foot12_indices:
            return
        pts_f = _label_frame_points(f)
        lf_pts, lf_nm = _finite_label_cloud(pts_f[leg_foot12_indices], leg_foot12_names)
        if lf_pts.shape[0] == 0:
            return
        lf_cloud = pv.PolyData(lf_pts)
        lf_cloud["names"] = lf_nm
        plotter.add_point_labels(
            lf_cloud,
            "names",
            font_size=int(label_font_size - 3),
            show_points=False,
            text_color=leg_foot12_text_color,
            shape_color=leg_foot12_shape_color,
            shape_opacity=0.85,
            always_visible=True,
            name="leg_foot12_labels",
        )

    def add_other_labels(f: int) -> None:
        try:
            plotter.remove_actor("other_labels")
        except Exception:
            pass
        if not other_indices:
            return
        pts_f = _label_frame_points(f)
        other_pts, other_nm = _finite_label_cloud(pts_f[other_indices], other_names)
        if other_pts.shape[0] == 0:
            return
        other_cloud = pv.PolyData(other_pts)
        other_cloud["names"] = other_nm
        _text = "grey" if background == "white" else "lightgrey"
        _shape = "lightgrey" if background == "white" else "dimgrey"
        plotter.add_point_labels(
            other_cloud,
            "names",
            font_size=int(label_font_size - 4),
            show_points=False,
            text_color=_text,
            shape_color=_shape,
            shape_opacity=0.85,
            always_visible=True,
            name="other_labels",
        )

    def frame_text_str(f: int) -> str:
        base = f"Frame {f} / {n_frames}  (rate: {rate:.1f} Hz)"
        if best_frame_1based is not None:
            base += f"  Best frame: {best_frame_1based}"
        if use_y_clip:
            parts: list[str] = []
            if y_clip_min_mm is not None:
                parts.append(f"Y≥{y_clip_min_mm:g} mm")
            if y_clip_max_mm is not None:
                parts.append(f"Y≤{y_clip_max_mm:g} mm")
            base += "  [" + ", ".join(parts) + "]"
        return base

    add_obstacle_labels(0)
    add_head_labels(0)
    add_c7_shoulder_labels(0)
    add_clav_rbak_labels(0)
    add_strn_t10_arm_labels(0)
    add_pelvis_arm12_labels(0)
    add_leg_foot12_labels(0)
    add_other_labels(0)
    plotter.add_text(frame_text_str(0), font_size=12, name="frame_text")

    # Shared state: current frame index, playing flag, optional slider widget
    frame_idx = [0]
    playing = [False]
    slider_widget: Any = [None]  # ref so we can set value from update_display

    def set_frame(f: int) -> None:
        f = max(0, min(n_frames - 1, f))
        frame_idx[0] = f
        pos, valid = frame_for_cloud(f)
        cloud.points = pos
        cloud["valid"] = valid
        # Update segment lines for this frame (per-segment colors)
        remove_segment_meshes()
        add_segment_meshes(frame_for_segments(f))
        add_obstacle_labels(f)
        add_head_labels(f)
        add_c7_shoulder_labels(f)
        add_clav_rbak_labels(f)
        add_strn_t10_arm_labels(f)
        add_pelvis_arm12_labels(f)
        add_leg_foot12_labels(f)
        add_other_labels(f)
        plotter.add_text(frame_text_str(f), font_size=12, name="frame_text")
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

    _ZOOM_FACTOR = 1.25

    def zoom_in() -> None:
        plotter.zoom_camera(_ZOOM_FACTOR)
        plotter.update()

    def zoom_out() -> None:
        plotter.zoom_camera(1.0 / _ZOOM_FACTOR)
        plotter.update()

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
    x += btn_size + 8
    # Zoom In / Zoom Out buttons
    plotter.add_checkbox_button_widget(
        lambda _: zoom_in(),
        value=False,
        position=(x, btn_y),
        size=btn_size,
        border_size=4,
        color_on="blue",
        color_off="grey",
        background_color="white",
    )
    plotter.add_text("Zoom +", position=(x, btn_y + btn_size + 2), font_size=9, name="label_zoom_in")
    x += btn_size + 8
    plotter.add_checkbox_button_widget(
        lambda _: zoom_out(),
        value=False,
        position=(x, btn_y),
        size=btn_size,
        border_size=4,
        color_on="blue",
        color_off="grey",
        background_color="white",
    )
    plotter.add_text("Zoom -", position=(x, btn_y + btn_size + 2), font_size=9, name="label_zoom_out")

    # Keyboard: +/= zoom in, - zoom out
    plotter.add_key_event("=", zoom_in)
    plotter.add_key_event("+", zoom_in)
    plotter.add_key_event("-", zoom_out)

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
    # Zoom keys for compare viewer
    def _zoom_in() -> None:
        plotter.zoom_camera(1.25)
        plotter.update()
    def _zoom_out() -> None:
        plotter.zoom_camera(1.0 / 1.25)
        plotter.update()
    plotter.add_key_event("equal", _zoom_in)
    plotter.add_key_event("plus", _zoom_in)
    plotter.add_key_event("minus", _zoom_out)
    plotter.show()


def main() -> None:
    """CLI entry point for 3D viewer."""
    import argparse
    parser = argparse.ArgumentParser(description="3D viewer for labeled C3D/CSV with segments (PyVista).")
    parser.add_argument("file", help="Labeled .c3d or labeled .csv file")
    parser.add_argument("--point-size", type=float, default=12.0, help="Marker size (default 12)")
    parser.add_argument("--font-size", type=float, default=18.0, metavar="SIZE", help="Point label font size (default 18)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default 1)")
    parser.add_argument("--background", choices=("white", "black"), default="white", help="Background color")
    parser.add_argument("--segment-color", default="darkblue", help="Color of segment lines (default darkblue)")
    parser.add_argument("--scale", type=float, default=1.0, metavar="FACTOR", help="Multiply coordinates by FACTOR (e.g. 1000 if file is in meters; default 1)")
    parser.add_argument(
        "--y-clip-min",
        type=float,
        default=None,
        metavar="MM",
        help="Hide markers and segment endpoints with lab Y below this (mm, after --scale). Example: -1000",
    )
    parser.add_argument(
        "--y-clip-max",
        type=float,
        default=None,
        metavar="MM",
        help="Hide markers with lab Y above this (mm, after --scale).",
    )
    args = parser.parse_args()
    run_viewer(
        args.file,
        point_size=args.point_size,
        playback_speed=args.speed,
        background=args.background,
        segment_color=args.segment_color,
        scale_factor=args.scale,
        label_font_size=args.font_size,
        y_clip_min_mm=args.y_clip_min,
        y_clip_max_mm=args.y_clip_max,
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
