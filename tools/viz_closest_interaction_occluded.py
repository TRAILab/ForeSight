#!/usr/bin/env python3
"""Show camera + BEV views of occluded GT objects whose 3-s future swept
corridor comes closest to the ego's 3-s future swept corridor.

For every frame in the dataset the script:
  1. Collects all occluded GT objects (valid_flag=False; optionally also
     interpolated / extrapolated synthetic boxes) that were *previously
     visible* earlier in the same scene.
  2. Builds the ego's swept footprint corridor over the next ~3 s from
     ``gt_ego_fut_trajs``.
  3. Builds the object's swept footprint corridor over the next ~3 s from
     ``gt_agent_fut_trajs``.
  4. Scores each (sample, object) pair by polygon-to-polygon distance
     between the two corridors (0 when corridors overlap — i.e. the
     agents share ground inside the horizon).
  5. Sorts globally ascending and renders per-object views for the
     top-N: [annotated camera | last-visible camera | BEV corridor],
     plus a summary grid.

Compared to viz_closest_occluded.py:
  - Score is corridor-to-corridor distance rather than the current ego-to-
    object Euclidean distance. This surfaces yield / cross-traffic
    interactions whose paths share ground over the horizon even when no
    single frame has the agents radially close.
  - Adds a top-down BEV panel showing both corridors.

Usage
-----
    python tools/viz_closest_interaction_occluded.py \\
        --pkl data/infos/nuscenes_infos_val.pkl

    python tools/viz_closest_interaction_occluded.py \\
        --pkl data/infos/nuscenes_infos_val_occ.pkl --include-synthetic \\
        --classes car truck bus pedestrian bicycle --top-n 24
"""

import argparse
import os
import pickle
import numpy as np

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
except ImportError as e:
    raise ImportError(
        "shapely is required for swept-corridor scoring. "
        "Install with `pip install shapely`."
    ) from e


IMG_W, IMG_H = 1600, 900

CAMERA_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]

_C_OCC_RGB   = (230,  81,   0)   # orange-red : valid_flag=False
_C_SYNTH_RGB = (123,  31, 162)   # purple     : interpolated / extrapolated

# nuScenes ego (Renault Zoe) footprint, used to build the ego corridor.
EGO_LENGTH_DEFAULT = 4.084
EGO_WIDTH_DEFAULT  = 1.730


# ===========================================================================
# 3-D camera geometry  (same conventions as viz_closest_occluded.py)
# ===========================================================================

def _box_corners_3d(cx, cy, cz, length, width, height, yaw):
    l, w, h = length / 2.0, width / 2.0, height / 2.0
    local = np.array([
        [ l,  w,  h], [-l,  w,  h], [-l, -w,  h], [ l, -w,  h],
        [ l,  w, -h], [-l,  w, -h], [-l, -w, -h], [ l, -w, -h],
    ], dtype=np.float64)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    Rz   = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return local @ Rz.T + np.array([cx, cy, cz])


_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _project_to_cam(pts_lidar, M, T, K):
    """Project (N, 3) lidar points to camera pixels using the stored
    ``sensor2lidar`` extrinsics: ``p_lidar = p_cam @ M + T``."""
    M = np.asarray(M, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    pts_cam = (pts_lidar - T) @ M
    depth   = pts_cam[:, 2].copy()
    z_safe  = np.where(depth > 0, depth, 1e-6)
    proj    = pts_cam @ K.T
    uv      = proj[:, :2] / z_safe[:, None]
    return uv, depth


def _select_best_camera(box_lidar, cams):
    cx, cy, cz = float(box_lidar[0]), float(box_lidar[1]), float(box_lidar[2])
    corners    = _box_corners_3d(cx, cy, cz,
                                 float(box_lidar[3]), float(box_lidar[4]),
                                 float(box_lidar[5]), float(box_lidar[6]))
    centre     = np.array([[cx, cy, cz]])

    best_result   = None
    best_n_in     = -1
    best_dist_img = float('inf')

    for cam_name in CAMERA_NAMES:
        if cam_name not in cams:
            continue
        ci  = cams[cam_name]
        M   = ci['sensor2lidar_rotation']
        T   = ci['sensor2lidar_translation']
        K   = ci['cam_intrinsic']

        _, depth_c = _project_to_cam(centre, M, T, K)
        if depth_c[0] <= 0:
            continue

        uv, depth = _project_to_cam(corners, M, T, K)
        in_frame  = (
            (depth > 0) &
            (uv[:, 0] >= 0) & (uv[:, 0] <= IMG_W) &
            (uv[:, 1] >= 0) & (uv[:, 1] <= IMG_H)
        )
        n_in = int(in_frame.sum())

        uv_c, _ = _project_to_cam(centre, M, T, K)
        dist_img = float(np.hypot(uv_c[0, 0] - IMG_W / 2, uv_c[0, 1] - IMG_H / 2))

        if n_in > best_n_in or (n_in == best_n_in and dist_img < best_dist_img):
            best_n_in     = n_in
            best_dist_img = dist_img
            best_result   = (cam_name, uv, depth)

    return best_result


# ===========================================================================
# Swept-corridor geometry  (the new scoring primitive)
# ===========================================================================

def _footprint_polygon(cx, cy, length, width, yaw):
    """Oriented 2-D rectangle (BEV) centred at (cx, cy), rotated by yaw."""
    l, w = length / 2.0, width / 2.0
    local = np.array([[ l,  w], [-l,  w], [-l, -w], [ l, -w]], dtype=np.float64)
    c, s  = float(np.cos(yaw)), float(np.sin(yaw))
    R     = np.array([[c, -s], [s, c]])
    return Polygon(local @ R.T + np.array([cx, cy]))


def _trajectory_headings(positions, init_heading):
    """Heading at each step: motion direction; fall back to previous step
    (or init_heading at t=0) when displacement is below 1 mm."""
    headings = np.full(len(positions), float(init_heading), dtype=np.float64)
    for k in range(1, len(positions)):
        dx, dy = positions[k] - positions[k - 1]
        if np.hypot(dx, dy) > 1e-3:
            headings[k] = float(np.arctan2(dy, dx))
        else:
            headings[k] = headings[k - 1]
    return headings


def _swept_corridor(positions, headings, length, width):
    """Union of footprint rectangles along a trajectory."""
    polys = [_footprint_polygon(positions[t, 0], positions[t, 1],
                                 length, width, headings[t])
             for t in range(len(positions))]
    return unary_union(polys)


def _build_ego_corridor(ego_fut_trajs, n_steps, ego_length, ego_width,
                        ego_offset_x=0.0):
    """Ego corridor in lidar XY. Ego is at (ego_offset_x, 0) heading +x at t=0."""
    deltas = np.asarray(ego_fut_trajs, dtype=np.float64)[:n_steps]
    if len(deltas) == 0:
        return _footprint_polygon(ego_offset_x, 0.0, ego_length, ego_width, 0.0)
    cum  = np.cumsum(deltas, axis=0)
    pts  = np.concatenate([np.zeros((1, 2)), cum], axis=0) + np.array([ego_offset_x, 0.0])
    yaws = _trajectory_headings(pts, init_heading=0.0)
    return _swept_corridor(pts, yaws, ego_length, ego_width)


def _build_object_corridor(box, agent_fut_trajs, agent_fut_mask, n_steps):
    """Object corridor in lidar XY, anchored at the box's current XY."""
    cx, cy = float(box[0]), float(box[1])
    length = float(box[3])
    width  = float(box[4])
    yaw0   = float(box[6])

    if agent_fut_trajs is None:
        return _footprint_polygon(cx, cy, length, width, yaw0)

    deltas = np.asarray(agent_fut_trajs, dtype=np.float64)[:n_steps]
    if agent_fut_mask is not None:
        mask = np.asarray(agent_fut_mask, dtype=bool)[:len(deltas)]
    else:
        mask = np.ones(len(deltas), dtype=bool)

    cum   = np.cumsum(deltas, axis=0)
    pts   = np.concatenate([np.array([[0.0, 0.0]]), cum], axis=0) + np.array([cx, cy])
    valid = np.concatenate([[True], mask], axis=0)
    pts   = pts[valid]
    if len(pts) == 0:
        pts = np.array([[cx, cy]])
    yaws = _trajectory_headings(pts, init_heading=yaw0)
    return _swept_corridor(pts, yaws, length, width)


# ===========================================================================
# Data collection
# ===========================================================================

def find_interacting_occluded(infos, n_steps, ego_length, ego_width,
                               ego_offset_x=0.0,
                               include_synthetic=False, classes=None,
                               min_dist=1.0, max_dist=55.0,
                               candidate_ceiling=20.0):
    """Score occluded GT objects by ego↔object swept-corridor distance.

    Returns ALL candidate entries with ``corridor_dist <= candidate_ceiling``,
    sorted ascending. The downstream ``max_score`` (interaction threshold) is
    applied at the *output* stage — events.json / subset / render — so the
    diagnostic plot can still see the full distribution beyond the threshold.
    """
    from collections import defaultdict

    has_synthetic = 'is_interpolated' in infos[0]

    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])

    entries          = []
    skipped_no_traj  = 0
    skipped_no_ego   = 0

    for sc_tok, gidxs in scene_map.items():
        last_visible = {}  # inst_ind -> (box, cams) from most recent visible frame

        for gi in gidxs:
            info  = infos[gi]
            boxes = info.get('gt_boxes')
            names = info.get('gt_names')
            if boxes is None or len(boxes) == 0:
                continue
            if 'cams' not in info or not info['cams']:
                continue

            ego_fut = info.get('gt_ego_fut_trajs')
            if ego_fut is None or len(ego_fut) == 0:
                skipped_no_ego += 1
                continue
            ego_corridor = _build_ego_corridor(
                ego_fut, n_steps, ego_length, ego_width, ego_offset_x)

            agent_fut  = info.get('gt_agent_fut_trajs')
            agent_mask = info.get('gt_agent_fut_masks')

            n     = len(boxes)
            valid = info.get('valid_flag', np.ones(n, dtype=bool))
            is_i  = info['is_interpolated'] if has_synthetic else np.zeros(n, dtype=bool)
            is_e  = info['is_extrapolated'] if has_synthetic else np.zeros(n, dtype=bool)

            for bi in range(n):
                inst_ind = int(info['instance_inds'][bi])
                is_occ   = not bool(valid[bi])
                is_synth = bool(is_i[bi]) or bool(is_e[bi])

                # Refresh last-visible memory BEFORE the emit gate so the
                # currently-occluded frame can look itself up.
                if bool(valid[bi]) and not is_synth:
                    last_visible[inst_ind] = (
                        boxes[bi].copy(), info['cams'],
                        gi, info.get('timestamp', 0), info.get('token', ''),
                    )

                if not is_occ and not (include_synthetic and is_synth):
                    continue
                if classes is not None and names[bi] not in classes:
                    continue

                dist = float(np.hypot(float(boxes[bi, 0]), float(boxes[bi, 1])))
                if dist < min_dist or dist > max_dist:
                    continue

                traj_bi = (agent_fut[bi]
                           if agent_fut is not None and bi < len(agent_fut)
                           else None)
                mask_bi = (agent_mask[bi]
                           if agent_mask is not None and bi < len(agent_mask)
                           else None)
                if traj_bi is None:
                    skipped_no_traj += 1
                obj_corridor = _build_object_corridor(
                    boxes[bi], traj_bi, mask_bi, n_steps)

                score = float(ego_corridor.distance(obj_corridor))
                if score > candidate_ceiling:
                    continue

                prev = last_visible.get(inst_ind)
                if prev is not None:
                    prev_box, prev_cams, prev_gi, prev_ts, prev_tok = prev
                    prev_dist_ego = float(np.hypot(float(prev_box[0]),
                                                    float(prev_box[1])))
                    cur_ts  = info.get('timestamp', 0)
                    prev_dt = (cur_ts - prev_ts) / 1e6 if prev_ts else None
                else:
                    prev_box = prev_cams = prev_gi = prev_tok = None
                    prev_dist_ego = None
                    prev_dt       = None

                entries.append(dict(
                    scene_token             = sc_tok,
                    global_idx              = gi,
                    inst_ind                = inst_ind,
                    class_name              = str(names[bi]),
                    dist_ego                = dist,
                    corridor_dist           = score,
                    box_lidar               = boxes[bi].copy(),
                    is_interp               = bool(is_i[bi]),
                    is_extrap               = bool(is_e[bi]),
                    valid_flag              = bool(valid[bi]),
                    cams_info               = info['cams'],
                    token                   = info.get('token', ''),
                    timestamp               = info.get('timestamp', 0),
                    last_visible_box        = prev_box,
                    last_visible_cams       = prev_cams,
                    last_visible_global_idx = prev_gi,
                    last_visible_token      = prev_tok,
                    last_visible_dt_s       = prev_dt,
                    last_visible_dist_ego   = prev_dist_ego,
                    ego_corridor            = ego_corridor,
                    obj_corridor            = obj_corridor,
                ))

    # Require a previously-visible reference frame (same scene, earlier sample
    # with valid_flag=True and not synthetic). Drops first-appearance and
    # ever-occluded instances.
    before = len(entries)
    entries = [e for e in entries if e['last_visible_box'] is not None]
    dropped_no_prev = before - len(entries)

    entries.sort(key=lambda x: (x['corridor_dist'], x['dist_ego']))

    if skipped_no_ego:
        print(f'  (note: {skipped_no_ego} samples skipped: no gt_ego_fut_trajs)')
    if skipped_no_traj:
        print(f'  (note: {skipped_no_traj} occluded boxes had no agent traj '
              f'and used a stationary corridor)')
    if dropped_no_prev:
        print(f'  (note: {dropped_no_prev} candidates dropped: no previously-'
              f'visible frame in this scene)')
    return entries


# ===========================================================================
# Rendering
# ===========================================================================

def _draw_3d_box(draw, uv, depth, color, line_width=3):
    visible = {i for i in range(8) if depth[i] > 0}
    for i, j in _BOX_EDGES:
        if i in visible and j in visible:
            draw.line(
                [(int(round(uv[i, 0])), int(round(uv[i, 1]))),
                 (int(round(uv[j, 0])), int(round(uv[j, 1])))],
                fill=color, width=line_width,
            )


def _draw_2d_bbox(draw, uv, depth, color, line_width=3):
    vis = uv[depth > 0]
    if len(vis) == 0:
        return
    w, h = draw.im.size
    u_min = max(0, int(vis[:, 0].min()))
    u_max = min(w - 1, int(vis[:, 0].max()))
    v_min = max(0, int(vis[:, 1].min()))
    v_max = min(h - 1, int(vis[:, 1].max()))
    if u_min >= u_max or v_min >= v_max:
        return
    draw.rectangle([u_min, v_min, u_max, v_max],
                   outline=color, width=line_width)


def _plot_corridor_bev(ax, ego_corridor, obj_corridor,
                       ego_offset_x=0.0, title=None, pad=4.0):
    """Top-down view of both swept corridors."""
    def _fill(geom, **kw):
        if geom is None or geom.is_empty:
            return
        geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for g in geoms:
            if g.is_empty:
                continue
            x, y = g.exterior.xy
            ax.fill(x, y, **kw)

    _fill(ego_corridor, facecolor='#2e7d32', edgecolor='#1b5e20',
          alpha=0.40, linewidth=1.0)
    _fill(obj_corridor, facecolor='#e65100', edgecolor='#7b1fa2',
          alpha=0.55, linewidth=1.0)

    ax.plot([ego_offset_x], [0], marker='s', markersize=7,
            markerfacecolor='#1b5e20', markeredgecolor='black')

    bounds_union = ego_corridor.union(obj_corridor).bounds
    minx, miny, maxx, maxy = bounds_union
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    r  = max(maxx - minx, maxy - miny) / 2 + pad
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)', fontsize=7)
    ax.set_ylabel('y (m)', fontsize=7)
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def render_interactions(entries, top_n, output_dir, dataroot,
                         ego_offset_x=0.0):
    """Render annotated camera + BEV views for the top-N entries."""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    Image.MAX_IMAGE_PIXELS = None

    os.makedirs(output_dir, exist_ok=True)
    cam_dir = os.path.join(output_dir, 'cameras')
    os.makedirs(cam_dir, exist_ok=True)

    entries = entries[:top_n]
    if not entries:
        print('Nothing to render.')
        return

    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    except Exception:
        font = ImageFont.load_default()

    def _load_annotated(box_lidar, cams_info, color, label, stem):
        if box_lidar is None:
            return None, 'box_lidar is None'
        if cams_info is None:
            return None, 'cams_info is None'
        result = _select_best_camera(box_lidar, cams_info)
        if result is None:
            return None, 'no camera has box centre in front (depth<=0 in all 6 cams)'
        cam_name, uv, depth = result
        ci       = cams_info[cam_name]
        rel_path = ci['data_path']
        # Try a few candidate locations: absolute, as-is from cwd
        # (handles pkls that store './data/nuscenes/samples/...'), and
        # joined with --dataroot (handles pkls that store bare 'samples/...').
        candidates = [rel_path]
        if not os.path.isabs(rel_path):
            candidates.append(os.path.normpath(rel_path))
            candidates.append(os.path.join(dataroot, rel_path))
            # If the stored path already starts with the dataroot, also try
            # stripping it (some converters double-prefix).
            stripped = rel_path
            for prefix in ('./', dataroot + '/', './' + dataroot + '/'):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
            if stripped != rel_path:
                candidates.append(stripped)
                candidates.append(os.path.join(dataroot, stripped))
        img_path = next((p for p in candidates if os.path.isfile(p)), None)
        if img_path is None:
            return None, (f'image not found — tried {candidates!r} '
                          f'(cam={cam_name})')
        img = Image.open(img_path).convert('RGB')

        raw_path = os.path.join(cam_dir, f'{stem}_raw_{cam_name}.png')
        img.save(raw_path)

        img_ann = img.copy()
        draw = ImageDraw.Draw(img_ann)
        _draw_3d_box(draw, uv, depth, color, line_width=3)
        _draw_2d_bbox(draw, uv, depth, color, line_width=2)
        draw.rectangle([0, 0, img_ann.width, 26], fill=(0, 0, 0))
        draw.text((5, 5), label, fill=(255, 255, 255), font=font)
        ann_path = os.path.join(cam_dir, f'{stem}_{cam_name}.png')
        img_ann.save(ann_path)
        return (ann_path, raw_path, cam_name), None

    saved_items = []  # (rank, occ_path, occ_cam, vis_path, vis_cam, entry)

    for rank, entry in enumerate(entries):
        is_synth = entry['is_interp'] or entry['is_extrap']
        color    = _C_SYNTH_RGB if is_synth else _C_OCC_RGB
        occ_tag  = ('interp' if entry['is_interp'] else
                    'extrap' if entry['is_extrap'] else 'occluded')
        stem_occ = f'occ_{rank+1:03d}_{entry["scene_token"][:8]}'
        stem_vis = f'vis_{rank+1:03d}_{entry["scene_token"][:8]}'

        occ_label = (f"#{rank+1} OCCLUDED  {entry['class_name']}  "
                     f"corridor={entry['corridor_dist']:.2f}m  "
                     f"dist={entry['dist_ego']:.1f}m  [{occ_tag}]")
        vis_label = f"#{rank+1} LAST VISIBLE  {entry['class_name']}"

        occ_result, occ_reason = _load_annotated(
            entry['box_lidar'], entry['cams_info'], color, occ_label, stem_occ)
        vis_result, _vis_reason = _load_annotated(
            entry['last_visible_box'], entry['last_visible_cams'],
            (0, 120, 40), vis_label, stem_vis)

        if occ_result is None:
            print(f'  [{rank+1:3d}]  {entry["class_name"]:<20s}  '
                  f'corridor={entry["corridor_dist"]:.2f}m  '
                  f'— skipped: {occ_reason}')
            continue

        occ_path, _occ_raw, occ_cam = occ_result
        if vis_result is not None:
            vis_path, _vis_raw, vis_cam = vis_result
        else:
            vis_path = vis_cam = None
        saved_items.append((rank, occ_path, occ_cam, vis_path, vis_cam, entry))
        print(f'  [{rank+1:3d}]  {entry["class_name"]:<20s}  '
              f'corridor={entry["corridor_dist"]:5.2f}m  '
              f'dist={entry["dist_ego"]:5.1f}m  [{occ_tag}]  '
              f'occ={occ_cam}  vis={vis_cam or "none"}')

    if not saved_items:
        print('No images were rendered (check --dataroot and image paths).')
        return

    # Summary grid: 2 objects per row × 3 columns each (occ | vis | bev).
    n               = len(saved_items)
    objects_per_row = 2
    cols_per_obj    = 3
    ncols           = objects_per_row * cols_per_obj
    nrows           = (n + objects_per_row - 1) // objects_per_row
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows),
                              facecolor='white')
    axes = np.array(axes).reshape(nrows, ncols)

    for k, (rank, occ_path, occ_cam, vis_path, vis_cam, e) in enumerate(saved_items):
        row  = k // objects_per_row
        col0 = (k % objects_per_row) * cols_per_obj

        is_synth    = e['is_interp'] or e['is_extrap']
        title_color = '#7b1fa2' if is_synth else '#e65100'
        occ_tag     = ('interp' if e['is_interp'] else
                       'extrap' if e['is_extrap'] else 'occluded')

        ax_occ = axes[row, col0]
        ax_occ.imshow(plt.imread(occ_path))
        ax_occ.set_axis_off()
        ax_occ.set_title(
            f"#{rank+1} {e['class_name']}  corridor={e['corridor_dist']:.2f}m\n"
            f"dist={e['dist_ego']:.1f}m  [{occ_tag}]  {occ_cam}",
            color=title_color, fontsize=7, pad=2)

        ax_vis = axes[row, col0 + 1]
        if vis_path is not None:
            ax_vis.imshow(plt.imread(vis_path))
            ax_vis.set_title(f"last visible  {vis_cam}",
                             color='#2e7d32', fontsize=7, pad=2)
        else:
            ax_vis.set_facecolor('#f0f0f0')
            ax_vis.text(0.5, 0.5, 'No previous\nvisible frame',
                        ha='center', va='center', fontsize=9,
                        color='#888888', transform=ax_vis.transAxes)
            ax_vis.set_title('last visible  —', color='#888888', fontsize=7, pad=2)
        ax_vis.set_axis_off()

        ax_bev = axes[row, col0 + 2]
        _plot_corridor_bev(
            ax_bev, e['ego_corridor'], e['obj_corridor'],
            ego_offset_x=ego_offset_x,
            title=f"BEV  ego (green) vs obj (orange)  d={e['corridor_dist']:.2f}m")

    # Hide unused cells in the last row.
    for k in range(n, nrows * objects_per_row):
        row  = k // objects_per_row
        col0 = (k % objects_per_row) * cols_per_obj
        for c in range(cols_per_obj):
            axes[row, col0 + c].set_visible(False)

    fig.suptitle(
        f'Top {n} previously-visible-then-occluded GT objects '
        f'by ego↔object swept-corridor distance',
        fontsize=11, color='#111111', y=1.005)
    fig.subplots_adjust(hspace=0.45, wspace=0.1)

    grid_out = os.path.join(output_dir, 'closest_interactions.png')
    plt.savefig(grid_out, dpi=130, bbox_inches='tight', facecolor='white')
    print(f'\nGrid saved → {grid_out}')
    plt.close(fig)


# ===========================================================================
# Event summary dashboard
# ===========================================================================

def _plot_event_summary(entries, output_dir, max_score):
    """High-level 2x3 dashboard of the interaction events.

    Panels: corridor-distance hist (with threshold line), current-distance
    hist, occlusion-duration hist, class bar, per-scene-event-count bar,
    corridor-vs-current scatter coloured by class.
    """
    if not entries:
        return
    import matplotlib.pyplot as plt
    from collections import Counter

    corr = np.array([e['corridor_dist'] for e in entries])
    dist = np.array([e['dist_ego']      for e in entries])
    dt   = np.array([e['last_visible_dt_s'] for e in entries
                     if e['last_visible_dt_s'] is not None])
    cls  = [e['class_name']  for e in entries]
    sc   = [e['scene_token'] for e in entries]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), facecolor='white')

    # 1. Corridor distance histogram (severity).
    ax = axes[0, 0]
    upper = max(float(corr.max()) + 0.5, float(max_score) + 0.5)
    ax.hist(corr, bins=np.linspace(0, upper, 31),
            color='#1976d2', edgecolor='white')
    ax.axvline(max_score, color='red', linestyle='--', linewidth=1.2,
               label=f'max_score = {max_score:g} m')
    ax.set_xlabel('corridor distance (m)')
    ax.set_ylabel('events')
    ax.set_title('Corridor distance — interaction severity')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Current ego-object distance.
    ax = axes[0, 1]
    ax.hist(dist, bins=np.linspace(0, float(dist.max()) + 1, 31),
            color='#388e3c', edgecolor='white')
    ax.set_xlabel('current ego–object distance (m)')
    ax.set_ylabel('events')
    ax.set_title('Current radial distance')
    ax.grid(alpha=0.3)

    # 3. Time since last visible.
    ax = axes[0, 2]
    if len(dt) > 0:
        ax.hist(dt, bins=np.linspace(0, max(float(dt.max()), 1.0), 21),
                color='#f57c00', edgecolor='white')
    ax.set_xlabel('seconds since last visible')
    ax.set_ylabel('events')
    ax.set_title('Occlusion duration')
    ax.grid(alpha=0.3)

    # 4. Class breakdown.
    ax       = axes[1, 0]
    cls_hist = Counter(cls).most_common()
    if cls_hist:
        labels, counts = zip(*cls_hist)
        ax.barh(range(len(labels)), counts, color='#7b1fa2')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
    ax.set_xlabel('events')
    ax.set_title('Class breakdown')
    ax.grid(alpha=0.3, axis='x')

    # 5. Top-30 scenes by event count.
    ax           = axes[1, 1]
    scene_counts = Counter(sc).most_common(30)
    if scene_counts:
        _, c = zip(*scene_counts)
        ax.bar(range(len(c)), c, color='#0097a7')
        ax.set_xlabel(f'scene rank (top {len(c)} of {len(set(sc))})')
    ax.set_ylabel('events')
    ax.set_title('Events per scene')
    ax.grid(alpha=0.3, axis='y')

    # 6. Corridor vs current distance scatter, coloured by top classes.
    ax          = axes[1, 2]
    top_classes = [c for c, _ in Counter(cls).most_common(5)]
    palette     = plt.get_cmap('tab10')
    cls_arr     = np.array(cls)
    for i, c in enumerate(top_classes):
        m = cls_arr == c
        ax.scatter(dist[m], corr[m], s=22, alpha=0.65,
                   color=palette(i), label=c, edgecolors='none')
    other = ~np.isin(cls_arr, top_classes)
    if other.any():
        ax.scatter(dist[other], corr[other], s=18, alpha=0.40,
                   color='gray', label='other', edgecolors='none')
    ax.axhline(max_score, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('current distance (m)')
    ax.set_ylabel('corridor distance (m)')
    ax.set_title('Corridor distance vs current distance')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    fig.suptitle(f'Interaction event summary — n={len(entries)} '
                 f'({len(set(sc))} scenes, {len(set(e["token"] for e in entries))} samples)',
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'events_summary.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Wrote summary plot      → {out_path}')


# ===========================================================================
# Persistence: event records + token lists for downstream metric filtering
# ===========================================================================

def _save_event_records(entries, infos, output_dir, src_pkl, n_steps,
                         min_dist, max_dist, max_score,
                         classes, include_synthetic, neighbor_window):
    """Dump per-event interaction records + sample/scene token lists.

    Always writes (into ``output_dir``):
      - events.json              every interaction event with full metadata
      - interaction_samples.txt  unique sample tokens with >=1 event
      - interaction_scenes.txt   unique scene tokens with >=1 event

    These are the canonical artefacts for post-hoc filtering of full-val
    eval metrics: load the model's per-sample metrics JSON, then average
    only over the tokens in ``interaction_samples.txt`` to get the
    "interaction-relevant" headline number.
    """
    import json
    os.makedirs(output_dir, exist_ok=True)

    events = []
    for rank, e in enumerate(entries):
        bx  = e['box_lidar']
        rec = {
            'rank':            rank + 1,
            'scene_token':     e['scene_token'],
            'sample_token':    e['token'],
            'global_idx':      int(e['global_idx']),
            'instance_ind':    int(e['inst_ind']),
            'class_name':      e['class_name'],
            'corridor_dist':   float(e['corridor_dist']),
            'dist_ego':        float(e['dist_ego']),
            'is_occluded':     not bool(e['valid_flag']),
            'is_interpolated': bool(e['is_interp']),
            'is_extrapolated': bool(e['is_extrap']),
            'pose': {
                'x':       float(bx[0]),
                'y':       float(bx[1]),
                'z':       float(bx[2]),
                'length':  float(bx[3]),
                'width':   float(bx[4]),
                'height':  float(bx[5]),
                'yaw_rad': float(bx[6]),
                'yaw_deg': float(np.degrees(float(bx[6]))),
            },
        }
        if e['last_visible_dt_s'] is not None:
            rec['last_visible'] = {
                'sample_token': e['last_visible_token'],
                'global_idx':   int(e['last_visible_global_idx']),
                'dt_s':         float(e['last_visible_dt_s']),
                'dist_ego':     float(e['last_visible_dist_ego']),
            }
        else:
            rec['last_visible'] = None
        events.append(rec)

    # Unique sample / scene tokens (insertion-order preserving).
    sample_tokens, seen_s = [], set()
    scene_tokens,  seen_c = [], set()
    for e in entries:
        if e['token'] not in seen_s:
            seen_s.add(e['token'])
            sample_tokens.append(e['token'])
        if e['scene_token'] not in seen_c:
            seen_c.add(e['scene_token'])
            scene_tokens.append(e['scene_token'])

    # Optional temporal-neighbor expansion: include samples within ±N frames
    # (in scene-temporal order) of any interaction-bearing sample.
    n_core_samples = len(sample_tokens)
    if neighbor_window > 0:
        from collections import defaultdict
        scene_order = defaultdict(list)
        for gi, info in enumerate(infos):
            scene_order[info['scene_token']].append((info.get('timestamp', 0),
                                                      info.get('token', ''),
                                                      gi))
        for sc in scene_order:
            scene_order[sc].sort(key=lambda x: x[0])

        token_to_scene = {e['token']: e['scene_token'] for e in entries}
        expanded       = set(sample_tokens)
        for tok in list(sample_tokens):
            sc      = token_to_scene[tok]
            ordered = scene_order[sc]
            idx     = next((i for i, x in enumerate(ordered) if x[1] == tok), None)
            if idx is None:
                continue
            lo = max(0, idx - neighbor_window)
            hi = min(len(ordered), idx + neighbor_window + 1)
            for _, t, _ in ordered[lo:hi]:
                expanded.add(t)
        # Preserve original ordering, append new neighbors in scene-temporal order.
        new_only = [t for sc in scene_tokens
                     for _, t, _ in scene_order[sc]
                     if t in expanded and t not in seen_s]
        sample_tokens = sample_tokens + new_only

    doc = {
        'config': {
            'pkl':               src_pkl,
            'horizon_steps':     int(n_steps),
            'horizon_s_approx':  float(n_steps) * 0.5,
            'min_dist_m':        float(min_dist),
            'max_dist_m':        float(max_dist),
            'max_score_m':       float(max_score),
            'classes':           list(classes) if classes else None,
            'include_synthetic': bool(include_synthetic),
            'neighbor_window':   int(neighbor_window),
        },
        'summary': {
            'n_events':            len(events),
            'n_core_samples':      n_core_samples,
            'n_unique_samples':    len(sample_tokens),
            'n_unique_scenes':     len(scene_tokens),
            'n_overlap_corridors': sum(1 for e in entries
                                        if e['corridor_dist'] <= 1e-6),
        },
        'events': events,
    }

    json_path    = os.path.join(output_dir, 'events.json')
    samples_path = os.path.join(output_dir, 'interaction_samples.txt')
    scenes_path  = os.path.join(output_dir, 'interaction_scenes.txt')

    with open(json_path, 'w') as f:
        json.dump(doc, f, indent=2)
    with open(samples_path, 'w') as f:
        f.write('\n'.join(sample_tokens) + '\n')
    with open(scenes_path, 'w') as f:
        f.write('\n'.join(scene_tokens) + '\n')

    expansion = ('' if neighbor_window == 0
                  else f' (core={n_core_samples}, +{len(sample_tokens) - n_core_samples} '
                       f'within ±{neighbor_window} frames)')
    print(f'\nWrote interaction records:')
    print(f'  {len(events):4d} events                 → {json_path}')
    print(f'  {len(sample_tokens):4d} unique samples         → {samples_path}{expansion}')
    print(f'  {len(scene_tokens):4d} unique scenes          → {scenes_path}')


# ===========================================================================
# Scene-subset construction
# ===========================================================================

def _build_interaction_subset(entries, infos, data, n_samples, src_pkl, output_dir):
    """Aggregate interaction events into scenes and write a subset pkl.

    Selection strategy: scenes are ordered by ascending minimum corridor
    distance (most-severe event per scene), tie-broken by descending event
    count. Scenes are then added greedily in that order until the union of
    their interaction-bearing samples reaches ``n_samples`` (or all
    candidate scenes are exhausted).

    Writes (into ``output_dir``):
      - subset_scenes.txt              one scene_token per line, in
                                       severity order
      - <pkl_base>_interactions.pkl    filtered copy of the source pkl
        containing every sample of the selected scenes (drop-in for val).

    Distinct from ``interaction_scenes.txt`` produced by
    ``_save_event_records``, which lists *every* scene with at least one
    event; ``subset_scenes.txt`` is the severity-ranked, sample-budgeted
    selection.
    """
    from collections import defaultdict, Counter

    per_scene = defaultdict(list)
    for e in entries:
        per_scene[e['scene_token']].append(e)

    scored = sorted(
        per_scene.items(),
        key=lambda kv: (min(e['corridor_dist'] for e in kv[1]),
                        -len(kv[1])),
    )

    # Greedy: add scenes in severity order until interaction-sample budget hit.
    selected         = []
    covered_samples  = set()
    for sc, evts in scored:
        selected.append(sc)
        covered_samples.update(e['token'] for e in evts)
        if len(covered_samples) >= n_samples:
            break
    selected_set = set(selected)
    k            = len(selected)

    cls_hist = Counter(
        e['class_name'] for sc in selected for e in per_scene[sc]
    )
    n_events     = sum(len(per_scene[sc]) for sc in selected)
    total_events = sum(len(v) for v in per_scene.values())
    sub_infos    = [i for i in infos if i['scene_token'] in selected_set]

    all_interaction_samples      = {e['token']
                                     for v in per_scene.values() for e in v}
    selected_interaction_samples = covered_samples

    hit_target = len(selected_interaction_samples) >= n_samples
    status     = ('target reached'
                  if hit_target
                  else f'target {n_samples} not reachable — used all candidates')

    print(f'\nBuilding interaction-scene subset (target {n_samples} samples) ...')
    print(f'  {status}.')
    print(f'  Selected {k}/{len(per_scene)} interaction-bearing scenes '
          f'({n_events}/{total_events} events, {len(sub_infos)} total samples).')
    print(f'  Selected {len(selected_interaction_samples)}/'
          f'{len(all_interaction_samples)} interaction-bearing samples.')
    print(f'  Class breakdown of subset events: '
          + ', '.join(f'{c}={n}' for c, n in cls_hist.most_common()))

    os.makedirs(output_dir, exist_ok=True)

    scenes_path = os.path.join(output_dir, 'subset_scenes.txt')
    with open(scenes_path, 'w') as f:
        for sc in selected:
            f.write(sc + '\n')
    print(f'  Wrote scene list  → {scenes_path}')

    sub_data          = dict(data)
    sub_data['infos'] = sub_infos
    pkl_base  = os.path.splitext(os.path.basename(src_pkl))[0]
    pkl_out   = os.path.join(output_dir, f'{pkl_base}_interactions.pkl')
    with open(pkl_out, 'wb') as f:
        pickle.dump(sub_data, f)
    print(f'  Wrote filtered pkl → {pkl_out}')


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pkl', default='data/infos/nuscenes_infos_val.pkl',
                    help='Path to nuScenes info pkl (must contain '
                         'gt_ego_fut_trajs and gt_agent_fut_trajs).')
    ap.add_argument('--include-synthetic', action='store_true',
                    help='Also include interpolated/extrapolated boxes '
                         '(requires an _occ pkl with is_interpolated flags).')
    ap.add_argument('--classes', nargs='+', metavar='CLS',
                    default=['car', 'truck', 'bus', 'trailer',
                             'construction_vehicle', 'motorcycle',
                             'bicycle', 'pedestrian'],
                    help='Restrict to these GT class names. Default: dynamic '
                         'agents only (excludes barrier and traffic_cone). '
                         'To include everything, pass all 10 classes '
                         'explicitly.')
    ap.add_argument('--min-dist', type=float, default=1.0, metavar='M',
                    help='Minimum current ego-to-object distance (default: 1 m). '
                         'Filters self/sensor-noise hits.')
    ap.add_argument('--max-dist', type=float, default=55.0, metavar='M',
                    help='Maximum current ego-to-object distance (default: 55 m, '
                         'matching the detector range — objects past this point '
                         'are not actionable by the planner regardless of '
                         'corridor overlap).')
    ap.add_argument('--max-score', type=float, default=8.0, metavar='M',
                    help='Interaction threshold (default: 8 m). Only events '
                         'with corridor distance below this are written to '
                         'events.json, the subset pkl, and the renders. The '
                         'diagnostic plot shows all candidates up to '
                         '--candidate-ceiling regardless, so you can see '
                         'where this threshold sits in the full distribution.')
    ap.add_argument('--candidate-ceiling', type=float, default=20.0, metavar='M',
                    help='Compute / plot candidates up to this corridor '
                         'distance (default: 20 m). Beyond this, objects are '
                         'definitely not interacting and are skipped at '
                         'collection time. Increase to inspect the far tail '
                         'of the distribution.')
    ap.add_argument('--horizon-steps', type=int, default=6, metavar='N',
                    help='Future timesteps integrated into each corridor '
                         '(default: 6 ≈ 3 s @ 0.5 s). Capped by the pkl.')
    ap.add_argument('--ego-length', type=float, default=EGO_LENGTH_DEFAULT, metavar='M')
    ap.add_argument('--ego-width',  type=float, default=EGO_WIDTH_DEFAULT,  metavar='M')
    ap.add_argument('--ego-offset-x', type=float, default=0.0, metavar='M',
                    help='Ego centre x-offset from lidar origin in lidar frame '
                         '(nuScenes LIDAR_TOP sits ~0.985 m forward of ego centre; '
                         'pass -0.985 to place the ego footprint behind the lidar). '
                         'Default 0 treats lidar = ego.')
    ap.add_argument('--top-n', type=int, default=12, metavar='N',
                    help='Number of top-ranked objects to render (default: 12).')
    ap.add_argument('--output-dir', default='vis/interaction_occluded', metavar='DIR')
    ap.add_argument('--dataroot', default='data/nuscenes', metavar='DIR')
    ap.add_argument('--build-subset', action='store_true',
                    help='Also write subset_scenes.txt and a filtered pkl '
                         'containing every sample of the selected scenes — a '
                         'drop-in replacement for the val ann_file focused on '
                         'occluded interactions.')
    ap.add_argument('--n-samples', type=int, default=500, metavar='N',
                    help='Target number of interaction-bearing samples in the '
                         'subset (default: 500). Scenes are added greedily in '
                         'descending severity order until the union of their '
                         'interaction-bearing samples reaches this target '
                         '(or all candidate scenes are exhausted). Only used '
                         'with --build-subset.')
    ap.add_argument('--neighbor-window', type=int, default=0, metavar='N',
                    help='When >0, expand interaction_samples.txt to also '
                         'include samples within ±N frames of any '
                         'interaction-bearing sample (in scene-temporal '
                         'order). Useful for inflating the sample count for '
                         'metric-aggregation CI without changing the '
                         'interaction definition. Default 0 (no expansion).')
    args = ap.parse_args()

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos    = data['infos']
    n_scenes = len({i['scene_token'] for i in infos})
    print(f'  {len(infos)} samples across {n_scenes} scenes')

    sample0 = infos[0]
    if 'gt_ego_fut_trajs' not in sample0:
        raise SystemExit(
            'gt_ego_fut_trajs missing from pkl — this script requires a '
            'SparseDrive-style pkl with ego/agent future trajectories.')
    ego_avail   = len(sample0['gt_ego_fut_trajs'])
    agent_avail = (len(sample0['gt_agent_fut_trajs'][0])
                   if sample0.get('gt_agent_fut_trajs') is not None
                   and len(sample0['gt_agent_fut_trajs']) > 0 else 0)
    n_steps = min(args.horizon_steps, ego_avail,
                  agent_avail if agent_avail > 0 else args.horizon_steps)
    print(f'  Horizon available: ego={ego_avail} steps, agent={agent_avail} steps; '
          f'using {n_steps}')

    if args.classes:
        print(f'  Class filter : {args.classes}')

    print(f'\nScoring occluded objects by swept-corridor distance ...')
    all_entries = find_interacting_occluded(
        infos,
        n_steps           = n_steps,
        ego_length        = args.ego_length,
        ego_width         = args.ego_width,
        ego_offset_x      = args.ego_offset_x,
        include_synthetic = args.include_synthetic,
        classes           = args.classes,
        min_dist          = args.min_dist,
        max_dist          = args.max_dist,
        candidate_ceiling = args.candidate_ceiling,
    )

    if not all_entries:
        print('No occluded objects within the candidate ceiling.')
        return

    # Diagnostic plot uses ALL candidates; everything else uses interaction-
    # threshold filter.
    entries = [e for e in all_entries if e['corridor_dist'] <= args.max_score]

    n_overlap = sum(1 for e in entries if e['corridor_dist'] <= 1e-6)
    print(f'  Found {len(all_entries)} candidates within {args.candidate_ceiling:g} m '
          f'corridor distance.')
    print(f'  Of these, {len(entries)} pass --max-score={args.max_score:g} m '
          f'(={n_overlap} overlapping corridors / score 0).')

    if not entries:
        print('No events pass the interaction threshold; plot only.')
        _plot_event_summary(all_entries, args.output_dir, max_score=args.max_score)
        return

    # Class-frequency breakdown for the top-N about to be rendered.
    from collections import Counter
    topN     = entries[:args.top_n]
    cls_hist = Counter(e['class_name'] for e in topN)
    print(f'\n  Class breakdown of top {len(topN)}: '
          + ', '.join(f'{c}={n}' for c, n in cls_hist.most_common()))

    print(f'\n  Top {len(topN)} interaction details:\n')
    for i, e in enumerate(topN):
        occ_tag = ('interp' if e['is_interp'] else
                   'extrap' if e['is_extrap'] else 'occluded')
        bx      = e['box_lidar']
        pos     = f'({float(bx[0]):+6.1f}, {float(bx[1]):+6.1f}, {float(bx[2]):+5.1f})'
        yaw_deg = float(np.degrees(float(bx[6])))
        size    = f'{float(bx[3]):.2f}×{float(bx[4]):.2f}'  # length × width
        overlap = 'yes' if e['corridor_dist'] <= 1e-6 else 'no'
        print(f'  #{i+1:>3d}  {e["class_name"]:<22s}  '
              f'corridor={e["corridor_dist"]:5.2f}m  '
              f'dist={e["dist_ego"]:5.1f}m  [{occ_tag}]  overlap={overlap}')
        print(f'        scene_token   = {e["scene_token"]}')
        print(f'        sample_token  = {e["token"]}  (global_idx={e["global_idx"]})')
        print(f'        instance_ind  = {e["inst_ind"]}')
        print(f'        pose          = pos={pos}  yaw={yaw_deg:+6.1f}°  L×W={size}')
        if e['last_visible_dt_s'] is not None:
            print(f'        last_visible  = {e["last_visible_dt_s"]:.1f}s ago  '
                  f'(global_idx={e["last_visible_global_idx"]}, '
                  f'dist={e["last_visible_dist_ego"]:.1f}m)')
            print(f'        last_visible_token = {e["last_visible_token"]}')
        else:
            print(f'        last_visible  = (none — should have been filtered)')
        print()

    _save_event_records(
        entries, infos, args.output_dir,
        src_pkl           = args.pkl,
        n_steps           = n_steps,
        min_dist          = args.min_dist,
        max_dist          = args.max_dist,
        max_score         = args.max_score,
        classes           = args.classes,
        include_synthetic = args.include_synthetic,
        neighbor_window   = args.neighbor_window,
    )
    _plot_event_summary(all_entries, args.output_dir, max_score=args.max_score)

    if args.build_subset:
        _build_interaction_subset(
            entries, infos, data,
            n_samples  = args.n_samples,
            src_pkl    = args.pkl,
            output_dir = args.output_dir,
        )

    print(f'\nRendering top {args.top_n} ...')
    render_interactions(
        entries,
        top_n        = args.top_n,
        output_dir   = args.output_dir,
        dataroot     = args.dataroot,
        ego_offset_x = args.ego_offset_x,
    )


if __name__ == '__main__':
    main()
