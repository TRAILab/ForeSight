#!/usr/bin/env python3
"""Show camera views of the closest occluded GT objects to the ego vehicle.

For every frame in the dataset the script:
  1. Collects all objects with valid_flag=False (occluded in GT annotations)
     or synthetic boxes (interpolated/extrapolated, if --include-synthetic).
  2. Sorts them globally by distance to the ego (lidar-frame XY distance).
  3. Projects each object's 3-D bounding box into all 6 cameras to pick the
     best view (most corners visible, centre closest to image centre).
  4. Saves per-object annotated camera crops and a summary grid image.

Usage
-----
    # Basic: top-12 occluded objects in the val set
    python tools/viz_closest_occluded.py \\
        --pkl data/infos/nuscenes_infos_val.pkl

    # Include synthetic (interpolated/extrapolated) boxes from an _occ pkl
    python tools/viz_closest_occluded.py \\
        --pkl data/infos/nuscenes_infos_val_occ.pkl --include-synthetic

    # Restrict to vehicle classes, show top-20
    python tools/viz_closest_occluded.py \\
        --pkl data/infos/nuscenes_infos_val.pkl \\
        --classes car truck bus trailer --top-n 20

    # Wider search radius, no crop (show full camera image)
    python tools/viz_closest_occluded.py \\
        --pkl data/infos/nuscenes_infos_val.pkl \\
        --max-dist 80 --no-crop
"""

import argparse
import os
import pickle
import numpy as np


IMG_W, IMG_H = 1600, 900

CAMERA_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]

_C_OCC_RGB   = (230,  81,   0)   # orange-red : valid_flag=False
_C_SYNTH_RGB = (123,  31, 162)   # purple     : interpolated / extrapolated


# ===========================================================================
# Geometry
# ===========================================================================

def _box_corners_3d(cx, cy, cz, length, width, height, yaw):
    """Return (8, 3) corners of an oriented 3-D box in the lidar frame.

    Corner order: 0-3 top face (+z), 4-7 bottom face (-z),
    both wound counter-clockwise from the (+l, +w) corner.
    """
    l, w, h = length / 2.0, width / 2.0, height / 2.0
    local = np.array([
        [ l,  w,  h], [-l,  w,  h], [-l, -w,  h], [ l, -w,  h],
        [ l,  w, -h], [-l,  w, -h], [-l, -w, -h], [ l, -w, -h],
    ], dtype=np.float64)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    Rz   = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return local @ Rz.T + np.array([cx, cy, cz])


_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # top face
    (4, 5), (5, 6), (6, 7), (7, 4),   # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def _project_to_cam(pts_lidar, M, T, K):
    """Project (N, 3) lidar points into camera pixel coordinates.

    Storage convention (from nuscenes_converter.py):
      ``p_lidar = p_cam @ M + T``   (M = sensor2lidar_rotation)
    So the inverse is:
      ``p_cam = (p_lidar - T) @ M.T``

    Returns
    -------
    uv    : (N, 2)  pixel coordinates (may be outside image bounds)
    depth : (N,)    camera z-depth (positive = in front of camera)
    """
    M = np.asarray(M, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    pts_cam = (pts_lidar - T) @ M            # (N, 3)
    depth   = pts_cam[:, 2].copy()
    z_safe  = np.where(depth > 0, depth, 1e-6)
    proj    = pts_cam @ K.T                  # (N, 3)  [u·z, v·z, z]
    uv      = proj[:, :2] / z_safe[:, None]
    return uv, depth


def _select_best_camera(box_lidar, cams):
    """Return (cam_name, uv_8corners, depth_8corners) for the best camera view.

    Selection criteria (lexicographic):
      1. Maximum number of corners that project inside the image with depth>0.
      2. Minimum distance from projected box centre to image centre.

    Returns None if no camera has the box centre in front of it.
    """
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
# Data collection
# ===========================================================================

def find_closest_occluded(infos, include_synthetic=False, classes=None,
                           min_dist=2.0, max_dist=50.0):
    """Collect occluded GT objects, sorted ascending by distance to ego.

    Returns a list of dicts with keys:
      scene_token, global_idx, inst_ind, class_name, dist_ego,
      box_lidar (7,), is_interp, is_extrap, valid_flag, cams_info, token,
      last_visible_box (7,) or None, last_visible_cams or None
    """
    from collections import defaultdict
    has_synthetic = 'is_interpolated' in infos[0]

    # Process scene by scene in temporal order so we can track last visible frame
    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])

    entries = []
    for sc_tok, gidxs in scene_map.items():
        last_visible = {}  # inst_ind -> (box_lidar, cams_info) from most recent visible frame

        for gi in gidxs:
            info  = infos[gi]
            boxes = info.get('gt_boxes')
            names = info.get('gt_names')
            if boxes is None or len(boxes) == 0:
                continue
            if 'cams' not in info or not info['cams']:
                continue

            n     = len(boxes)
            valid = info.get('valid_flag', np.ones(n, dtype=bool))
            is_i  = info['is_interpolated'] if has_synthetic else np.zeros(n, dtype=bool)
            is_e  = info['is_extrapolated'] if has_synthetic else np.zeros(n, dtype=bool)

            for bi in range(n):
                inst_ind = int(info['instance_inds'][bi])
                is_occ   = not bool(valid[bi])
                is_synth = bool(is_i[bi]) or bool(is_e[bi])

                # Update last-visible record before processing occlusion
                if bool(valid[bi]) and not is_synth:
                    last_visible[inst_ind] = (boxes[bi].copy(), info['cams'])

                if not is_occ and not (include_synthetic and is_synth):
                    continue
                if classes is not None and names[bi] not in classes:
                    continue

                dist = float(np.hypot(float(boxes[bi, 0]), float(boxes[bi, 1])))
                if dist < min_dist or dist > max_dist:
                    continue

                prev = last_visible.get(inst_ind)
                entries.append(dict(
                    scene_token        = sc_tok,
                    global_idx         = gi,
                    inst_ind           = inst_ind,
                    class_name         = str(names[bi]),
                    dist_ego           = dist,
                    box_lidar          = boxes[bi].copy(),
                    is_interp          = bool(is_i[bi]),
                    is_extrap          = bool(is_e[bi]),
                    valid_flag         = bool(valid[bi]),
                    cams_info          = info['cams'],
                    token              = info.get('token', ''),
                    last_visible_box   = prev[0] if prev else None,
                    last_visible_cams  = prev[1] if prev else None,
                ))

    entries.sort(key=lambda x: x['dist_ego'])
    return entries


# ===========================================================================
# Rendering helpers
# ===========================================================================

def _draw_3d_box(draw, uv, depth, color, line_width=3):
    """Draw the 12 edges of a projected 3-D box onto a PIL ImageDraw canvas."""
    visible = {i for i in range(8) if depth[i] > 0}
    for i, j in _BOX_EDGES:
        if i in visible and j in visible:
            draw.line(
                [(int(round(uv[i, 0])), int(round(uv[i, 1]))),
                 (int(round(uv[j, 0])), int(round(uv[j, 1])))],
                fill=color, width=line_width,
            )


def _draw_2d_bbox(draw, uv, depth, color, line_width=3):
    """Draw an axis-aligned 2-D bounding rectangle around visible projected corners.

    This is a reliable fallback when some corners are behind the camera,
    drawn on top of (or instead of) the 3-D edges.
    """
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


def _crop_around_box(img, uv, depth, pad=100):
    """Return a padded crop around the projected box corners (depth > 0)."""
    vis = uv[depth > 0]
    if len(vis) == 0:
        return img

    u_min = int(vis[:, 0].min()) - pad
    u_max = int(vis[:, 0].max()) + pad
    v_min = int(vis[:, 1].min()) - pad
    v_max = int(vis[:, 1].max()) + pad

    # Enforce minimum size
    if u_max - u_min < 2 * pad:
        mid_u = (u_min + u_max) // 2
        u_min, u_max = mid_u - pad, mid_u + pad
    if v_max - v_min < 2 * pad:
        mid_v = (v_min + v_max) // 2
        v_min, v_max = mid_v - pad, mid_v + pad

    # Clamp to actual image dimensions (not hardcoded IMG_W/H)
    w, h  = img.size
    u_min = max(0, u_min)
    u_max = min(w, u_max)
    v_min = max(0, v_min)
    v_max = min(h, v_max)

    if u_min >= u_max or v_min >= v_max:
        return img

    return img.crop((u_min, v_min, u_max, v_max))


# ===========================================================================
# Main rendering pipeline
# ===========================================================================

def render_occluded_views(entries, top_n=12, output_dir='vis/occluded_closest',
                           dataroot='data/nuscenes'):
    """Render annotated camera images for the top-N closest occluded objects.

    Each object is shown as a pair: [current occluded frame | last visible frame].
    Saves:
      output_dir/cameras/occ_NNN_<scene8>_<CAM>.png  – annotated occluded frame
      output_dir/cameras/vis_NNN_<scene8>_<CAM>.png  – annotated last-visible frame
      output_dir/closest_occluded.png                 – summary grid (pairs)
    """
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
        """Project box, save raw + annotated images. Returns (ann_path, raw_path, cam_name) or None."""
        if box_lidar is None or cams_info is None:
            return None
        result = _select_best_camera(box_lidar, cams_info)
        if result is None:
            return None
        cam_name, uv, depth = result
        ci       = cams_info[cam_name]
        rel_path = ci['data_path']
        img_path = rel_path if os.path.isabs(rel_path) \
                   else os.path.join(dataroot, rel_path)
        if not os.path.isfile(img_path):
            img_path = rel_path
        if not os.path.isfile(img_path):
            return None
        img = Image.open(img_path).convert('RGB')

        # Save raw (no annotation)
        raw_path = os.path.join(cam_dir, f'{stem}_raw_{cam_name}.png')
        img.save(raw_path)

        # Annotate and save
        img_ann = img.copy()
        draw = ImageDraw.Draw(img_ann)
        _draw_3d_box(draw, uv, depth, color, line_width=3)
        _draw_2d_bbox(draw, uv, depth, color, line_width=2)
        draw.rectangle([0, 0, img_ann.width, 26], fill=(0, 0, 0))
        draw.text((5, 5), label, fill=(255, 255, 255), font=font)
        ann_path = os.path.join(cam_dir, f'{stem}_{cam_name}.png')
        img_ann.save(ann_path)

        return ann_path, raw_path, cam_name

    # Each item: (rank, occ_path, occ_cam, vis_path, vis_cam, entry)
    saved_items = []

    for rank, entry in enumerate(entries):
        is_synth = entry['is_interp'] or entry['is_extrap']
        color    = _C_SYNTH_RGB if is_synth else _C_OCC_RGB
        occ_tag  = ('interp' if entry['is_interp'] else
                    'extrap' if entry['is_extrap'] else 'occluded')
        stem_occ = f'occ_{rank+1:03d}_{entry["scene_token"][:8]}'
        stem_vis = f'vis_{rank+1:03d}_{entry["scene_token"][:8]}'

        occ_label = (f"#{rank+1} OCCLUDED  {entry['class_name']}  "
                     f"{entry['dist_ego']:.1f} m  [{occ_tag}]")
        vis_label = f"#{rank+1} LAST VISIBLE  {entry['class_name']}"

        occ_result = _load_annotated(
            entry['box_lidar'], entry['cams_info'], color, occ_label, stem_occ)
        vis_result = _load_annotated(
            entry['last_visible_box'], entry['last_visible_cams'],
            (0, 120, 40), vis_label, stem_vis)

        if occ_result is None:
            print(f'  [{rank+1:3d}]  {entry["class_name"]:<20s}  '
                  f'{entry["dist_ego"]:5.1f} m  — no valid camera projection, skipped')
            continue

        occ_path, occ_raw_path, occ_cam = occ_result
        if vis_result is not None:
            vis_path, vis_raw_path, vis_cam = vis_result
        else:
            vis_path = vis_raw_path = vis_cam = None
        saved_items.append((rank, occ_path, occ_raw_path, occ_cam,
                             vis_path, vis_raw_path, vis_cam, entry))
        print(f'  [{rank+1:3d}]  {entry["class_name"]:<20s}  '
              f'{entry["dist_ego"]:5.1f} m  [{occ_tag}]  '
              f'occ={occ_cam}  vis={vis_cam or "none"}')

    if not saved_items:
        print('No images were rendered (check --dataroot and image paths).')
        return

    # ── Summary grid: 3 objects per row, 2 columns per object (occ | visible) ─
    n              = len(saved_items)
    objects_per_row = 3
    ncols          = objects_per_row * 2
    nrows          = (n + objects_per_row - 1) // objects_per_row
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.5 * nrows),
                              facecolor='white')
    axes = np.array(axes).reshape(nrows, ncols)

    for k, (rank, occ_path, occ_raw_path, occ_cam,
             vis_path, vis_raw_path, vis_cam, e) in enumerate(saved_items):
        row     = k // objects_per_row
        col_occ = (k % objects_per_row) * 2
        col_vis = col_occ + 1

        is_synth    = e['is_interp'] or e['is_extrap']
        title_color = '#7b1fa2' if is_synth else '#e65100'
        occ_tag     = ('interp' if e['is_interp'] else
                       'extrap' if e['is_extrap'] else 'occluded')

        # Occluded panel (annotated)
        ax_occ = axes[row, col_occ]
        ax_occ.imshow(plt.imread(occ_path))
        ax_occ.set_axis_off()
        ax_occ.set_title(
            f"#{rank+1} {e['class_name']} {e['dist_ego']:.1f}m [{occ_tag}]\n"
            f"scene {e['scene_token'][:8]}…  {occ_cam}",
            color=title_color, fontsize=7, pad=2)

        # Last-visible panel (annotated)
        ax_vis = axes[row, col_vis]
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

    # Hide unused axes
    for k in range(n, nrows * objects_per_row):
        row     = k // objects_per_row
        col_occ = (k % objects_per_row) * 2
        axes[row, col_occ].set_visible(False)
        axes[row, col_occ + 1].set_visible(False)

    fig.suptitle(f'Top {n} closest occluded GT objects  (left: occluded | right: last visible)',
                 fontsize=10, color='#111111', y=1.01)
    fig.subplots_adjust(hspace=0.4, wspace=0.05)

    grid_out = os.path.join(output_dir, 'closest_occluded.png')
    plt.savefig(grid_out, dpi=130, bbox_inches='tight', facecolor='white')
    print(f'\nGrid saved → {grid_out}')
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pkl', default='data/infos/nuscenes_infos_val.pkl',
                    help='Path to nuScenes info pkl file')
    ap.add_argument('--include-synthetic', action='store_true',
                    help='Also include interpolated/extrapolated boxes '
                         '(requires an _occ pkl with is_interpolated flags)')
    ap.add_argument('--classes', nargs='+', default=None, metavar='CLS',
                    help='Restrict to these GT class names (default: all). '
                         'E.g. --classes car truck bus trailer')
    ap.add_argument('--min-dist', type=float, default=2.0, metavar='M',
                    help='Minimum ego-to-object distance to consider (default: 2 m)')
    ap.add_argument('--max-dist', type=float, default=50.0, metavar='M',
                    help='Maximum ego-to-object distance to consider (default: 50 m)')
    ap.add_argument('--top-n', type=int, default=12, metavar='N',
                    help='Number of closest occluded objects to render (default: 12)')
    ap.add_argument('--output-dir', default='vis/occluded_closest', metavar='DIR',
                    help='Directory to save images (default: vis/occluded_closest)')
    ap.add_argument('--dataroot', default='data/nuscenes', metavar='DIR',
                    help='nuScenes dataset root for resolving image paths '
                         '(default: data/nuscenes)')
    args = ap.parse_args()

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos    = data['infos']
    n_scenes = len({i['scene_token'] for i in infos})
    print(f'  {len(infos)} samples across {n_scenes} scenes')
    if args.classes:
        print(f'  Class filter : {args.classes}')
    if args.include_synthetic:
        has_flags = 'is_interpolated' in infos[0]
        if not has_flags:
            print('  WARNING: --include-synthetic set but pkl has no '
                  'is_interpolated/is_extrapolated flags — ignoring.')

    print(f'\nSearching for occluded objects between {args.min_dist:.0f}–{args.max_dist:.0f} m ...')
    occluded = find_closest_occluded(
        infos,
        include_synthetic=args.include_synthetic,
        classes=args.classes,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
    )

    if not occluded:
        print('No occluded objects found.')
        return

    n_invalid = sum(1 for e in occluded if not e['valid_flag'])
    n_interp  = sum(1 for e in occluded if e['is_interp'])
    n_extrap  = sum(1 for e in occluded if e['is_extrap'])
    print(f'  Found {len(occluded)} occluded object instances.')
    print(f'    invalid annotations (valid_flag=False) : {n_invalid}')
    if args.include_synthetic:
        print(f'    interpolated (synthetic)              : {n_interp}')
        print(f'    extrapolated (synthetic)              : {n_extrap}')

    print(f'\n  Closest {min(10, len(occluded))} occluded objects:')
    for e in occluded[:10]:
        occ_tag = ('interp' if e['is_interp'] else
                   'extrap' if e['is_extrap'] else 'occluded')
        print(f'    {e["class_name"]:<22s}  {e["dist_ego"]:5.2f} m  '
              f'[{occ_tag}]  scene {e["scene_token"][:8]}…')

    print(f'\nRendering top {args.top_n} ...')
    render_occluded_views(
        occluded,
        top_n      = args.top_n,
        output_dir = args.output_dir,
        dataroot   = args.dataroot,
    )


if __name__ == '__main__':
    main()
