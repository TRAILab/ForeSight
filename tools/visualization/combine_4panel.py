"""Compose 4-panel figure: top row = cam_front (base | ours), bottom row = BEV (base | ours)."""
import argparse, glob, os, cv2


def crop_cam_front(cam_img):
    """The cam_plan_pred_only image is a 2x3 grid; CAM_FRONT is row 0, col 1."""
    h, w = cam_img.shape[:2]
    cell_h, cell_w = h // 2, w // 3
    return cam_img[0:cell_h, cell_w:2 * cell_w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-dir', required=True)
    ap.add_argument('--ours-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bev_glob = sorted(glob.glob(os.path.join(args.baseline_dir, 'bev_plan_pred_only', '*.jpg')))
    if not bev_glob:
        raise SystemExit(f'no images in {args.baseline_dir}/bev_plan_pred_only')

    for path in bev_glob:
        idx = os.path.basename(path).split('.')[0]
        b_bev = cv2.imread(os.path.join(args.baseline_dir, 'bev_plan_pred_only', f'{idx}.jpg'))
        o_bev = cv2.imread(os.path.join(args.ours_dir, 'bev_plan_pred_only', f'{idx}.jpg'))
        b_cam = cv2.imread(os.path.join(args.baseline_dir, 'cam_plan_pred_only', f'{idx}.jpg'))
        o_cam = cv2.imread(os.path.join(args.ours_dir, 'cam_plan_pred_only', f'{idx}.jpg'))
        if any(im is None for im in [b_bev, o_bev, b_cam, o_cam]):
            print(f'skip {idx}: missing image')
            continue

        b_front = crop_cam_front(b_cam)
        o_front = crop_cam_front(o_cam)

        # Resize cams to match BEV width so columns line up.
        bev_w = b_bev.shape[1]
        scale = bev_w / b_front.shape[1]
        cam_h = int(round(b_front.shape[0] * scale))
        b_front_r = cv2.resize(b_front, (bev_w, cam_h))
        o_front_r = cv2.resize(o_front, (bev_w, cam_h))

        # Resize ours BEV to match baseline BEV (should already match).
        if o_bev.shape != b_bev.shape:
            o_bev = cv2.resize(o_bev, (b_bev.shape[1], b_bev.shape[0]))

        top = cv2.hconcat([b_front_r, o_front_r])
        bot = cv2.hconcat([b_bev, o_bev])
        # Annotate column headers.
        header_h = 60
        header = 255 * (top[:1].astype('uint8') * 0 + 1)  # white strip
        header = header.repeat(header_h, axis=0)[:, :, :].copy()
        if header.ndim == 2:
            header = cv2.cvtColor(header, cv2.COLOR_GRAY2BGR)
        cv2.putText(header, 'SparseDrive baseline', (40, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(header, 'Ours (K3408316)', (bev_w + 40, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

        out = cv2.vconcat([header, top, bot])
        out_path = os.path.join(args.out_dir, f'{idx}.jpg')
        cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('wrote to', args.out_dir)


if __name__ == '__main__':
    main()
