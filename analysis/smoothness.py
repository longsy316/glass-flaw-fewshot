"""Background roughness of each glass dataset: mean optical-contrast response |R| = |I - B| / (B + floor) over pixels
outside every ground-truth box (i.e. flaw-free glass), with the same median background as the optical contrast
encoding. Smooth glass gives a small value; textured coatings, print and edges give a large one."""
import json
import sys

import cv2
import numpy as np

from glassdet.data import optical_contrast

SETS = {
    'SSGD': ('data/ssgd_coco/all_val.json', 'data/ssgd/SSGD'),
    'GlassCover': ('data/glasscover_coco/val.json', 'data/glasscover/GlassCover_datset'),
    'MSD': ('data/msd_coco/test.json', 'data/msd/MSD'),
    'AGDD': ('data/agdd_coco/k10_s0_test_fwd.json', 'data/agdd/data'),
    'bottle': ('data/mvtec_bottle/k9_s0_test.json', 'data/mvtec_bottle'),
    'vial': ('data/vial_coco/k10_s0_test.json', 'data/vial/test_public'),
    'CCB': ('data/ccb_tiles/val.json', 'data/ccb_tiles'),
}
out = {}
for name, (ann, root) in SETS.items():
    d = json.load(open(ann))
    by = {}
    for a in d['annotations']:
        by.setdefault(a['image_id'], []).append(a['bbox'])
    vals = []
    for im in d['images'][:200]:
        img = cv2.imread(f"{root}/{im['file_name']}", cv2.IMREAD_COLOR)
        o = optical_contrast(img).astype(np.float32) / 255.0
        resp = o[..., 1] + o[..., 0]  # |R| * gain, clipped
        keep = np.ones(resp.shape, bool)
        for x, y, w, h in by.get(im['id'], []):
            m = 0.25 * max(w, h) + 4
            keep[int(max(0, y - m)):int(y + h + m), int(max(0, x - m)):int(x + w + m)] = False
        if keep.any():
            vals.append(float(resp[keep].mean()))
    out[name] = {'roughness': float(np.mean(vals)), 'n_images': len(vals)}
    print(f'{name:11s} background roughness {100 * out[name]["roughness"]:.2f} (x1e-2, n={len(vals)})', flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'runs/smoothness.json', 'w'), indent=1)
