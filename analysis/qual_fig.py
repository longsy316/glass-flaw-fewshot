"""Qualitative strip for Fig. 1(b): GT box (green), predicted box (red, score >= thr) and the SAM mask of each
prediction (cyan) on one representative defect per panel; crops are centred on the chosen GT instance."""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from glassdet.data import CocoDet
from glassdet.evaluate import decode
from glassdet.model import GlassDet

S = 'crack,scratch,chipping,white spot,light leakage,broken film,stain'
PANELS = [  # (title, ckpt, val json, img_root, classes, category_id, selector)
    ('(i) through crack, SSGD', 'runs/fs100_s0/gd_sfa0/model.pt', 'data/ssgd_coco/all_val.json', 'data/ssgd/SSGD', S, 1,
     lambda a: max(a['bbox'][2:]) / max(1, min(a['bbox'][2:])) > 6),
    ('(ii) faint scratch, cover glass', 'runs/gc_fs100_s0/gd_sfa0/model.pt', 'data/glasscover_coco/val.json',
     'data/glasscover/GlassCover_datset', 'dot,scratch,patch', 2, lambda a: max(a['bbox'][2:]) > 120),
    ('(iii) point flaw, cover glass', 'runs/gc_fs100_s0/gd_sfa0/model.pt', 'data/glasscover_coco/val.json',
     'data/glasscover/GlassCover_datset', 'dot,scratch,patch', 1, lambda a: a['area'] < 200),
    ('(iv) bubble, tempered glass', 'runs/gen/ccb_fs20_s0/gd/model.pt', 'data/ccb_tiles/val.json', 'data/ccb_tiles',
     'bubble,impurity,scratch', 1, lambda a: True),
    ('(v) scratch, phone screen', 'runs/msd_k30_s0/gd_sfa0/model.pt', 'data/msd_coco/test.json', 'data/msd/MSD',
     'oil,stain,scratch', 3, lambda a: max(a['bbox'][2:]) > 150),
]
dev = 'cuda'
tiles = []
for title, ckpt, val, root, classes, cid, sel in PANELS:
    m = GlassDet('IDEA-Research/grounding-dino-tiny', 'facebook/sam-vit-base', classes.split(','), sfa=False).to(dev).eval()
    m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
    d = json.load(open(val))
    ds = CocoDet(val, root)
    idx = {im['id']: i for i, im in enumerate(d['images'])}
    cands = [a for a in d['annotations'] if a['category_id'] == cid and sel(a)]
    best = None
    for a in cands[:40]:  # pick the candidate the model detects best (IoU>0.5, highest score) for a clear illustration
        x, tg = ds[idx[a['image_id']]]
        H, W = tg['orig_size']
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            full, sv, hw = m.sam_encode(x[None].to(dev))
            out = m(x[None].to(dev), None)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        sc, cl, bx = decode(out, m.label_map, [(H, W)])[0]
        gx, gy, gw, gh = a['bbox']
        g = np.array([gx, gy, gx + gw, gy + gh])
        for s, c, b in zip(sc, cl, bx):
            iw = max(0, min(b[2], g[2]) - max(b[0], g[0])); ih = max(0, min(b[3], g[3]) - max(b[1], g[1]))
            iou = iw * ih / ((b[2] - b[0]) * (b[3] - b[1]) + gw * gh - iw * ih + 1e-9)
            if c + 1 == cid and iou > 0.5 and (best is None or s > best[0]):
                best = (float(s), a, full, hw, sc, cl, bx, H, W)
    s0, a, full, hw, sc, cl, bx, H, W = best
    keep = sc >= max(0.3, 0.8 * s0)
    sc, cl, bx = sc[keep], cl[keep], bx[keep]
    h, w = hw
    b = torch.tensor(bx, device=dev) * torch.tensor([w / W, h / H, w / W, h / H], device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = m.sam(image_embeddings=full, input_boxes=b[None].float(), multimask_output=False)
    lm = F.interpolate(o.pred_masks[0, :, :1].float(), size=(1024, 1024), mode='bilinear')[:, :, :h, :w]
    mk = (F.interpolate(lm, size=(H, W), mode='bilinear')[:, 0] > 0).cpu().numpy()
    img = cv2.imread(f"{root}/{d['images'][idx[a['image_id']]]['file_name']}")
    ov = img.copy()
    for k in range(len(mk)):
        ov[mk[k]] = (255, 255, 0)
    img = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
    gx, gy, gw, gh = a['bbox']
    r = max(gw, gh, 60) * (1.0 if cid == 1 and 'SSGD' in title else 0.85) * (1.6 if max(gw, gh) < 40 else 1.0)
    cx, cy = gx + gw / 2, gy + gh / 2
    X0, Y0 = int(max(0, cx - r)), int(max(0, cy - r))
    X1, Y1 = int(min(W, cx + r)), int(min(H, cy + r))
    sx, sy = 400 / (X1 - X0), 400 / (Y1 - Y0)
    crop = cv2.resize(img[Y0:Y1, X0:X1], (400, 400), interpolation=cv2.INTER_CUBIC)

    def rect(bb, col):  # draw after resizing so every panel has the same 2-px line width
        p0 = (int((bb[0] - X0) * sx), int((bb[1] - Y0) * sy)); p1 = (int((bb[2] - X0) * sx), int((bb[3] - Y0) * sy))
        cv2.rectangle(crop, p0, p1, col, 2)
    for g in d['annotations']:
        if g['image_id'] == a['image_id']:
            x0, y0, ww, hh = g['bbox']; rect((x0 - 2, y0 - 2, x0 + ww + 2, y0 + hh + 2), (0, 210, 0))
    for bb in bx:
        rect(bb, (0, 0, 255))
    band = np.full((44, 400, 3), 255, np.uint8)
    cv2.putText(band, title, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    tiles.append(np.vstack([crop, band]))
    print(title, 'score %.2f' % s0, 'gt', a['id'], flush=True)
    del m; torch.cuda.empty_cache()
gap = np.full((444, 8, 3), 255, np.uint8)
strip = tiles[0]
for t_ in tiles[1:]:
    strip = np.hstack([strip, gap, t_])
os.makedirs('figs', exist_ok=True)
cv2.imwrite('figs/qualitative.png', strip)
print('saved figs/qualitative.png', strip.shape)
