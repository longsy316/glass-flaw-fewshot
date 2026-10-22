"""Flaw metrology from predicted masks vs. ground-truth masks (MSD, MVTec bottle, MVTec AD 2 vial).

For every GT flaw the best-overlapping detection (score >= thr) is taken; its SAM mask (DINO box prompt) and,
as the detector-only alternative, its box are converted into
  length L : max(skeleton length, long side of the min-area rectangle)  [px]
  area   A : pixel count (box: w*h)                                       [px^2]
  width  W : A / L                                                        [px]
and compared with the same quantities of the GT mask. Through Griffith's relation sigma_f ~ K_IC / sqrt(pi a), with
the flaw size a proportional to L, the predicted/true strength ratio is sqrt(L_gt / L_pred); it is independent of the
unknown pixel size, so no physical scale is assumed."""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

from glassdet.data import CocoDet, collate
from glassdet.evaluate import decode
from glassdet.model import GlassDet

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--val', required=True)
ap.add_argument('--img_root', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--thr', type=float, default=0.3)
ap.add_argument('--out', required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = 'cuda'
m = GlassDet('IDEA-Research/grounding-dino-tiny', 'facebook/sam-vit-base', args.classes.split(','), sfa=False).to(dev).eval()
m.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
AMP = torch.autocast('cuda', dtype=torch.bfloat16)
gt = COCO(args.val)


def skel_len(mk):
    sk = skeletonize(mk)
    if sk.sum() == 0:
        return 0.0
    p = np.pad(sk, 1).astype(np.uint8)
    straight = (p[1:-1, 1:-1] & p[1:-1, 2:]).sum() + (p[1:-1, 1:-1] & p[2:, 1:-1]).sum()
    diag = (p[1:-1, 1:-1] & p[2:, 2:]).sum() + (p[1:-1, 1:-1] & p[2:, :-2]).sum()
    return float(straight + np.sqrt(2) * diag)


def feret(pts):
    """Maximum Feret diameter = largest distance between convex-hull vertices (robust to boundary noise)."""
    h = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    if len(h) < 2:
        return 1.0
    d = np.sqrt(((h[:, None, :] - h[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def geom(mk):
    A = float(mk.sum())
    if A == 0:
        return None
    cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack(cs).reshape(-1, 2)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    sk = skel_len(mk)
    return {'skel': sk, 'feret': max(feret(pts), 1.0), 'L_old': max(sk, max(rw, rh), 1.0), 'A': A,
            'aspect': max(rw, rh) / max(min(rw, rh), 1.0)}


def flaw_size(g, ref_aspect):
    """Griffith flaw size: centre-line length for elongated flaws (aspect >= 3), max Feret diameter otherwise.
    The definition is chosen from the GT shape so that prediction and GT are measured identically."""
    return g['skel'] if ref_aspect >= 3 else g['feret']


rows, images = [], []
ds = CocoDet(args.val, args.img_root)
for pix, tg in DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate):
    H, W = tg[0]['orig_size']
    anns = gt.loadAnns(gt.getAnnIds(imgIds=tg[0]['image_id']))
    if not anns:
        continue
    with torch.no_grad(), AMP:
        full, _, hw = m.sam_encode(pix.to(dev))
        out = m(pix.to(dev), None)
    out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
    sc, cl, bx = decode(out, m.label_map, [(H, W)])[0]
    keep = sc >= args.thr
    sc, cl, bx = sc[keep], cl[keep], bx[keep]
    masks = np.zeros((0, H, W), bool)
    if len(bx):
        h, w = hw
        b = torch.tensor(bx, device=dev) * torch.tensor([w / W, h / H, w / W, h / H], device=dev)
        ms = []
        for j in range(0, len(b), 20):
            with torch.no_grad(), AMP:
                o = m.sam(image_embeddings=full, input_boxes=b[None, j:j + 20].float(), multimask_output=False)
            lm = F.interpolate(o.pred_masks[0, :, :1].float(), size=(1024, 1024), mode='bilinear')[:, :, :h, :w]
            ms.append((F.interpolate(lm, size=(H, W), mode='bilinear')[:, 0] > 0).cpu().numpy())
        masks = np.concatenate(ms)
    for a in anns:
        g = gt.annToMask(a).astype(bool)
        gg = geom(g)
        if gg is None:
            continue
        r = {'cat': a['category_id'], 'gt': gg, 'pred': None, 'box': None}
        if len(masks):
            ious = [(np.logical_and(g, mk).sum() / max(np.logical_or(g, mk).sum(), 1)) for mk in masks]
            j = int(np.argmax(ious))
            if ious[j] > 0.1:
                r['pred'] = geom(masks[j])
                x0, y0, x1, y1 = bx[j]
                bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
                r['box'] = {'L': float(np.hypot(bw, bh)), 'A': float(bw * bh)}
                r['iou'] = float(ious[j])
        if r['pred'] is not None:
            a_ref = gg['aspect']
            r['gt']['L'] = flaw_size(gg, a_ref)
            r['pred']['L'] = flaw_size(r['pred'], a_ref)
            r['gt']['W'], r['pred']['W'] = gg['A'] / r['gt']['L'], r['pred']['A'] / r['pred']['L']
            r['box']['W'] = r['box']['A'] / r['box']['L']
        rows.append(r)
    # image-level record for weakest-link analysis: ALL predictions count (false positives included),
    # each flaw sized with its own shape class
    gts = [(geom(gt.annToMask(a).astype(bool)), a['category_id']) for a in anns]
    preds = [(geom(mk), int(c) + 1) for mk, c in zip(masks, cl)]
    images.append({'image_id': int(tg[0]['image_id']),
                   'gt': [flaw_size(g, g['aspect']) for g, _ in gts if g],
                   'gt_cat': [c for g, c in gts if g],
                   'pred': [flaw_size(p, p['aspect']) for p, _ in preds if p],
                   'pred_cat': [c for p, c in preds if p],
                   'box': [float(np.hypot(max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0))) for b in bx],
                   'box_cat': [int(c) + 1 for c in cl]})

json.dump(rows, open(f'{args.out}/rows.json', 'w'))
json.dump(images, open(f'{args.out}/images.json', 'w'))


def summary(sub):
    hit = [r for r in sub if r['pred']]
    s = {'n_gt': len(sub), 'detected': len(hit) / max(len(sub), 1)}
    e_old = np.array([abs(r['pred']['L_old'] - r['gt']['L_old']) / r['gt']['L_old'] for r in hit])
    s['pred_Lold_MAPE'] = float(np.median(e_old)) if len(e_old) else float('nan')  # skeleton-for-all definition
    for src in ('pred', 'box'):
        for k in ('L', 'A', 'W'):
            e = np.array([abs(r[src][k] - r['gt'][k]) / r['gt'][k] for r in hit])
            s[f'{src}_{k}_MAPE'] = float(np.median(e)) if len(e) else float('nan')
        ratio = np.array([np.sqrt(r['gt']['L'] / r[src]['L']) for r in hit])
        s[f'{src}_sigma_within10'] = float(np.mean(np.abs(ratio - 1) <= 0.10)) if len(ratio) else float('nan')
        s[f'{src}_sigma_within20'] = float(np.mean(np.abs(ratio - 1) <= 0.20)) if len(ratio) else float('nan')
        s[f'{src}_sigma_median_err'] = float(np.median(np.abs(ratio - 1))) if len(ratio) else float('nan')
    return s


res = {'all': summary(rows), 'elongated': summary([r for r in rows if r['gt']['aspect'] >= 3]),
       'compact': summary([r for r in rows if r['gt']['aspect'] < 3])}
res['AP50'] = res['all']['detected']  # queue completion marker
json.dump(res, open(f'{args.out}/eval_final.json', 'w'), indent=1)
for k in ('all', 'elongated', 'compact'):
    s = res[k]
    print(f"{k:9s} n={s['n_gt']:4d} det={100 * s['detected']:5.1f}% | median rel. error mask/box: "
          f"L {100 * s['pred_L_MAPE']:5.1f}/{100 * s['box_L_MAPE']:5.1f}  A {100 * s['pred_A_MAPE']:5.1f}/{100 * s['box_A_MAPE']:5.1f}  "
          f"W {100 * s['pred_W_MAPE']:5.1f}/{100 * s['box_W_MAPE']:5.1f} | strength within 10%: "
          f"{100 * s['pred_sigma_within10']:5.1f}/{100 * s['box_sigma_within10']:5.1f}", flush=True)
