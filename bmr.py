"""BMR: box-mask mutual refinement at inference (no retraining).

  1. box -> mask : DINO boxes prompt the SAM decoder (shared SAM embedding) -> mask + predicted mask IoU q.
  2. mask -> box : tight box of the mask, mapped to the dataset's annotation style by a per-class side-margin
                   calibration fitted on TRAINING images (GT boxes prompt SAM; margins = GT - tight, relative to
                   tight w/h, robust median).  Loose-box datasets (SSGD) get margins > 0, mask-derived GT ~ 0.
  3. fuse        : box <- (1-b) box + b cal_box,  b = b_max * clip((q - t0)/(1 - t0), 0, 1)  (trust good masks only)
  4. iterate     : the fused box re-prompts SAM (K rounds), so box and mask converge together.
  5. rescore     : s <- s * q^a.
All configurations reuse one DINO pass and one SAM image embedding per image."""
import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from glassdet.data import CocoDet, collate
from glassdet.evaluate import coco_eval, decode, gt_strata, strata_recall
from glassdet.model import GlassDet

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--sfa', type=int, default=1)
ap.add_argument('--sqf', type=int, default=0)
ap.add_argument('--train', required=True, help='training json used only for the margin calibration')
ap.add_argument('--val', required=True)
ap.add_argument('--img_root', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--short', type=int, default=800)
ap.add_argument('--n_cal', type=int, default=150)
ap.add_argument('--out', required=True)
args = ap.parse_args()
dev = 'cuda'
CL = args.classes.split(',')
os.makedirs(args.out, exist_ok=True)

m = GlassDet('IDEA-Research/grounding-dino-tiny', 'facebook/sam-vit-base', CL, sfa=bool(args.sfa),
             sqf=bool(args.sqf)).to(dev).eval()
sd = torch.load(args.ckpt, map_location='cpu')
if '__lang_free__' in sd:
    m.convert_language_free(sd['static_text'].to(dev)); sd.pop('__lang_free__')
m.load_state_dict(sd, strict=False)
AMP = torch.autocast('cuda', dtype=torch.bfloat16)


@torch.no_grad()
def sam_boxes(full, hw, boxes, H, W):
    """boxes (N,4) xyxy in original pixels -> (tight boxes (N,4) with nan if empty, q (N,))."""
    h, w = hw
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    sc = torch.tensor([w / W, h / H, w / W, h / H], device=dev)
    b = torch.as_tensor(boxes, dtype=torch.float32, device=dev) * sc
    tight, qs = [], []
    for j in range(0, len(b), 25):
        with AMP:
            o = m.sam(image_embeddings=full, input_boxes=b[None, j:j + 25], multimask_output=False)
        qs.append(o.iou_scores[0, :, 0].float().clamp(0, 1))
        lm = F.interpolate(o.pred_masks[0, :, :1].float(), size=(1024, 1024), mode='bilinear')[:, 0, :h, :w] > 0
        cols, rows = lm.any(1), lm.any(2)  # (n,w), (n,h)
        for k in range(lm.shape[0]):
            xs, ys = torch.nonzero(cols[k]).flatten(), torch.nonzero(rows[k]).flatten()
            if len(xs) == 0:
                tight.append([np.nan] * 4)
            else:
                tight.append([xs[0].item() * W / w, ys[0].item() * H / h, (xs[-1].item() + 1) * W / w,
                              (ys[-1].item() + 1) * H / h])
    return np.array(tight, np.float32), torch.cat(qs).cpu().numpy()


def encode(pix):
    with torch.no_grad(), AMP:
        full, sv, hw = m.sam_encode(pix)
        out = m(pix, sv if (m.sfa is not None or m.sqf is not None) else None)
    return full, hw, {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}


# ------------------------------------------------------------------ 1. margin calibration on training images
cal_ds = CocoDet(args.train, args.img_root, short=args.short)
gtj = json.load(open(args.train))
by_img = {}
for a in gtj['annotations']:
    by_img.setdefault(a['image_id'], []).append(a)
marg = {c: [] for c in range(len(CL))}
for i in range(min(len(cal_ds), args.n_cal)):
    pix, tg = cal_ds[i]
    anns = by_img.get(tg['image_id'], [])
    if not anns:
        continue
    H, W = tg['orig_size']
    with torch.no_grad(), AMP:
        full, _, hw = m.sam_encode(pix[None].to(dev))
    g = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns],
                 np.float32)
    t, q = sam_boxes(full, hw, g, H, W)
    for a, gb, tb, qq in zip(anns, g, t, q):
        if np.isnan(tb[0]) or qq < 0.5:
            continue
        tw, th = max(tb[2] - tb[0], 1.0), max(tb[3] - tb[1], 1.0)
        marg[a['category_id'] - 1].append([(tb[0] - gb[0]) / tw, (tb[1] - gb[1]) / th, (gb[2] - tb[2]) / tw,
                                           (gb[3] - tb[3]) / th])
allm = np.concatenate([np.array(v) for v in marg.values() if len(v)], 0)
glob_med = np.median(allm, 0)
MARGIN = {c: (np.median(np.array(v), 0) if len(v) >= 5 else glob_med) for c, v in marg.items()}
print('calibrated side margins (l,t,r,b as fraction of tight w/h):',
      {CL[c]: np.round(v, 3).tolist() for c, v in MARGIN.items()}, flush=True)


def calibrate(tb, cls):
    l, t, r, b = MARGIN[int(cls)]
    tw, th = max(tb[2] - tb[0], 1.0), max(tb[3] - tb[1], 1.0)
    return np.array([tb[0] - l * tw, tb[1] - t * th, tb[2] + r * tw, tb[3] + b * th], np.float32)


# ------------------------------------------------------------------ 2. inference with all BMR configurations
CFGS = [dict(K=0, cal=0, bmax=0.0, t0=0.0)]
for K, cal, bmax, t0 in itertools.product((1, 2), (0, 1), (0.5, 1.0), (0.5, 0.8)):
    CFGS.append(dict(K=K, cal=cal, bmax=bmax, t0=t0))
results = {i: [] for i in range(len(CFGS))}  # cfg -> list of (iid, score, cls, box, q)

ds = CocoDet(args.val, args.img_root, short=args.short)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)
for pix, tg in dl:
    H, W = tg[0]['orig_size']
    iid = tg[0]['image_id']
    full, hw, out = encode(pix.to(dev))
    sc, cl, xyxy = decode(out, m.label_map, [(H, W)])[0]
    keep = sc >= 0.01
    sc, cl, xyxy = sc[keep], cl[keep], xyxy[keep]
    t1, q1 = sam_boxes(full, hw, xyxy, H, W)  # first round is shared by every configuration
    for ci, c in enumerate(CFGS):
        box, q = xyxy.copy(), q1.copy()
        tight = t1
        for k in range(c['K']):
            if k > 0:
                tight, q = sam_boxes(full, hw, box, H, W)
            for j in range(len(box)):
                if np.isnan(tight[j][0]):
                    continue
                target = calibrate(tight[j], cl[j]) if c['cal'] else tight[j]
                beta = c['bmax'] * np.clip((q[j] - c['t0']) / (1 - c['t0']), 0, 1)
                box[j] = (1 - beta) * box[j] + beta * target
        for j in range(len(box)):
            results[ci].append((iid, float(sc[j]), int(cl[j]), box[j].copy(), float(q[j])))

strata = gt_strata(args.val, args.img_root, args.val.replace('.json', '_strata.json'))
summary = {}
for ci, c in enumerate(CFGS):
    for a in (0.0, 0.5):
        dets = [{'image_id': int(i), 'category_id': k + 1, 'score': s * (q ** a),
                 'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]}
                for i, s, k, b, q in results[ci]]
        r = coco_eval(args.val, dets, CL)
        r.update(strata_recall(args.val, dets, strata))
        key = f"K{c['K']}_cal{c['cal']}_b{c['bmax']}_t{c['t0']}_a{a}"
        summary[key] = r
        print(f'{key:28s} AP50 {100 * r["AP50"]:5.1f} mAP {100 * r["mAP"]:5.1f} APs {100 * r["APs"]:4.1f} '
              f'F1 {100 * r["F1"]:4.1f} R_small {100 * r["R_small"]:4.1f} R_elong {100 * r["R_elong"]:4.1f} '
              f'R_weak {100 * r["R_weak"]:4.1f}', flush=True)
summary['_margins'] = {CL[c]: v.tolist() for c, v in MARGIN.items()}
json.dump(summary, open(f'{args.out}/eval_final.json', 'w'), indent=1)
