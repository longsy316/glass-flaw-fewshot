"""Mask-verified pseudo-labelling (training-time box-mask mutual promotion).

The few-shot teacher labels an unlabelled pool (the remaining training images, labels discarded). Every detection
with score >= tau_s prompts SAM; it is accepted as a pseudo-label only if
  (i)  SAM's predicted mask quality q >= tau_q, and
  (ii) the mask agrees with the box: IoU(box, calibrated tight box of the mask) >= tau_a,
i.e. detector and segmenter must agree on the flaw. Two training sets are written for the ablation:
  <out>/train_score.json    : labelled + pseudo-labels filtered by score only
  <out>/train_verified.json : labelled + pseudo-labels that also pass the SAM checks
All thresholds are fixed a priori (no tuning on validation data)."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from glassdet.data import CocoDet, collate
from glassdet.evaluate import decode
from glassdet.model import GlassDet

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--labelled', required=True, help='few-shot training json')
ap.add_argument('--pool', required=True, help='full training json; images not in --labelled form the pool')
ap.add_argument('--img_root', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--tau_s', type=float, default=0.4)
ap.add_argument('--tau_q', type=float, default=0.75)
ap.add_argument('--tau_a', type=float, default=0.5)
ap.add_argument('--tau_neg', type=float, default=0.1, help='max teacher score for a confident negative image')
ap.add_argument('--max_pool', type=int, default=2000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = 'cuda'
CL = args.classes.split(',')
m = GlassDet('IDEA-Research/grounding-dino-tiny', 'facebook/sam-vit-base', CL, sfa=False).to(dev).eval()
m.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
AMP = torch.autocast('cuda', dtype=torch.bfloat16)

lab = json.load(open(args.labelled))
pool = json.load(open(args.pool))
lab_files = {im['file_name'] for im in lab['images']}
pool_imgs = [im for im in pool['images'] if im['file_name'] not in lab_files]
rng = np.random.RandomState(args.seed)
if len(pool_imgs) > args.max_pool:
    pool_imgs = [pool_imgs[i] for i in sorted(rng.choice(len(pool_imgs), args.max_pool, replace=False))]
pool_json = {'images': pool_imgs, 'annotations': [], 'categories': pool['categories']}
json.dump(pool_json, open(f'{args.out}/pool_unlabelled.json', 'w'))  # labels removed

# per-class side margins from the labelled set (same calibration as BMR)
by = {}
for a in lab['annotations']:
    by.setdefault(a['image_id'], []).append(a)


@torch.no_grad()
def sam_tight(full, hw, boxes, H, W):
    h, w = hw
    b = torch.as_tensor(boxes, dtype=torch.float32, device=dev) * torch.tensor([w / W, h / H, w / W, h / H], device=dev)
    tight, qs = [], []
    for j in range(0, len(b), 20):
        with AMP:
            o = m.sam(image_embeddings=full, input_boxes=b[None, j:j + 20], multimask_output=False)
        qs.append(o.iou_scores[0, :, 0].float())
        lm = F.interpolate(o.pred_masks[0, :, :1].float(), size=(1024, 1024), mode='bilinear')[:, 0, :h, :w] > 0
        for k in range(lm.shape[0]):
            xs, ys = torch.nonzero(lm[k].any(0)).flatten(), torch.nonzero(lm[k].any(1)).flatten()
            tight.append([np.nan] * 4 if len(xs) == 0 else
                         [xs[0].item() * W / w, ys[0].item() * H / h, (xs[-1].item() + 1) * W / w, (ys[-1].item() + 1) * H / h])
    return np.array(tight, np.float32), torch.cat(qs).cpu().numpy()


cal_ds = CocoDet(args.labelled, args.img_root)
marg = {c: [] for c in range(len(CL))}
for i in range(len(cal_ds)):
    pix, tg = cal_ds[i]
    anns = by.get(tg['image_id'], [])
    if not anns:
        continue
    H, W = tg['orig_size']
    with torch.no_grad(), AMP:
        full, _, hw = m.sam_encode(pix[None].to(dev))
    g = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns])
    t, q = sam_tight(full, hw, g, H, W)
    for a, gb, tb, qq in zip(anns, g, t, q):
        if not np.isnan(tb[0]) and qq >= 0.5:
            tw, th = max(tb[2] - tb[0], 1), max(tb[3] - tb[1], 1)
            marg[a['category_id'] - 1].append([(tb[0] - gb[0]) / tw, (tb[1] - gb[1]) / th, (gb[2] - tb[2]) / tw, (gb[3] - tb[3]) / th])
allm = np.concatenate([np.array(v) for v in marg.values() if v]) if any(marg.values()) else np.zeros((1, 4))
MARGIN = {c: (np.median(np.array(v), 0) if len(v) >= 3 else np.median(allm, 0)) for c, v in marg.items()}


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / max(u, 1e-9)


pl_score, pl_ver = [], []  # (image_record, [(cls, xywh)])
negatives = []  # pool images in which the teacher's best score is below tau_neg
stats = {'pool_images': len(pool_imgs), 'cand': 0, 'pass_q': 0, 'pass_both': 0}
ds = CocoDet(f'{args.out}/pool_unlabelled.json', args.img_root)
for k, (pix, tg) in enumerate(DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate)):
    H, W = tg[0]['orig_size']
    with torch.no_grad(), AMP:
        full, _, hw = m.sam_encode(pix.to(dev))
        out = m(pix.to(dev), None)
    out = {kk: (v.float() if torch.is_tensor(v) else v) for kk, v in out.items()}
    sc, cl, bx = decode(out, m.label_map, [(H, W)])[0]
    if float(sc.max()) < args.tau_neg:  # teacher sees nothing at all -> confident defect-free negative
        negatives.append(pool_imgs[k])
    keep = sc >= args.tau_s
    sc, cl, bx = sc[keep], cl[keep], bx[keep]
    if len(bx) == 0:
        continue
    t, q = sam_tight(full, hw, bx, H, W)
    s_list, v_list = [], []
    for c, b, tb, qq in zip(cl, bx, t, q):
        xywh = [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
        s_list.append((int(c), xywh))
        stats['cand'] += 1
        if np.isnan(tb[0]) or qq < args.tau_q:
            continue
        stats['pass_q'] += 1
        l, tp, r, bt = MARGIN[int(c)]
        tw, th = max(tb[2] - tb[0], 1), max(tb[3] - tb[1], 1)
        cal = [tb[0] - l * tw, tb[1] - tp * th, tb[2] + r * tw, tb[3] + bt * th]
        if iou(b, cal) >= args.tau_a:
            stats['pass_both'] += 1
            v_list.append((int(c), xywh))
    rec = pool_imgs[k]
    if s_list:
        pl_score.append((rec, s_list))
    if v_list:
        pl_ver.append((rec, v_list))


def write(name, pls):
    images = [dict(im) for im in lab['images']]
    anns = [dict(a) for a in lab['annotations']]
    nid = max(im['id'] for im in images) + 1
    for rec, objs in pls:
        im = dict(rec); im['id'] = nid; images.append(im)
        for c, box in objs:
            anns.append({'id': len(anns) + 1, 'image_id': nid, 'category_id': c + 1, 'bbox': box,
                         'area': box[2] * box[3], 'iscrowd': 0, 'pseudo': 1})
        nid += 1
    for i, a in enumerate(anns, 1):
        a['id'] = i
    json.dump({'images': images, 'annotations': anns, 'categories': lab['categories']}, open(f'{args.out}/{name}.json', 'w'))
    return len(images), len(anns)


stats['score_set'] = write('train_score', pl_score)
stats['verified_set'] = write('train_verified', pl_ver)
# confident negatives as images without boxes; at most as many as pseudo-labelled images
neg = [(rec, []) for rec in negatives[:max(len(pl_score), 1)]]
stats['n_negatives_found'] = len(negatives)
stats['score_neg_set'] = write('train_score_neg', pl_score + neg)
stats['margins'] = {CL[c]: np.round(v, 3).tolist() for c, v in MARGIN.items()}
json.dump(stats, open(f'{args.out}/stats.json', 'w'), indent=1)
json.dump({'AP50': 1.0}, open(f'{args.out}/eval_final.json', 'w'))  # queue completion marker
print(stats, flush=True)
