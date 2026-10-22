"""Mask quality on pixel-annotated glass sets (MSD, MVTec bottle, MVTec AD 2 vial).

Compares three ways of obtaining a defect mask:
  box-fill : the DINO box itself filled (what a detector alone can deliver)
  DINO->SAM: DINO boxes prompt SAM through the shared embedding (ours)
  GT->SAM  : ground-truth boxes prompt SAM (upper bound of the promptable segmenter)
Reports COCO segm AP / AP50 and the mean IoU of the best-matching predicted mask per GT instance."""
import argparse
import json
import os

import numpy as np
import pycocotools.mask as mu
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from glassdet.data import CocoDet, collate
from glassdet.evaluate import decode
from glassdet.model import GlassDet

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--val', required=True)
ap.add_argument('--img_root', required=True)
ap.add_argument('--classes', required=True)
ap.add_argument('--short', type=int, default=800)
ap.add_argument('--out', required=True)
args = ap.parse_args()
dev = 'cuda'
CL = args.classes.split(',')
os.makedirs(args.out, exist_ok=True)
m = GlassDet('IDEA-Research/grounding-dino-tiny', 'facebook/sam-vit-base', CL, sfa=False).to(dev).eval()
sd = torch.load(args.ckpt, map_location='cpu')
if '__lang_free__' in sd:
    m.convert_language_free(sd['static_text'].to(dev)); sd.pop('__lang_free__')
m.load_state_dict(sd, strict=False)
AMP = torch.autocast('cuda', dtype=torch.bfloat16)
gt = COCO(args.val)


@torch.no_grad()
def sam_masks(full, hw, boxes, H, W):
    h, w = hw
    if len(boxes) == 0:
        return np.zeros((0, H, W), bool)
    b = torch.as_tensor(boxes, dtype=torch.float32, device=dev) * torch.tensor([w / W, h / H, w / W, h / H], device=dev)
    outs = []
    for j in range(0, len(b), 20):
        with AMP:
            o = m.sam(image_embeddings=full, input_boxes=b[None, j:j + 20], multimask_output=False)
        lm = F.interpolate(o.pred_masks[0, :, :1].float(), size=(1024, 1024), mode='bilinear')[:, :, :h, :w]
        outs.append((F.interpolate(lm, size=(H, W), mode='bilinear')[:, 0] > 0).cpu().numpy())
    return np.concatenate(outs)


def rle(mk):
    r = mu.encode(np.asfortranarray(mk.astype(np.uint8)))
    r['counts'] = r['counts'].decode()
    return r


dets = {'box-fill': [], 'DINO->SAM': [], 'GT->SAM': []}
ious = {k: [] for k in dets}
ds = CocoDet(args.val, args.img_root, short=args.short)
for pix, tg in DataLoader(ds, batch_size=1, num_workers=4, collate_fn=collate):
    H, W = tg[0]['orig_size']
    iid = tg[0]['image_id']
    with torch.no_grad(), AMP:
        full, _, hw = m.sam_encode(pix.to(dev))
        out = m(pix.to(dev), None)
    out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
    sc, cl, xyxy = decode(out, m.label_map, [(H, W)])[0]
    keep = sc >= 0.05
    sc, cl, xyxy = sc[keep], cl[keep], xyxy[keep]
    fill = np.zeros((len(xyxy), H, W), bool)
    for j, b in enumerate(xyxy):
        x0, y0, x1, y1 = np.clip(np.round(b).astype(int), 0, [W, H, W, H])
        fill[j, y0:y1, x0:x1] = True
    pred = {'box-fill': (fill, sc, cl), 'DINO->SAM': (sam_masks(full, hw, xyxy, H, W), sc, cl)}
    anns = gt.loadAnns(gt.getAnnIds(imgIds=iid))
    gb = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns])
    pred['GT->SAM'] = (sam_masks(full, hw, gb, H, W), np.ones(len(anns)), np.array([a['category_id'] - 1 for a in anns]))
    gmasks = [gt.annToMask(a).astype(bool) for a in anns]
    for k, (mk, s, c) in pred.items():
        for j in range(len(mk)):
            dets[k].append({'image_id': iid, 'category_id': int(c[j]) + 1, 'score': float(s[j]), 'segmentation': rle(mk[j])})
        for gm in gmasks:  # best IoU of any predicted mask with this GT instance
            best = 0.0
            for j in range(len(mk)):
                inter = np.logical_and(gm, mk[j]).sum()
                if inter:
                    best = max(best, inter / np.logical_or(gm, mk[j]).sum())
            ious[k].append(best)

res = {}
for k, d in dets.items():
    E = COCOeval(gt, gt.loadRes(d) if d else gt.loadRes([]), 'segm')
    E.evaluate(); E.accumulate(); E.summarize()
    res[k] = {'segm_AP': float(E.stats[0]), 'segm_AP50': float(E.stats[1]), 'mIoU_matched': float(np.mean(ious[k]))}
    print(f'{k:10s} segm AP {100 * res[k]["segm_AP"]:5.1f}  AP50 {100 * res[k]["segm_AP50"]:5.1f}  '
          f'mean best-IoU {100 * res[k]["mIoU_matched"]:5.1f}', flush=True)
res['AP50'] = res['DINO->SAM']['segm_AP50']  # queue completion marker
json.dump(res, open(f'{args.out}/eval_final.json', 'w'), indent=1)
