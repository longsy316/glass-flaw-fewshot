"""Diagnostic only: precision/recall of pseudo-labels against the hidden GT of the pool (IoU>=0.5, class-aware)."""
import json, sys
import numpy as np
run, gt_json = sys.argv[1], sys.argv[2]
gt = json.load(open(gt_json)); fid = {im['file_name']: im['id'] for im in gt['images']}
G = {}
for a in gt['annotations']:
    G.setdefault(a['image_id'], []).append(a)
pool = {im['file_name'] for im in json.load(open(f'{run}/pool_unlabelled.json'))['images']}
n_gt = sum(len(G.get(fid[f], [])) for f in pool)
def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx)); ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    return iw * ih / (aw * ah + bw * bh - iw * ih + 1e-9)
for name in ('train_score', 'train_verified'):
    d = json.load(open(f'{run}/{name}.json')); ims = {im['id']: im for im in d['images']}
    tp = fp = 0; hit = set()
    for a in d['annotations']:
        if not a.get('pseudo'): continue
        f = ims[a['image_id']]['file_name']; gl = G.get(fid[f], [])
        m = [g for g in gl if g['category_id'] == a['category_id'] and iou(a['bbox'], g['bbox']) >= 0.5]
        if m: tp += 1; hit.add(m[0]['id'])
        else: fp += 1
    print(f'{name:15s} pseudo={tp + fp:5d}  precision={tp / max(tp + fp, 1):.3f}  recall_of_pool_GT={len(hit) / max(n_gt, 1):.3f}')
