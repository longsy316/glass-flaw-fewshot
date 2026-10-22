import contextlib
import io
import json
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def decode(out, label_map, orig_hw, topk=100):
    """Grounding-DINO token logits -> per-class scores -> top-k (score, cls, xyxy in original pixels)."""
    prob = out['logits'].sigmoid()  # (B,Q,T)
    lm = label_map.to(prob.device).bool()  # (C,T)
    cls_score = torch.stack([prob[..., lm[c]].max(-1).values for c in range(lm.shape[0])], -1)  # (B,Q,C)
    res = []
    for b in range(prob.shape[0]):
        s = cls_score[b].flatten()
        sc, idx = s.topk(min(topk, s.numel()))
        q, c = idx // lm.shape[0], idx % lm.shape[0]
        bx = out['pred_boxes'][b, q]
        H, W = orig_hw[b]
        xyxy = torch.stack([(bx[:, 0] - bx[:, 2] / 2) * W, (bx[:, 1] - bx[:, 3] / 2) * H,
                            (bx[:, 0] + bx[:, 2] / 2) * W, (bx[:, 1] + bx[:, 3] / 2) * H], 1)
        res.append((sc.cpu().numpy(), c.cpu().numpy(), xyxy.cpu().numpy()))
    return res


def coco_eval(gt_json, dets, cat_names):
    """dets: list of dict(image_id, category_id, bbox[xywh], score)."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        if not dets:
            return {'mAP': 0, 'AP50': 0, 'APs': 0, 'per_class_AP50': {n: 0 for n in cat_names}}
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, 'bbox')
        E.evaluate(); E.accumulate(); E.summarize()
    pc = {}
    prec = E.eval['precision']  # T,R,K,A,M
    for k, cid in enumerate(E.params.catIds):
        p = prec[0, :, k, 0, -1]
        p = p[p > -1]
        pc[cat_names[cid - 1]] = float(p.mean()) if p.size else float('nan')
    return {'mAP': float(E.stats[0]), 'AP50': float(E.stats[1]), 'APs': float(E.stats[3]), 'per_class_AP50': pc}


def iou_xyxy(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def gt_strata(gt_json, img_root, cache=None):
    """Per-GT flags: small (sqrt area < 32 px), elongated (aspect >= 4), weak (lowest-tercile local contrast)."""
    if cache:
        try:
            return json.load(open(cache))
        except FileNotFoundError:
            pass
    d = json.load(open(gt_json))
    fn = {i['id']: i['file_name'] for i in d['images']}
    by = {}
    for a in d['annotations']:
        by.setdefault(a['image_id'], []).append(a)
    rec = {}
    for iid, anns in by.items():
        g = cv2.imread(f'{img_root}/{fn[iid]}', cv2.IMREAD_GRAYSCALE).astype(np.float32)
        H, W = g.shape
        for a in anns:
            x, y, w, h = a['bbox']
            x0, y0, x1, y1 = int(x), int(y), int(np.ceil(x + w)), int(np.ceil(y + h))
            m = max(4, int(0.5 * max(w, h)))
            X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
            outer = g[Y0:Y1, X0:X1]
            inner = g[y0:y1, x0:x1]
            ring_sum = outer.sum() - inner.sum(); ring_n = outer.size - inner.size
            ring_mean = ring_sum / max(ring_n, 1)
            ring_std = outer.std() + 1.0
            # defect signal = strongest deviation inside the box relative to its surroundings
            contrast = float(np.abs(inner - ring_mean).max() / ring_std) if inner.size else 0.0
            rec[a['id']] = {'small': bool(np.sqrt(w * h) < 32), 'elong': bool(max(w, h) / max(1, min(w, h)) >= 4),
                            'contrast': contrast}
    cs = np.array([r['contrast'] for r in rec.values()])
    thr = float(np.quantile(cs, 1 / 3))
    for r in rec.values():
        r['weak'] = bool(r['contrast'] <= thr)
    rec = {str(k): v for k, v in rec.items()}
    if cache:
        json.dump(rec, open(cache, 'w'))
    return rec


def strata_recall(gt_json, dets, strata, iou_thr=0.5):
    """Recall of each GT stratum at the score threshold that maximises overall F1 (class-aware matching)."""
    d = json.load(open(gt_json))
    gts = {}
    for a in d['annotations']:
        x, y, w, h = a['bbox']
        gts.setdefault(a['image_id'], []).append((a['id'], a['category_id'], [x, y, x + w, y + h]))
    dd = {}
    for t in dets:
        x, y, w, h = t['bbox']
        dd.setdefault(t['image_id'], []).append((t['score'], t['category_id'], [x, y, x + w, y + h]))
    n_gt = sum(len(v) for v in gts.values())
    best = None
    for thr in np.arange(0.05, 0.95, 0.025):
        tp, fp, hit = 0, 0, set()
        for iid in set(gts) | set(dd):
            G = gts.get(iid, []); P = sorted([p for p in dd.get(iid, []) if p[0] >= thr], key=lambda t: -t[0])
            used = set()
            for s, c, b in P:
                cand = [(i, g) for i, g in enumerate(G) if g[1] == c and i not in used]
                if cand:
                    ious = iou_xyxy(np.array([b]), np.array([g[2] for _, g in cand]))[0]
                    j = int(ious.argmax())
                    if ious[j] >= iou_thr:
                        used.add(cand[j][0]); hit.add(cand[j][1][0]); tp += 1
                        continue
                fp += 1
        prec = tp / max(tp + fp, 1); recl = tp / max(n_gt, 1)
        f1 = 2 * prec * recl / max(prec + recl, 1e-9)
        if best is None or f1 > best[0]:
            best = (f1, thr, prec, recl, hit)
    f1, thr, prec, recl, hit = best
    out = {'F1': f1, 'thr': float(thr), 'P': prec, 'R': recl}
    for k in ('small', 'elong', 'weak'):
        ids = [gid for gid, r in strata.items() if r[k]]
        out[f'R_{k}'] = sum(int(g) in hit for g in ids) / max(len(ids), 1)
        out[f'n_{k}'] = len(ids)
    return out


def dets_from_decoded(decoded, image_ids, score_min=0.01):
    dets = []
    for (sc, c, xyxy), iid in zip(decoded, image_ids):
        for s, k, b in zip(sc, c, xyxy):
            if s < score_min:
                continue
            dets.append({'image_id': int(iid), 'category_id': int(k) + 1, 'score': float(s),
                         'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    return dets
