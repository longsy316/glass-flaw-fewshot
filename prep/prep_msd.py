"""MSD (phone-screen glass, PKU): VOC palette masks (1 oil, 2 stain, 3 scratch) -> COCO instances.
One instance = one connected component per class after a 5-px dilation (re-joins broken scratches);
boxes/polygons come from the original (undilated) pixels of that component."""
import json, os, random, collections
import cv2
import numpy as np
from PIL import Image

R = 'data/msd/MSD'
OUT = 'data/msd_coco'
CLASSES = ['oil', 'stain', 'scratch']  # palette index 1,2,3


def instances(mask):
    out = []
    for c in (1, 2, 3):
        m = (mask == c).astype(np.uint8)
        if not m.any():
            continue
        n, cc = cv2.connectedComponents(cv2.dilate(m, np.ones((11, 11), np.uint8)), connectivity=8)
        for k in range(1, n):
            ck = ((cc == k) & (m > 0)).astype(np.uint8)
            if ck.sum() < 10:
                continue
            ys, xs = np.nonzero(ck)
            cs, _ = cv2.findContours(ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly = [p.flatten().astype(float).tolist() for p in cs if len(p) >= 3]
            out.append((c, [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1),
                            float(ys.max() - ys.min() + 1)], poly, int(ck.sum())))
    return out


def build(names):
    images, anns = [], []
    for iid, nm in enumerate(names, 1):
        mask = np.array(Image.open(f'{R}/SegmentationClass/{nm}.png'))
        H, W = mask.shape
        images.append({'id': iid, 'file_name': f'JPEGImages/{nm}.jpg', 'width': W, 'height': H,
                       'mask_file': f'SegmentationClass/{nm}.png'})
        for c, box, poly, area in instances(mask):
            anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': c, 'bbox': box, 'area': float(area),
                         'segmentation': poly, 'iscrowd': 0})
    return {'images': images, 'annotations': anns, 'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(CLASSES)]}


def save(name, d):
    json.dump(d, open(f'{OUT}/{name}.json', 'w'))
    cnt = collections.Counter(a['category_id'] for a in d['annotations'])
    print(f'{name:12s} imgs={len(d["images"]):4d} anns={len(d["annotations"]):5d}',
          {CLASSES[k - 1]: v for k, v in sorted(cnt.items())})


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    sp = {s: open(f'{R}/ImageSets/Segmentation/{s}.txt').read().split() for s in ('train', 'val', 'test')}
    for s in sp:
        save(s, build(sp[s]))
    for k in (10, 30):  # balanced K-shot: k/3 images per defect type
        for seed in (0, 1, 2):
            rng = random.Random(seed)
            pick = []
            for pre in ('Oil', 'Sta', 'Scr'):
                cand = [n for n in sp['train'] if n.startswith(pre)]
                pick += rng.sample(cand, k // 3 + (1 if pre == 'Oil' and k % 3 else 0))
            save(f'k{k}_s{seed}', build(pick))
