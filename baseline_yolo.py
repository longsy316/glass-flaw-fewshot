"""Ultralytics baselines (YOLO11s / RT-DETR-L) on the same COCO splits, scored with our evaluator."""
import argparse
import json
import os

import yaml
from ultralytics import RTDETR, YOLO

from glassdet.evaluate import coco_eval, gt_strata, strata_recall

p = argparse.ArgumentParser()
p.add_argument('--train', required=True)
p.add_argument('--val', required=True)
p.add_argument('--img_root', default='data/ssgd/SSGD')
p.add_argument('--model', default='yolo11s.pt')
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--imgsz', type=int, default=1280)
p.add_argument('--bs', type=int, default=8)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', required=True)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
args.out = os.path.abspath(args.out)
args.img_root = os.path.abspath(args.img_root)


def to_yolo(ann, name):
    """Symlink images and write YOLO txt labels into <out>/yolo/<name>."""
    d = json.load(open(ann))
    root = f'{args.out}/yolo/{name}'
    os.makedirs(f'{root}/images', exist_ok=True)
    os.makedirs(f'{root}/labels', exist_ok=True)
    by = {}
    for a in d['annotations']:
        by.setdefault(a['image_id'], []).append(a)
    for im in d['images']:
        stem = im['file_name'].replace('/', '__').rsplit('.', 1)[0]
        dst = f'{root}/images/{stem}.jpg'
        if not os.path.lexists(dst):
            os.symlink(f"{args.img_root}/{im['file_name']}", dst)
        W, H = im['width'], im['height']
        with open(f'{root}/labels/{stem}.txt', 'w') as f:
            for a in by.get(im['id'], []):
                x, y, w, h = a['bbox']
                f.write(f"{a['category_id'] - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
    return d, f'{root}/images'


dtr, tr_dir = to_yolo(args.train, 'train')
dva, va_dir = to_yolo(args.val, 'val')
names = [c['name'] for c in sorted(dva['categories'], key=lambda c: c['id'])]
cfg = f'{args.out}/data.yaml'
yaml.safe_dump({'train': tr_dir, 'val': va_dir, 'names': {i: n for i, n in enumerate(names)}}, open(cfg, 'w'))

M = RTDETR if 'rtdetr' in args.model else YOLO
model = M(args.model)
model.train(data=cfg, epochs=args.epochs, imgsz=args.imgsz, batch=args.bs, seed=args.seed, project=args.out,
            name='train', exist_ok=True, workers=6, verbose=False, plots=False, patience=0, amp=True)

best = f'{args.out}/train/weights/last.pt'
model = M(best)
stem2id = {im['file_name'].replace('/', '__').rsplit('.', 1)[0]: im['id'] for im in dva['images']}
dets = []
for r in model.predict(va_dir, imgsz=args.imgsz, conf=0.01, max_det=100, stream=True, verbose=False):
    iid = stem2id[os.path.basename(r.path).rsplit('.', 1)[0]]
    for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        dets.append({'image_id': iid, 'category_id': int(c) + 1, 'score': float(s),
                     'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
res = coco_eval(args.val, dets, names)
res.update(strata_recall(args.val, dets, gt_strata(args.val, args.img_root, args.val.replace('.json', '_strata.json'))))
res['tag'] = 'final'
json.dump(res, open(f'{args.out}/eval_final.json', 'w'), indent=1)
json.dump(dets, open(f'{args.out}/dets_final.json', 'w'))
print('final', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != 'per_class_AP50'})
print('  per-class AP50', {k: round(v, 3) for k, v in res['per_class_AP50'].items()})
