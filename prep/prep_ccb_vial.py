"""CCB (curved tempered washing-machine glass, 5120x5120, YOLO boxes: 0 bubble, 1 impurity, 2 scratch) and
MVTec AD 2 'vial' (glass vials, pixel masks, defects only in test_public).

CCB: bubbles are ~20-45 px at 5120 px, so images are resized to 2560 and cut into 2x2 tiles of 1280;
a box is kept in a tile if >=50 % of it lies inside (clipped to the tile). Evaluation is tile-level.
vial: no defective training images exist, so K-shot support sets are drawn from test_public/bad; the
remaining bad + all good images form the test set. Instances = connected mask components (single class)."""
import glob, json, os, random, collections
import cv2
import numpy as np

D = 'data'


# ----------------------------------------------------------------------------------------- CCB
def prep_ccb():
    src, out = f'{D}/ccb', f'{D}/ccb_tiles'
    CL = ['bubble', 'impurity', 'scratch']
    os.makedirs(f'{out}/images', exist_ok=True)
    tiles = {}  # split -> list of (tile_name, boxes)
    for split in ('train', 'val'):
        tiles[split] = []
        for f in sorted(glob.glob(f'{src}/images/{split}/*.jpg')):
            stem = os.path.basename(f)[:-4]
            lab = f'{src}/labels/{split}/{stem}.txt'
            boxes = []
            for l in open(lab) if os.path.exists(lab) else []:
                c, cx, cy, w, h = l.split()
                cx, cy, w, h = (float(v) * 2560 for v in (cx, cy, w, h))
                boxes.append((int(c) + 1, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            img = cv2.resize(cv2.imread(f), (2560, 2560), interpolation=cv2.INTER_AREA)
            for ty in range(2):
                for tx in range(2):
                    x0, y0 = tx * 1280, ty * 1280
                    name = f'{split}_{stem}_{ty}{tx}.jpg'
                    cv2.imwrite(f'{out}/images/{name}', img[y0:y0 + 1280, x0:x0 + 1280], [cv2.IMWRITE_JPEG_QUALITY, 95])
                    tb = []
                    for c, a, b, cc, d in boxes:
                        ca, cb, ccc, cd = max(a, x0), max(b, y0), min(cc, x0 + 1280), min(d, y0 + 1280)
                        if ccc <= ca or cd <= cb:
                            continue
                        if (ccc - ca) * (cd - cb) >= 0.5 * (cc - a) * (d - b):
                            tb.append((c, [ca - x0, cb - y0, ccc - ca, cd - cb]))
                    tiles[split].append((stem, name, tb))

    def build(items):
        images, anns = [], []
        for iid, (_, name, tb) in enumerate(items, 1):
            images.append({'id': iid, 'file_name': f'images/{name}', 'width': 1280, 'height': 1280})
            for c, box in tb:
                anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': c, 'bbox': box,
                             'area': box[2] * box[3], 'iscrowd': 0})
        return {'images': images, 'annotations': anns, 'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(CL)]}

    def save(name, d):
        json.dump(d, open(f'{out}/{name}.json', 'w'))
        cnt = collections.Counter(a['category_id'] for a in d['annotations'])
        print(f'ccb {name:10s} imgs={len(d["images"]):5d} anns={len(d["annotations"]):5d}', {CL[k - 1]: v for k, v in sorted(cnt.items())})

    save('train', build(tiles['train']))
    save('val', build(tiles['val']))
    stems = sorted({s for s, _, _ in tiles['train']})
    for n in (20, 50):  # n source images (4n tiles) per few-shot set
        for seed in (0, 1, 2):
            pick = set(random.Random(seed).sample(stems, n))
            save(f'fs{n}_s{seed}', build([t for t in tiles['train'] if t[0] in pick]))


# ----------------------------------------------------------------------------------------- vial
def prep_vial():
    src, out = f'{D}/vial/test_public', f'{D}/vial_coco'
    os.makedirs(out, exist_ok=True)
    bad = sorted(glob.glob(f'{src}/bad/*.png'))
    good = sorted(glob.glob(f'{src}/good/*.png'))

    def build(files):
        images, anns = [], []
        for iid, f in enumerate(files, 1):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
            rel = os.path.relpath(f, src)
            images.append({'id': iid, 'file_name': rel, 'width': W, 'height': H})
            mf = f'{src}/ground_truth/bad/{os.path.basename(f)[:-4]}_mask.png'
            if '/bad/' not in f or not os.path.exists(mf):
                continue
            m = (cv2.imread(mf, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            n, cc, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((7, 7), np.uint8)), connectivity=8)
            for k in range(1, n):
                ck = ((cc == k) & (m > 0)).astype(np.uint8)
                if ck.sum() < 10:
                    continue
                ys, xs = np.nonzero(ck)
                cs, _ = cv2.findContours(ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': 1,
                             'bbox': [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                             'area': float(ck.sum()), 'iscrowd': 0,
                             'segmentation': [p.flatten().astype(float).tolist() for p in cs if len(p) >= 3]})
        return {'images': images, 'annotations': anns, 'categories': [{'id': 1, 'name': 'defect'}]}

    for k in (5, 10):
        for seed in (0, 1, 2):
            sup = sorted(random.Random(seed).sample(bad, k))
            for name, files in ((f'k{k}_s{seed}_train', sup), (f'k{k}_s{seed}_test', [f for f in bad if f not in sup] + good)):
                d = build(files)
                json.dump(d, open(f'{out}/{name}.json', 'w'))
                print(f'vial {name:14s} imgs={len(d["images"]):4d} anns={len(d["annotations"]):4d} size={d["images"][0]["width"]}x{d["images"][0]["height"]}')


if __name__ == '__main__':
    prep_vial()
    prep_ccb()
