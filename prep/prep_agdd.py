"""AGDD (aircraft canopy glass): 'images' = forward lighting, 'image' = backward lighting (same views).
K-shot training sets are drawn from forward-lit images; every image not used for training forms the test
set, evaluated under forward light (in-domain) and backward light (illumination shift)."""
import glob, json, os, random, collections

R = 'data/agdd/data'
OUT = 'data/agdd_coco'
CLASSES = ['contusion', 'scratch', 'crack', 'spot']
CATS = [{'id': i + 1, 'name': n} for i, n in enumerate(CLASSES)]


def labels(stem_path):
    objs = []
    for l in open(stem_path):
        c, cx, cy, w, h = l.split()
        objs.append((int(c) + 1, float(cx), float(cy), float(w), float(h)))
    return objs


items = []  # (split, stem)
for sp in ('train', 'val'):
    for f in sorted(glob.glob(f'{R}/labels_rect/{sp}/*.txt')):
        items.append((sp, os.path.basename(f)[:-4]))
lab = {it: labels(f'{R}/labels_rect/{it[0]}/{it[1]}.txt') for it in items}


def build(its, folder):
    images, anns = [], []
    for iid, (sp, st) in enumerate(its, 1):
        images.append({'id': iid, 'file_name': f'{folder}/{sp}/{st}.png', 'width': 640, 'height': 640})
        for c, cx, cy, w, h in lab[(sp, st)]:
            box = [(cx - w / 2) * 640, (cy - h / 2) * 640, w * 640, h * 640]
            anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': c, 'bbox': box,
                         'area': box[2] * box[3], 'iscrowd': 0})
    return {'images': images, 'annotations': anns, 'categories': CATS}


def save(name, d):
    json.dump(d, open(f'{OUT}/{name}.json', 'w'))
    cnt = collections.Counter(a['category_id'] for a in d['annotations'])
    print(f'{name:18s} imgs={len(d["images"]):4d} anns={len(d["annotations"]):4d}',
          {CLASSES[k - 1]: v for k, v in sorted(cnt.items())})


os.makedirs(OUT, exist_ok=True)
for k in (10, 30):
    for s in (0, 1, 2):
        rng = random.Random(s)
        pool = items[:]
        rng.shuffle(pool)
        chosen = []
        for c in range(1, 5):  # cover every class first
            if not any(c == o[0] for it in chosen for o in lab[it]):
                chosen.append(next(it for it in pool if it not in chosen and any(o[0] == c for o in lab[it])))
        chosen += [it for it in pool if it not in chosen][:k - len(chosen)]
        test = [it for it in items if it not in chosen]
        save(f'k{k}_s{s}_train', build(chosen, 'images'))
        save(f'k{k}_s{s}_test_fwd', build(test, 'images'))
        save(f'k{k}_s{s}_test_bwd', build(test, 'image'))
