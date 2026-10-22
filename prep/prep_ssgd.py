"""Rebuild SSGD annotations from the original XML (pinyin) labels.

The English category names shipped in SSGD's COCO json are scrambled relative to
the XML labels (e.g. lb101 'broken' == huahen/scratch), so we ignore the json
categories and only reuse its official 5-fold image lists.
"""
import json, os, random, collections
import xml.etree.ElementTree as ET

ROOT = 'data/ssgd/SSGD'
OUT = 'data/ssgd_coco'
PINYIN = ['liewen', 'huahen', 'bengque', 'baidian', 'louguang', 'pomo', 'wuzi']
CLASSES = ['crack', 'scratch', 'chipping', 'white spot', 'light leakage', 'broken film', 'stain']
CATS = [{'id': i + 1, 'name': n} for i, n in enumerate(CLASSES)]


def parse(lb, fn):
    x = f'{ROOT}/{lb}/' + fn.rsplit('.', 1)[0] + '.xml'
    objs = []
    if not os.path.exists(x):
        return objs
    for o in ET.parse(x).getroot().iter('object'):
        name = o.find('name').text.strip()
        b = o.find('bndbox')
        x0, y0, x1, y1 = (float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        if name not in PINYIN or x1 <= x0 or y1 <= y0:
            print('skip', lb, fn, name)
            continue
        objs.append((PINYIN.index(name) + 1, [x0, y0, x1 - x0, y1 - y0]))
    return objs


def build(entries):
    """entries: list of (subset, file_name)."""
    images, anns = [], []
    for iid, (lb, fn) in enumerate(entries, 1):
        images.append({'id': iid, 'file_name': f'{lb}/{fn}', 'width': 1500, 'height': 1000})
        for cid, box in parse(lb, fn):
            anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': cid, 'bbox': box,
                         'area': box[2] * box[3], 'iscrowd': 0})
    return {'images': images, 'annotations': anns, 'categories': CATS}


def split_list(lb, name):
    d = json.load(open(f'{ROOT}/annotations_{lb}/{name}.json'))
    return [(lb, i['file_name']) for i in d['images']]


def fewshot(entries, n, seed):
    """Sample n defect images; first cover every class once, then fill randomly."""
    rng = random.Random(seed)
    lab = {e: {c for c, _ in parse(*e)} for e in entries}
    pool = [e for e in entries if lab[e]]
    rng.shuffle(pool)
    chosen = []
    for c in range(1, len(CLASSES) + 1):
        if any(c in lab[e] for e in chosen):
            continue
        cand = [e for e in pool if c in lab[e] and e not in chosen]
        if cand:
            chosen.append(cand[0])
    for e in pool:
        if len(chosen) >= n:
            break
        if e not in chosen:
            chosen.append(e)
    return chosen[:n]


def save(name, d):
    json.dump(d, open(f'{OUT}/{name}.json', 'w'))
    cnt = collections.Counter(a['category_id'] for a in d['annotations'])
    print(f'{name:22s} imgs={len(d["images"]):5d} anns={len(d["annotations"]):5d}',
          {CLASSES[k - 1]: v for k, v in sorted(cnt.items())})


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    tr = {lb: split_list(lb, 'train1') for lb in ('lb101', 'lb201')}
    va = {lb: split_list(lb, 'val1') for lb in ('lb101', 'lb201')}
    save('all_train', build(tr['lb101'] + tr['lb201']))
    save('all_val', build(va['lb101'] + va['lb201']))
    for lb in ('lb101', 'lb201'):
        save(f'{lb}_train', build(tr[lb]))
        save(f'{lb}_val', build(va[lb]))
    for n in (20, 50, 100):
        for s in (0, 1, 2):
            save(f'fs{n}_s{s}', build(fewshot(tr['lb101'] + tr['lb201'], n, s)))
