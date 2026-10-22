"""GlassCoverDefectDataset (SCUT, phone glass cover, 8437 VOC images): labels 0/1/2 = point/line/area defects.
Official ImageSets/Main train/val lists; few-shot subsets are drawn from train."""
import json, os, random, collections
import xml.etree.ElementTree as ET

R = 'data/glasscover/GlassCover_datset'
OUT = 'data/glasscover_coco'
CLASSES = ['dot', 'scratch', 'patch']  # prompts for point / line / area defects


def parse(stem):
    t = ET.parse(f'{R}/Annotations/{stem}.xml').getroot()
    W, H = int(t.find('size/width').text), int(t.find('size/height').text)
    objs = []
    for o in t.iter('object'):
        b = o.find('bndbox')
        x0, y0, x1, y1 = (float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        if x1 > x0 and y1 > y0:
            objs.append((int(o.find('name').text) + 1, [x0, y0, x1 - x0, y1 - y0]))
    return W, H, objs


def build(stems):
    images, anns = [], []
    for iid, st in enumerate(stems, 1):
        W, H, objs = parse(st)
        images.append({'id': iid, 'file_name': f'JPEGImages/{st}.jpg', 'width': W, 'height': H})
        for c, box in objs:
            anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': c, 'bbox': box,
                         'area': box[2] * box[3], 'iscrowd': 0})
    return {'images': images, 'annotations': anns, 'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(CLASSES)]}


def save(name, d):
    json.dump(d, open(f'{OUT}/{name}.json', 'w'))
    cnt = collections.Counter(a['category_id'] for a in d['annotations'])
    print(f'{name:12s} imgs={len(d["images"]):5d} anns={len(d["annotations"]):6d}',
          {CLASSES[k - 1]: v for k, v in sorted(cnt.items())})


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    sp = {s: [l.strip() for l in open(f'{R}/ImageSets/Main/{s}.txt') if l.strip()] for s in ('train', 'val')}
    save('train', build(sp['train']))
    save('val', build(sp['val']))
    lab = {st: {c for c, _ in parse(st)[2]} for st in sp['train']}
    for n in (20, 50, 100):
        for seed in (0, 1, 2):
            rng = random.Random(seed)
            pool = [s for s in sp['train'] if lab[s]]
            rng.shuffle(pool)
            pick = [next(s for s in pool if c in lab[s]) for c in (1, 2, 3)]
            pick += [s for s in pool if s not in pick][:n - len(pick)]
            save(f'fs{n}_s{seed}', build(pick))
