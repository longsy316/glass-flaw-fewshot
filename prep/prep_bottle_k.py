"""MVTec AD 'bottle': defects exist only in the test split, so K-shot support sets are drawn from defective
test images (K/3 per type, seeded); remaining defective + all good images form the test set."""
import json, random, sys

R = sys.argv[1] if len(sys.argv) > 1 else 'data/mvtec_bottle'
d = json.load(open(f'{R}/test.json'))
by = {}
for a in d['annotations']:
    by.setdefault(a['image_id'], []).append(a)
types = {}
for im in d['images']:
    t = im['file_name'].split('/')[-1].rsplit('_', 1)[0]
    types.setdefault(t, []).append(im)


def sub(ims):
    keep = {im['id'] for im in ims}
    return {'images': ims, 'annotations': [a for a in d['annotations'] if a['image_id'] in keep],
            'categories': d['categories']}


for k in (6, 9):
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        sup = []
        for t in ('broken_large', 'broken_small', 'contamination'):
            sup += rng.sample(types[t], k // 3)
        ids = {im['id'] for im in sup}
        test = [im for im in d['images'] if im['id'] not in ids]
        for name, ims in ((f'k{k}_s{seed}_train', sup), (f'k{k}_s{seed}_test', test)):
            s = sub(ims)
            json.dump(s, open(f'{R}/{name}.json', 'w'))
            print(name, len(s['images']), 'imgs', len(s['annotations']), 'anns')
