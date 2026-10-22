"""Export MVTec AD 'bottle' (glass bottle) from the FiftyOne copy into images/ + COCO json with boxes and polygons
(one instance per connected mask component)."""
import json, os, shutil, sys
import cv2
import numpy as np
SRC, OUT = sys.argv[1], sys.argv[2]
CLASSES = ['broken_large', 'broken_small', 'contamination']
d = json.load(open(f'{SRC}/samples.json'))
s = [x for x in d['samples'] if x['category']['label'] == 'bottle' and x['split'] == 'test']
os.makedirs(f'{OUT}/images', exist_ok=True); os.makedirs(f'{OUT}/masks', exist_ok=True)
images, anns = [], []
for iid, x in enumerate(sorted(s, key=lambda t: t['filepath']), 1):
    lab = x['defect']['label']
    fn = f"{lab}_{os.path.basename(x['filepath'])}"
    shutil.copy(f"{SRC}/{x['filepath']}", f'{OUT}/images/{fn}')
    img = cv2.imread(f'{OUT}/images/{fn}')
    H, W = img.shape[:2]
    images.append({'id': iid, 'file_name': f'images/{fn}', 'width': W, 'height': H})
    if lab == 'good':
        continue
    m = (cv2.imread(f"{SRC}/{x['defect_mask']['mask_path']}", cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    cv2.imwrite(f'{OUT}/masks/{fn}', m * 255)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 20:
            continue
        ck = (cc == k).astype(np.uint8)
        cs, _ = cv2.findContours(ck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = [c.flatten().astype(float).tolist() for c in cs if len(c) >= 3]
        xx, yy, ww, hh = st[k, :4]
        anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': CLASSES.index(lab) + 1,
                     'bbox': [float(xx), float(yy), float(ww), float(hh)], 'area': float(st[k, cv2.CC_STAT_AREA]),
                     'segmentation': poly, 'iscrowd': 0})
json.dump({'images': images, 'anns_note': 'test split only', 'annotations': anns,
           'categories': [{'id': i + 1, 'name': c} for i, c in enumerate(CLASSES)]}, open(f'{OUT}/test.json', 'w'))
print(len(images), 'images', len(anns), 'instances')
