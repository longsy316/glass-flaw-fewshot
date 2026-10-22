"""Weakest-part ranking: inspection must rank parts by strength (largest flaw). Spearman correlation between the
part strengths a_max^-1/2 from predictions (all detections) and from GT masks, and the share of the true weakest
10 % of parts that are also flagged as weakest 10 % (scale-free, no calibration needed)."""
import glob
import json
import sys

import numpy as np
from scipy.stats import spearmanr

for ds in sys.argv[1:]:
    res = {'mask': [], 'box': [], 'mask_top': [], 'box_top': []}
    for f in sorted(glob.glob(f'runs/geom2/{ds}_*/images.json')):
        ims = [r for r in json.load(open(f)) if r['gt']]
        g = np.array([max(r['gt']) for r in ims])
        for src in ('pred', 'box'):
            p = np.array([max(r[src]) if r[src] else 0.0 for r in ims])  # undetected part = no visible flaw
            key = 'mask' if src == 'pred' else 'box'
            res[key].append(spearmanr(g, p).correlation)
            k = max(1, int(round(0.1 * len(ims))))
            top_g, top_p = set(np.argsort(-g)[:k]), set(np.argsort(-p)[:k])
            res[key + '_top'].append(len(top_g & top_p) / k)
    print(f"{ds:7s} Spearman rho  mask {np.mean(res['mask']):.3f}±{np.std(res['mask']):.3f}  box {np.mean(res['box']):.3f}±{np.std(res['box']):.3f}"
          f" | weakest-10% recovered  mask {100 * np.mean(res['mask_top']):.0f}%  box {100 * np.mean(res['box_top']):.0f}%")
