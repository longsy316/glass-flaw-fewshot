"""Flaw-size statistics vs. strength statistics (Jayatilaka & Trustrum): if flaw sizes have a power-law tail
P(A > a) ~ a^-r, Griffith (sigma ~ a^-1/2) gives Weibull strengths with modulus m = 2r.
r is estimated with the Hill estimator over the largest k flaws (k = top 50 %), for GT masks, predicted masks
(all detections) and predicted boxes, and compared with the part-level Weibull modulus from weibull.py."""
import glob
import json
import sys

import numpy as np
from scipy.stats import weibull_min


def hill(a, frac=0.5):
    a = np.sort(np.asarray(a, float))[::-1]
    k = max(5, int(len(a) * frac))
    return 1.0 / np.mean(np.log(a[:k] / a[k]))


def m_parts(amax):
    s = np.asarray(amax, float) ** -0.5
    return weibull_min.fit(s / np.median(s), floc=0)[0]


for ds in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(f'runs/geom2/{ds}_*/images.json')):
        ims = [r for r in json.load(open(f)) if r['gt']]
        out = {}
        for src in ('gt', 'pred', 'box'):
            flaws = [x for r in ims for x in r[src]]
            amax = [max(r[src]) for r in ims if r[src]]
            out[src] = (2 * hill(flaws), m_parts(amax), len(flaws))
        rows.append(out)
        print(f"{f.split('/')[-2]:16s} " + '  '.join(
            f"{s}: 2r={v[0]:.2f} m={v[1]:.2f} (n_flaws={v[2]})" for s, v in out.items()), flush=True)
    for s in ('gt', 'pred', 'box'):
        tr = np.array([r[s][0] for r in rows]); mm = np.array([r[s][1] for r in rows])
        print(f'  {ds:7s} {s:4s} mean 2r = {tr.mean():.2f}  mean m = {mm.mean():.2f}  |2r-m|/m = {np.mean(np.abs(tr - mm) / mm):.2f}')
