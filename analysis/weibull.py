"""Weakest-link strength statistics from inspection images.

Each image (glass part) fails from its largest flaw, so its relative strength is s = a_max^(-1/2). A two-parameter
Weibull distribution is fitted by maximum likelihood (ASTM C1239) to the relative strengths of all defective parts;
the Weibull modulus m does not depend on pixel size or K_IC. We compare m obtained from ground-truth masks, from
predicted SAM masks (all detections, false positives included) and from predicted boxes, with bootstrap 95% CIs,
plus the share of parts whose critical-flaw strength is estimated within +-10%."""
import glob
import json
import sys

import numpy as np
from scipy.stats import weibull_min

rng = np.random.default_rng(0)


def fit_m(s):
    c, _, _ = weibull_min.fit(s / np.median(s), floc=0)
    return c


def boot(s, n=1000):
    ms = [fit_m(rng.choice(s, len(s), replace=True)) for _ in range(n)]
    return np.percentile(ms, [2.5, 97.5])


out = {}
for ds in sys.argv[1:]:
    runs = sorted(glob.glob(f'runs/geom2/{ds}_*/images.json'))
    per_seed = []
    for f in runs:
        ims = [r for r in json.load(open(f)) if r['gt']]
        a_gt = np.array([max(r['gt']) for r in ims])
        a_pr = np.array([max(r['pred']) if r['pred'] else np.nan for r in ims])
        a_bx = np.array([max(r['box']) if r['box'] else np.nan for r in ims])
        res = {'n_parts': len(ims), 'coverage': float(np.mean(~np.isnan(a_pr)))}
        s_gt = a_gt ** -0.5
        res['m_gt'] = fit_m(s_gt); res['m_gt_ci'] = boot(s_gt).tolist()
        for name, a in (('mask', a_pr), ('box', a_bx)):
            ok = ~np.isnan(a)
            s = a[ok] ** -0.5
            res[f'm_{name}'] = fit_m(s); res[f'm_{name}_ci'] = boot(s).tolist()
            ratio = (a_gt[ok] / a[ok]) ** 0.5
            res[f'crit_within10_{name}'] = float(np.mean(np.abs(ratio - 1) <= 0.10) * ok.mean())
        per_seed.append(res)
        print(f"{f.split('/')[-2]:16s} parts={res['n_parts']:4d} detected={100 * res['coverage']:5.1f}%  "
              f"m_GT={res['m_gt']:.2f} [{res['m_gt_ci'][0]:.2f},{res['m_gt_ci'][1]:.2f}]  "
              f"m_mask={res['m_mask']:.2f} [{res['m_mask_ci'][0]:.2f},{res['m_mask_ci'][1]:.2f}]  "
              f"m_box={res['m_box']:.2f} [{res['m_box_ci'][0]:.2f},{res['m_box_ci'][1]:.2f}]  "
              f"critical flaw within 10%: mask {100 * res['crit_within10_mask']:.1f}% box {100 * res['crit_within10_box']:.1f}%",
              flush=True)
    out[ds] = per_seed
json.dump(out, open('runs/weibull.json', 'w'), indent=1, default=float)
