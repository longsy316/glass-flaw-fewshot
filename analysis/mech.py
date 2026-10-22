"""Fracture-relevant flaws only. Griffith's relation applies to mechanical flaws (scratches, cracks, chips, breakage),
not to contamination such as oil or stains. Per-flaw metrology uses ground-truth mechanical flaws; part strengths use the
largest mechanical flaw of each part - for predictions, every detection that the detector itself classifies as a
mechanical class (false positives included), as an inspection system would.
MSD classes: 1 oil, 2 stain, 3 scratch -> {3}; MVTec bottle: 1 broken_large, 2 broken_small, 3 contamination -> {1, 2}."""
import glob
import json

import numpy as np
from scipy.stats import weibull_min

MECH = {'msd': {3}, 'bottle': {1, 2}}
rng = np.random.default_rng(0)


def fit_m(s):
    return weibull_min.fit(s / np.median(s), floc=0)[0]


def ci(s, n=1000):
    return np.percentile([fit_m(rng.choice(s, len(s), replace=True)) for _ in range(n)], [2.5, 97.5])


def s1(m):  # strength at 1 % failure probability relative to the median
    return (np.log(1 / 0.99) / np.log(2)) ** (1 / m)


out = {}
for ds, mech in MECH.items():
    runs = []
    for d in sorted(glob.glob(f'runs/geom3/{ds}_*')):
        rows = [r for r in json.load(open(f'{d}/rows.json')) if r['cat'] in mech]
        hit = [r for r in rows if r['pred']]
        res = {'n_flaws': len(rows), 'detected': len(hit) / max(len(rows), 1)}
        for src in ('pred', 'box'):
            res[f'{src}_size_err'] = float(np.median([abs(r[src]['L'] - r['gt']['L']) / r['gt']['L'] for r in hit]))
            res[f'{src}_area_err'] = float(np.median([abs(r[src]['A'] - r['gt']['A']) / r['gt']['A'] for r in hit]))
            ratio = np.array([np.sqrt(r['gt']['L'] / r[src]['L']) for r in hit])
            res[f'{src}_sig10'] = float(np.mean(np.abs(ratio - 1) <= 0.10))
        ims = json.load(open(f'{d}/images.json'))
        parts = []
        for r in ims:
            g = [a for a, c in zip(r['gt'], r['gt_cat']) if c in mech]
            if not g:
                continue
            p = [a for a, c in zip(r['pred'], r['pred_cat']) if c in mech]
            b = [a for a, c in zip(r['box'], r['box_cat']) if c in mech]
            parts.append((max(g), max(p) if p else np.nan, max(b) if b else np.nan))
        parts = np.array(parts)
        res['n_parts'] = len(parts)
        res['part_detected'] = float(np.mean(~np.isnan(parts[:, 1])))
        s_gt = parts[:, 0] ** -0.5
        res['m_gt'] = fit_m(s_gt); res['m_gt_ci'] = ci(s_gt).tolist()
        for j, src in ((1, 'mask'), (2, 'box')):
            ok = ~np.isnan(parts[:, j])
            s = parts[ok, j] ** -0.5
            res[f'm_{src}'] = fit_m(s); res[f'm_{src}_ci'] = ci(s).tolist()
            res[f'in_ci_{src}'] = bool(res['m_gt_ci'][0] <= res[f'm_{src}'] <= res['m_gt_ci'][1])
        runs.append(res)
        print(f"{d.split('/')[-1]:14s} flaws={res['n_flaws']:4d} det={100 * res['detected']:.0f}% | size err mask/box "
              f"{100 * res['pred_size_err']:.1f}/{100 * res['box_size_err']:.1f} area {100 * res['pred_area_err']:.1f}/"
              f"{100 * res['box_area_err']:.1f} sig10 {100 * res['pred_sig10']:.1f}/{100 * res['box_sig10']:.1f} | parts={res['n_parts']}"
              f" m GT {res['m_gt']:.2f} [{res['m_gt_ci'][0]:.2f},{res['m_gt_ci'][1]:.2f}] mask {res['m_mask']:.2f}"
              f"({'in' if res['in_ci_mask'] else 'out'}) box {res['m_box']:.2f}({'in' if res['in_ci_box'] else 'out'})", flush=True)
    g = lambda k: float(np.mean([r[k] for r in runs]))
    mg, mm, mb = g('m_gt'), g('m_mask'), g('m_box')
    print(f"== {ds}: size err {100 * g('pred_size_err'):.1f}/{100 * g('box_size_err'):.1f}  area {100 * g('pred_area_err'):.1f}/"
          f"{100 * g('box_area_err'):.1f}  sig10 {100 * g('pred_sig10'):.1f}/{100 * g('box_sig10'):.1f}  m GT {mg:.2f} mask {mm:.2f}"
          f" box {mb:.2f}  in-CI mask {sum(r['in_ci_mask'] for r in runs)}/{len(runs)} box {sum(r['in_ci_box'] for r in runs)}/{len(runs)}"
          f"  sigma1% dev mask {100 * (s1(mm) / s1(mg) - 1):+.0f}% box {100 * (s1(mb) / s1(mg) - 1):+.0f}%")
    out[ds] = runs
json.dump(out, open('runs/mech.json', 'w'), indent=1, default=float)
