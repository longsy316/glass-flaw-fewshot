"""Collect every finished run into the paper tables (mean +- std over seeds)."""
import glob, json, os, re, collections
import numpy as np

BASE, BMR = 'K0_cal0_b0.0_t0.0_a0.0', 'K1_cal1_b0.5_t0.5_a0.5'
KEYS = ('AP50', 'mAP', 'APs', 'R_small', 'R_elong', 'R_weak')


def load(p, key=None):
    try:
        d = json.load(open(p))
    except FileNotFoundError:
        return None
    if key:
        d = d.get(key)
    return d if d and 'AP50' in d else None


def fmt(vals, k):
    v = np.array([x[k] for x in vals]) * 100
    return f'{v.mean():5.1f}±{v.std():3.1f}'


def table(title, rows):
    print(f'\n===== {title}')
    print(f'{"setting":10s} {"method":22s} n  ' + '  '.join(f'{k:>9s}' for k in KEYS))
    for setting, meth, vals in rows:
        if vals:
            print(f'{setting:10s} {meth:22s} {len(vals)}  ' + '  '.join(fmt(vals, k) for k in KEYS))


# ---------------------------------------------------------------- main tables (SSGD, GlassCover)
for ds, pre, bpre in (('SSGD', '', 'ssgd_'), ('GlassCover', 'gc_', 'gc_')):
    rows = []
    for n in (20, 50, 100):
        def runs(m):
            return [x for s in range(3) if (x := load(f'runs/{pre}fs{n}_s{s}/{m}/eval_final.json'))]

        def bmr(m, key):
            return [x for s in range(3) if (x := load(f'runs/bmr/{bpre}fs{n}_s{s}_{m}/eval_final.json', key))]
        for name, vals in (('YOLO11s', runs('yolo11s')), ('RT-DETR-L', runs('rtdetr')), ('GDINO-T ft', runs('gd_sfa0')),
                           ('+BMR', bmr('gd_sfa0', BMR)), ('+SFA+SQF', runs('sfa_sqf')), ('+SFA+SQF+BMR', bmr('sfa_sqf', BMR)),
                           ('LF', runs('gd_lf')), ('LF+BMR', bmr('gd_lf', BMR))):
            rows.append((f'{n} img', name, vals))
    table(ds, rows)

# ---------------------------------------------------------------- generalisation to new glass types
rows = []
for tag, gen, bm in (('AGDD fwd', 'agdd_k10', 'agdd_k10'), ('AGDD bwd', 'agdd_k10', 'agdd_bwd_k10'),
                     ('bottle', 'bottle_k9', 'bottle_k9'), ('vial', 'vial_k10', 'vial_k10'), ('CCB', 'ccb_fs20', 'ccb_fs20')):
    g = lambda m: [x for s in range(3) if (x := load(f'runs_a2/runs/gen/{gen}_s{s}/{m}/eval_final.json')
                                           or load(f'runs/gen/{gen}_s{s}/{m}/eval_final.json'))]
    gd = 'gd_bwd' if tag == 'AGDD bwd' else 'gd'
    bv = [x for s in range(3) if (x := load(f'runs_a2/runs/bmr/{bm}_s{s}/eval_final.json', BMR))]
    for name, vals in (('YOLO11s', [] if tag == 'AGDD bwd' else g('yolo11s')),
                       ('RT-DETR-L', [] if tag == 'AGDD bwd' else g('rtdetr')), ('GDINO-T ft', g(gd)), ('+BMR', bv)):
        rows.append((tag, name, vals))
table('Generalisation (new glass types, few-shot)', rows)

# ---------------------------------------------------------------- speed
b = json.load(open('runs/bench2/eval_final.json'))
print('\n===== Speed (RTX 4090, bs1, 1200x800, bf16, median of 5 rounds)')
for k, v in b.items():
    print(f'{k:40s} {v["ms_median"]:6.1f} ms {v["fps"]:5.1f} FPS {v["params_M"]:6.1f} M')
