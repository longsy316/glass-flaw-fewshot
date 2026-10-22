import glob, json, collections, numpy as np
BASE = 'K0_cal0_b0.0_t0.0_a0.0'
groups = {'SSGD': 'runs/bmr/ssgd_fs*_gd_sfa0', 'GlassCover': 'runs/bmr/gc_fs*_gd_sfa0',
          'new-glass': 'runs_a2/runs/bmr/*_s[0-9]'}
delta = collections.defaultdict(lambda: collections.defaultdict(list))
for g, pat in groups.items():
    for d in glob.glob(pat):
        try: r = json.load(open(d + '/eval_final.json'))
        except FileNotFoundError: continue
        b = r[BASE]
        for k, v in r.items():
            if k.startswith('K'):
                delta[k][g].append([100 * (v['AP50'] - b['AP50']), 100 * (v['mAP'] - b['mAP']), 100 * (v['APs'] - b['APs'])])
rows = []
for k, gs in delta.items():
    allv = np.concatenate([np.array(v) for v in gs.values()])
    rows.append((allv[:, 1].mean(), k, {g: np.round(np.array(v).mean(0), 2).tolist() for g, v in gs.items()}, (allv[:, 1] > 0).mean()))
for m, k, gs, frac in sorted(rows, reverse=True)[:8]:
    print(f'{k:28s} mean dmAP {m:+.2f}  win-rate {frac:.0%}  [dAP50,dmAP,dAPs] by group: {gs}')
print('n runs per group:', {g: len(v) for g, v in delta[BASE].items()})
