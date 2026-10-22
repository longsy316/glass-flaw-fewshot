import json, numpy as np
cfgs = {'base': 'K0_cal0_b0.0_t0.0_a0.0', 'rescore only': 'K0_cal0_b0.0_t0.0_a0.5',
        'BMR w/o calibration': 'K1_cal0_b0.5_t0.5_a0.5', 'BMR (full)': 'K1_cal1_b0.5_t0.5_a0.5', 'BMR 2 rounds': 'K2_cal1_b0.5_t0.5_a0.5'}
for ds, pre in (('SSGD', 'ssgd'), ('GlassCover', 'gc')):
    for n in (50, 100):
        print(f'== {ds} {n}-shot (GDINO-T ft, 3 seeds)')
        for name, k in cfgs.items():
            v = [json.load(open(f'runs/bmr/{pre}_fs{n}_s{s}_gd_sfa0/eval_final.json'))[k] for s in range(3)]
            g = lambda key: np.array([x[key] for x in v]) * 100
            print(f'  {name:22s} AP50 {g("AP50").mean():5.1f}±{g("AP50").std():.1f}  mAP {g("mAP").mean():5.1f}±{g("mAP").std():.1f}  APs {g("APs").mean():4.1f}±{g("APs").std():.1f}')
