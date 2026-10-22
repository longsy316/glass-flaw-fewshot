"""Fig. 2: Weibull probability plots of part strengths from the largest MECHANICAL flaw per part (seed 0), from
ground-truth masks, predicted SAM masks and predicted boxes (predictions: detections classified as a mechanical class).
Strengths a_max^-1/2 are normalised by their own median, so the slope is the Weibull modulus m."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SRC = [('gt', 'Ground-truth mask', INK, 'o'), ('pred', 'Predicted mask (ours)', '#2a78d6', 's'),
       ('box', 'Predicted box', '#eb6834', '^')]
PANELS = [('msd_k30_s0', 'Phone screen: scratches', {3}), ('bottle_k9_s0', 'Glass bottle: breakage', {1, 2})]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK})
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7), constrained_layout=True)
for ax, (run, title, mech) in zip(axes, PANELS):
    ims = json.load(open(f'runs/geom3/{run}/images.json'))
    ms = []
    for key, label, col, mk in SRC:
        a = []
        for r in ims:
            if not any(c in mech for c in r['gt_cat']):
                continue
            v = [x for x, c in zip(r[key], r[f'{key}_cat']) if c in mech]
            if v:
                a.append(max(v))
        s = np.sort((np.array(a, float) ** -0.5) / np.median(np.array(a, float) ** -0.5))
        m, _, scale = weibull_min.fit(s, floc=0)
        ms.append(m)
        F = (np.arange(1, len(s) + 1) - 0.5) / len(s)
        x, y = np.log(s), np.log(-np.log(1 - F))
        ax.scatter(x, y, s=10, marker=mk, facecolor='none' if key == 'gt' else col, edgecolor=col, linewidth=0.7,
                   alpha=0.85, zorder=3, label=label)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, m * (xx - np.log(scale)), color=col, linewidth=1.4, zorder=2)
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.set_xlabel(r'ln($\sigma_f/\sigma_{f,\mathrm{median}}$)')
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(0.03, 0.97, f'$m$: GT {ms[0]:.2f}   mask {ms[1]:.2f}   box {ms[2]:.2f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=6.8, color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside lower center', ncol=3, frameon=False, fontsize=7, markerscale=1.4)
axes[0].set_ylabel(r'ln ln[1/(1$-F$)]')
fig.savefig('figs/weibull.pdf')
fig.savefig('figs/weibull.png', dpi=300)
print('saved', [round(v, 2) for v in ms])
