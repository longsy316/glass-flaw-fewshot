"""Latency / parameter benchmark on an otherwise idle GPU (batch 1, 1200x800 input, bf16 autocast).
Configurations are timed in 5 interleaved rounds and the median per-image latency is reported, so that
transient load cannot bias one configuration."""
import argparse
import json
import os
import time

import numpy as np
import torch

from glassdet.model import GlassDet

CL = 'crack,scratch,chipping,white spot,light leakage,broken film,stain'.split(',')
dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='runs/bench'); OUT = ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
x = torch.randn(1, 3, 800, 1200, device=dev)
boxes = torch.tensor([[100., 100., 300., 160.]] * 20, device=dev)  # 20 BMR prompts per image

CFG = {  # name: (load SAM, SFA, SQF, language-free, run BMR)
    'GDINO-T (text, box only)': (0, 0, 0, 0, 0),
    'GDINO-T (language-free, box only)': (0, 0, 0, 1, 0),
    'GDINO-T + SAM BMR (text)': (1, 0, 0, 0, 1),
    'GDINO-T + SAM BMR (language-free)': (1, 0, 0, 1, 1),
    'GDINO-T + SFA + SQF + BMR (text)': (1, 1, 1, 0, 1),
}
models = {}
for name, (sam, sfa, sqf, lf, _) in CFG.items():
    m = GlassDet('IDEA-Research/grounding-dino-tiny', 'facebook/sam-vit-base' if sam else None, CL,
                 sfa=bool(sfa), sqf=bool(sqf)).to(dev).eval()
    text_M = sum(p.numel() for p in m.gd.model.text_backbone.parameters()) / 1e6
    if lf:
        L = len(m.gd.model.encoder.layers)
        m.convert_language_free(torch.zeros(L + 1, m.input_ids.shape[1], 256, device=dev))
    models[name] = (m, text_M)


@torch.no_grad()
def run(name):
    m = models[name][0]
    sam, sfa, sqf, lf, bmr = CFG[name]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        full = sv = None
        if sam:
            full, sv, hw = m.sam_encode(x)
        m(x, sv if (sfa or sqf) else None)
        if bmr:
            m.sam(image_embeddings=full, input_boxes=boxes[None] * 0.68, multimask_output=False)


times = {n: [] for n in CFG}
for n in CFG:  # warm-up
    for _ in range(20):
        run(n)
for rnd in range(5):
    for n in CFG:
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50):
            run(n)
        torch.cuda.synchronize(); times[n].append((time.perf_counter() - t) / 50 * 1000)
res = {}
for n in CFG:
    m, text_M = models[n]
    ms = float(np.median(times[n]))
    res[n] = {'ms_median': ms, 'ms_rounds': times[n], 'fps': 1000 / ms,
              'params_M': sum(p.numel() for p in m.parameters()) / 1e6,
              'text_branch_M': 0.0 if CFG[n][3] else text_M}
    print(f'{n:38s} {ms:6.1f} ms  ({min(times[n]):.1f}-{max(times[n]):.1f})  {1000 / ms:5.1f} FPS  '
          f'params {res[n]["params_M"]:.1f}M', flush=True)
json.dump(res, open(f'{OUT}/eval_final.json', 'w'), indent=1)
