"""Build a flaw bank for mask-based flaw synthesis: every labelled flaw is segmented by SAM from its GT box and stored
as a crop (GT box enlarged by 25 %), its soft alpha mask, the mean intensity of the crop background (outside the
mask) and the GT box inside the crop, so that pasted flaws keep the dataset's annotation style."""
import argparse
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from glassdet.data import CocoDet
from glassdet.model import GlassDet

ap = argparse.ArgumentParser()
ap.add_argument('--labelled', required=True)
ap.add_argument('--img_root', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
dev = 'cuda'
m = GlassDet('IDEA-Research/grounding-dino-tiny', 'facebook/sam-vit-base', ['flaw'], sfa=False).to(dev).eval()
import json
d = json.load(open(args.labelled))
by = {}
for a in d['annotations']:
    by.setdefault(a['image_id'], []).append(a)
ds = CocoDet(args.labelled, args.img_root)
bank = []
for i, im in enumerate(d['images']):
    anns = by.get(im['id'], [])
    if not anns:
        continue
    pix, tg = ds[i]
    H, W = tg['orig_size']
    img = cv2.imread(f"{args.img_root}/{im['file_name']}", cv2.IMREAD_COLOR).astype(np.float32)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        full, _, (h, w) = m.sam_encode(pix[None].to(dev))
        g = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns],
                         device=dev, dtype=torch.float32)
        o = m.sam(image_embeddings=full, input_boxes=(g * torch.tensor([w / W, h / H, w / W, h / H], device=dev))[None],
                  multimask_output=False)
    lm = F.interpolate(o.pred_masks[0, :, :1].float(), size=(1024, 1024), mode='bilinear')[:, :, :h, :w]
    prob = torch.sigmoid(F.interpolate(lm, size=(H, W), mode='bilinear')[:, 0]).cpu().numpy()
    q = o.iou_scores[0, :, 0].float().cpu().numpy()
    for a, p, qq in zip(anns, prob, q):
        x, y, bw, bh = a['bbox']
        mx, my = 0.25 * bw + 4, 0.25 * bh + 4
        X0, Y0 = int(max(0, x - mx)), int(max(0, y - my))
        X1, Y1 = int(min(W, x + bw + mx)), int(min(H, y + bh + my))
        crop, alpha = img[Y0:Y1, X0:X1], p[Y0:Y1, X0:X1]
        hard = alpha > 0.5
        if hard.sum() < 4 or (~hard).sum() < 4 or qq < 0.5:
            alpha = np.zeros_like(alpha)
            alpha[int(y - Y0):int(y + bh - Y0), int(x - X0):int(x + bw - X0)] = 1.0  # fall back to the GT box region
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.0)
        bg = crop[alpha < 0.2].mean(0) if (alpha < 0.2).any() else crop.reshape(-1, 3).mean(0)
        bank.append({'cat': a['category_id'], 'crop': crop.astype(np.float32), 'alpha': alpha, 'bg': bg.astype(np.float32),
                     'box': [x - X0, y - Y0, bw, bh]})
pickle.dump(bank, open(args.out, 'wb'))
print('flaws in bank:', len(bank))
