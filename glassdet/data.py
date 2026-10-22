import json
import pickle
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def target_size(h, w, short=800, long_max=1333):
    s = short / min(h, w)
    if max(h, w) * s > long_max:
        s = long_max / max(h, w)
    return int(round(h * s)), int(round(w * s))


def add_glare(img, rng):
    """Synthetic specular reflection: anisotropic Gaussian highlights and thin bright streaks."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(0.03, 0.25) * w, rng.uniform(0.01, 0.1) * h
        th = rng.uniform(0, np.pi)
        xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        g = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
        out += rng.uniform(40, 140) * g[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def optical_contrast(img, sigma_frac=0.012, gain=4.0):
    """Optical contrast encoding for smooth glass.
    Glass surfaces carry almost no texture, so the local background B (illumination gradients, specular bands) is
    smooth and can be estimated by a wide low-pass filter. Flaws are local relative deviations R = (I - B) / B:
    scattering flaws (scratches, chips, cracks) brighten the image like dark-field imaging (R > 0), absorbing or
    blocking flaws (stains, inclusions) darken it like bright-field imaging (R < 0). The three input channels become
    [intensity, scattering contrast max(R,0), absorption contrast max(-R,0)] (scaled by `gain`)."""
    g8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = g8.astype(np.float32)
    # Background by a median filter at 1/4 resolution: it removes compact/thin flaws (minority pixels) but keeps
    # step edges of the part (glass borders, coatings), so these structural edges cancel in the residual.
    small = cv2.resize(g8, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = int(max(5, sigma_frac * 4 * max(small.shape))) | 1
    bg = cv2.resize(cv2.medianBlur(small, min(k, 255)), (g.shape[1], g.shape[0]),
                    interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # noise floor: relative contrast is not amplified where the background is almost black
    r = (g - bg) / (bg + 0.1 * float(g.mean()) + 4.0)
    bright = np.clip(r * gain, 0, 1) * 255
    dark = np.clip(-r * gain, 0, 1) * 255
    return np.stack([dark, bright, g], -1).astype(np.uint8)  # BGR order -> RGB = (intensity, bright, dark)


def paste_flaws(img, boxes, labels, bank, rng, kmax=3):
    """Mask-based flaw synthesis: transfer the intensity deviation of banked flaws onto new locations,
    new = target + alpha * (flaw - flaw_background). Pasted boxes keep the GT annotation style."""
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    nb, nl = [boxes], [labels]
    for _ in range(rng.randint(1, kmax)):
        f = bank[rng.randrange(len(bank))]
        crop, alpha, box = f['crop'] - f['bg'], f['alpha'], list(f['box'])
        s = rng.uniform(0.75, 1.25)
        ch, cw = max(2, int(crop.shape[0] * s)), max(2, int(crop.shape[1] * s))
        crop = cv2.resize(crop, (cw, ch)); alpha = cv2.resize(alpha, (cw, ch))
        box = [v * s for v in box]
        if rng.random() < 0.5:
            crop, alpha = crop[:, ::-1], alpha[:, ::-1]; box[0] = cw - box[0] - box[2]
        if rng.random() < 0.5:
            crop, alpha = crop[::-1], alpha[::-1]; box[1] = ch - box[1] - box[3]
        if rng.random() < 0.5:  # 90 degree rotation (flaws have no preferred orientation on glass)
            crop, alpha = np.ascontiguousarray(np.rot90(crop)), np.ascontiguousarray(np.rot90(alpha))
            ch, cw = crop.shape[:2]; box = [box[1], cw - box[0] - box[2], box[3], box[2]]
        if ch >= H or cw >= W:
            continue
        bg_level = float(np.mean(f['bg']))
        for _try in range(20):  # similar background brightness (same imaging context) and no overlap with flaws
            x0, y0 = rng.randint(0, W - cw), rng.randint(0, H - ch)
            region = out[y0:y0 + ch, x0:x0 + cw]
            if abs(float(region.mean()) - bg_level) > 40 or region.std() > 45:
                continue
            nbx = np.array([x0 + box[0], y0 + box[1], x0 + box[0] + box[2], y0 + box[1] + box[3]], np.float32)
            allb = np.concatenate(nb, 0)
            if len(allb) == 0 or not ((np.minimum(allb[:, 2], nbx[2]) > np.maximum(allb[:, 0], nbx[0])) &
                                      (np.minimum(allb[:, 3], nbx[3]) > np.maximum(allb[:, 1], nbx[1]))).any():
                break
        else:
            continue
        out[y0:y0 + ch, x0:x0 + cw] += alpha[..., None] * crop
        nb.append(nbx[None]); nl.append(np.array([f['cat'] - 1], np.int64))
    return np.clip(out, 0, 255).astype(np.uint8), np.concatenate(nb, 0), np.concatenate(nl, 0)


class CocoDet(Dataset):
    def __init__(self, ann, img_root, train=False, repeat=1, glare=0.0, short=800, seed=0, bank=None, paste_p=0.0,
                 oce=False):
        d = json.load(open(ann))
        self.root = img_root
        self.imgs = d['images']
        self.anns = {i['id']: [] for i in self.imgs}
        for a in d['annotations']:
            self.anns[a['image_id']].append(a)
        self.train, self.repeat, self.glare, self.short = train, repeat, glare, short
        self.rng = random.Random(seed)
        self.bank = pickle.load(open(bank, 'rb')) if bank else None
        self.paste_p = paste_p
        self.oce = oce

    def __len__(self):
        return len(self.imgs) * self.repeat

    def load(self, i):
        info = self.imgs[i % len(self.imgs)]
        img = cv2.imread(f"{self.root}/{info['file_name']}", cv2.IMREAD_COLOR)
        boxes = np.array([a['bbox'] for a in self.anns[info['id']]], np.float32).reshape(-1, 4)
        labels = np.array([a['category_id'] - 1 for a in self.anns[info['id']]], np.int64)
        return info, img, boxes, labels

    def __getitem__(self, i):
        info, img, boxes, labels = self.load(i)
        H, W = img.shape[:2]
        b = boxes.copy()
        b[:, 2:] += b[:, :2]  # xyxy
        if self.train:
            r = self.rng
            if self.bank and r.random() < self.paste_p:
                img, b, labels = paste_flaws(np.ascontiguousarray(img), b, labels, self.bank, r)
            if r.random() < 0.5:
                img = img[:, ::-1]
                b[:, [0, 2]] = W - b[:, [2, 0]]
            if r.random() < 0.5:
                img = img[::-1]
                b[:, [1, 3]] = H - b[:, [3, 1]]
            if r.random() < 0.8:
                a, c = r.uniform(0.75, 1.25), r.uniform(-25, 25)
                img = np.clip(img.astype(np.float32) * a + c, 0, 255).astype(np.uint8)
            if self.glare > 0 and r.random() < self.glare:
                img = add_glare(np.ascontiguousarray(img), r)
        extra = None
        if self.oce == 1:  # replace the replicated grey channels (v1)
            img = optical_contrast(np.ascontiguousarray(img))
        elif self.oce == 2:  # keep the natural image and append the two contrast channels (v2)
            o = optical_contrast(np.ascontiguousarray(img))
            extra = o[..., [1, 0]]  # (scattering, absorption)
        h, w = target_size(H, W, self.short)
        if extra is not None:
            extra = cv2.resize(np.ascontiguousarray(extra), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        img = cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=cv2.INTER_LINEAR)
        x = (img[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        if extra is not None:  # channels 4-5: optical contrast, consumed by zero-initialised stem weights
            x = torch.cat([x, torch.from_numpy(np.ascontiguousarray(extra.transpose(2, 0, 1)))], 0)
        cx = (b[:, 0] + b[:, 2]) / 2 / W
        cy = (b[:, 1] + b[:, 3]) / 2 / H
        bw = (b[:, 2] - b[:, 0]) / W
        bh = (b[:, 3] - b[:, 1]) / H
        tgt = {'class_labels': torch.from_numpy(labels), 'boxes': torch.from_numpy(np.stack([cx, cy, bw, bh], 1)),
               'image_id': info['id'], 'orig_size': (H, W)}
        return x, tgt


def collate(batch):
    xs, ts = zip(*batch)
    return torch.stack(xs), list(ts)
