import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from glassdet.data import CocoDet, collate
from glassdet.evaluate import coco_eval, decode, dets_from_decoded, gt_strata, strata_recall
from glassdet.losses import DetLoss, kd_loss
from glassdet.model import GlassDet

FROZEN = ('sam.vision_encoder', 'sam.prompt_encoder', 'gd.model.backbone', 'gd.model.text_backbone')

p = argparse.ArgumentParser()
p.add_argument('--train', default='')
p.add_argument('--val', required=True)
p.add_argument('--img_root', default='data/ssgd/SSGD')
p.add_argument('--classes', default='crack,scratch,chipping,white spot,light leakage,broken film,stain')
p.add_argument('--gd', default='IDEA-Research/grounding-dino-tiny')
p.add_argument('--sam', default='facebook/sam-vit-base')
p.add_argument('--sfa', type=int, default=1)
p.add_argument('--sqf', type=int, default=0)
p.add_argument('--iters', type=int, default=2000)
p.add_argument('--bs', type=int, default=4)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--glare', type=float, default=0.0)
p.add_argument('--bank', default='', help='flaw bank (flaw_bank.py) for mask-based flaw synthesis')
p.add_argument('--paste_p', type=float, default=0.0)
p.add_argument('--oce', type=int, default=0, help='optical contrast encoding for smooth glass')
p.add_argument('--train_stem', type=int, default=0, help='unfreeze the backbone patch embedding')
p.add_argument('--short', type=int, default=800)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', required=True)
p.add_argument('--eval_only', action='store_true')
p.add_argument('--ckpt', default='')
p.add_argument('--lf_from', default='', help='teacher checkpoint -> train language-free student')
p.add_argument('--kd', type=float, default=1.0)
p.add_argument('--sfa_lr_mult', type=float, default=10.0)
p.add_argument('--gate_init', type=float, default=0.0)
p.add_argument('--amp', type=int, default=1)
args = p.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = 'cuda'
os.makedirs(args.out, exist_ok=True)
classes = args.classes.split(',')
amp = torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(args.amp))


def build(sfa):
    m = GlassDet(args.gd, args.sam if (sfa or args.sqf) else None, classes, sfa=bool(sfa), gate_init=args.gate_init,
                 sqf=bool(args.sqf)).to(dev)
    if args.oce == 2:  # 5-channel input: RGB + optical contrast through zero-initialised stem filters
        m.expand_stem(2)
    if args.train_stem:  # let the patch embedding adapt to the optical-contrast channels
        for n, q in m.gd.model.backbone.named_parameters():
            if 'embeddings' in n:
                q.requires_grad_(True)
    return m


def keep_in_ckpt(k):
    return not k.startswith(FROZEN) or ((args.train_stem or args.oce == 2) and k.startswith('gd.model.backbone')
                                        and 'embeddings' in k)


def save(model, path):
    sd = {k: v.cpu() for k, v in model.state_dict().items() if keep_in_ckpt(k)}
    if model.lang_free:
        sd['__lang_free__'] = torch.tensor(1)
    torch.save(sd, path)


def load(model, path):
    sd = torch.load(path, map_location='cpu')
    if '__lang_free__' in sd:
        model.convert_language_free(sd['static_text'].to(dev))
        sd.pop('__lang_free__')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(FROZEN)]
    assert not bad and not unexpected, (bad[:5], unexpected[:5])


@torch.no_grad()
def evaluate(model, tag):
    model.eval()
    ds = CocoDet(args.val, args.img_root, train=False, short=args.short, oce=args.oce)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=6, collate_fn=collate)
    dets = []
    for pix, tg in dl:
        pix = pix.to(dev)
        with amp:
            sv = model.sam_encode(pix)[1] if model.sam is not None else None
            out = model(pix, sv)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        dec = decode(out, model.label_map, [t['orig_size'] for t in tg])
        dets += dets_from_decoded(dec, [t['image_id'] for t in tg])
    res = coco_eval(args.val, dets, classes)
    cache = args.val.replace('.json', '_strata.json')
    res.update(strata_recall(args.val, dets, gt_strata(args.val, args.img_root, cache)))
    res['tag'] = tag
    json.dump(res, open(f'{args.out}/eval_{tag}.json', 'w'), indent=1)
    json.dump(dets, open(f'{args.out}/dets_{tag}.json', 'w'))
    print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != 'per_class_AP50'},
          flush=True)
    print('  per-class AP50', {k: round(v, 3) for k, v in res['per_class_AP50'].items()}, flush=True)
    model.train()
    return res


def loader():
    ds = CocoDet(args.train, args.img_root, train=True, repeat=1000, glare=args.glare, short=args.short,
                 seed=args.seed, bank=args.bank or None, paste_p=args.paste_p, oce=args.oce)

    def wif(wid):
        ds.rng = random.Random(args.seed * 1000 + wid + int(time.time()))
    return DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=6, collate_fn=collate,
                      drop_last=True, worker_init_fn=wif, persistent_workers=True)


def lr_at(it):
    warm = min(100, args.iters // 10)
    if it < warm:
        return args.lr * (it + 1) / warm
    return args.lr * 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, args.iters - warm)))


def train(model, teacher=None):
    params = [q for q in model.parameters() if q.requires_grad]
    print('trainable params %.1fM' % (sum(q.numel() for q in params) / 1e6), flush=True)
    # newly added SFA modules start from scratch (gate ~0) -> larger LR, otherwise the gates never open
    new = [q for n, q in model.named_parameters() if q.requires_grad and n.startswith(('sfa.', 'sqf.'))]
    old = [q for n, q in model.named_parameters() if q.requires_grad and not n.startswith(('sfa.', 'sqf.'))]
    groups = [{'params': old, 'mult': 1.0}] + ([{'params': new, 'mult': args.sfa_lr_mult}] if new else [])
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    crit = DetLoss(model.label_map)
    it, t0 = 0, time.time()
    model.train()
    for pix, tg in loader():
        if it >= args.iters:
            break
        for g in opt.param_groups:
            g['lr'] = lr_at(it) * g['mult']
        pix = pix.to(dev)
        tgt = [{'class_labels': t['class_labels'].to(dev), 'boxes': t['boxes'].to(dev)} for t in tg]
        with amp:
            sv = model.sam_encode(pix)[1] if model.sam is not None else None
            out = model(pix, sv)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        loss = crit(out, tgt)
        if teacher is not None:
            with torch.no_grad(), amp:
                tout = teacher(pix, sv)
            tout = {k: (v.float() if torch.is_tensor(v) else v) for k, v in tout.items()}
            loss = loss + args.kd * kd_loss(out, tout, out['text_mask'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.1)
        opt.step()
        if it % 50 == 0:
            print(f'it {it} loss {loss.item():.3f} lr {lr_at(it):.2e} {(time.time() - t0) / (it + 1):.2f}s/it',
                  flush=True)
        it += 1
    return model


if __name__ == '__main__':
    json.dump(vars(args), open(f'{args.out}/args.json', 'w'), indent=1)
    if args.lf_from:
        teacher = build(args.sfa)
        load(teacher, args.lf_from)
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        student = build(args.sfa)
        load(student, args.lf_from)
        bank_dl = DataLoader(CocoDet(args.train, args.img_root, train=False, short=args.short, oce=args.oce), batch_size=4,
                             num_workers=6, collate_fn=collate)
        with amp:
            bank = student.collect_text_states(bank_dl, dev).float()
        student.convert_language_free(bank)
        evaluate(student, 'lf_init')
        if args.iters > 0:
            train(student, teacher)
        save(student, f'{args.out}/model.pt')
        evaluate(student, 'final')
    else:
        model = build(args.sfa)
        if args.ckpt:
            load(model, args.ckpt)
        if not args.eval_only:
            train(model)
            save(model, f'{args.out}/model.pt')
        evaluate(model, 'final')
