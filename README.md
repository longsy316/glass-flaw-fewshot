# Few-shot detection, segmentation and metrology of surface flaws in glass

Code, data splits and per-run results for the Materials Letters manuscript
*"Few-shot detection, segmentation and metrology of surface flaws in glass with language-grounded foundation models"*
(S. Long, H. Huang, N. Liu, M. Pei).

## Method in brief

1. **Few-shot detection with flaw-name prompts.** Grounding DINO-T is fine-tuned on 9-100 labelled images with the
   prompt formed by the flaw names (backbone and text encoder frozen).
2. **Self-training.** The fine-tuned detector pseudo-labels up to 2000 unlabelled images (score >= 0.4); a student is
   trained on labelled + pseudo-labelled images (`pseudo_label.py`).
3. **Shared-encoder SAM masks.** Every detected box prompts SAM ViT-B; masks give mask AP and flaw metrology.
4. **Flaw metrology and weakest-link statistics.** Flaw size = skeleton length (aspect >= 3) or maximum Feret diameter;
   relative Griffith strength `(a_GT / a)^0.5`; part strength from the largest flaw and a two-parameter Weibull fit
   (`geom_eval.py`, `analysis/weibull.py`). All quantities are independent of pixel size and K_IC.
5. **Language-free inference.** The text encoder is replaced by distilled static tokens (`train.py --lf_from`).

Ablations reported in the paper are switches of the same code: `--sfa/--sqf` (SAM feature fusion), `bmr.py`
(box-mask refinement), `--glare` (synthetic specular-glare augmentation), `--oce 1|2` (optical contrast encoding),
`--bank/--paste_p` (mask-based flaw synthesis), `pseudo_label.py` outputs `train_verified.json` (SAM-verified
pseudo-labels) and `train_score_neg.json` (with confident negatives). Several of these did **not** give consistent
gains over three seeds; see the paper and `results/`.

## What is in this repository

| Path | Content |
|---|---|
| `glassdet/` | Grounding DINO + SAM wrapper (`model.py`), data loading and augmentation (`data.py`), loss with a fixed label offset for batches >= 3 (`losses.py`), COCO / strata evaluation (`evaluate.py`) |
| `train.py` | Few-shot fine-tuning with flaw-name prompts; `--lf_from` converts a trained model to language-free inference |
| `pseudo_label.py` | Self-training: pseudo-labels on unlabelled images |
| `mask_eval.py`, `geom_eval.py` | Mask AP and mask-based flaw metrology (flaw size, area, Griffith strength ratio) |
| `bmr.py` | Optional box-mask refinement |
| `baseline_yolo.py` | YOLO11 / RT-DETR baselines on the same splits |
| `bench.py` | Latency and parameter benchmark |
| `prep/` | Converters from the original dataset releases to COCO json (deterministic, seeded few-shot splits) |
| `analysis/` | Weibull and aggregation scripts used for the tables |
| `splits/` | Image-file lists of every split used in the paper |
| `results/` | `eval_final.json` of every run reported in the paper |

## Datasets

The original images and annotations are **not redistributed**; please obtain them from their authors under their licences
and place them under `data/` as listed below. `prep/*.py` then rebuild exactly the splits in `splits/`.

| Dataset | Glass product | Source | Licence / terms | Expected path |
|---|---|---|---|---|
| SSGD | phone cover glass | https://github.com/VincentHancoder/SSGD | academic use only | `data/ssgd/SSGD` |
| GlassCoverDefectDataset | phone cover glass | https://github.com/Johnhushwill/GlassCoverDefectDataset | cite the source | `data/glasscover/GlassCover_datset` |
| MSD | phone screen | https://github.com/jianzhang96/MSD | GPL-3.0 | `data/msd/MSD` |
| AGDD | aircraft canopy | https://github.com/core128/AGDD | CC BY-NC-SA 4.0 | `data/agdd/data` |
| MVTec AD (bottle) | glass bottle | https://www.mvtec.com/company/research/datasets/mvtec-ad | CC BY-NC-SA 4.0 | `data/mvtec_bottle` (via `prep/export_bottle.py`) |
| MVTec AD 2 (vial) | glass vial | https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2 | CC BY-NC-SA 4.0 | `data/vial` |
| CCB_GlassDefectDataset | curved tempered glass | https://github.com/shaoqilyx/CCB_GlassDefectDataset | CC BY-NC-SA 4.0 | `data/ccb` |

**SSGD label correction.** The English category names in SSGD's released COCO json are permuted with respect to the
original pinyin XML labels (e.g. the json class `broken` of part lb101 holds `huahen`, i.e. scratches).
`prep/prep_ssgd.py` therefore rebuilds all annotations from the XML files with the mapping
`liewen->crack, huahen->scratch, bengque->chipping, baidian->white spot, louguang->light leakage, pomo->broken film, wuzi->stain`.

## Setup

Tested with Python 3.11, PyTorch 2.5.1 (CUDA 12.1), transformers 5.17, ultralytics 8.4, pycocotools, scikit-image, scipy, opencv.
Pre-trained weights: `IDEA-Research/grounding-dino-tiny` and `facebook/sam-vit-base` from the Hugging Face Hub.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==5.17.0 ultralytics pycocotools scikit-image scipy opencv-python-headless
```

Run all commands from the repository root (`PYTHONPATH=.` for scripts in `analysis/`).

## Reproducing the main results

```bash
# 1. data
python prep/prep_ssgd.py && python prep/prep_glasscover.py && python prep/prep_msd.py
# 2. few-shot fine-tuning (example: SSGD, 50 images, seed 0)
python train.py --train data/ssgd_coco/fs50_s0.json --val data/ssgd_coco/all_val.json --sfa 0 --iters 2000 --out runs/fs50_s0/gd
# 3. self-training on the unlabelled remainder of the training split
python pseudo_label.py --ckpt runs/fs50_s0/gd/model.pt --labelled data/ssgd_coco/fs50_s0.json --pool data/ssgd_coco/all_train.json \
    --img_root data/ssgd/SSGD --classes "crack,scratch,chipping,white spot,light leakage,broken film,stain" --out runs/pl/ssgd_fs50_s0
python train.py --train runs/pl/ssgd_fs50_s0/train_score.json --val data/ssgd_coco/all_val.json --sfa 0 --iters 3000 --out runs/pl/ssgd_fs50_s0_score
# 4. language-free conversion
python train.py --train data/ssgd_coco/fs50_s0.json --val data/ssgd_coco/all_val.json --sfa 0 --lf_from runs/fs50_s0/gd/model.pt \
    --iters 1000 --lr 5e-5 --out runs/fs50_s0/gd_lf
# 5. masks, flaw metrology and Weibull statistics (MSD example)
python geom_eval.py --ckpt runs/msd_k30_s0/gd/model.pt --val data/msd_coco/test.json --img_root data/msd/MSD \
    --classes oil,stain,scratch --out runs/geom2/msd_k30_s0
PYTHONPATH=. python analysis/weibull.py msd
```

## Licence

Code: MIT (see `LICENSE`). Data splits and results: CC BY 4.0. The original datasets keep their own licences.
