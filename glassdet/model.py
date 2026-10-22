"""Grounding DINO + SAM for glass defects.

GlassDet wraps HF GroundingDinoForObjectDetection and SamModel:
  * SFA  - frozen SAM ViT embedding is injected into every DINO feature level
           through zero-initialised gates (starts exactly at the pretrained model).
  * mask - DINO boxes prompt the SAM mask decoder, which re-uses the same SAM
           embedding (one image encoding for both detection and segmentation).
  * LF   - language-free inference: the BERT branch is removed and the text path
           inside every encoder layer is replaced by static learned class tokens,
           so only the image->text half of each fusion layer is computed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, GroundingDinoForObjectDetection, SamModel
from torchvision.ops import roi_align
from transformers.models.grounding_dino.modeling_grounding_dino import (
    build_label_maps, generate_masks_with_special_tokens_and_transfer_map)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SamFusion(nn.Module):
    def __init__(self, n_levels=4, d=256, sam_dim=256, gate_init=0.0):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(sam_dim, d, 3, padding=1), nn.GroupNorm(32, d))
                                   for _ in range(n_levels)])
        self.gate = nn.ParameterList([nn.Parameter(torch.full((1, d, 1, 1), gate_init)) for _ in range(n_levels)])

    def forward(self, level, feat, sam_emb):
        s = F.interpolate(sam_emb, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        return feat + self.gate[level] * self.proj[level](s)


class SamQueryFusion(nn.Module):
    """Two-stage Grounding DINO starts every decoder query from the same learned content embedding.
    SQF gives each query the SAM description of its own proposal box (RoIAlign 7x7 -> mean|max -> MLP),
    added through a zero-initialised gate."""

    def __init__(self, d=256, sam_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * sam_dim, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Parameter(torch.zeros(d))

    def forward(self, target, ref_cxcywh, sam_valid):
        B, Q, _ = ref_cxcywh.shape
        h, w = sam_valid.shape[-2:]
        cx, cy, bw, bh = ref_cxcywh.float().unbind(-1)
        xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
        f = roi_align(sam_valid.float(), list(xyxy), output_size=7, spatial_scale=1.0, sampling_ratio=2, aligned=True)
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1).view(B, Q, -1).to(target.dtype)
        return target + self.gate * self.norm(self.mlp(f))


class GlassDet(nn.Module):
    def __init__(self, gd_path, sam_path=None, classes=(), sfa=True, freeze_backbone=True, gate_init=0.0, sqf=False):
        super().__init__()
        self.gd = GroundingDinoForObjectDetection.from_pretrained(gd_path)
        self.gd.config.auxiliary_loss = True
        self.tok = AutoTokenizer.from_pretrained(gd_path)
        self.classes = list(classes)
        self.use_sfa = sfa
        self.sam = SamModel.from_pretrained(sam_path) if sam_path else None
        if self.sam is not None:
            for p in self.sam.vision_encoder.parameters():
                p.requires_grad_(False)
            for p in self.sam.prompt_encoder.parameters():
                p.requires_grad_(False)
        self.sfa = SamFusion(self.gd.config.num_feature_levels, gate_init=gate_init) if sfa else None
        self.sqf = SamQueryFusion() if sqf else None
        if freeze_backbone:
            for p in self.gd.model.backbone.parameters():
                p.requires_grad_(False)
            for p in self.gd.model.text_backbone.parameters():
                p.requires_grad_(False)
        # tokenised prompt "crack. scratch. ..." is fixed for the whole run
        prompt = '. '.join(self.classes) + '.'
        t = self.tok(prompt, return_tensors='pt')
        self.register_buffer('input_ids', t['input_ids'], persistent=False)
        self.register_buffer('attn_mask', t['attention_mask'], persistent=False)
        self.label_map = build_label_maps(torch.zeros(1, 1, self.gd.config.max_text_len), t['input_ids'])[0].float()
        self.lang_free = False
        self.static_text = None  # (L+1, T, d) after convert_language_free
        # pure-PyTorch deformable attention over ~20k tokens dominates memory -> checkpoint it
        for layer in self.gd.model.encoder.layers:
            f = layer.deformable_layer.forward
            layer.deformable_layer.forward = (
                lambda *a, _f=f, **k: checkpoint(_f, *a, use_reentrant=False, **k) if torch.is_grad_enabled()
                else _f(*a, **k))

    # ------------------------------------------------------------------ SAM
    def expand_stem(self, extra_ch):
        """Zero-initialised channel expansion of the Swin patch embedding: the pretrained RGB filters are kept and the
        filters of the extra (optical-contrast) channels start at zero, so training starts from the pretrained model."""
        emb = self.gd.model.backbone.conv_encoder.model.swin.embeddings.patch_embeddings
        old = emb.projection
        new = nn.Conv2d(old.in_channels + extra_ch, old.out_channels, old.kernel_size, old.stride, old.padding,
                        bias=old.bias is not None).to(old.weight.device)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_channels] = old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
        emb.projection = new
        emb.num_channels = new.in_channels
        for q in new.parameters():
            q.requires_grad_(True)

    def sam_encode(self, pix):
        """pix: normalised (B,3[+extra],H,W) DINO input -> (full SAM embedding, valid crop, SAM-input hw)."""
        pix = pix[:, :3]  # SAM always sees the natural image
        B, _, H, W = pix.shape
        s = 1024 / max(H, W)
        h, w = int(round(H * s)), int(round(W * s))
        x = F.interpolate(pix, size=(h, w), mode='bilinear', align_corners=False)
        x = F.pad(x, (0, 1024 - w, 0, 1024 - h))
        with torch.no_grad():
            emb = self.sam.get_image_embeddings(x)  # (B,256,64,64)
        vh, vw = int(round(h / 16)), int(round(w / 16))
        return emb, emb[:, :, :vh, :vw], (h, w)

    # ------------------------------------------------------------- text
    def text_inputs(self, B):
        ids = self.input_ids.expand(B, -1)
        am = self.attn_mask.expand(B, -1)
        tsam, pos_ids = generate_masks_with_special_tokens_and_transfer_map(ids)
        return ids, am, tsam, pos_ids

    # ------------------------------------------------------------- forward
    def forward(self, pix, sam_valid=None, return_text_states=False):
        m = self.gd.model
        cfg = self.gd.config
        B, _, H, W = pix.shape
        ids, am, tsam, pos_ids = self.text_inputs(B)
        text_token_mask = am.bool()
        if not self.lang_free:
            tout = m.text_backbone(ids, tsam[:, None], torch.zeros_like(ids), pos_ids, return_dict=True)
            text_features = m.text_projection(tout.last_hidden_state)
        else:
            text_features = self.static_text[0].unsqueeze(0).expand(B, -1, -1)

        pixel_mask = torch.ones((B, H, W), dtype=torch.long, device=pix.device)
        vision_features, pos_list = m.backbone(pix, pixel_mask)
        feature_maps, masks = [], []
        for level, (source, mask) in enumerate(vision_features):
            f = m.input_proj_vision[level](source)
            if self.sfa is not None and sam_valid is not None:
                f = self.sfa(level, f, sam_valid)
            feature_maps.append(f)
            masks.append(mask)
        n0 = len(feature_maps)
        for level in range(n0, cfg.num_feature_levels):
            source = m.input_proj_vision[level](vision_features[-1][0] if level == n0 else feature_maps[-1])
            if self.sfa is not None and sam_valid is not None:
                source = self.sfa(level, source, sam_valid)
            mask = F.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
            pos_list.append(m.backbone.position_embedding(source, mask).to(source.dtype))
            feature_maps.append(source)
            masks.append(mask)

        src, msk, lpos, shapes = [], [], [], []
        for level, (s, mk, pe) in enumerate(zip(feature_maps, masks, pos_list)):
            shapes.append(tuple(s.shape[-2:]))
            src.append(s.flatten(2).transpose(1, 2))
            msk.append(mk.flatten(1))
            lpos.append(pe.flatten(2).transpose(1, 2) + m.level_embed[level].view(1, 1, -1))
        src, msk, lpos = torch.cat(src, 1), torch.cat(msk, 1), torch.cat(lpos, 1)
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=pix.device)
        level_start = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([m.get_valid_ratio(mk, dtype=src.dtype) for mk in masks], 1)

        text_states = None
        if not self.lang_free:
            enc = m.encoder(vision_features=src, vision_attention_mask=~msk, vision_position_embedding=lpos,
                            spatial_shapes=spatial_shapes, spatial_shapes_list=shapes, level_start_index=level_start,
                            valid_ratios=valid_ratios, text_features=text_features,
                            text_attention_mask=~text_token_mask, text_position_embedding=None,
                            text_self_attention_masks=~tsam, text_position_ids=pos_ids,
                            output_hidden_states=return_text_states, return_dict=True)
            enc_v, enc_t = enc.last_hidden_state_vision, enc.last_hidden_state_text
            if return_text_states:
                text_states = enc.text_hidden_states  # L+1 tensors
        else:
            enc_v = self._lf_encoder(src, msk, lpos, spatial_shapes, shapes, level_start, valid_ratios, B)
            enc_t = self.static_text[-1].unsqueeze(0).expand(B, -1, -1)

        # two-stage query selection (as in GroundingDinoModel.forward)
        obj_q, proposals = m.generate_encoder_output_proposals(enc_v, ~msk, shapes)
        enc_cls = m.encoder_output_class_embed(obj_q, enc_t, text_token_mask)
        enc_coord = m.encoder_output_bbox_embed(obj_q) + proposals
        topk = torch.topk(enc_cls.max(-1)[0], cfg.num_queries, dim=1)[1]
        enc_boxes = torch.gather(enc_coord, 1, topk.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()
        ref = enc_boxes.detach()
        target = m.query_position_embeddings.weight.unsqueeze(0).repeat(B, 1, 1)
        if self.sqf is not None and sam_valid is not None:
            target = self.sqf(target, ref, sam_valid)
        # intermediate (encoder) supervision on the selected proposals, as in the original
        # Grounding DINO; HF instead scores the learned query embeddings against the
        # pre-encoder text, which yields logits of +-160 and a 1e5-scale loss.
        enc_logits = torch.gather(enc_cls, 1, topk.unsqueeze(-1).repeat(1, 1, enc_cls.shape[-1]))

        dec = m.decoder(inputs_embeds=target, vision_encoder_hidden_states=enc_v, vision_encoder_attention_mask=msk,
                        text_encoder_hidden_states=enc_t, text_encoder_attention_mask=~text_token_mask,
                        reference_points=ref, spatial_shapes=spatial_shapes, spatial_shapes_list=shapes,
                        level_start_index=level_start, valid_ratios=valid_ratios, return_dict=True)
        hs, inter_ref = dec.intermediate_hidden_states, dec.intermediate_reference_points
        cls_all, box_all = [], []
        for lv in range(hs.shape[1]):
            r = ref if lv == 0 else inter_ref[:, lv - 1]
            r = torch.special.logit(r, eps=1e-5)
            cls_all.append(self.gd.class_embed[lv](vision_hidden_state=hs[:, lv], text_hidden_state=enc_t,
                                                   text_token_mask=text_token_mask))
            box_all.append((self.gd.bbox_embed[lv](hs[:, lv]) + r).sigmoid())
        out = {'logits': cls_all[-1], 'pred_boxes': box_all[-1], 'cls_all': torch.stack(cls_all),
               'box_all': torch.stack(box_all), 'enc_logits': enc_logits, 'enc_boxes': enc_boxes,
               'text_mask': text_token_mask, 'enc_v': enc_v}
        if text_states is not None:
            out['text_states'] = text_states
        return out

    # ------------------------------------------------------- language-free
    def _lf_encoder(self, v, msk, vpos, spatial_shapes, shapes, level_start, valid_ratios, B):
        enc = self.gd.model.encoder
        refp = enc.get_reference_points(shapes, valid_ratios, device=v.device)
        tmask = ~self.attn_mask.bool().expand(B, -1)
        for li, layer in enumerate(enc.layers):
            fl = layer.fusion_layer
            a = fl.attn
            vn = fl.layer_norm_vision(v)
            tn = fl.layer_norm_text(self.static_text[li]).unsqueeze(0).expand(B, -1, -1)
            N = vn.shape[1]
            q = a._reshape(a.vision_proj(vn) * a.scale, N, B).flatten(0, 1)
            k = a._reshape(a.text_proj(tn), -1, B).flatten(0, 1)
            val = a._reshape(a.values_text_proj(tn), -1, B).flatten(0, 1)
            w = torch.bmm(q, k.transpose(1, 2))
            tm = tmask[:, None, None, :].repeat(1, a.num_heads, 1, 1).flatten(0, 1)
            w = w.masked_fill(tm, float('-inf')).softmax(-1)
            o = torch.bmm(w, val).view(B, a.num_heads, N, a.head_dim).transpose(1, 2).reshape(B, N, a.embed_dim)
            v = vn + fl.vision_param * a.out_vision_proj(o)
            v, _ = layer.deformable_layer(hidden_states=v, attention_mask=msk, position_embeddings=vpos,
                                          reference_points=refp, spatial_shapes=spatial_shapes,
                                          spatial_shapes_list=shapes, level_start_index=level_start)
        return v

    @torch.no_grad()
    def collect_text_states(self, loader, device, n_batches=50, use_sam=True):
        """Mean encoder text states over training images (initialises the static bank)."""
        acc, n = None, 0
        for i, (pix, _) in enumerate(loader):
            if i >= n_batches:
                break
            pix = pix.to(device)
            sv = self.sam_encode(pix)[1] if (use_sam and self.sam is not None and self.sfa is not None) else None
            st = torch.stack(self.forward(pix, sv, return_text_states=True)['text_states'])  # (L+1,B,T,d)
            s = st.sum(1)
            acc = s if acc is None else acc + s
            n += pix.shape[0]
        return acc / n

    def convert_language_free(self, bank):
        self.static_text = nn.Parameter(bank.clone())
        self.lang_free = True
        self.gd.model.text_backbone = None  # BERT is gone

    # ------------------------------------------------------------ masks
    @torch.no_grad()
    def predict_masks(self, sam_full, boxes_xyxy, sam_hw, out_hw):
        """boxes_xyxy in the pixel frame of out_hw (same aspect as the SAM input); returns (N,H,W) bool masks."""
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0,) + tuple(out_hw), dtype=torch.bool, device=boxes_xyxy.device)
        h, w = sam_hw
        H, W = out_hw
        b = boxes_xyxy * torch.tensor([w / W, h / H, w / W, h / H], device=boxes_xyxy.device)
        o = self.sam(image_embeddings=sam_full, input_boxes=b[None], multimask_output=False)
        lm = o.pred_masks[0, :, 0]  # (N,256,256)
        mk = F.interpolate(lm[:, None], size=(1024, 1024), mode='bilinear', align_corners=False)[:, 0, :h, :w]
        mk = F.interpolate(mk[:, None], size=(H, W), mode='bilinear', align_corners=False)[:, 0]
        return mk > 0
