import torch
import torch.nn.functional as F
from transformers.loss.loss_grounding_dino import GroundingDinoHungarianMatcher, GroundingDinoImageLoss

W = {'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}


class SharedPromptLoss(GroundingDinoImageLoss):
    """HF loss offsets class labels by len(label_maps[i]) instead of a running sum, which is
    wrong for batch >= 3. Every image here shares one prompt, so index the one label map."""

    def _get_target_classes_one_hot(self, outputs, targets, indices):
        logits = outputs['logits']
        lm = outputs['label_maps'][0]
        idx = self._get_source_permutation_idx(indices)
        cls = torch.cat([t['class_labels'][J] for t, (_, J) in zip(targets, indices)])
        onehot = torch.zeros_like(logits, dtype=torch.long)
        onehot[idx] = lm[cls].long()
        return onehot


class DetLoss:
    def __init__(self, label_map):
        self.crit = SharedPromptLoss(GroundingDinoHungarianMatcher(class_cost=1.0, bbox_cost=5.0, giou_cost=2.0),
                                     focal_alpha=0.25, losses=['labels', 'boxes'])
        self.label_map = label_map

    def _one(self, logits, boxes, targets, text_mask):
        B = logits.shape[0]
        o = {'logits': logits, 'pred_boxes': boxes, 'label_maps': (self.label_map.to(logits.device),) * B,
             'text_mask': text_mask}
        d = self.crit(o, targets)
        return sum(d[k] * W[k] for k in W)

    def __call__(self, out, targets):
        B = out['logits'].shape[0]
        tm = torch.zeros_like(out['logits'], dtype=torch.bool)
        tm[:, :, :out['text_mask'].shape[1]] = out['text_mask'][:, None, :]
        loss = 0.0
        for lv in range(out['cls_all'].shape[0]):  # decoder layers (last + auxiliary)
            loss = loss + self._one(out['cls_all'][lv], out['box_all'][lv], targets, tm)
        loss = loss + self._one(out['enc_logits'], out['enc_boxes'], targets, tm)
        return loss


def kd_loss(student, teacher, text_mask):
    """Soft-label focal-free BCE on valid text tokens + MSE on encoder vision features."""
    T = student['logits'].shape[-1]
    tm = torch.zeros_like(student['logits'], dtype=torch.bool)
    tm[:, :, :text_mask.shape[1]] = text_mask[:, None, :]
    s = student['logits'][tm]
    t = teacher['logits'][tm].sigmoid()
    l_cls = F.binary_cross_entropy_with_logits(s, t)
    l_feat = F.mse_loss(student['enc_v'], teacher['enc_v'])
    return l_cls * 10.0 + l_feat
