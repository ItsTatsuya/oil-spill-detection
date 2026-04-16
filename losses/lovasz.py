from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-8)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


class LovaszHingeLoss(nn.Module):

    def __init__(self, per_image: bool = False) -> None:
        super().__init__()
        self.per_image = per_image

    def _flatten_binary_scores(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)
        return logits, labels

    def _lovasz_hinge_flat(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.numel() == 0:
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = labels[perm]
        grad = _lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.per_image:
            losses = []
            for logit, label in zip(logits, labels):
                logit_f, label_f = self._flatten_binary_scores(logit, label)
                losses.append(self._lovasz_hinge_flat(logit_f, label_f))
            return torch.stack(losses).mean() if losses else logits.sum() * 0.0
        logits_f, labels_f = self._flatten_binary_scores(logits, labels)
        return self._lovasz_hinge_flat(logits_f, labels_f)


class LovaszSoftmaxLoss(nn.Module):

    def __init__(self, classes: str = "present", per_image: bool = False) -> None:
        super().__init__()
        self.classes = classes
        self.per_image = per_image

    def _flatten_probas(
        self, probas: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = probas.shape[1]
        probas = probas.permute(0, 2, 3, 1).reshape(-1, c)
        labels = labels.reshape(-1)
        valid = (labels >= 0) & (labels < c)
        if torch.any(valid):
            probas = probas[valid]
            labels = labels[valid]
        else:
            probas = probas[:0]
            labels = labels[:0]
        return probas, labels

    def _lovasz_softmax_flat(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if probas.numel() == 0:
            return probas.sum() * 0.0
        c = probas.shape[1]
        losses = []
        class_iter = range(c)
        for cls in class_iter:
            fg = (labels == cls).float()
            if self.classes == "present" and fg.sum() == 0:
                continue
            class_pred = probas[:, cls]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
        if not losses:
            return probas.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.per_image:
            losses = []
            for prob, label in zip(probas, labels):
                prob_f, label_f = self._flatten_probas(prob.unsqueeze(0), label.unsqueeze(0))
                losses.append(self._lovasz_softmax_flat(prob_f, label_f))
            return torch.stack(losses).mean() if losses else probas.sum() * 0.0
        prob_f, label_f = self._flatten_probas(probas, labels)
        return self._lovasz_softmax_flat(prob_f, label_f)
