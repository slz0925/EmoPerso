"""Shared interaction model, style supervision and stopped-teacher consistency."""
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from .augmentation import DEFAULT_STYLES, style_categories, validate_pairs
from .errors import DataContractError
from .interaction import EmoPersoInteractionModel, InteractionLoss, _weight


def bernoulli_kl(teacher_logits, student_logits):
    """KL(Ber(sigmoid(teacher.detach())) || Ber(sigmoid(student))), mean over 4.

    BCE(student,p)-BCE(teacher,p) avoids log(0), 0*log(0), and probability
    clipping. Returns one value per view. FP16/BF16 accumulate in FP32.
    """
    if (not isinstance(teacher_logits, torch.Tensor) or not isinstance(student_logits, torch.Tensor)
            or teacher_logits.ndim != 2 or teacher_logits.shape[1] != 4
            or teacher_logits.shape != student_logits.shape or teacher_logits.device != student_logits.device
            or not teacher_logits.is_floating_point() or not student_logits.is_floating_point()
            or not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all()):
        raise DataContractError("Expected finite matching teacher/student logits [V,4]")
    dtype = torch.float64 if torch.float64 in (teacher_logits.dtype, student_logits.dtype) else torch.float32
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        t, s = teacher_logits.detach().to(dtype), student_logits.to(dtype)
        p = t.sigmoid()
        cross = F.binary_cross_entropy_with_logits(s, p, reduction="none")
        entropy = F.binary_cross_entropy_with_logits(t, p, reduction="none")
        return (cross-entropy).clamp_min(0).mean(-1)


def user_mean(values, indices):
    """Average each represented user's views first, then average those users."""
    if values.ndim != 1 or len(values) != len(indices) or not len(indices):
        raise DataContractError("Nonempty view losses require one user index each")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
        raise DataContractError("View user indices must be nonnegative integers")
    return torch.stack([values[[j for j, i in enumerate(indices) if i == k]].mean()
                        for k in sorted(set(indices))]).mean()


@dataclass(frozen=True)
class ViewPrediction:
    personality_logits: torch.Tensor
    style_logits: torch.Tensor | None
    original_indices: tuple[int, ...]
    styles: tuple[str | None, ...]
    view_ids: tuple[str, ...]
    batch_signature: str
    intermediate: object | None = field(default=None, repr=False)


@dataclass(frozen=True)
class AugmentationOutput:
    original: object = field(repr=False)
    views: tuple[ViewPrediction, ...]
    styles: tuple[str, ...]


class EmoPersoAugmentationModel(nn.Module):
    """One registered interaction model and an optional two-layer style head.

    Pass a PairedBatches for training views; ordinary inference can call the
    original interaction_model directly. Full view Token tensors are returned
    only with return_view_intermediates=True. Autograd still retains the
    activations required for backward; cap generated views for large models.
    Generation is external to this module and its checkpoint.
    styles=() omits the style head for completion-only training; completion
    groups have style_logits=None, including in a model with styles enabled.
    """
    def __init__(self, interaction_model, *, styles=DEFAULT_STYLES, style_hidden_size=None):
        super().__init__()
        if not isinstance(interaction_model, EmoPersoInteractionModel):
            raise TypeError("Expected EmoPersoInteractionModel")
        self.styles = style_categories(styles, allow_empty=True)
        self.interaction_model = interaction_model
        w = interaction_model.mtl.personality_head.weight
        hidden = w.shape[1] if style_hidden_size is None else style_hidden_size
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("style_hidden_size must be a positive integer")
        self.style_head = (nn.Sequential(nn.Linear(w.shape[1], hidden, device=w.device, dtype=w.dtype),
            nn.ReLU(), nn.Linear(hidden, len(self.styles), device=w.device, dtype=w.dtype))
            if self.styles else None)

    def get_extra_state(self):
        return {"style_categories": self.styles}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("style_categories") != self.styles:
            raise DataContractError("Checkpoint style categories/order differ from this model")

    def forward(self, paired, *, return_view_intermediates=False):
        validate_pairs(paired)
        if not isinstance(return_view_intermediates, bool):
            raise ValueError("return_view_intermediates must be boolean")
        original = self.interaction_model(paired.original)
        outputs = []
        for group in paired.views:
            styles = tuple(v.style for v in group.views)
            if any(s is not None and s not in self.styles for s in styles):
                raise DataContractError("View style is not in the model category order")
            out = self.interaction_model(group.batch)
            logits = (self.style_head(out.shared_embedding.to(self.style_head[0].weight.dtype))
                      if self.style_head is not None and any(s is not None for s in styles) else None)
            outputs.append(ViewPrediction(out.personality_logits, logits, group.original_indices,
                styles, tuple(v.view_id for v in group.views), out.batch_signature,
                out if return_view_intermediates else None))
        return AugmentationOutput(original, tuple(outputs), self.styles)


@dataclass(frozen=True)
class AugmentationLosses:
    base_loss: object = field(repr=False)
    style_loss: torch.Tensor
    consistency_loss: torch.Tensor
    total_loss: torch.Tensor
    valid_style_views: int
    valid_consistency_views: int
    consistency_users: int
    has_active_supervision: bool


@dataclass(frozen=True)
class AugmentationAuxiliaryLosses:
    style_loss: torch.Tensor
    consistency_loss: torch.Tensor
    total_loss: torch.Tensor
    valid_style_views: int
    valid_consistency_views: int
    consistency_users: int
    has_active_supervision: bool


class AugmentationLoss(nn.Module):
    """Original-only InteractionLoss + .1*style CE + .1*personality KL.

    Both added terms average views per user, then users. Completion has no style
    target. Zero-weight terms never connect to total_loss. If no active terms,
    return a detached zero and skip both backward and optimizer.step.
    """
    def __init__(self, *, base_loss=None, styles=DEFAULT_STYLES, style_weight=.1, consistency_weight=.1):
        super().__init__()
        self.base_loss = InteractionLoss() if base_loss is None else base_loss
        if not isinstance(self.base_loss, InteractionLoss):
            raise TypeError("Expected InteractionLoss")
        self.styles = style_categories(styles, allow_empty=True)
        self.style_weight = _weight(style_weight, "style_weight")
        self.consistency_weight = _weight(consistency_weight, "consistency_weight")

    def auxiliary(self, output):
        """Compute style and consistency objectives separately from base supervision."""
        if not isinstance(output, AugmentationOutput) or output.styles != self.styles:
            raise DataContractError("Expected augmentation output with matching style category order")
        pl = output.original.personality_logits
        zero = torch.zeros((), device=pl.device, dtype=torch.float64 if pl.dtype == torch.float64 else torch.float32)
        kl_values, kl_indices, style_values, style_indices = [], [], [], []
        seen = set()
        for group in output.views:
            n = len(group.original_indices)
            if (not n or len(group.styles) != n or len(group.view_ids) != n
                    or group.personality_logits.shape != (n, 4)
                    or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(pl) for i in group.original_indices)
                    or any(s is not None and s not in self.styles for s in group.styles)
                    or len(set(group.view_ids)) != n or seen.intersection(group.view_ids)):
                raise DataContractError("Invalid augmented predictions or original-user mapping")
            valid = [i for i, s in enumerate(group.styles) if s is not None]
            sl = group.style_logits
            if sl is None:
                if valid:
                    raise DataContractError("Style-labelled views require style logits")
            elif (not self.styles or not isinstance(sl, torch.Tensor) or sl.shape != (n, len(self.styles))
                    or sl.device != pl.device or not sl.is_floating_point() or not torch.isfinite(sl).all()):
                raise DataContractError("Invalid style logits or disabled style head")
            seen.update(group.view_ids)
            teacher = pl[list(group.original_indices)]
            kl_values.append(bernoulli_kl(teacher, group.personality_logits))
            kl_indices.extend(group.original_indices)
            if valid:
                dtype = torch.float64 if group.style_logits.dtype == torch.float64 else torch.float32
                labels = torch.tensor([self.styles.index(group.styles[i]) for i in valid], device=pl.device)
                with torch.autocast(device_type=pl.device.type, enabled=False):
                    style_values.append(F.cross_entropy(group.style_logits[valid].to(dtype), labels, reduction="none"))
                style_indices.extend(group.original_indices[i] for i in valid)
        style = user_mean(torch.cat(style_values), style_indices) if style_values else zero
        consistency = user_mean(torch.cat(kl_values), kl_indices) if kl_values else zero
        terms = []
        if self.style_weight and style_indices:
            terms.append(self.style_weight*style)
        if self.consistency_weight and kl_indices:
            terms.append(self.consistency_weight*consistency)
        total = sum(terms[1:], terms[0]) if terms else zero
        return AugmentationAuxiliaryLosses(style, consistency, total, len(style_indices), len(kl_indices),
            len(set(kl_indices)), bool(terms))

    def forward(self, output, targets):
        if not isinstance(output, AugmentationOutput) or output.styles != self.styles:
            raise DataContractError("Expected augmentation output with matching style category order")
        base = self.base_loss(output.original, targets)
        aux = self.auxiliary(output)
        terms = [base.total_loss] if base.has_active_supervision else []
        if self.style_weight and aux.valid_style_views:
            terms.append(self.style_weight*aux.style_loss)
        if self.consistency_weight and aux.valid_consistency_views:
            terms.append(self.consistency_weight*aux.consistency_loss)
        total = sum(terms[1:], terms[0]) if terms else torch.zeros_like(base.total_loss, requires_grad=False)
        return AugmentationLosses(base, aux.style_loss, aux.consistency_loss, total,
            aux.valid_style_views, aux.valid_consistency_views, aux.consistency_users, bool(terms))
