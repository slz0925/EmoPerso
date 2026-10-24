"""Personality/emotion heads and separate, source-aligned multi-task BCE."""
from dataclasses import dataclass, field
from numbers import Real
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import EmoPersoEncoder
from .errors import DataContractError
from .emotion import DEFAULT_EMOTIONS, EmotionLabeler, emotion_categories
from .token_inputs import batch_signature, _integer


@dataclass(frozen=True)
class MTLOutput:
    shared_embedding: torch.Tensor
    personality_embedding: torch.Tensor
    emotion_embedding: torch.Tensor
    personality_logits: torch.Tensor
    emotion_logits: torch.Tensor
    personality_probabilities: torch.Tensor
    emotion_probabilities: torch.Tensor
    token_features: torch.Tensor
    content_mask: torch.Tensor
    attention_weights: torch.Tensor | None
    categories: tuple[str, ...]
    batch_signature: str
    metadata: dict = field(repr=False)


class EmoPersoMTL(nn.Module):
    """Compose an Encoder with independent two-layer task MLPs and sigmoid heads.

    New parameters follow the shared projection's device/dtype; the frozen
    Backbone may remain dispatched elsewhere. Forward never prepares targets.
    """
    def __init__(self, encoder, *, hidden_size=None, categories=DEFAULT_EMOTIONS, dropout=0.2):
        super().__init__()
        if not isinstance(encoder, EmoPersoEncoder):
            raise TypeError("Expected EmoPersoEncoder")
        d = encoder.projection.out_features
        hidden_size = d if hidden_size is None else hidden_size
        if not _integer(hidden_size) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite in [0,1)")
        self.categories = emotion_categories(categories)
        self.encoder = encoder
        options = dict(device=encoder.projection.weight.device, dtype=encoder.projection.weight.dtype)
        def mlp():
            return nn.Sequential(nn.Linear(d, hidden_size, **options), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_size, d, **options))
        self.personality_mlp, self.emotion_mlp = mlp(), mlp()
        self.personality_head = nn.Linear(d, 4, **options)
        self.emotion_head = nn.Linear(d, len(self.categories), **options)

    def get_extra_state(self):
        return {"emotion_categories": self.categories}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("emotion_categories") != self.categories:
            raise DataContractError("Checkpoint emotion categories/order differ from this model")

    def forward(self, batch, *, return_attention_weights=False):
        signature = batch_signature(batch)
        encoded = self.encoder(batch, return_attention_weights=return_attention_weights)
        z = encoded.pooled_features
        pers = self.personality_mlp(z.to(dtype=self.personality_head.weight.dtype))
        emo = self.emotion_mlp(z.to(dtype=self.emotion_head.weight.dtype))
        pl, el = self.personality_head(pers), self.emotion_head(emo)
        if not torch.isfinite(pl).all() or not torch.isfinite(el).all():
            raise DataContractError("Nonfinite MTL logits")
        return MTLOutput(z, pers, emo, pl, el, pl.sigmoid(), el.sigmoid(),
                         encoded.token_features, encoded.content_mask, encoded.attention_weights,
                         self.categories, signature, encoded.metadata)


@dataclass(frozen=True)
class ExternalEmotionTargets:
    """Explicit user-level auxiliary targets, in exactly the current Batch order."""
    identities: tuple[tuple[str, str], ...]
    categories: tuple[str, ...]
    targets: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class MTLTargets:
    personality_labels: np.ndarray = field(repr=False)
    emotion_targets: np.ndarray = field(repr=False)
    emotion_valid: np.ndarray = field(repr=False)
    categories: tuple[str, ...]
    batch_signature: str


def _binary(value, shape, name, *, boolean=False):
    if (not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind not in "biuf"
            or not np.isfinite(value).all() or not np.isin(value, (0, 1)).all()
            or boolean and value.dtype != np.bool_):
        raise DataContractError(f"{name} must be {'boolean' if boolean else 'binary numeric'} {shape}")
    return value


def prepare_targets(batch, *, labeler=None, categories=DEFAULT_EMOTIONS, external=None):
    """Prepare source-aligned CPU targets independently from model inference.

    Use on train batches for optimization, validation batches only for evaluation.
    No persona label is ever used to infer an emotion target. Generic tokenizers
    without retained offsets can use explicit ExternalEmotionTargets instead.
    """
    signature = batch_signature(batch)
    categories = emotion_categories(categories)
    labels = batch.get("labels")
    if labels is None:
        raise DataContractError("MTL supervision requires personality labels; inference needs no targets")
    n = len(batch["user_keys"])
    if external is None:
        labeler = EmotionLabeler(categories) if labeler is None else labeler
        if not isinstance(labeler, EmotionLabeler) or labeler.categories != categories:
            raise DataContractError("Pseudo-label categories/order differ from target configuration")
        targets, valid = labeler.label_batch(batch)
    else:
        if labeler is not None:
            raise ValueError("Choose heuristic or external emotion supervision, not both")
        if (not isinstance(external, ExternalEmotionTargets) or external.categories != categories
                or external.identities != tuple(zip(batch["dataset_ids"], batch["user_keys"]))):
            raise DataContractError("External emotion targets must match Batch identities/order and categories")
        targets, valid = external.targets, external.valid
    _binary(labels, (n,4), "personality_labels")
    _binary(targets, (n,len(categories)), "emotion_targets")
    _binary(valid, (n,), "emotion_valid", boolean=True)
    def snapshot(array):
        result = array.copy()
        result.flags.writeable = False
        return result
    return MTLTargets(snapshot(labels), snapshot(targets), snapshot(valid), categories, signature)


@dataclass(frozen=True)
class MTLLosses:
    personality_loss: torch.Tensor
    emotion_loss: torch.Tensor
    total_loss: torch.Tensor
    valid_emotion_count: int
    has_active_supervision: bool


class MultiTaskLoss(nn.Module):
    """Mean BCE over B*4 and valid_B*C, followed by configurable task weights.

    Half/BF16 logits accumulate BCE in float32, double remains double. When no
    emotion rows are valid, its loss is a detached zero. Only positively weighted
    supervised terms enter total_loss. With no active supervision, skip backward
    and optimizer.step; total_loss is a detached scalar.
    """
    def __init__(self, *, categories=DEFAULT_EMOTIONS, personality_weight=0.7, emotion_weight=0.3):
        super().__init__()
        self.categories = emotion_categories(categories)
        weights = (personality_weight, emotion_weight)
        if (any(isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w < 0 for w in weights)
                or not any(weights)):
            raise ValueError("Loss weights must be finite, nonnegative, with a positive total")
        self.personality_weight, self.emotion_weight = map(float, weights)

    def forward(self, output, targets):
        if not isinstance(output, MTLOutput) or not isinstance(targets, MTLTargets):
            raise TypeError("Expected MTLOutput and MTLTargets")
        if (output.categories != self.categories or targets.categories != self.categories
                or output.batch_signature != targets.batch_signature):
            raise DataContractError("Output and supervision must match Batch identity and emotion category order")
        pl, el = output.personality_logits, output.emotion_logits
        if (not isinstance(pl, torch.Tensor) or not pl.is_floating_point() or pl.ndim != 2
                or not pl.shape[0] or pl.shape[1] != 4 or not isinstance(el, torch.Tensor)
                or not el.is_floating_point() or el.shape != (pl.shape[0],len(self.categories))
                or el.device != pl.device or not torch.isfinite(pl).all() or not torch.isfinite(el).all()):
            raise DataContractError("Expected finite personality [B,4] and emotion [B,C] logits on one device")
        n = pl.shape[0]
        _binary(targets.personality_labels, (n,4), "personality_labels")
        _binary(targets.emotion_targets, (n,len(self.categories)), "emotion_targets")
        _binary(targets.emotion_valid, (n,), "emotion_valid", boolean=True)
        dtype = torch.float64 if pl.dtype == torch.float64 or el.dtype == torch.float64 else torch.float32
        p = torch.tensor(targets.personality_labels, device=pl.device, dtype=dtype)
        e = torch.tensor(targets.emotion_targets, device=pl.device, dtype=dtype)
        valid = torch.tensor(targets.emotion_valid, device=pl.device, dtype=torch.bool)
        pers = F.binary_cross_entropy_with_logits(pl.to(dtype), p)
        count = int(targets.emotion_valid.sum())
        zero = torch.zeros((), dtype=dtype, device=pl.device)
        emo = F.binary_cross_entropy_with_logits(el[valid].to(dtype), e[valid]) if count else zero
        terms = []
        if self.personality_weight:
            terms.append(self.personality_weight*pers)
        if self.emotion_weight and count:
            terms.append(self.emotion_weight*emo)
        total = sum(terms[1:], terms[0]) if terms else zero
        return MTLLosses(pers, emo, total, count, bool(terms))
