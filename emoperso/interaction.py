"""Personality-conditioned context, emotion token pooling, and joint training."""
from dataclasses import dataclass, field
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import _positive
from .errors import DataContractError
from .emotion import DEFAULT_EMOTIONS
from .mtl import EmoPersoMTL, MTLOutput, MultiTaskLoss


def _weight(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _floating(value, shape, device, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or tuple(value.shape) != tuple(shape) or value.device != device
            or not torch.isfinite(value).all()):
        raise DataContractError(f"{name} must be finite floating {tuple(shape)} on the feature device")


@dataclass(frozen=True)
class InteractionFeatures:
    context: torch.Tensor
    contextual_token_features: torch.Tensor
    refined_personality_embedding: torch.Tensor
    emotion_attention_weights: torch.Tensor | None
    cross_attention_weights: torch.Tensor | None


class PersonalityEmotionInteraction(nn.Module):
    """One personality query -> global context -> residual token fusion.

    Emotion pooling scores are v*tanh(W_h*h + W_e*emotion + b). Scoring width
    defaults to min(d,256); all weights are independent of sequence length.
    Attention has no bias: a key projection bias would be a softmax-cancelled
    common shift. FP16/BF16/FP32 compute in FP32, double remains double.
    Optional cross weights are head-averaged with shape [B,T].
    """
    def __init__(self, hidden_size, *, num_heads=4, attention_size=None, device=None, dtype=None):
        super().__init__()
        self.hidden_size = _positive(hidden_size, "hidden_size")
        self.num_heads = _positive(num_heads, "num_heads")
        if hidden_size < 2 or hidden_size % num_heads:
            raise ValueError("hidden_size must be >=2 and divisible by num_heads")
        attention_size = min(hidden_size,256) if attention_size is None else _positive(attention_size,"attention_size")
        if dtype is not None and dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            raise ValueError("Expected a floating interaction dtype")
        options = dict(device=device,dtype=dtype)
        self.cross_attention = nn.MultiheadAttention(hidden_size,num_heads,bias=False,
                                                     dropout=0.,batch_first=True,**options)
        self.token_norm = nn.LayerNorm(hidden_size,**options)
        self.refined_norm = nn.LayerNorm(hidden_size,**options)
        self.token_score = nn.Linear(hidden_size,attention_size,bias=True,**options)
        self.emotion_score = nn.Linear(hidden_size,attention_size,bias=False,**options)
        self.score_vector = nn.Parameter(torch.empty(attention_size,**options))
        nn.init.normal_(self.score_vector,std=attention_size**-.5)

    def forward(self, personality_embedding, emotion_embedding, token_features, content_mask,
                *, return_attention_weights=False):
        h,p,e = token_features,personality_embedding,emotion_embedding
        if (not isinstance(h,torch.Tensor) or not h.is_floating_point() or h.ndim!=3
                or not h.shape[0] or not h.shape[1] or h.shape[2]!=self.hidden_size):
            raise DataContractError("Expected floating Token features [B,T,d]")
        if (not isinstance(content_mask,torch.Tensor) or content_mask.dtype!=torch.bool
                or content_mask.shape!=h.shape[:2] or content_mask.device!=h.device
                or not content_mask.any(1).all()):
            raise DataContractError("Expected bool content_mask [B,T] with content in every row")
        if not isinstance(return_attention_weights,bool):
            raise ValueError("return_attention_weights must be boolean")
        _floating(p,(h.shape[0],self.hidden_size),h.device,"personality_embedding")
        _floating(e,p.shape,h.device,"emotion_embedding")
        if self.score_vector.device!=h.device:
            raise DataContractError("Interaction parameters must share the feature device")
        dtype = torch.float64 if torch.float64 in (h.dtype,p.dtype,e.dtype,self.score_vector.dtype) else torch.float32
        with torch.autocast(device_type=h.device.type,enabled=False):
            safe = h.masked_fill(~content_mask.unsqueeze(-1),0).to(dtype)
            if not torch.isfinite(safe).all():
                raise DataContractError("Nonfinite valid interaction Token features")
            p,e = p.to(dtype),e.to(dtype)
            attn = self.cross_attention
            # Functional attention allows differentiable parameter casts without
            # mutating a dispatched encoder.
            context,weights = F.multi_head_attention_forward(
                p.unsqueeze(0),safe.transpose(0,1),safe.transpose(0,1),self.hidden_size,self.num_heads,
                attn.in_proj_weight.to(dtype),None,None,None,False,0.,
                attn.out_proj.weight.to(dtype),None,training=self.training,
                key_padding_mask=~content_mask,need_weights=return_attention_weights,
                average_attn_weights=True)
            context = context.squeeze(0)
            fused = safe+context.unsqueeze(1)
            contextual = F.layer_norm(fused,(self.hidden_size,),self.token_norm.weight.to(dtype),
                                      self.token_norm.bias.to(dtype),self.token_norm.eps)
            contextual = contextual.masked_fill(~content_mask.unsqueeze(-1),0)
            scores_hidden = F.linear(contextual,self.token_score.weight.to(dtype),self.token_score.bias.to(dtype))
            scores_hidden = scores_hidden+F.linear(e,self.emotion_score.weight.to(dtype)).unsqueeze(1)
            scores = F.linear(torch.tanh(scores_hidden),self.score_vector.to(dtype).unsqueeze(0)).squeeze(-1)
            if not torch.isfinite(scores).all() or not torch.isfinite(contextual).all():
                raise DataContractError("Nonfinite interaction scores or contextual features")
            beta = torch.softmax(scores.masked_fill(~content_mask,-torch.inf),dim=1)
            beta = beta.masked_fill(~content_mask,0)
            pooled = (beta.unsqueeze(-1)*contextual).sum(dim=1)
            refined = F.layer_norm(p+pooled,(self.hidden_size,),self.refined_norm.weight.to(dtype),
                                    self.refined_norm.bias.to(dtype),self.refined_norm.eps)
            if not torch.isfinite(refined).all():
                raise DataContractError("Nonfinite refined personality features")
        return InteractionFeatures(context,contextual,refined,beta if return_attention_weights else None,
                                   weights.squeeze(1) if weights is not None else None)


@dataclass(frozen=True)
class InteractionOutput(MTLOutput):
    """Main personality_logits/probabilities always use the refined embedding."""
    refined_personality_embedding: torch.Tensor
    contextual_token_features: torch.Tensor
    emotion_attention_weights: torch.Tensor | None
    cross_attention_weights: torch.Tensor | None
    mtl_output: MTLOutput = field(repr=False)


class EmoPersoInteractionModel(nn.Module):
    """Single public Batch forward, reusing MTL and its personality classifier."""
    def __init__(self, mtl, *, num_heads=4, attention_size=None):
        super().__init__()
        if not isinstance(mtl,EmoPersoMTL):
            raise TypeError("Expected EmoPersoMTL")
        self.mtl = mtl
        w = mtl.personality_head.weight
        self.interaction = PersonalityEmotionInteraction(w.shape[1],num_heads=num_heads,
            attention_size=attention_size,device=w.device,dtype=w.dtype)

    def forward(self,batch,*,return_attention_weights=False):
        original = self.mtl(batch,return_attention_weights=return_attention_weights)
        features = self.interaction(original.personality_embedding,original.emotion_embedding,
            original.token_features,original.content_mask,return_attention_weights=return_attention_weights)
        logits = self.mtl.personality_head(features.refined_personality_embedding.to(self.mtl.personality_head.weight.dtype))
        if not torch.isfinite(logits).all():
            raise DataContractError("Nonfinite refined personality logits")
        return InteractionOutput(shared_embedding=original.shared_embedding,
            personality_embedding=original.personality_embedding,emotion_embedding=original.emotion_embedding,
            personality_logits=logits,emotion_logits=original.emotion_logits,
            personality_probabilities=logits.sigmoid(),emotion_probabilities=original.emotion_probabilities,
            token_features=original.token_features,content_mask=original.content_mask,
            attention_weights=original.attention_weights,categories=original.categories,
            batch_signature=original.batch_signature,metadata=original.metadata,
            refined_personality_embedding=features.refined_personality_embedding,
            contextual_token_features=features.contextual_token_features,
            emotion_attention_weights=features.emotion_attention_weights,cross_attention_weights=features.cross_attention_weights,
            mtl_output=original)


def cosine_consistency(refined,emotion,valid):
    """Mean (1-cosine) on valid rows. Zero vectors have cosine=0 (eps=1e-8).

    Scale before the norm to avoid overflow on large finite vectors. No valid
    rows returns a detached zero, preserving the optimizer skip contract.
    """
    if not isinstance(refined,torch.Tensor) or refined.ndim!=2 or not all(refined.shape):
        raise DataContractError("Expected refined features [B,d]")
    _floating(refined,refined.shape,refined.device,"refined features")
    _floating(emotion,refined.shape,refined.device,"emotion features")
    if (not isinstance(valid,torch.Tensor) or valid.dtype!=torch.bool or valid.shape!=(refined.shape[0],)
            or valid.device!=refined.device):
        raise DataContractError("Expected bool consistency validity [B] on feature device")
    dtype = torch.float64 if torch.float64 in (refined.dtype,emotion.dtype) else torch.float32
    if not valid.any():
        return torch.zeros((),dtype=dtype,device=refined.device)
    def unit(x):
        x = x[valid].to(dtype)
        scale = x.abs().amax(-1,keepdim=True).clamp_min(1).detach()
        scaled = x/scale
        norm = torch.linalg.vector_norm(scaled,dim=-1,keepdim=True)
        return scaled/torch.maximum(norm,1e-8/scale)
    with torch.autocast(device_type=refined.device.type,enabled=False):
        cosine = (unit(refined)*unit(emotion)).sum(-1).clamp(-1,1)
        return (1-cosine).mean()


@dataclass(frozen=True)
class InteractionLosses:
    personality_loss: torch.Tensor
    emotion_loss: torch.Tensor
    task_loss: torch.Tensor
    cross_loss: torch.Tensor
    total_loss: torch.Tensor
    valid_emotion_count: int
    has_active_supervision: bool


class InteractionLoss(nn.Module):
    """lambda_mtl*(w_p*refined_BCE+w_e*emotion_BCE)+lambda_cross*valid_cosine.

    Disabling a classifier's supervision does not freeze its representation:
    refined personality BCE uses both task embeddings; consistency uses both
    embeddings and the interaction, but neither classifier. Only active loss
    terms connect to total. Both outer lambdas may be zero to skip optimization.
    """
    def __init__(self,*,categories=DEFAULT_EMOTIONS,personality_weight=.7,emotion_weight=.3,
                 lambda_mtl=1.,lambda_cross=.1):
        super().__init__()
        self.task_loss = MultiTaskLoss(categories=categories,personality_weight=personality_weight,emotion_weight=emotion_weight)
        self.lambda_mtl,self.lambda_cross = _weight(lambda_mtl,"lambda_mtl"),_weight(lambda_cross,"lambda_cross")

    def forward(self,output,targets):
        if not isinstance(output,InteractionOutput):
            raise TypeError("Expected InteractionOutput")
        task = self.task_loss(output,targets)
        valid = torch.tensor(targets.emotion_valid,dtype=torch.bool,device=output.personality_logits.device)
        cross = cosine_consistency(output.refined_personality_embedding,output.emotion_embedding,valid)
        terms = []
        if self.lambda_mtl and task.has_active_supervision:
            terms.append(self.lambda_mtl*task.total_loss)
        if self.lambda_cross and task.valid_emotion_count:
            terms.append(self.lambda_cross*cross)
        total = sum(terms[1:],terms[0]) if terms else torch.zeros_like(task.total_loss,requires_grad=False)
        return InteractionLosses(task.personality_loss,task.emotion_loss,task.total_loss,cross,total,
                                 task.valid_emotion_count,bool(terms))
