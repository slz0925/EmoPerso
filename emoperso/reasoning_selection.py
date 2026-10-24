"""Stable Bernoulli entropy, bounded association, hard choice and preference."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .augmentation_model import user_mean
from .backbone import _positive
from .errors import DataContractError
from .interaction import _weight


def _float(value,ndim,name):
    if (not isinstance(value,torch.Tensor) or not value.is_floating_point() or value.ndim!=ndim
            or not torch.isfinite(value).all()):raise DataContractError(f'{name} must be finite floating {ndim}D')


def _temperature(value):
    value=_weight(value,'temperature')
    if not 1e-6<=value<=1e6:raise ValueError('temperature must be in [1e-6,1e6]')
    return value


def _indices(indices,n,b):
    if (not isinstance(indices,tuple) or len(indices)!=n
            or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<b for i in indices)):
        raise DataContractError('Expected one valid original-user index per candidate')


def bernoulli_entropy(logits):
    """Independent MBTI Bernoulli entropy, averaged over four dimensions, nats."""
    _float(logits,2,'logits')
    if logits.shape[1]!=4:raise DataContractError('Personality logits must be [N,4]')
    dtype=torch.float64 if logits.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=logits.device.type,enabled=False):
        x=logits.to(dtype).abs()
        return (F.softplus(-x)+x*torch.sigmoid(-x)).mean(-1)


def information_gain(base_logits,candidate_logits,candidate_to_user):
    _indices(candidate_to_user,len(candidate_logits),len(base_logits))
    if base_logits.device!=candidate_logits.device:raise DataContractError('IG logits must share a device')
    return bernoulli_entropy(base_logits)[list(candidate_to_user)]-bernoulli_entropy(candidate_logits)


@dataclass(frozen=True)
class AssociationFeatures:
    scores: torch.Tensor
    reasoning: torch.Tensor
    emotion: torch.Tensor
    candidate_to_user: tuple[int,...]
    temperature: float


class EmotionReasoningAssociation(nn.Module):
    """Projected emotion and reasoning vectors with bounded cosine scores.

    Score for a candidate uses only its own user's emotion, independent of other
    batch members. InfoNCE below separately uses valid cross-user negatives.
    """
    def __init__(self,hidden_size,*,projection_size=128,temperature=.1,device=None,dtype=None):
        super().__init__();_positive(hidden_size,'hidden_size');_positive(projection_size,'projection_size')
        self.temperature=_temperature(temperature)
        self.reasoning_projection=nn.Linear(hidden_size,projection_size,bias=False,device=device,dtype=dtype)
        self.emotion_projection=nn.Linear(hidden_size,projection_size,bias=False,device=device,dtype=dtype)

    def forward(self,reasoning,emotion,candidate_to_user):
        _float(reasoning,2,'reasoning');_float(emotion,2,'emotion')
        _indices(candidate_to_user,len(reasoning),len(emotion))
        if (reasoning.shape[1]!=self.reasoning_projection.in_features or emotion.shape[1]!=reasoning.shape[1]
                or reasoning.device!=emotion.device or reasoning.device!=self.reasoning_projection.weight.device):
            raise DataContractError('Association feature dimensions/device differ')
        dtype=torch.float64 if torch.float64 in (reasoning.dtype,emotion.dtype,self.reasoning_projection.weight.dtype) else torch.float32
        def projected(x,layer):
            # Scaling before projection/norm avoids overflow on finite large inputs.
            x=x.to(dtype);scale=x.abs().amax(-1,keepdim=True).clamp_min(1).detach()
            y=F.linear(x/scale,layer.weight.to(dtype))
            if not torch.isfinite(y).all():raise DataContractError('Nonfinite association projection')
            return F.normalize(y,p=2,dim=-1,eps=1e-8)
        with torch.autocast(device_type=reasoning.device.type,enabled=False):
            r=projected(reasoning,self.reasoning_projection);e=projected(emotion,self.emotion_projection)
            score=(r*e[list(candidate_to_user)]).sum(-1).clamp(-1,1)
        return AssociationFeatures(score,r,e,candidate_to_user,self.temperature)


@dataclass(frozen=True)
class ContrastiveLoss:
    loss: torch.Tensor
    users: int
    candidates: int
    has_active_supervision: bool


def emotion_reasoning_contrastive(features,emotion_valid):
    """Candidate->unique-user InfoNCE; mean candidates within user, then users.

    Only emotion-valid users WITH candidates supply positive/negative emotion
    rows. Multiple candidates of a user never become each other's negatives.
    Fewer than two eligible users returns a detached zero.
    """
    if not isinstance(features,AssociationFeatures):raise TypeError('Expected AssociationFeatures')
    r,e=features.reasoning,features.emotion
    _float(r,2,'projected reasoning');_float(e,2,'projected emotion')
    _indices(features.candidate_to_user,len(r),len(e))
    if (not isinstance(emotion_valid,torch.Tensor) or emotion_valid.dtype!=torch.bool
            or emotion_valid.shape!=(len(e),) or emotion_valid.device!=e.device or r.device!=e.device):
        raise DataContractError('Expected aligned bool emotion validity [B]')
    users=sorted({i for i in features.candidate_to_user if bool(emotion_valid[i])})
    zero=torch.zeros((),device=r.device,dtype=r.dtype)
    if len(users)<2:return ContrastiveLoss(zero,len(users),0,False)
    rows=[j for j,i in enumerate(features.candidate_to_user) if i in users]
    positive=torch.tensor([users.index(features.candidate_to_user[j]) for j in rows],device=r.device)
    with torch.autocast(device_type=r.device.type,enabled=False):
        logits=r[rows]@e[users].T/_temperature(features.temperature)
        loss=F.cross_entropy(logits,positive,reduction='none')
    return ContrastiveLoss(user_mean(loss,tuple(features.candidate_to_user[j] for j in rows)),len(users),len(rows),True)


@dataclass(frozen=True)
class ChainSelection:
    scores: torch.Tensor
    preferences: torch.Tensor
    selected_indices: torch.Tensor
    selected_ids: tuple[str | None,...]
    valid_users: torch.Tensor
    entropies: torch.Tensor
    entropy_loss: torch.Tensor
    multi_candidate_users: int


def select_chains(scores,candidate_to_user,candidate_ids,*,batch_size,valid=None,temperature=1.):
    """Per-user soft preference and hard argmax; exact ties use smallest ID.

    Invalid score values are ignored, not included in reductions. Entropy is
    averaged over users with any candidates, including zero for singletons;
    no multi-candidate users returns a detached zero (no active entropy target).
    """
    if not isinstance(scores,torch.Tensor) or not scores.is_floating_point() or scores.ndim!=1:
        raise DataContractError('Expected floating candidate scores [N]')
    _positive(batch_size,'batch_size');_indices(candidate_to_user,len(scores),batch_size)
    if (not isinstance(candidate_ids,tuple) or len(candidate_ids)!=len(scores) or len(set(candidate_ids))!=len(candidate_ids)
            or any(not isinstance(s,str) or not s for s in candidate_ids)):raise DataContractError('Candidate IDs must be unique nonempty strings')
    temperature=_temperature(temperature)
    if valid is None:valid=torch.ones(len(scores),dtype=torch.bool,device=scores.device)
    if not isinstance(valid,torch.Tensor) or valid.dtype!=torch.bool or valid.shape!=scores.shape or valid.device!=scores.device:
        raise DataContractError('Expected bool candidate validity [N]')
    if not torch.isfinite(scores[valid]).all():raise DataContractError('Nonfinite valid candidate score')
    dtype=torch.float64 if scores.dtype==torch.float64 else torch.float32
    p=torch.zeros_like(scores,dtype=dtype);selected=torch.full((batch_size,),-1,dtype=torch.long,device=scores.device)
    zero=torch.zeros((),device=scores.device,dtype=dtype)
    entropies=[];chosen=[];active=[];valid_count=0
    with torch.autocast(device_type=scores.device.type,enabled=False):
        for user in range(batch_size):
            indices=sorted((j for j,i in enumerate(candidate_to_user) if i==user and bool(valid[j])),key=lambda j:candidate_ids[j])
            if not indices:entropies.append(zero);chosen.append(None);continue
            valid_count+=1
            if len(indices)==1:
                p[indices[0]]=1;entropy=zero;best=indices[0]
            else:
                x=scores[indices].to(dtype)
                logp=F.log_softmax((x-x.max())/temperature,dim=0)
                weights=logp.exp();p=p.index_copy(0,torch.tensor(indices,device=p.device),weights)
                positive=weights>0;entropy=-(weights[positive]*logp[positive]).sum()
                best=indices[int(torch.argmax(x).detach())];active.append(entropy)
            selected[user]=best;chosen.append(candidate_ids[best]);entropies.append(entropy)
    total=sum(active[1:],active[0])/valid_count if active else zero
    return ChainSelection(scores,p,selected,tuple(chosen),selected>=0,torch.stack(entropies),total,len(active))


class ReasoningFusion(nn.Module):
    """Bottleneck gated residual fusion; invalid rows bypass every fusion layer."""
    def __init__(self,hidden_size,*,bottleneck_size=None,device=None,dtype=None):
        super().__init__();_positive(hidden_size,'hidden_size')
        width=min(256,hidden_size) if bottleneck_size is None else _positive(bottleneck_size,'bottleneck_size')
        options=dict(device=device,dtype=dtype)
        self.down=nn.Linear(2*hidden_size,width,**options)
        self.up=nn.Linear(width,hidden_size,**options)
        self.gate=nn.Linear(width,1,**options)
        self.norm=nn.LayerNorm(hidden_size,**options)

    def forward(self,personality,reasoning,valid=None):
        _float(personality,2,'personality')
        if (not isinstance(reasoning,torch.Tensor) or not reasoning.is_floating_point() or reasoning.shape!=personality.shape
                or reasoning.device!=personality.device or self.down.weight.device!=personality.device
                or personality.shape[1]!=self.up.out_features):raise DataContractError('Fusion feature shape/device differs')
        if valid is None:valid=torch.ones(len(personality),dtype=torch.bool,device=personality.device)
        if not isinstance(valid,torch.Tensor) or valid.dtype!=torch.bool or valid.shape!=(len(personality),) or valid.device!=personality.device:
            raise DataContractError('Expected bool fusion validity [B]')
        if not valid.any():return personality
        _float(reasoning[valid],2,'valid reasoning')
        dtype=torch.float64 if torch.float64 in (personality.dtype,reasoning.dtype,self.down.weight.dtype) else torch.float32
        with torch.autocast(device_type=personality.device.type,enabled=False):
            p=personality[valid].to(dtype);r=reasoning[valid].to(dtype)
            h=F.relu(F.linear(torch.cat((p,r),-1),self.down.weight.to(dtype),self.down.bias.to(dtype)))
            delta=F.linear(h,self.up.weight.to(dtype),self.up.bias.to(dtype))
            gate=torch.sigmoid(F.linear(h,self.gate.weight.to(dtype),self.gate.bias.to(dtype)))
            result=F.layer_norm(p+gate*delta,(p.shape[1],),self.norm.weight.to(dtype),self.norm.bias.to(dtype),self.norm.eps)
            if not torch.isfinite(result).all():raise DataContractError('Nonfinite reasoning fusion')
        return personality.to(dtype).index_copy(0,valid.nonzero().flatten(),result)
