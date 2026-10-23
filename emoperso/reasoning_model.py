"""Shared Encoder, information-based hard chain choice and final MBTI fusion."""
from dataclasses import dataclass, field, replace
import math

import torch
from torch import nn

from .augmentation import DEFAULT_STYLES
from .augmentation_model import EmoPersoAugmentationModel,AugmentationOutput,AugmentationLoss
from .data import DataContractError
from .interaction import InteractionLoss,_weight
from .reasoning import validate_reasoning_batches
from .reasoning_selection import (EmotionReasoningAssociation,ReasoningFusion,information_gain,
    select_chains,emotion_reasoning_contrastive,_temperature)
from .token_inputs import batch_signature


@dataclass(frozen=True)
class ReasoningOutput:
    personality_logits: torch.Tensor
    personality_probabilities: torch.Tensor
    final_personality_embedding: torch.Tensor
    reasoning_embeddings: torch.Tensor
    candidate_personality_logits: torch.Tensor
    information_gain: torch.Tensor
    association: object
    selection: object
    candidate_to_user: tuple[int,...]
    candidate_ids: tuple[str,...]
    augmentation: AugmentationOutput = field(repr=False)
    candidate_token_metadata: tuple[dict,...]
    score_objective_enabled: bool

    @property
    def interaction_output(self):return self.augmentation.original

    @property
    def refined_personality_embedding(self):return self.interaction_output.refined_personality_embedding

    @property
    def emotion_embedding(self):return self.interaction_output.emotion_embedding

    @property
    def metadata(self):return self.interaction_output.metadata

    @property
    def batch_signature(self):return self.interaction_output.batch_signature


class EmoPersoReasoningModel(nn.Module):
    """Shared interaction, optional style supervision and reasoning fusion.

    Generation and candidate tokenization are explicit preparation steps. The
    same trainable Encoder projects and pools candidates on each forward pass.
    Forward returns pooled candidate vectors and their token metadata.
    Hard choice is identical in train/eval; its argmax has no gradient. Final
    BCE updates selected fusion paths; entropy and contrastive train scoring.
    """
    def __init__(self,interaction_model,*,styles=None,bottleneck_size=None,
                 association_size=128,association_temperature=.1,
                 ig_weight=1.,association_weight=1.,selection_temperature=1.):
        super().__init__()
        if isinstance(interaction_model,EmoPersoAugmentationModel):
            if styles is not None and tuple(styles)!=interaction_model.styles:
                raise DataContractError('Existing augmentation model style categories differ')
            self.augmentation_model=interaction_model
        else:
            self.augmentation_model=EmoPersoAugmentationModel(interaction_model,styles=DEFAULT_STYLES if styles is None else styles)
        w=self.interaction_model.mtl.personality_head.weight;d=w.shape[1]
        self.fusion=ReasoningFusion(d,bottleneck_size=bottleneck_size,device=w.device,dtype=w.dtype)
        self.association=EmotionReasoningAssociation(d,projection_size=association_size,
            temperature=association_temperature,device=w.device,dtype=w.dtype)
        self.ig_weight=_weight(ig_weight,'ig_weight');self.association_weight=_weight(association_weight,'association_weight')
        self.selection_temperature=_temperature(selection_temperature)

    @property
    def interaction_model(self):return self.augmentation_model.interaction_model

    def get_extra_state(self):
        return {'ig_weight':self.ig_weight,'association_weight':self.association_weight,
            'selection_temperature':self.selection_temperature,'association_temperature':self.association.temperature}

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise DataContractError('Checkpoint reasoning score configuration differs')

    def forward(self,original,*,reasoning=None,paired=None):
        signature=batch_signature(original)
        if paired is not None:
            if paired.original_signature!=signature:raise DataContractError('Augmentation and reasoning originals differ')
            aug=self.augmentation_model(paired)
        else:
            aug=AugmentationOutput(self.interaction_model(original),(),self.augmentation_model.styles)
        base=aug.original;p=base.refined_personality_embedding
        vectors=[];indices=[];ids=[];token_metadata=[]
        if reasoning is not None:
            validate_reasoning_batches(reasoning,original)
            for group in reasoning.groups:
                encoded=self.interaction_model.mtl.encoder(group.batch)
                vectors.append(encoded.pooled_features)
                indices.extend(group.original_indices)
                ids.extend(c.candidate_id for c in group.candidates)
                token_metadata.extend({'tokens':len(row.input_ids),'truncated':row.truncated,
                    'retained_characters':row.retained_characters} for row in group.batch['provenance'])
        indices,ids=tuple(indices),tuple(ids)
        if vectors:
            r=torch.cat(vectors,dim=0)
            candidate_features=self.fusion(p[list(indices)],r)
            head=self.interaction_model.mtl.personality_head
            logits=head(candidate_features.to(head.weight.dtype))
            ig=information_gain(base.personality_logits,logits,indices)
            assoc=self.association(r,base.emotion_embedding,indices)
            # Both components lie in [-1,1] before explicit weights. No batch
            # statistics or labels enter score calibration or inference.
            components=[]
            if self.ig_weight:components.append(self.ig_weight*(ig/math.log(2.)))
            if self.association_weight:components.append(self.association_weight*assoc.scores)
            score=sum(components[1:],components[0]) if components else torch.zeros_like(ig,requires_grad=False)
        else:
            r=torch.empty((0,p.shape[1]),device=p.device,dtype=p.dtype)
            logits=torch.empty((0,4),device=p.device,dtype=base.personality_logits.dtype)
            ig=torch.empty((0,),device=p.device,dtype=p.dtype)
            assoc=None;score=ig
        selection=select_chains(score,indices,ids,batch_size=len(p),temperature=self.selection_temperature)
        valid_rows=selection.valid_users.nonzero().flatten()
        if len(valid_rows):
            chosen=selection.selected_indices[valid_rows]
            final=p.to(candidate_features.dtype).index_copy(0,valid_rows,candidate_features[chosen])
            final_logits=base.personality_logits.index_copy(0,valid_rows,logits[chosen])
        else:
            final,final_logits=p,base.personality_logits
        return ReasoningOutput(final_logits,final_logits.sigmoid(),final,r,logits,ig,assoc,selection,indices,ids,aug,tuple(token_metadata),
            bool(self.ig_weight or self.association_weight))


@dataclass(frozen=True)
class ReasoningLosses:
    base_loss: object = field(repr=False)
    augmentation_loss: object = field(repr=False)
    reasoning_entropy: torch.Tensor
    contrastive_loss: torch.Tensor
    total_loss: torch.Tensor
    contrastive_users: int
    entropy_users: int
    has_active_supervision: bool


class ReasoningLoss(nn.Module):
    """Final-personality BCE with interaction and augmentation objectives.

    Default total: lambda_mtl*(.7 final BCE+.3 emotion BCE)+.1 cross
      +.1 style CE+.1 augmentation prediction KL+.1 entropy+.1 InfoNCE.
    Only positively weighted, actually supervised objectives connect to total.
    Entropy is inactive for only-singleton choices or both score weights zero.
    """
    def __init__(self,*,base_loss=None,augmentation_loss=None,star_weight=.1,contrastive_weight=.1,styles=DEFAULT_STYLES):
        super().__init__()
        self.base_loss=InteractionLoss() if base_loss is None else base_loss
        self.augmentation_loss=AugmentationLoss(styles=styles) if augmentation_loss is None else augmentation_loss
        if not isinstance(self.base_loss,InteractionLoss) or not isinstance(self.augmentation_loss,AugmentationLoss):
            raise TypeError('Expected existing InteractionLoss and AugmentationLoss')
        self.star_weight=_weight(star_weight,'star_weight');self.contrastive_weight=_weight(contrastive_weight,'contrastive_weight')

    def forward(self,output,targets):
        if not isinstance(output,ReasoningOutput):raise TypeError('Expected ReasoningOutput')
        base_output=replace(output.interaction_output,personality_logits=output.personality_logits,
                            personality_probabilities=output.personality_probabilities)
        base=self.base_loss(base_output,targets)
        aux=self.augmentation_loss.auxiliary(output.augmentation)
        zero=torch.zeros_like(base.total_loss,requires_grad=False)
        if output.association is not None:
            valid=torch.tensor(targets.emotion_valid,device=output.personality_logits.device,dtype=torch.bool)
            contrast=emotion_reasoning_contrastive(output.association,valid)
            closs,cusers,cactive=contrast.loss,contrast.users,contrast.has_active_supervision
        else:closs,cusers,cactive=zero,0,False
        entropy=output.selection.entropy_loss
        terms=[base.total_loss] if base.has_active_supervision else []
        if self.augmentation_loss.style_weight and aux.valid_style_views:
            terms.append(self.augmentation_loss.style_weight*aux.style_loss)
        if self.augmentation_loss.consistency_weight and aux.valid_consistency_views:
            terms.append(self.augmentation_loss.consistency_weight*aux.consistency_loss)
        entropy_active=output.selection.multi_candidate_users>0 and output.score_objective_enabled
        if self.star_weight and entropy_active:terms.append(self.star_weight*entropy)
        if self.contrastive_weight and cactive:terms.append(self.contrastive_weight*closs)
        total=sum(terms[1:],terms[0]) if terms else zero
        return ReasoningLosses(base,aux,entropy,closs,total,cusers,output.selection.multi_candidate_users,bool(terms))
