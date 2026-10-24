"""Multi-epoch EmoPerso training with validation and epoch-boundary resume."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from .checkpointing import atomic_json,save_checkpoint,load_checkpoint,model_signature
from .errors import DataContractError
from .datasets import iter_batches
from .evaluation import (collect_predictions,check_disjoint,require_partition,
                         select_thresholds,thresholds_tuple)
from .mtl import prepare_targets
from .reasoning_model import EmoPersoReasoningModel,ReasoningLoss


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    max_epochs: int = 20
    seed: int = 42
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
    weight_decay: float = 0.
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.
    scheduler: str = 'constant'
    schedule_steps: int | None = None
    warmup_steps: int = 0
    precision: str = 'fp32'
    validation_frequency: int = 1
    patience: int = 5
    min_improvement: float = 0.
    threshold_search: bool = False
    threshold_grid: tuple[float,...] = (.3,.4,.5,.6,.7)

    def __post_init__(self):
        for key in ('batch_size','max_epochs','gradient_accumulation_steps','validation_frequency'):
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError(key+' must be positive integer')
        for key in ('seed','warmup_steps','patience'):
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,int) or v<0:raise ValueError(key+' must be nonnegative integer')
        if self.seed>=2**63:raise ValueError('seed exceeds supported range')
        if self.schedule_steps is not None and (isinstance(self.schedule_steps,bool) or not isinstance(self.schedule_steps,int) or self.schedule_steps<1):
            raise ValueError('schedule_steps must be positive integer')
        for key in ('learning_rate','weight_decay','min_improvement','gradient_clip_norm'):
            v=getattr(self,key)
            if key=='gradient_clip_norm' and v is None:continue
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
                raise ValueError(key+' must be finite nonnegative')
            if key in ('learning_rate','gradient_clip_norm') and v==0:raise ValueError(key+' must be positive')
        if self.optimizer not in ('adam','adamw') or self.scheduler not in ('constant','linear','cosine'):
            raise ValueError('Unsupported optimizer/scheduler')
        if self.precision not in ('fp32','bf16','fp16'):raise ValueError('Unsupported precision')
        if not isinstance(self.threshold_search,bool):raise ValueError('threshold_search must be boolean')
        if not self.threshold_grid:raise ValueError('threshold_grid cannot be empty')
        for t in self.threshold_grid:thresholds_tuple(t)
        object.__setattr__(self,'threshold_grid',tuple(self.threshold_grid))


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed % 2**32);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def precision_context(model, precision):
    device=model.interaction_model.mtl.encoder.projection.weight.device
    if device.type not in ('cpu','cuda'):raise ValueError('Training currently supports CPU/CUDA')
    if precision!='fp32' and device.type!='cuda':raise ValueError('Mixed precision requires CUDA; CPU execution uses FP32')
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA device is not available')
    if precision=='bf16' and not torch.cuda.is_bf16_supported():raise ValueError('CUDA BF16 is not supported')
    return (lambda: nullcontext()) if precision=='fp32' else (
        lambda:torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16))


class Trainer:
    """Accumulate the mean joint loss over active microbatches.

    Each existing joint loss retains its own valid-user/view denominators.
    Average those scalar losses over the active microbatches in each actual
    optimizer window, including a short tail. Inactive batches do not advance
    windows, optimizer state or learning rate.
    """
    def __init__(self,model,criterion,*,tokenizer,config=None,compatibility=None,preparer=None,temp_dir=None):
        if not isinstance(model,EmoPersoReasoningModel) or not isinstance(criterion,ReasoningLoss):
            raise TypeError('Trainer requires EmoPersoReasoningModel and ReasoningLoss')
        self.model,self.criterion,self.tokenizer=model,criterion,tokenizer
        self.config=TrainingConfig() if config is None else config
        if not isinstance(self.config,TrainingConfig):raise TypeError('Expected TrainingConfig')
        self.compatibility={'model':model_signature(model),'external':{} if compatibility is None else compatibility}
        self.preparer,self.temp_dir=preparer,temp_dir
        self.autocast=precision_context(model,self.config.precision)
        self.parameters=[p for p in model.parameters() if p.requires_grad]
        if not self.parameters:raise ValueError('Model has no trainable parameters')
        device=model.interaction_model.mtl.encoder.projection.weight.device
        if any(p.device!=device for p in self.parameters):
            raise ValueError('Trainable modules must share one feature device; frozen Backbone may be dispatched')
        opt=torch.optim.Adam if self.config.optimizer=='adam' else torch.optim.AdamW
        self.optimizer=opt(self.parameters,lr=self.config.learning_rate,weight_decay=self.config.weight_decay)
        self.scaler=torch.amp.GradScaler('cuda') if self.config.precision=='fp16' else None
        self.scheduler=None;self.progress=None;self.history=[]

    def _schedule(self,total):
        c=self.config
        if c.warmup_steps>=total and c.warmup_steps:raise ValueError('Warmup must be shorter than schedule horizon')
        def factor(step):
            if step<c.warmup_steps:return (step+1)/c.warmup_steps
            if c.scheduler=='constant':return 1.
            fraction=min(1.,max(0.,(step-c.warmup_steps)/max(1,total-c.warmup_steps)))
            return 1.-fraction if c.scheduler=='linear' else .5*(1.+math.cos(math.pi*fraction))
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,factor)

    def _step(self,active):
        if self.scaler:self.scaler.unscale_(self.optimizer)
        for p in self.parameters:
            if p.grad is not None:p.grad.div_(active)
        finite=all(torch.isfinite(p.grad).all().item() for p in self.parameters if p.grad is not None)
        if not finite:
            if self.scaler:
                self.scaler.step(self.optimizer);self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                return False
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite gradients; no optimizer/scheduler step')
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters,self.config.gradient_clip_norm,error_if_nonfinite=True)
        if self.scaler:self.scaler.step(self.optimizer);self.scaler.update()
        else:self.optimizer.step()
        self.scheduler.step();self.optimizer.zero_grad(set_to_none=True)
        self.progress['global_step']+=1
        return True

    def train_epoch(self,dataset,epoch):
        require_partition(dataset,'train')
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        batches=iter_batches(dataset,tokenizer=self.tokenizer,batch_size=self.config.batch_size,
            shuffle=True,seed=(self.config.seed+epoch)%2**63,temp_dir=self.temp_dir)
        active=pending=seen=skipped=steps=overflows=0;totals={};candidate_count=views=0
        start=time.monotonic()
        try:
            for batch in batches:
                if set(batch['partitions'])!={'train'}:raise DataContractError('Only train may supply gradients')
                seen+=len(batch['user_keys'])
                options=self.preparer.prepare(batch,training=True) if self.preparer is not None else {}
                with self.autocast():
                    output=self.model(batch,**options)
                    targets=prepare_targets(batch,categories=self.model.interaction_model.mtl.categories)
                    losses=self.criterion(output,targets)
                loss=losses.total_loss
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite training loss; refusing optimizer step')
                values={'total':loss,'personality':losses.base_loss.personality_loss,'emotion':losses.base_loss.emotion_loss,
                    'cross':losses.base_loss.cross_loss,'style':losses.augmentation_loss.style_loss,
                    'augmentation_KL':losses.augmentation_loss.consistency_loss,
                    'reasoning_entropy':losses.reasoning_entropy,'contrastive':losses.contrastive_loss}
                candidate_count+=len(output.candidate_ids);views+=losses.augmentation_loss.valid_consistency_views
                if not losses.has_active_supervision:skipped+=1;continue
                active+=1;pending+=1
                for k,v in values.items():totals[k]=totals.get(k,0.)+float(v.detach())
                if self.scaler:self.scaler.scale(loss).backward()
                else:loss.backward()
                if pending==self.config.gradient_accumulation_steps:
                    ok=self._step(pending);steps+=int(ok);overflows+=int(not ok);pending=0
            if pending:
                ok=self._step(pending);steps+=int(ok);overflows+=int(not ok)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True);raise
        finally:batches.close()
        if seen!=len(dataset):raise DataContractError('Training did not cover all users')
        return {'users':seen,'active_microbatches':active,'inactive_microbatches':skipped,'optimizer_steps':steps,
            'AMP_overflow_windows':overflows,'loss_mean_over_active_microbatches':{k:v/active for k,v in totals.items()},
            'candidates':candidate_count,'augmentation_views':views,'seconds':time.monotonic()-start}

    def fit(self,train,validation,*,output_dir,checkpoint_dir=None,resume=None,until_epoch=None):
        require_partition(train,'train');require_partition(validation,'validation');check_disjoint(train,validation)
        c=self.config
        total=c.schedule_steps or c.max_epochs*math.ceil(math.ceil(len(train)/c.batch_size)/c.gradient_accumulation_steps)
        self._schedule(total)
        semantics={**asdict(c),'schedule_horizon':total,'training_users':len(train),
            'preparation':getattr(self.preparer,'identity',None)}
        base=self.criterion.base_loss;aux=self.criterion.augmentation_loss
        semantics['loss_weights']={'personality':base.task_loss.personality_weight,'emotion':base.task_loss.emotion_weight,
            'lambda_mtl':base.lambda_mtl,'cross':base.lambda_cross,'style':aux.style_weight,
            'consistency':aux.consistency_weight,'star':self.criterion.star_weight,'contrastive':self.criterion.contrastive_weight}
        # Record data identity without hashing/storing all text in RAM.
        semantics['data_versions']={name:[{'name':Path(p).name,'bytes':v[0],'mtime_ns':v[1]} for p,v in d.source_versions.items()]
                                    for name,d in (('train',train),('validation',validation))}
        semantics['data_processing']={name:{'layout':d.layout,'preprocess':d.preprocess,'order':d.order,'text':asdict(d.text_config)}
                                      for name,d in (('train',train),('validation',validation))}
        out=Path(output_dir);checkpoints=Path(checkpoint_dir) if checkpoint_dir is not None else out/'checkpoints'
        self.history=[]
        if resume is None:
            if out.exists() and any(out.iterdir()):raise FileExistsError('Experiment output is not empty; use a new directory or explicit resume')
            if checkpoints.exists() and any(checkpoints.iterdir()):raise FileExistsError('Checkpoint directory is not empty')
            out.mkdir(parents=True,exist_ok=True);checkpoints.mkdir(parents=True,exist_ok=True)
            seed_everything(c.seed)
            self.progress={'epoch':0,'global_step':0,'best_epoch':0,'best_score':None,'bad_validations':0,
                           'thresholds':list(thresholds_tuple()),'run_id':uuid.uuid4().hex,'stopped_early':False}
            atomic_json(out/'run.json',{'run_id':self.progress['run_id'],'training':asdict(c),
                'compatibility':self.compatibility,'trainable_parameters':sum(p.numel() for p in self.parameters),
                'frozen_parameters':sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
                'device':str(self.parameters[0].device),'precision':c.precision,
                'accumulation':'mean of active microbatch joint losses; no cross-microbatch InfoNCE negatives'})
        else:
            owner=json.loads((out/'run.json').read_text(encoding='utf8'))
            self.progress=load_checkpoint(resume,self.model,compatibility=self.compatibility,optimizer=self.optimizer,
                scheduler=self.scheduler,scaler=self.scaler,training_config=semantics,run_id=owner['run_id'])
            if self.progress['run_id']!=owner['run_id']:raise DataContractError('Resume checkpoint belongs to another experiment')
            if self.progress['stopped_early']:raise DataContractError('Early-stopped run is complete; use weights-only transfer for a new experiment')
            if self.progress['best_epoch'] and not (checkpoints/'best.pt').is_file():
                raise DataContractError('Resume requires this experiment\'s saved best checkpoint')
        end=c.max_epochs if until_epoch is None else until_epoch
        if isinstance(end,bool) or not isinstance(end,int) or not self.progress['epoch']<end<=c.max_epochs:
            raise ValueError('until_epoch must advance within configured total max_epochs')
        for epoch in range(self.progress['epoch']+1,end+1):
            train_stats=self.train_epoch(train,epoch);record={'epoch':epoch,'train':train_stats}
            improved=False
            if epoch%c.validation_frequency==0 or epoch==c.max_epochs:
                prediction=collect_predictions(self.model,validation,tokenizer=self.tokenizer,batch_size=c.batch_size,
                    preparer=self.preparer,temp_dir=self.temp_dir,autocast=self.autocast)
                thresholds=select_thresholds(prediction.labels,prediction.probabilities,partition='validation',grid=c.threshold_grid) if c.threshold_search else thresholds_tuple()
                metric=prediction.metrics(thresholds);record['validation']={'fixed_threshold_metrics':prediction.metrics(),
                    'selection_metrics':metric,'thresholds':list(thresholds),'users':len(prediction.identities)}
                best=self.progress['best_score']
                improved=best is None or metric['average']>best+c.min_improvement
                if improved:
                    self.progress.update(best_score=metric['average'],best_epoch=epoch,bad_validations=0,thresholds=list(thresholds))
                else:self.progress['bad_validations']+=1
                if c.patience and self.progress['bad_validations']>=c.patience:self.progress['stopped_early']=True
            self.progress['epoch']=epoch
            record.update(global_step=self.progress['global_step'],learning_rates=self.scheduler.get_last_lr(),
                best_epoch=self.progress['best_epoch'],best_score=self.progress['best_score'],
                bad_validations=self.progress['bad_validations'],stopped_early=self.progress['stopped_early'])
            common=dict(compatibility=self.compatibility,progress=self.progress,optimizer=self.optimizer,
                scheduler=self.scheduler,scaler=self.scaler,training_config=semantics,overwrite=True)
            if improved:save_checkpoint(checkpoints/'best.pt',self.model,**common)
            save_checkpoint(checkpoints/'latest.pt',self.model,**common)
            with (out/'epochs.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
            self.history.append(record)
            if self.progress['stopped_early']:break
        # Latest retains all optimizer/RNG/progress; best restore is prediction-only.
        if self.progress['best_epoch']:
            load_checkpoint(checkpoints/'best.pt',self.model,compatibility=self.compatibility,weights_only=True,restore_random=False,
                            run_id=self.progress['run_id'])
        result={**self.progress,'history':self.history,'best_checkpoint':str(checkpoints/'best.pt') if self.progress['best_epoch'] else None,
                'latest_checkpoint':str(checkpoints/'latest.pt')}
        atomic_json(out/'result.json',result)
        return result
