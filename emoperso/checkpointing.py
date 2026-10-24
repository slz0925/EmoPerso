"""Atomic local checkpoints, excluding the frozen Backbone subtree."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .backbone import FrozenBackbone
from .errors import DataContractError


def model_signature(model):
    core=model.interaction_model;encoder=core.mtl.encoder
    config=encoder.backbone.model.config.to_dict()
    for key in ('_name_or_path','transformers_version','dtype','torch_dtype','_commit_hash'):
        config.pop(key,None)
    # JSON-normalize tuple/list metadata without tying loading to machine paths.
    return json.loads(json.dumps({'model_class':type(model).__name__,'backbone':config,
        'tokenizer':getattr(encoder.backbone.tokenizer,'encoding_fingerprint',None),
        'shared_size':encoder.projection.out_features,'emotion_categories':core.mtl.categories,
        'style_categories':model.augmentation_model.styles,'heads':core.interaction.num_heads,
        'dropout':[m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)],
        'score_config':model.get_extra_state()},sort_keys=True))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def lightweight_state(model):
    """Collect local state without traversing enormous/dispatched frozen weights.

    Save all other persistent buffers, extra state and parameters, including
    currently inactive heads. A generator is never owned by this model.
    """
    state = {}
    def visit(module, prefix=''):
        if isinstance(module, FrozenBackbone):return
        module._save_to_state_dict(state, prefix, keep_vars=False)
        for name, child in module.named_children():visit(child, prefix+name+'.')
    visit(model)
    return state


def _cpu(value):
    if isinstance(value, torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value, dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value, tuple):return tuple(_cpu(v) for v in value)
    if isinstance(value, list):return [_cpu(v) for v in value]
    return copy.deepcopy(value)


def rng_state():
    n = np.random.get_state()
    return {'python':random.getstate(), 'numpy':(n[0], n[1].tolist(), n[2], n[3], n[4]),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if state['cuda'] and (not torch.cuda.is_available() or len(state['cuda'])!=torch.cuda.device_count()):
        raise DataContractError('CUDA RNG device count differs; use weights-only transfer for migration')
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, *, compatibility, progress, optimizer=None,
                    scheduler=None, scaler=None, training_config=None, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError('Checkpoint exists; use a new experiment or explicitly replace its own state')
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = {'format':'emoperso-lightweight-v1','model':_cpu(lightweight_state(model)),
        'compatibility':copy.deepcopy(compatibility),'progress':copy.deepcopy(progress),
        'optimizer':_cpu(optimizer.state_dict()) if optimizer is not None else None,
        'scheduler':scheduler.state_dict() if scheduler is not None else None,
        'scaler':scaler.state_dict() if scaler is not None else None,
        'training_config':training_config,'rng':rng_state()}
    handle,temporary = tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(handle,'wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def load_checkpoint(path, model, *, compatibility, optimizer=None, scheduler=None,
                    scaler=None, training_config=None, weights_only=False, restore_random=True,run_id=None):
    """Use torch's restricted loader; never unpickle arbitrary Python classes.

    Validate every model key/shape/extra state before copying any parameters.
    Weights-only transfer keeps structural compatibility, ignores training state.
    Only load checkpoints from a trusted local source, even with weights_only.
    """
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('format')!='emoperso-lightweight-v1':
        raise DataContractError('Unsupported lightweight checkpoint format')
    if payload.get('compatibility') != compatibility:
        raise DataContractError('Checkpoint model/Backbone/tokenizer semantics differ')
    if run_id is not None and payload.get('progress',{}).get('run_id')!=run_id:
        raise DataContractError('Checkpoint belongs to another experiment')
    current = lightweight_state(model); supplied = payload.get('model',{})
    if current.keys()!=supplied.keys():raise DataContractError('Checkpoint model state keys differ')
    for key,value in current.items():
        other = supplied[key]
        if isinstance(value,torch.Tensor):
            if not isinstance(other,torch.Tensor) or value.shape!=other.shape or value.dtype!=other.dtype:
                raise DataContractError('Checkpoint tensor shape/dtype differs: '+key)
            if other.is_floating_point() and not torch.isfinite(other).all():
                raise DataContractError('Nonfinite checkpoint tensor: '+key)
        elif value!=other:raise DataContractError('Checkpoint category/score state differs: '+key)
    if not weights_only:
        if payload.get('training_config')!=training_config:
            raise DataContractError('Resume training semantics differ; use explicit weights-only transfer')
        for name,obj in (('optimizer',optimizer),('scheduler',scheduler),('scaler',scaler)):
            if (obj is None)!=(payload.get(name) is None):raise DataContractError('Resume '+name+' state differs')
    with torch.no_grad():
        for key,value in current.items():
            if isinstance(value,torch.Tensor):value.copy_(supplied[key].to(value.device))
    if not weights_only:
        if optimizer is not None:optimizer.load_state_dict(payload['optimizer'])
        if scheduler is not None:scheduler.load_state_dict(payload['scheduler'])
        if scaler is not None:scaler.load_state_dict(payload['scaler'])
        if restore_random:restore_rng(payload['rng'])
    return copy.deepcopy(payload['progress'])
