"""Public MBTI evaluation/thresholds and identity-preserving prediction exports.

Metrics/configuration helpers need NumPy only. Torch is imported at execution.
"""
from contextlib import nullcontext
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import DataContractError
from .datasets import UserDataset, iter_batches
from .labels import DIMENSIONS, bits_to_mbti
from .metrics import evaluate_mbti, predictions_to_bits
from .token_inputs import validate_identifier


def thresholds_tuple(values=.5):
    v = np.asarray(values,dtype=float)
    if v.ndim==0:v=np.repeat(v,4)
    if v.shape!=(4,) or not np.isfinite(v).all() or not ((v>0)&(v<1)).all():
        raise ValueError('Thresholds must contain four finite values strictly between 0 and 1')
    return tuple(float(x) for x in v)


def threshold_bits(probabilities, thresholds=.5):
    thresholds=thresholds_tuple(thresholds)
    # Reuse the existing score contract (including shape, finite and range checks).
    predictions_to_bits(probabilities,prediction_type='probabilities')
    return (np.asarray(probabilities)>=np.asarray(thresholds)).astype(np.int64)


def select_thresholds(labels, probabilities, *, partition, grid=(.3,.4,.5,.6,.7)):
    if partition!='validation':raise DataContractError('Threshold fitting requires validation, never test/inference')
    candidates=tuple(grid)
    if not candidates:raise ValueError('Threshold grid cannot be empty')
    for t in candidates:thresholds_tuple(t)
    evaluate_mbti(labels,probabilities,prediction_type='probabilities')
    # Deterministic ties prefer closest to .5, then smaller threshold.
    result=[]
    for column,key in enumerate(DIMENSIONS):
        scores=[(evaluate_mbti(labels,probabilities,prediction_type='probabilities',threshold=t)[key],t) for t in candidates]
        result.append(min(scores,key=lambda x:(-x[0],abs(x[1]-.5),x[1]))[1])
    return tuple(result)


def require_partition(dataset, partition):
    if not isinstance(dataset,UserDataset) or dataset.partition!=partition or len(dataset)==0:
        raise DataContractError('Provide a nonempty explicit '+partition+' UserDataset view')
    dataset._check()
    for source,(size,mtime) in dataset.source_versions.items():
        st=Path(source).stat()
        if (st.st_size,st.st_mtime_ns)!=(size,mtime):raise DataContractError('Dataset changed since indexing')


def check_disjoint(*datasets):
    """Compare indexed identities with bounded memory; never read corpus text."""
    for i,left in enumerate(datasets):
        for right in datasets[i+1:]:
            if left.partition is None or right.partition is None:
                raise DataContractError('Choose explicit partitions before checking user isolation')
            for dataset,uid in left.db.execute('SELECT dataset,uid FROM users WHERE partition=?',(left.partition,)):
                if right.db.execute('SELECT 1 FROM users WHERE dataset=? AND uid=? AND partition=?',
                                    (dataset,uid,right.partition)).fetchone():
                    raise DataContractError('User identity overlaps experiment partitions')


@dataclass(frozen=True)
class Predictions:
    identities: tuple[tuple[str,str],...] = field(repr=False)
    probabilities: np.ndarray = field(repr=False)
    labels: np.ndarray | None = field(repr=False)
    reasoning_used: tuple[bool,...]
    partition: str

    def metrics(self,thresholds=.5):
        if self.labels is None:raise DataContractError('Unlabelled prediction has no evaluation targets')
        return evaluate_mbti(self.labels,threshold_bits(self.probabilities,thresholds))


def collect_predictions(model,dataset,*,tokenizer,batch_size=8,preparer=None,temp_dir=None,
                        partition='validation',autocast=None):
    import torch
    if partition not in ('validation','test','inference'):
        raise DataContractError('Evaluation partition must be validation/test/inference')
    require_partition(dataset,partition)
    was_training=model.training;identities=[];scores=[];truth=[];reasoning=[]
    batches=iter_batches(dataset,tokenizer=tokenizer,batch_size=batch_size,
                         require_supervised=partition!='inference',temp_dir=temp_dir)
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                options=preparer.prepare(batch,training=False) if preparer is not None else {}
                with (autocast() if autocast else nullcontext()):out=model(batch,**options)
                probabilities=out.personality_probabilities.detach().float().cpu().numpy()
                predictions_to_bits(probabilities,prediction_type='probabilities')
                identities.extend(zip(batch['dataset_ids'],batch['user_keys']));scores.append(probabilities)
                if batch['labels'] is not None:truth.append(batch['labels'].copy())
                reasoning.extend(bool(x) for x in out.selection.valid_users.detach().cpu().tolist())
    finally:
        batches.close();model.train(was_training)
    if len(identities)!=len(dataset):raise DataContractError('Evaluation user coverage differs from dataset')
    return Predictions(tuple(identities),np.concatenate(scores),np.concatenate(truth) if truth else None,
                       tuple(reasoning),partition)


def write_predictions(path,predictions,*,thresholds=.5):
    """Export only identity and final predictions, never text or hidden states."""
    path=Path(path)
    if path.suffix not in ('.jsonl','.csv'):raise ValueError('Prediction output must be .jsonl or .csv')
    if not isinstance(predictions,Predictions):raise TypeError('Expected Predictions')
    count=len(predictions.identities)
    if (np.shape(predictions.probabilities)!=(count,4) or len(predictions.reasoning_used)!=count
            or predictions.labels is not None and np.shape(predictions.labels)!=(count,4)):
        raise DataContractError('Prediction fields must contain the same number of users, with probabilities shaped [N,4]')
    seen=set()
    for identity in predictions.identities:
        if not isinstance(identity,(tuple,list)) or len(identity)!=2:
            raise DataContractError('Prediction identities must contain dataset_id and user_id')
        for name,value in zip(('dataset_id','user_id'),identity):validate_identifier(value,name)
        if tuple(identity) in seen:raise DataContractError('Duplicate prediction identity')
        seen.add(tuple(identity))
    if any(not isinstance(value,(bool,np.bool_)) for value in predictions.reasoning_used):
        raise DataContractError('reasoning_used must contain boolean values')
    bits=threshold_bits(predictions.probabilities,thresholds)
    path.parent.mkdir(parents=True,exist_ok=True)
    handle,temporary=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(handle,'w',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['dataset_id','user_id','probabilities','predictions','mbti','reasoning_used']) if path.suffix=='.csv' else None
            if writer:writer.writeheader()
            for (dataset,uid),prob,pred,used in zip(predictions.identities,predictions.probabilities,bits,predictions.reasoning_used):
                row={'dataset_id':dataset,'user_id':uid,'probabilities':prob.tolist(),
                     'predictions':pred.tolist(),'mbti':bits_to_mbti(pred),'reasoning_used':bool(used)}
                if writer:
                    row['probabilities']=json.dumps(row['probabilities']);row['predictions']=json.dumps(row['predictions']);writer.writerow(row)
                else:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
