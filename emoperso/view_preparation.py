"""Text generation and bounded per-user caching of source-bound text results."""
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from .augmentation import (AugmentedView,TextEdit,_validate_view,_digest,
                          source_signature,prepare_augmented_batches)
from .errors import DataContractError
from .reasoning import ReasoningCandidate,source_binding,prepare_reasoning_batches
from .token_inputs import batch_signature


class ViewPreparer:
    """on_demand generates per visit; reuse writes misses; prepared requires hits.

    Each cache record is bound to the current dataset/user/partition, observed
    text/tokens and generation settings.
    Records contain private text: keep cache_dir outside any public package.
    Empty generation results are valid cached outcomes. Pairing validates
    source bindings for both generated and cached records.
    """
    def __init__(self,tokenizer,*,augmenter=None,reasoner=None,cache_dir=None,
                 policy='on_demand',candidate_microbatch=8,max_cache_bytes=4*2**20):
        if policy not in ('on_demand','reuse','prepared'):raise ValueError('Invalid generation preparation policy')
        if policy!='on_demand' and cache_dir is None:raise ValueError('Reusable/prepared generation needs cache_dir')
        if isinstance(candidate_microbatch,bool) or not isinstance(candidate_microbatch,int) or candidate_microbatch<1:
            raise ValueError('candidate_microbatch must be a positive integer')
        if isinstance(max_cache_bytes,bool) or not isinstance(max_cache_bytes,int) or max_cache_bytes<1:
            raise ValueError('max_cache_bytes must be a positive integer')
        self.tokenizer,self.augmenter,self.reasoner=tokenizer,augmenter,reasoner
        self.cache_dir=Path(cache_dir) if cache_dir is not None else None
        self.policy,self.candidate_microbatch,self.max_cache_bytes=policy,candidate_microbatch,max_cache_bytes
        self.statistics=Counter()

    @property
    def identity(self):
        return {'augmentation':self.augmenter.config_id if self.augmenter else None,
            'reasoning':self.reasoner.config_id if self.reasoner else None,
            'policy':self.policy,'candidate_microbatch':self.candidate_microbatch}

    def _row(self,row,training):
        aug=self.augmenter if training else None
        binding={'policy':'emoperso-generated-inputs-v1','reasoning':source_binding(row),
            'augmentation':source_signature(row) if aug else None,
            'augmentation_config':aug.config_id if aug else None,
            'reasoning_config':self.reasoner.config_id if self.reasoner else None}
        path=self.cache_dir/(_digest(binding)+'.json') if self.cache_dir is not None else None
        if self.policy!='on_demand' and path.is_file():
            if path.stat().st_size>self.max_cache_bytes:raise DataContractError('Generation cache record exceeds limit')
            try:
                value=json.loads(path.read_text(encoding='utf8'))
                if value['binding']!=binding:raise ValueError('Cache binding differs')
                views=[]
                for v in value['views']:
                    v=dict(v);v['edits']=tuple(TextEdit(**e) for e in v['edits'])
                    v['generation_metadata']=tuple(v['generation_metadata']);view=AugmentedView(**v)
                    _validate_view(view);views.append(view)
                chains=[]
                for c in value['candidates']:
                    c=dict(c);c['steps']=tuple(c['steps']);chains.append(ReasoningCandidate(**c))
            except (KeyError,ValueError,TypeError,UnicodeError) as exc:
                raise DataContractError('Invalid generated input cache record') from exc
            self.statistics['cache_hits']+=1
            return views,chains
        if self.policy=='prepared':raise DataContractError('Missing prepared generation record for current source/configuration')
        views=[];chains=[]
        if aug:
            result=aug.generate_row(row);views.extend(result.views)
            self.statistics.update('augmentation:'+v['status'] for v in result.attempts)
        if self.reasoner:
            one=self.tokenizer.batch([row],require_supervised=False)
            result=self.reasoner.generate(one);chains.extend(result.candidates)
            self.statistics.update('reasoning:'+v['status'] for v in result.attempts)
        self.statistics['generated_users']+=1
        if self.policy=='reuse':
            value={'binding':binding,'views':[asdict(v) for v in views],'candidates':[asdict(c) for c in chains]}
            content=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf8')
            if len(content)>self.max_cache_bytes:raise DataContractError('Generated input exceeds cache record limit')
            path.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
            try:
                with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
                os.replace(tmp,path)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        return views,chains

    def prepare(self,batch,*,training):
        batch_signature(batch)
        if training and set(batch['partitions'])!={'train'}:
            raise DataContractError('Only train inputs may use training augmentation')
        if not self.reasoner and not (training and self.augmenter):return {}
        views=[];chains=[]
        for row in batch['provenance']:
            v,c=self._row(row,training);views.extend(v);chains.extend(c)
        options={}
        if training and self.augmenter:
            options['paired']=prepare_augmented_batches(batch,views,tokenizer=self.tokenizer,config_id=self.augmenter.config_id)
        if self.reasoner:
            options['reasoning']=prepare_reasoning_batches(batch,chains,tokenizer=self.tokenizer,
                config_id=self.reasoner.config_id,microbatch_size=self.candidate_microbatch)
        return options
