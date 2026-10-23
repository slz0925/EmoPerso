"""Configured train/evaluate/predict/prepare commands; local resources by default."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict,dataclass,field,replace
import json
from pathlib import Path
import tomllib

import numpy as np

from .data import DataContractError


@dataclass(frozen=True)
class ExperimentConfig:
    data: dict = field(default_factory=dict)
    tokenizer: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)
    output_dir: str = 'runs/emoperso'
    checkpoint_dir: str | None = None
    temp_dir: str | None = None
    seeds: tuple[int,...] = ()

    def __post_init__(self):
        sections={'data':{'path','train','validation','test','inference','dataset_id','layout','preprocess','order','text'},
            'model':{'backbone','backbone_id','shared_hidden_size','shared_device','hidden_size','dropout',
                     'num_heads','attention_size','bottleneck_size','association_size','association_temperature',
                     'ig_weight','association_weight','selection_temperature'},
            'views':{'augmentation','reasoning','generator','policy','cache_dir',
                     'augmentation_config','reasoning_config','candidate_microbatch'},
            'losses':{'personality_weight','emotion_weight','lambda_mtl','lambda_cross','style_weight','consistency_weight',
                      'star_weight','contrastive_weight'}}
        for section in ('data','tokenizer','model','training','views','losses'):
            value=getattr(self,section)
            if not isinstance(value,dict):raise ValueError(section+' must be an object/table')
            if section in sections and value.keys()-sections[section]:raise ValueError('Unknown '+section+' settings: '+str(sorted(value.keys()-sections[section])))
        for key in ('reasoning',):
            if key in self.views and not isinstance(self.views[key],bool):raise ValueError(key+' must be boolean')
        if self.views.get('augmentation','off') not in ('off','style','completion','both'):
            raise ValueError('augmentation must be off/style/completion/both')
        if not isinstance(self.output_dir,str) or not self.output_dir.strip():raise ValueError('output_dir is required')
        if any(isinstance(s,bool) or not isinstance(s,int) or not 0<=s<2**63 for s in self.seeds) or len(set(self.seeds))!=len(self.seeds):
            raise ValueError('seeds must contain distinct nonnegative supported integers')
        object.__setattr__(self,'seeds',tuple(self.seeds))


def read_config(path):
    """JSON/TOML. Resource paths resolve relative to the config, not the cwd."""
    path=Path(path).resolve()
    with path.open('rb') as stream:
        raw=tomllib.load(stream) if path.suffix=='.toml' else json.load(stream)
    if not isinstance(raw,dict):raise ValueError('Configuration must be an object')
    config=ExperimentConfig(**raw)
    def resource(value):
        if isinstance(value,list):return [resource(x) for x in value]
        return str((path.parent/Path(value)).resolve())
    data=dict(config.data)
    for key in ('path','train','validation','test','inference'):
        if data.get(key) is not None:data[key]=resource(data[key])
    tok=dict(config.tokenizer)
    if tok.get('local_files_only',True):
        for key in ('tokenizer_name_or_path','model_name_or_path'):
            if tok.get(key):tok[key]=resource(tok[key])
    model=dict(config.model);backbone=dict(model.get('backbone',{}))
    if backbone.get('local_files_only',True) and backbone.get('model_name_or_path'):
        backbone['model_name_or_path']=resource(backbone['model_name_or_path'])
    if backbone.get('offload_folder'):
        backbone['offload_folder']=resource(backbone['offload_folder'])
    if backbone:model['backbone']=backbone
    views=dict(config.views)
    if views.get('cache_dir'):views['cache_dir']=resource(views['cache_dir'])
    generator=dict(views.get('generator',{}))
    if generator.get('local_files_only',True):
        for key in ('model_name_or_path','tokenizer_name_or_path'):
            if generator.get(key):generator[key]=resource(generator[key])
    if generator:views['generator']=generator
    return replace(config,data=data,tokenizer=tok,model=model,views=views,output_dir=resource(config.output_dir),
        checkpoint_dir=resource(config.checkpoint_dir) if config.checkpoint_dir else None,
        temp_dir=resource(config.temp_dir) if config.temp_dir else None)


def build_components(config,*,prepare_only=False,training_augmentation=True,
                     tokenizer=None,backbone=None,generator=None,backbone_identity=None):
    """Construct configured components, optionally reusing caller-owned resources.

    An injected FrozenBackbone must share the tokenizer's encoding fingerprint.
    Supply a stable JSON object as backbone_identity for checkpoint compatibility.
    Generators implement generate(prompt, seed=...) and a JSON identity.
    """
    from .augmentation import AugmentationConfig,DEFAULT_STYLES,TextAugmenter
    from .augmentation_model import AugmentationLoss
    from .backbone import BackboneConfig,EmoPersoEncoder,FrozenBackbone
    from .deepseek_tokenizer import TokenizerConfig,load_tokenizer
    from .generation import GenerationConfig,HFCausalLMGenerator
    from .interaction import EmoPersoInteractionModel,InteractionLoss
    from .mtl import EmoPersoMTL
    from .preparation import ViewPreparer
    from .reasoning import ReasoningConfig,ReasoningGenerator
    from .reasoning_model import EmoPersoReasoningModel,ReasoningLoss
    from .training import TrainingConfig,seed_everything

    training=TrainingConfig(**config.training);seed_everything(training.seed)
    if tokenizer is None:
        tokenizer=load_tokenizer(TokenizerConfig(**{'local_files_only':True,**config.tokenizer}))
    v=config.views;mode=v.get('augmentation','off');styles=DEFAULT_STYLES if mode in ('style','both') else ()
    aug_options=dict(v.get('augmentation_config',{}))
    if 'styles' in aug_options:styles=tuple(aug_options['styles'])
    if mode in ('off','completion') and styles:raise ValueError('Disabled style branch cannot specify style categories')
    if 'completion' in aug_options and aug_options['completion']!=(mode in ('completion','both')):
        raise ValueError('Augmentation branch configuration contradicts selected mode')
    if (training_augmentation and mode!='off') or v.get('reasoning',False):
        if generator is None:
            if not v.get('generator'):raise ValueError('Enabled views require a configured or injected generator')
            generator=HFCausalLMGenerator(GenerationConfig(**v['generator']))
    augmenter=TextAugmenter(generator,AugmentationConfig(**{'styles':styles,'completion':mode in ('completion','both'),
        'seed':training.seed,**aug_options})) if training_augmentation and mode!='off' else None
    reasoner=ReasoningGenerator(generator,ReasoningConfig(**{'seed':training.seed,**v.get('reasoning_config',{})})) if v.get('reasoning',False) else None
    preparer=ViewPreparer(tokenizer,augmenter=augmenter,reasoner=reasoner,
        cache_dir=v.get('cache_dir'),policy=v.get('policy','on_demand'),candidate_microbatch=v.get('candidate_microbatch',8))
    if prepare_only:return tokenizer,preparer
    m=config.model
    if backbone is None:
        if not m.get('backbone',{}).get('model_name_or_path'):
            raise ValueError('Configure model.backbone.model_name_or_path or inject a FrozenBackbone')
        b=BackboneConfig(**m['backbone'])
        encoder=EmoPersoEncoder.from_pretrained(b,tokenizer=tokenizer,shared_hidden_size=m.get('shared_hidden_size',4096),shared_device=m.get('shared_device'))
        backbone_identity={'kind':'pretrained','id':m.get('backbone_id') or b.revision or Path(b.model_name_or_path).name}
    else:
        if not isinstance(backbone,FrozenBackbone):raise TypeError('backbone must be a FrozenBackbone')
        if backbone.tokenizer.encoding_fingerprint!=tokenizer.encoding_fingerprint:
            raise ValueError('Injected Backbone and tokenizer encoding configurations differ')
        if not isinstance(backbone_identity,dict) or not backbone_identity:
            raise ValueError('An injected Backbone requires a nonempty backbone_identity object')
        backbone_identity=json.loads(json.dumps(backbone_identity,allow_nan=False))
        encoder=EmoPersoEncoder(backbone,shared_hidden_size=m.get('shared_hidden_size',4096),shared_device=m.get('shared_device'))
    core=EmoPersoInteractionModel(EmoPersoMTL(encoder,hidden_size=m.get('hidden_size'),dropout=m.get('dropout',.2)),
        num_heads=m.get('num_heads',4),attention_size=m.get('attention_size'))
    reasoning_keys=('bottleneck_size','association_size','association_temperature','ig_weight','association_weight','selection_temperature')
    model=EmoPersoReasoningModel(core,styles=styles,**{k:m[k] for k in reasoning_keys if k in m})
    l=config.losses
    criterion=ReasoningLoss(base_loss=InteractionLoss(**{k:l[k] for k in ('personality_weight','emotion_weight','lambda_mtl','lambda_cross') if k in l}),
        augmentation_loss=AugmentationLoss(styles=styles,**{k:l[k] for k in ('style_weight','consistency_weight') if k in l}),
        **{k:l[k] for k in ('star_weight','contrastive_weight') if k in l})
    return tokenizer,model,criterion,preparer,training,backbone_identity


def _dataset(stack,config,partition):
    from .datasets import load_dataset
    from .text_data import TextConfig
    d=config.data;path=d.get(partition) or d.get('path')
    if path is None:raise ValueError('No '+partition+' data path configured')
    return stack.enter_context(load_dataset(path,partition=partition,dataset_id=d.get('dataset_id','custom'),
        layout=d.get('layout','users'),preprocess=d.get('preprocess',False),order=d.get('order','source'),
        text_config=TextConfig(**d.get('text',{})),temp_dir=config.temp_dir))


def run_experiment(command,config,*,checkpoint=None,resume=None,partition=None,until_epoch=None):
    if command not in ('train','evaluate','predict','prepare'):
        raise ValueError('Unsupported experiment command')
    if checkpoint is not None and resume is not None:
        raise ValueError('Choose checkpoint transfer or resume, not both')
    if command in ('evaluate','predict') and checkpoint is None:
        raise ValueError('Evaluation/prediction require an explicitly selected checkpoint')
    if command!='train' and (resume is not None or until_epoch is not None):
        raise ValueError('resume and until_epoch are only for training')
    if command=='prepare' and checkpoint is not None:
        raise ValueError('prepare does not load checkpoints')
    allowed={'train':(None,), 'evaluate':(None,'validation','test'),
             'predict':(None,'inference'), 'prepare':(None,'train','validation','test','inference')}
    if partition not in allowed[command]:raise ValueError('Invalid partition for '+command)
    from .checkpointing import atomic_json,load_checkpoint,model_signature
    from .datasets import iter_batches
    from .evaluation import collect_predictions,write_predictions
    from .training import Trainer,precision_context
    if command=='prepare':
        if not config.views.get('cache_dir'):raise ValueError('prepare requires an explicit cache_dir')
        config=replace(config,views={**config.views,'policy':'reuse'})
        tokenizer,preparer=build_components(config,prepare_only=True,training_augmentation=(partition or 'train')=='train')
        with ExitStack() as stack:
            part=partition or 'train';dataset=_dataset(stack,config,part)
            batches=iter_batches(dataset,tokenizer=tokenizer,batch_size=config.training.get('batch_size',8),
                require_supervised=part!='inference',temp_dir=config.temp_dir)
            try:
                for batch in batches:preparer.prepare(batch,training=part=='train')
            finally:batches.close()
            return {'partition':part,'users':len(dataset),'preparation':dict(preparer.statistics)}
    tokenizer,model,criterion,preparer,training,identity=build_components(config,training_augmentation=command=='train')
    if command=='train':
        with ExitStack() as stack:
            train=_dataset(stack,config,'train');validation=_dataset(stack,config,'validation')
            trainer=Trainer(model,criterion,tokenizer=tokenizer,config=training,compatibility=identity,
                            preparer=preparer,temp_dir=config.temp_dir)
            if checkpoint:
                load_checkpoint(checkpoint,model,compatibility=trainer.compatibility,weights_only=True,restore_random=False)
            result=trainer.fit(train,validation,output_dir=config.output_dir,checkpoint_dir=config.checkpoint_dir,
                resume=resume,until_epoch=until_epoch)
            atomic_json(Path(config.output_dir)/'config.json',asdict(config))
            atomic_json(Path(config.output_dir)/'preparation_statistics.json',dict(preparer.statistics))
            return {k:v for k,v in result.items() if k!='history'}
    part=partition or ('inference' if command=='predict' else 'test')
    progress=load_checkpoint(checkpoint,model,compatibility={'model':model_signature(model),'external':identity},weights_only=True,restore_random=False)
    out=Path(config.output_dir)/(command+'-'+part)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Evaluation output already exists; choose another output_dir')
    with ExitStack() as stack:
        dataset=_dataset(stack,config,part)
        if command=='evaluate':
            # Test labels never enter fit. At explicit final evaluation, check
            # identity overlap with configured train/validation sources as well.
            from .evaluation import check_disjoint
            others=[]
            for other in ('train','validation'):
                if other!=part and (config.data.get(other) or config.data.get('path')):
                    others.append(_dataset(stack,config,other))
            check_disjoint(dataset,*others)
        prediction=collect_predictions(model,dataset,tokenizer=tokenizer,batch_size=training.batch_size,
            preparer=preparer,temp_dir=config.temp_dir,partition=part,autocast=precision_context(model,training.precision))
        thresholds=progress['thresholds']
        write_predictions(out/'predictions.jsonl',prediction,thresholds=thresholds)
        write_predictions(out/'predictions.csv',prediction,thresholds=thresholds)
        result={'partition':part,'users':len(prediction.identities),'thresholds':thresholds,
                'checkpoint_epoch':progress['epoch'],'metrics':prediction.metrics(thresholds) if command=='evaluate' else None}
        atomic_json(out/'metrics.json',result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','evaluate','predict','prepare'))
    parser.add_argument('--config',required=True);parser.add_argument('--tokenizer')
    parser.add_argument('--data');parser.add_argument('--partition',choices=('train','validation','test','inference'))
    parser.add_argument('--output');parser.add_argument('--checkpoint');parser.add_argument('--resume')
    parser.add_argument('--batch-size',type=int);parser.add_argument('--max-epochs',type=int);parser.add_argument('--seed',type=int)
    parser.add_argument('--until-epoch',type=int,help='Stop at this completed epoch while preserving the configured total schedule horizon')
    args=parser.parse_args(argv)
    try:
        config=read_config(args.config)
        if args.command=='train' and args.partition is not None:
            raise ValueError('Training uses configured train and validation partitions; omit --partition')
        if args.tokenizer:config=replace(config,tokenizer={**config.tokenizer,'tokenizer_name_or_path':str(Path(args.tokenizer).resolve()),'local_files_only':True})
        if args.output:config=replace(config,output_dir=str(Path(args.output).resolve()),checkpoint_dir=None)
        if args.data:
            if args.command=='train':raise ValueError('Set train and validation data in config; --data is for evaluate/predict/prepare')
            part=args.partition or ('inference' if args.command=='predict' else 'train' if args.command=='prepare' else 'test')
            config=replace(config,data={**config.data,part:str(Path(args.data).resolve())})
        overrides={k:getattr(args,k) for k in ('batch_size','max_epochs','seed') if getattr(args,k) is not None}
        config=replace(config,training={**config.training,**overrides})
        if args.seed is not None:config=replace(config,seeds=())
        if args.command=='train' and config.seeds:
            if args.resume or args.checkpoint:raise ValueError('Multi-seed train does not accept a shared checkpoint/resume')
            root=Path(config.output_dir)
            if root.exists() and any(root.iterdir()):raise FileExistsError('Multi-seed experiment output is not empty')
            records=[]
            for seed in config.seeds:
                single=replace(config,seeds=(),training={**config.training,'seed':seed},output_dir=str(root/f'seed_{seed}'),checkpoint_dir=None)
                records.append(run_experiment('train',single,until_epoch=args.until_epoch))
            scores=[r['best_score'] for r in records]
            if any(s is None for s in scores):raise ValueError('No validation metric yet; cannot summarize seeds')
            result={'seeds':list(config.seeds),'runs':records,'validation_average_macro_f1_mean':float(np.mean(scores)),
                'validation_average_macro_f1_std_population':float(np.std(scores,ddof=0)),'test_evaluated':False}
            from .checkpointing import atomic_json
            atomic_json(root/'seed_summary.json',result)
        else:
            result=run_experiment(args.command,config,checkpoint=args.checkpoint,resume=args.resume,
                                  partition=args.partition,until_epoch=args.until_epoch)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    except ImportError as exc:parser.exit(2,'Model execution requires emoperso[backbone]: '+str(exc)+'\n')
    except (ValueError,TypeError,OSError,DataContractError) as exc:parser.exit(2,str(exc)+'\n')
    except RuntimeError as exc:
        from .backbone import BackboneLoadError
        from .generation import GenerationLoadError
        if not isinstance(exc,(BackboneLoadError,GenerationLoadError)):raise
        parser.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
