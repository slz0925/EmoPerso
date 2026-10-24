# Training and configuration

[Overview](../README.md) · [Architecture](architecture.md) · [Data](data.md)

## Resources and command settings

The templates select the official post-trained `deepseek-ai/DeepSeek-V3` at revision `e815299b0bcbac849fa540c768ef21845365c9eb` for the Backbone, generator and tokenizer. With `local_files_only=true`, Hugging Face loads an already cached snapshot or local files and fails if required files are missing. It does not download weights. Prepare resources separately, set the data paths under `../local/` and choose devices with sufficient memory.

Backbone and generator use separate hidden-state and CausalLM interfaces. The generator uses the tokenizer from its model source unless `views.generator.tokenizer_name_or_path` is specified. Templates use GPU placement and BF16; CPU execution uses FP32. With Transformers 4.51.3, native FP8 execution requires Triton and CUDA GPUs with compute capability 8.9 or newer; setting BF16 does not convert FP8 weights.

TOML and JSON are supported. Unprefixed model names and `namespace/name` pairs are Hub IDs, including in offline mode. Prefix local relative model directories with `./` or `../`; absolute paths, Windows paths and direct cache snapshot directories are also accepted. Local paths resolve relative to the configuration file regardless of `local_files_only`. Data, output, cache and offload paths are always local. CLI path overrides resolve relative to the current directory.

For local resources, replace the three model/tokenizer sources with versioned directories and omit `revision`: Transformers does not use it to verify local files. A directory named after a commit is still a local directory, not proof of its contents. Record a meaningful `model.backbone_id` for the supplied weights and verify their origin when preparing them. Converted weights need their own identity. Checkpoints compare the source, effective Hub revision and declared Backbone ID; tokenizer encoding fingerprints describe the loaded tokenizer. Generation caches include configured sources, revisions and sampling settings. Use a new versioned directory or fresh cache when replacing local resources in place.

| Template | Data sources | Runtime settings |
|---|---|---|
| emoperso_train.toml | Train and validation; optional evaluation/inference paths | Full training configuration |
| emoperso_evaluate.toml | Test plus train/validation for identity overlap checks | batch_size, precision, seed |
| emoperso_predict.toml | Unlabelled inference | batch_size, precision, seed |

Evaluation and prediction use the shared `[training]` section for these three runtime fields. Other `TrainingConfig` fields take their defaults and do not run optimization, scheduling or early stopping. `seed` also controls reasoning generation. The model-compatible `views.augmentation` setting is retained to construct the Style Head; evaluation and prediction prepare reasoning candidates only.

Root fields are `output_dir`, optional `checkpoint_dir`/`temp_dir`, and training-only `seeds`. Keep model dimensions, tokenizer settings, style categories and resource identities consistent with the checkpoint. Python callers can inject compatible resources through `build_components`.

## Train and resume

Set optimization, accumulation and validation options in the [training template](../configs/emoperso_train.toml) for your dataset and hardware. `TrainingConfig` also supports AdamW, linear/cosine scheduling, warmup and threshold search. `patience=0` disables early stopping.

Run from the repository root after configuring resources, using a fresh output directory:

```shell
python -m emoperso.experiment train --config configs/emoperso_train.toml --output runs/my_run --until-epoch 1
python -m emoperso.experiment train --config configs/emoperso_train.toml --output runs/my_run --resume runs/my_run/checkpoints/latest.pt
```

`--until-epoch` stops at a completed epoch while preserving the configured `max_epochs` and schedule horizon. Resume restores that epoch boundary. To start a new run from compatible model weights, use `train --checkpoint path/to/checkpoint.pt --output path/to/new_run`.

Training checks train/validation user isolation and shuffles training users with seed+epoch, including the final short batch. Gradient accumulation averages active microbatch losses, including short accumulation windows. Loss denominators and InfoNCE negative sets remain local to each microbatch. Inactive batches and FP16 overflow skips leave optimizer and scheduler progress unchanged.

The training template uses two users per microbatch and four accumulation steps. InfoNCE requires at least two different emotion-valid users with accepted reasoning candidates in that microbatch; accumulation does not supply additional negatives. Epoch `InfoNCE` statistics distinguish eligible, active and successfully optimized microbatches, count contributing user occurrences, and report the active-batch mean loss. Check these alongside the other losses; short or insufficiently supervised batches may have no contrastive term.

All trainable modules share one device; frozen Backbone weights may use a separate device map. CUDA supports FP16/BF16 mixed precision. The [architecture guide](architecture.md) defines the joint loss and task activity rules.

## Views and preparation

`views.augmentation` accepts `off`, `style`, `completion` or `both`; `views.reasoning` independently enables reasoning candidates. Completion-only has no Style Head. Generation runs during view preparation, outside the differentiable model.

| Policy | Behaviour |
|---|---|
| on_demand | Generate on each visit |
| reuse | Store and reuse source-bound per-user text in cache_dir |
| prepared | Require previously prepared records |

With `cache_dir` configured, prepare a partition explicitly:

```shell
python -m emoperso.experiment prepare --config configs/emoperso_train.toml --partition train
```

Prepare other partitions separately. Training uses augmentation and reasoning; validation, test and inference use reasoning only. Prompts use visible source text, excluding target labels. Cache entries bind user identity, partition, source, configuration and tokenizer, and may contain private text. Embeddings are recomputed through the current trainable layers. `candidate_microbatch` controls candidate encoding group size; backward activations can span multiple groups.

Loading and execution failures stop the command without caching the failed user record. Individual responses that fail length or quality checks are skipped and may produce a valid empty cached result. Only `emoperso-generated-inputs-v2` records are reused; run `prepare` with the current configuration before using `prepared`. Model/tokenizer paths and generation settings also participate in cache identity. Use versioned resource paths or a fresh cache when replacing resource files in place.

Augmentation and reasoning share one lazy generator; the Backbone is a separate model instance. To avoid loading both sets of weights in the training process, run `prepare` for the required partitions in a separate process, then train with `policy="prepared"`. Cached text is re-encoded through the current trainable model.

## Evaluation and prediction

Validation selects checkpoints by average Macro-F1, requiring an increase greater than `min_improvement`. The default probability threshold is .5. Optional threshold search uses validation only and a fixed per-dimension grid; ties prefer the threshold nearest .5, then the smaller value.

Evaluation and prediction restore checkpoint thresholds:

```shell
python -m emoperso.experiment evaluate --config configs/emoperso_evaluate.toml --checkpoint runs/my_run/checkpoints/best.pt --output runs/final_evaluation
python -m emoperso.experiment predict --config configs/emoperso_predict.toml --checkpoint runs/my_run/checkpoints/best.pt --output runs/unlabelled_prediction
```

`evaluate` accepts validation or test. The supplied train/validation sources are checked for user overlap with the evaluation partition. `predict` reads the inference source without supervision targets or threshold fitting. `--data`, `--tokenizer`, `--batch-size`, `--seed` and `--output` override the corresponding settings; training data paths are configured in the file.

Both commands require checkpoint-compatible style categories, scoring settings, dimensions and weight identity. They export UTF-8 CSV/JSONL containing user identities, probabilities, bits, MBTI and reasoning-use flags. Keep identity-bearing outputs private.

## Checkpoints and records

A training run writes configuration, epoch records, results, preparation statistics and `checkpoints/best.pt`/`latest.pt`. `fit` restores the best model for prediction; `latest.pt` retains the last completed epoch's optimization state.

Lightweight checkpoints save trainable model state, training and random state, thresholds and progress. Supply frozen Backbone weights, tokenizer resources and generation models separately.

Checkpoint loading requires compatible model structure, tokenizer, weight identity, category order and scoring settings. Resume also requires the original run, configuration and input source versions; use weights-only transfer for a new compatible run. Load trusted local checkpoints through the restricted `torch.load(weights_only=True)` path.

For independent training runs, set `seeds=[42,43,44]` in the training configuration. Each run has its own state; the summary reports validation means and population standard deviations. An explicit `--seed` overrides this list and starts a single run. Final test evaluation is a separate command.
