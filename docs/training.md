# Training and configuration

[Overview](../README.md) · [Architecture](architecture.md) · [Data](data.md)

## Resources and command settings

The templates in `configs/` use external resources under `../local/`. Set the data, tokenizer, Backbone and generator paths, then choose devices with sufficient memory. Templates use GPU placement and BF16; CPU execution uses FP32. Keep model dimensions, tokenizer settings, style categories and weight identity consistent with the checkpoint.

TOML and JSON are supported. Resource paths, including the Backbone offload folder, resolve relative to the configuration file. CLI path overrides resolve relative to the current directory. Optional TOML fields can be omitted.

| Template | Data sources | Runtime settings |
|---|---|---|
| emoperso_train.toml | Train and validation; optional evaluation/inference paths | Full training configuration |
| emoperso_evaluate.toml | Test plus train/validation for identity overlap checks | batch_size, precision, seed |
| emoperso_predict.toml | Unlabelled inference | batch_size, precision, seed |

Evaluation and prediction use the shared `[training]` section for these three runtime fields. Other `TrainingConfig` fields take their defaults and do not run optimization, scheduling or early stopping. `seed` also controls reasoning generation. The model-compatible `views.augmentation` setting is retained to construct the Style Head; evaluation and prediction prepare reasoning candidates only.

Root fields are `output_dir`, optional `checkpoint_dir`/`temp_dir`, and training-only `seeds`. Specify a stable `model.backbone_id` or revision for external weights. Enabled generation requires compatible generator resources; Python callers can inject them through `build_components`.

## Train and resume

Set optimization, accumulation and validation options in the [training template](../configs/emoperso_train.toml) for your dataset and hardware. `TrainingConfig` also supports AdamW, linear/cosine scheduling, warmup and threshold search. `patience=0` disables early stopping.

Run from the repository root after configuring resources, using a fresh output directory:

```shell
python -m emoperso.experiment train --config configs/emoperso_train.toml --output runs/my_run --until-epoch 1
python -m emoperso.experiment train --config configs/emoperso_train.toml --output runs/my_run --resume runs/my_run/checkpoints/latest.pt
```

`--until-epoch` stops at a completed epoch while preserving the configured `max_epochs` and schedule horizon. Resume restores that epoch boundary. To start a new run from compatible model weights, use `train --checkpoint path/to/checkpoint.pt --output path/to/new_run`.

Training checks train/validation user isolation and shuffles training users with seed+epoch, including the final short batch. Gradient accumulation averages active microbatch losses, including short accumulation windows. Loss denominators and InfoNCE negative sets remain local to each microbatch. Inactive batches and FP16 overflow skips leave optimizer and scheduler progress unchanged.

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

Invalid generation resources stop the command with a loading error. Individual responses that fail length or quality checks are skipped and counted in preparation statistics.

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
