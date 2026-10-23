# Python API and package structure

[Overview](../README.md) · [Data](data.md) · [Architecture](architecture.md) · [Training](training.md)

## Construct an experiment

```python
from emoperso.experiment import read_config, build_components, run_experiment

config = read_config("configs/emoperso_train.toml")
tokenizer, model, criterion, preparer, training, identity = build_components(config)
```

`build_components` loads the configured frozen Backbone and creates the trainable model, joint loss and view preparer. Generation resources are loaded lazily when preparing views. `build_components(config, prepare_only=True)` returns just the tokenizer and preparer. `run_experiment("train", config)` runs the configured training workflow; `evaluate` and `predict` also require `checkpoint=...`.

To reuse already loaded resources, supply `tokenizer=...`, `backbone=...` and optionally `generator=...`. The Backbone must be a `FrozenBackbone` bound to the same tokenizer encoding. Supply a nonempty JSON-serializable `backbone_identity` describing its stable weight version; checkpoints compare this identity. Resource injection uses the same trainable architecture as configuration-based construction.

## Encode and optimize

```python
from emoperso.datasets import load_dataset, iter_batches
from emoperso.mtl import prepare_targets
import torch

optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
with load_dataset("/path/to/your/train.jsonl", partition="train", preprocess=True) as users:
    batches = iter_batches(users, tokenizer=tokenizer, batch_size=2)
    try:
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, **preparer.prepare(batch, training=True))
            loss = criterion(output, prepare_targets(batch))
            if loss.has_active_supervision:
                loss.total_loss.backward()
                optimizer.step()
    finally:
        batches.close()
```

Use `Trainer` for accumulation, scheduling, validation, checkpoints and resume. The example above shows the component-level contract. Inference uses `model.eval()`, `torch.no_grad()` and `preparer.prepare(batch, training=False)`; unlabelled batches use `require_supervised=False`.

`load_checkpoint(path, model, compatibility=..., weights_only=True, restore_random=False)` restores compatible trainable model state. `Trainer.compatibility` supplies the complete run contract; `run_experiment` handles it automatically.

## Custom generation

A generator implements `generate(prompt, *, seed=None) -> GenerationResult` and exposes a JSON-serializable `identity` containing its model version and generation settings. `HFCausalLMGenerator` loads a decoder-only model with its own tokenizer. `from_components(model, tokenizer, config)` supports a caller-owned CausalLM. Generation is outside the differentiable model; the model learns from accepted, source-bound views and candidates.

## Module map

| Modules | Responsibility |
|---|---|
| labels, data, metrics | MBTI encoding, cumulative feature files, Macro-F1 |
| raw_data, comment_index, conflict_audit, conflict_resolution | Source identities, comment indexing and conflict isolation |
| text_data, datasets | User text views, cleaning and CSV/JSONL loading |
| token_inputs, deepseek_tokenizer, token_statistics | Batch contracts, encoding and coverage summaries |
| backbone, emotion, mtl, interaction | Shared features, cue supervision and task interaction |
| generation, augmentation, augmentation_model | Generation, paired views and auxiliary objectives |
| reasoning, reasoning_selection, reasoning_model | Candidates, scores, selection and fusion |
| preparation, training, evaluation, checkpointing, experiment | View caches, training, prediction, recovery and CLI |

The package root exports lightweight data, label and metric functions. Import model modules explicitly when model dependencies are available.

## Dependencies and distributions

Use the root [requirements.txt](../requirements.txt) for the pinned model environment. [pyproject.toml](../pyproject.toml) defines the base dependency, tokenizer and Backbone extras, and build requirements. The base installation requires NumPy; the model extra adds PyTorch, Transformers, Accelerate and packaging.

Wheel contains the Python package, license metadata and three configuration templates under `<installation-prefix>/share/emoperso/configs`. Copy that `configs` directory into your experiment directory before editing resource paths. The source distribution also includes the input example and documentation. Commands use the configuration explicitly supplied with `--config`.
