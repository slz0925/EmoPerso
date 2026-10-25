# Inspect user batches

Install `emoperso[tokenizer]` and provide your CSV/JSONL files. Select a compatible DeepSeek-V3 tokenizer directory or Hub repository, then run from the repository root:

```shell
python -m examples.user_batches --data /path/to/your/train.jsonl --tokenizer /path/to/your/tokenizer --local-files-only --partition train --preprocess
```

To use the official Hub tokenizer and allow downloads:

```shell
python -m examples.user_batches --data /path/to/your/train.jsonl --tokenizer deepseek-ai/DeepSeek-V3 --partition train --preprocess
```

The source's default revision is used unless `--revision` selects a branch, tag or commit. `--local-files-only` restricts loading to cached or local files. No model weights are loaded.

The script validates the supplied files and prints batch shapes and counts. For post-level files add `--layout posts`; use `--order recent` when `created_utc` is available. For unlabelled input select `--partition inference`.

Use the [experiment CLI](../docs/training.md) for training, evaluation and prediction, or the [Python API](../docs/python_api.md) for custom workflows.
