# Inspect user batches

Install `emoperso[tokenizer]`, provide your CSV/JSONL files and a compatible local DeepSeek-V3 tokenizer, then run from the repository root:

```shell
python -m examples.user_batches --data /path/to/your/train.jsonl --tokenizer /path/to/your/tokenizer --partition train --preprocess
```

The script validates the supplied files and prints batch shapes and counts. For post-level files add `--layout posts`; use `--order recent` when `created_utc` is available. For unlabelled input select `--partition inference`.

Use the [experiment CLI](../docs/training.md) for training, evaluation and prediction, or the [Python API](../docs/development.md) for custom workflows.
