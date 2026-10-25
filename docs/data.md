# Data and model inputs

[Overview](../README.md) · [Architecture](architecture.md) · [Training](training.md)

## Labels and metrics

The canonical bit order is `(I/E, N/S, T/F, P/J)`, with class 1 equal to `E, S, F, J`. The prediction dimensions are displayed as I/E, S/N, T/F and P/J, with positive classes E, S, F and J.

| Dimension | 0 | 1 |
|---|---|---|
| 0 | I | E |
| 1 | N | S |
| 2 | T | F |
| 3 | P | J |

`emoperso.labels` converts strings, four bits and the original 16-class IDs. String input accepts lowercase and surrounding whitespace. The class-ID order is:

```text
ISTJ ISFJ INFJ INTJ ISTP ISFP INFP INTP
ESTP ESFP ENFP ENTP ESTJ ESFJ ENFJ ENTJ
```

The 16-class contract is for data interoperability; the model predicts four independent binary logits. `evaluate_mbti(truth, predictions, prediction_type="logits")` computes binary Macro-F1 including both classes in every dimension, then averages the four scores. Set `prediction_type` to match the input: `labels`, `probabilities` or `logits`. Labels have shape `[N,4]`; scores may be `[N,4]` class-1 scores or `[N,4,2]` paired scores with explicit `class_order=(0,1)` or `(1,0)`. Paired probabilities must sum to one. Values equal to the threshold select class 1. A zero F1 denominator contributes zero, so a perfectly predicted constant-zero column has Macro-F1 .5. Inputs must be nonempty and finite. See `metrics.py` for validation.

## Public CSV/JSONL

`load_dataset(path_or_paths, partition=..., layout="users")` validates all supplied files before yielding a selected partition. Use it as a context manager. An input record is:

```json
{"dataset_id":"demo","user_id":"sample-1","text":"I feel curious today!","mbti":"INFP","partition":"train"}
```

| Field | Contract |
|---|---|
| user_id | Required nonempty stable source identifier |
| dataset_id | Optional namespace, default `custom` |
| text | Required text; no other profile field is encoded |
| mbti | Complete four-letter type for supervised input |
| labels | Alternative four-element binary list in JSONL |
| label0..label3 | Alternative binary columns in CSV |
| partition | `train`, `validation`, `test`; unlabelled input may omit it or use `inference` |

If multiple label representations are present they must agree. Invalid, incomplete or contradictory labels are rejected; unlabelled records use the inference partition. `(dataset_id,user_id)` identifies one user. Duplicate user rows, cross-partition identities and label conflicts fail before the first batch. Identity checks cover the files supplied together.

For `layout="posts"`, each record also has `post_id`. Users may appear in nonadjacent rows or different files. Post IDs must be unique within a dataset/user, and may be reused in other namespaces. Source record/byte spans track provenance independently of post identity. Specify `order="recent"` only with valid nonnegative decimal UTC seconds in `created_utc`; ordering is numeric descending with source-order ties. Otherwise `order="source"` follows input file and record order.

The [batch inspection example](../examples/README.md) loads these files and prints batch shapes and user counts. For unlabelled batches, select `partition="inference"` and set `require_supervised=False` in `iter_batches`.

## Text policy and bounded processing

`preprocess=False` preserves input content apart from the required resource bounds. `True` enables URL, explicit MBTI and deleted-placeholder cleaning. Emotion words, negation, case, style and punctuation are preserved. Defaults in `TextConfig` are 64 posts, 4,096 characters per post and 65,536 per user. A user-layout row uses the overall user budget. Budgets are configurable independently of the maximum Token budget.

Character clipping retreats to a safe word boundary. An overlong first word can leave no usable text; eligibility/reason metadata records this outcome. Posts join with two newlines. Label detection respects real post boundaries: letters across different posts are not treated as a source-post MBTI mention. `user_text()` treats supplied text as a single unit and does not infer original posts or users.

Shuffle is explicit and seeded for training; validation/test shuffling is rejected. Tail batches are included. `UserDataset` uses disk indexing/sorting, and streaming UserText input uses bounded shuffling and disk-backed duplicate checks. The default maximum source record is 1 MiB (`max_record_bytes` can be configured). Private temporary SQLite files can contain identities and text: use a writable `temp_dir` with source-scale disk capacity, and close datasets/iterators. Interruptions may leave private temporary files.

## Tokenizer and Batch

`DeepSeekTokenizerAdapter` accepts compatible DeepSeek-V3 fast BPE resources from a Hub repository or local directory. `TokenizerConfig` defaults to `deepseek-ai/DeepSeek-V3`, revision `None`, max_length 2048, right padding and `local_files_only=False`. The experiment templates use the source's default revision and set `local_files_only=true` to read cached or local files. Set it to `false` to allow Hub downloads. Repository Python code is not executed.

Input is plain user text with one automatic BOS, no automatic EOS, chat template or personality prompt. The length budget includes specials and is in `[1,2048]`. Truncation retains a prefix and retreats when needed to avoid partial Unicode decoding or exposing a new explicit MBTI type inside a source post. Existing explicit types must be cleaned first or the adapter rejects them. Padding may be left or right; special/control Tokens are not pooled as content. Vocabulary, special IDs, offsets and model compatibility are validated; the supplied vocabulary and padding configuration are preserved.

| Batch key | Shape / meaning |
|---|---|
| input_ids, attention_mask, position_ids | `[B,T]`, NumPy int64 |
| content_mask, special_tokens_mask | `[B,T]`, NumPy bool |
| labels | `[B,4]` int64 for supervision; `None` for inference |
| dataset_ids, user_keys, partitions | Ordered identities, matching every row |
| provenance | TokenInput/UserText records, source tracking and certified visible character spans |

`attention_mask` includes actual specials but excludes padding; `content_mask` excludes both. A configuration/content fingerprint prevents incompatible token rows being combined. Certified `content_character_spans`, retained characters and post coverage support emotion/augmentation rules that only inspect visible text. Custom `TokenizerAdapter` implementations must supply certified spans for automatic emotion targets. Alternatively, supply identity-aligned `ExternalEmotionTargets` explicitly.

## Source integrity and user records

The source-processing APIs build user-level inputs from Kaggle records and Pandora profiles/comments:

- `raw_data` assigns stable source keys, validates complete MBTI labels and creates user-level splits. Kaggle keys identify dataset records; Pandora uses the source author field to link profiles and comments. Users with incomplete labels remain unsupervised.
- `comment_index` builds a disk-backed comment index, retaining source locations and duplicate versions. `conflict_audit` compares those versions; `conflict_resolution` produces a source-bound resolution overlay. Substantive body conflicts are isolated, while accepted duplicate IDs retain their first original version.
- `UserTextSource` reads a ready, source-verified bundle and constructs user text with the configured selection and cleaning policy. Bundle status, source versions and resolution evidence are checked before normal reads.

The general CSV/JSONL interface accepts caller-prepared user data directly. When preparing a cohort from raw records, preserve user identities, label eligibility, comment decisions and user-level partitions.

SQLite readers require stable committed input. `prepare_sqlite_snapshot(source, new_destination)` creates an independent committed snapshot from an eligible source; the destination needs its own source binding before use in a verified bundle. Active writers and nonempty WAL must be resolved before snapshot preparation. Source bundles, temporary indexes and generated caches can contain private user data.

## Token statistics

`emoperso.token_statistics.summarize_token_inputs(inputs)` consumes encoded `TokenInput` rows with certified character retention, content counts and post coverage. It returns dictionaries grouped by dataset, partition and supervision eligibility, with token-length distributions, retention ratios, post coverage and exclusion counts. `TokenStatistics.update(row)` supports incremental collection, followed by `result()`.

Use these summaries to inspect the effect of token budgets on a cohort. Exact percentiles retain scalar values per user, so memory grows linearly with user count.
