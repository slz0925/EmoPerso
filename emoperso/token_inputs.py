"""Tokenizer-adapter interfaces, provenance-aware user inputs and NumPy batches."""

import json
import hashlib
from collections.abc import Mapping

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .data import DataContractError
from .labels import bits_to_mbti, mbti_to_bits
from .text_data import TextPost, UserText, SEPARATOR, has_content, text_sha


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and 0 <= value < 2**63


def _vector(value, name, *, binary=False):
    try:
        values = tuple(value)
    except TypeError as exc:
        raise DataContractError(f"Tokenizer {name} must be a flat integer sequence") from exc
    if any(not _integer(v) or binary and v not in (0, 1) for v in values):
        raise DataContractError(f"Tokenizer {name} contains invalid values")
    return tuple(int(v) for v in values)


def validate_identifier(value, name):
    if (not isinstance(value, str) or not value.strip()
            or value != value.strip() or any(ord(c) < 32 for c in value)):
        raise DataContractError(f"{name} must be a nonempty identifier without edge whitespace/control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DataContractError(f"{name} contains malformed Unicode") from exc


def post_key(dataset_id, user_key, original_post_id):
    """Lossless namespaced identity, independent of source file ordering/paths."""
    for name, value in (("dataset_id", dataset_id), ("user_key", user_key), ("post_id", original_post_id)):
        validate_identifier(value, name)
    return "post:"+json.dumps([dataset_id, user_key, original_post_id], ensure_ascii=True, separators=(",", ":"))


def user_text(user_key, text, *, labels=None, partition=None, dataset_id="custom"):
    """Construct a UserText view from caller-provided text and identity.

    Text is preserved as one unit. Apply preprocessing upstream when needed.
    labels uses the canonical (I/E, N/S, T/F, P/J) binary class order.
    partition=None supports inference; supervised batches require an explicit split.
    """
    validate_identifier(user_key, "user_key")
    validate_identifier(dataset_id, "dataset_id")
    if not isinstance(text, str):
        raise DataContractError("User text must be a string")
    if partition not in (None, "train", "validation", "test"):
        raise DataContractError("Invalid user partition")
    try:
        mbti = bits_to_mbti(labels) if labels is not None else None
    except ValueError as exc:
        raise DataContractError("Expected four canonical binary MBTI labels") from exc
    bits = mbti_to_bits(mbti) if mbti else None
    eligible = bool(text.strip() and has_content(text))
    return UserText(dataset_id=dataset_id, user_key=user_key, partition=partition,
        mbti=mbti, bits=bits, label_eligible=bits is not None, policy="provided-user-text-v1",
        posts=(), text=text, raw_post_count=0, nonempty_raw_post_count=0,
        valid_post_count=0, candidate_post_count=0, operations=(), invalid_post_reasons=(),
        text_eligible=eligible, sample_eligible=eligible and bits is not None,
        exclusion_reasons=() if eligible else ("no_content_in_provided_text",))


@dataclass(frozen=True)
class TokenInput:
    dataset_id: str
    user_key: str
    partition: str | None
    labels: tuple[int, int, int, int] | None
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    content_mask: tuple[int, ...]
    special_tokens_mask: tuple[int, ...]
    tokenizer_id: str
    tokenizer_revision: str
    max_length: int
    original_token_count: int
    truncated: bool
    token_eligible: bool
    sample_eligible: bool
    text_sha256: str
    source: UserText = field(repr=False)
    encoded_post_coverage: str = "source candidates traced; exact token-to-post alignment unavailable"
    retained_characters: int | None = None
    original_content_token_count: int | None = None
    post_content_token_counts: tuple[int, ...] = ()
    unicode_boundary_tokens_removed: int = 0
    label_boundary_tokens_removed: int = 0
    token_exclusion_reasons: tuple[str, ...] = ()
    encoding_fingerprint: str | None = None
    # Union of retained non-special token offsets in source character space.
    # None means that this tokenizer cannot certify the visible text range.
    content_character_spans: tuple[tuple[int, int], ...] | None = None


class TokenizerAdapter:
    """Adapter for an explicitly supplied Hugging-Face-style callable tokenizer.

    Requires flat input_ids/attention_mask/special_tokens_mask, pad_token_id,
    all_special_ids and right truncation. Uses the supplied vocabulary and pad
    token. DeepSeek-specific compatibility checks belong to DeepSeekTokenizerAdapter.
    A full bounded text encoding measures true tokenizer length; if over budget,
    a second tokenizer call performs truncation and preserves its own specials.
    """

    def __init__(self, tokenizer, *, tokenizer_id, revision, max_length=2048, padding_side="right"):
        if not tokenizer_id or not isinstance(tokenizer_id, str) or not revision or not isinstance(revision, str):
            raise ValueError("Explicit tokenizer identity and revision are required")
        if not _integer(max_length) or not 1 <= max_length <= 2048:
            raise ValueError("max_length must be an integer in [1,2048], including specials")
        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be left or right")
        if getattr(tokenizer, "truncation_side", None) != "right":
            raise DataContractError("Tokenizer must explicitly use right truncation")
        if not _integer(getattr(tokenizer, "pad_token_id", None)):
            raise DataContractError("Tokenizer needs an explicitly configured pad_token_id")
        try:
            specials = _vector(tokenizer.all_special_ids, "all_special_ids")
        except AttributeError as exc:
            raise DataContractError("Tokenizer must expose all_special_ids") from exc
        self.pad_token_id = int(tokenizer.pad_token_id)
        if self.pad_token_id not in specials:
            raise DataContractError("Padding ID must be declared as a special token")
        self.tokenizer, self.tokenizer_id, self.revision = tokenizer, tokenizer_id, revision
        self.max_length, self.padding_side = int(max_length), padding_side
        self.special_ids = frozenset(specials)

    def _call(self, text, *, truncate):
        options = dict(add_special_tokens=True, truncation=truncate, padding=False,
                       return_attention_mask=True, return_special_tokens_mask=True)
        if truncate:
            options["max_length"] = self.max_length
        try:
            result = self.tokenizer(text, **options)
            ids = _vector(result["input_ids"], "input_ids")
            attention = _vector(result["attention_mask"], "attention_mask", binary=True)
            special = _vector(result["special_tokens_mask"], "special_tokens_mask", binary=True)
        except (KeyError, TypeError) as exc:
            raise DataContractError("Tokenizer must return flat IDs and both requested masks") from exc
        if len(ids) != len(attention) or len(ids) != len(special) or any(a != 1 for a in attention):
            raise DataContractError("Unpadded tokenizer output lengths/masks are inconsistent")
        if truncate and len(ids) > self.max_length:
            raise DataContractError("Tokenizer failed to enforce the requested total-token budget")
        if any(mark and token not in self.special_ids for token, mark in zip(ids, special)):
            raise DataContractError("Special-token mask refers to an undeclared special ID")
        # Literal registered special tokens in source text are not pooling content.
        special = tuple(int(mark or token in self.special_ids) for token, mark in zip(ids, special))
        return ids, attention, special

    @staticmethod
    def validate_user(sample):
        if not isinstance(sample, UserText):
            raise DataContractError("Expected a UserText view")
        validate_identifier(sample.user_key, "user_key")
        validate_identifier(sample.dataset_id, "dataset_id")
        validate_identifier(sample.policy, "text policy")
        if not isinstance(sample.text, str) or sample.partition not in (None, "train", "validation", "test"):
            raise DataContractError("Invalid user text or partition")
        if any(not isinstance(v, bool) for v in (sample.label_eligible, sample.text_eligible, sample.sample_eligible)):
            raise DataContractError("User eligibility flags must be booleans")
        try:
            sample.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DataContractError("User text contains malformed Unicode") from exc
        if sample.label_eligible:
            try:
                valid = isinstance(sample.bits, tuple) and bits_to_mbti(sample.bits) == bits_to_mbti(mbti_to_bits(sample.mbti))
            except (ValueError, TypeError):
                valid = False
            if not valid:
                raise DataContractError("User view label/partition contract mismatch")
        elif sample.bits is not None or sample.mbti is not None:
            raise DataContractError("Unlabelled user has inconsistent label fields")
        if sample.sample_eligible != bool(sample.text_eligible and sample.label_eligible):
            raise DataContractError("Supervised text eligibility is inconsistent")
        if sample.text_eligible and not has_content(sample.text):
            raise DataContractError("Content-eligible user text has no content")
        if not isinstance(sample.posts, tuple):
            raise DataContractError("User posts must be an ordered tuple")
        end = 0
        keys = set()
        for rank, post in enumerate(sample.posts):
            if (not isinstance(post, TextPost) or not _integer(post.user_text_start) or not _integer(post.user_text_end)
                    or post.user_text_start != end+(len(SEPARATOR) if rank else 0)
                    or not post.user_text_start < post.user_text_end <= len(sample.text)
                    or (rank and sample.text[end:post.user_text_start] != SEPARATOR)
                    or sample.text[post.user_text_start:post.user_text_end] != post.text):
                raise DataContractError("Post offsets do not match user text")
            validate_identifier(post.post_key, "post_key")
            if post.post_key in keys:
                raise DataContractError("Duplicate post identity within user")
            keys.add(post.post_key)
            if post.original_post_id is not None:
                if post.post_key != post_key(sample.dataset_id, sample.user_key, post.original_post_id):
                    raise DataContractError("Post identity does not belong to its source user")
            if post.source_file_id is not None and not _integer(post.source_file_id):
                raise DataContractError("Invalid post source file locator")
            end = post.user_text_end
        if sample.posts and end != len(sample.text):
            raise DataContractError("Post spans must cover the complete user text")

    def encode_user(self, sample):
        self.validate_user(sample)
        full = self._call(sample.text, truncate=False)
        original_count = len(full[0])
        truncated = original_count > self.max_length
        ids, attention, special = self._call(sample.text, truncate=True) if truncated else full
        content = tuple(int(a and not s) for a, s in zip(attention, special))
        token_eligible = bool(sample.text_eligible and any(content))
        return TokenInput(sample.dataset_id, sample.user_key, sample.partition, sample.bits,
            ids, attention, content, special, self.tokenizer_id, self.revision, self.max_length,
            original_count, truncated, token_eligible, token_eligible and sample.label_eligible,
            text_sha(sample.text), sample)

    def batch(self, inputs, *, pad_to_max_length=False, require_supervised=True):
        """Return int64 model arrays and provenance for every input row.

        One batch has a single partition and unique users. Invalid token inputs are
        rejected. Unlabelled inference uses labels=None.
        Specials and padding are excluded from content_mask; padding has attention=0.
        """
        inputs = tuple(inputs)
        if not inputs:
            raise DataContractError("Cannot collate an empty batch")
        if len({(row.dataset_id, row.user_key) for row in inputs}) != len(inputs):
            raise DataContractError("A user appears more than once in the batch")
        if len({row.partition for row in inputs}) != 1:
            raise DataContractError("Batch mixes data partitions")
        if len({row.labels is None for row in inputs}) != 1:
            raise DataContractError("Batch mixes labelled and unlabelled users")
        if require_supervised and inputs[0].partition not in ("train", "validation", "test"):
            raise DataContractError("Supervised batches require an explicit data partition")
        for row in inputs:
            self.validate_user(row.source)
            if ((row.tokenizer_id, row.tokenizer_revision, row.max_length) !=
                    (self.tokenizer_id, self.revision, self.max_length)):
                raise DataContractError("Batch mixes tokenizer contracts")
            if row.encoding_fingerprint != getattr(self, "encoding_fingerprint", None):
                raise DataContractError("Batch mixes tokenizer encoding configurations")
            if not row.token_eligible or require_supervised and not row.sample_eligible:
                raise DataContractError("Batch includes a token-ineligible or unsupervised user")
            if (not isinstance(row.token_eligible, bool) or not isinstance(row.sample_eligible, bool)
                    or row.sample_eligible != bool(row.token_eligible and row.source.label_eligible)):
                raise DataContractError("Token supervision eligibility is inconsistent")
            if not 0 < len(row.input_ids) <= self.max_length:
                raise DataContractError("Invalid encoded sequence length")
            if ((row.dataset_id, row.user_key, row.partition, row.labels, row.text_sha256) !=
                    (row.source.dataset_id, row.source.user_key, row.source.partition, row.source.bits, text_sha(row.source.text))):
                raise DataContractError("Token input lost its source user/label/text association")
            ids = _vector(row.input_ids, "input_ids")
            if getattr(self, "valid_token_ids", None) is not None and any(token not in self.valid_token_ids for token in ids):
                raise DataContractError("Encoded ID is outside the tokenizer vocabulary")
            masks = [_vector(mask, name, binary=True) for mask, name in (
                (row.attention_mask, "attention_mask"), (row.content_mask, "content_mask"),
                (row.special_tokens_mask, "special_tokens_mask"))]
            if any(len(mask) != len(ids) for mask in masks) or any(a != 1 for a in masks[0]):
                raise DataContractError("Encoded input contains invalid lengths or pre-padding")
            if any(c != int(a and not s) for a, c, s in zip(*masks)):
                raise DataContractError("Encoded content/special masks are inconsistent")
            if not row.source.text_eligible or not any(masks[1]):
                raise DataContractError("Batch input has no eligible content")
            if any(bool(s) != (token in self.special_ids) for token, s in zip(ids, masks[2])):
                raise DataContractError("Encoded special mask differs from registered token identities")
        width = self.max_length if pad_to_max_length else max(len(row.input_ids) for row in inputs)
        ids = np.full((len(inputs), width), self.pad_token_id, dtype=np.int64)
        attention = np.zeros_like(ids)
        content = np.zeros_like(ids, dtype=bool)
        special = np.ones_like(ids, dtype=bool)
        for index, row in enumerate(inputs):
            n = len(row.input_ids)
            start = 0 if self.padding_side == "right" else width-n
            ids[index, start:start+n] = row.input_ids
            attention[index, start:start+n] = row.attention_mask
            content[index, start:start+n] = row.content_mask
            special[index, start:start+n] = row.special_tokens_mask
        labels = (np.asarray([row.labels for row in inputs], dtype=np.int64)
                  if all(row.labels is not None for row in inputs) else None)
        return {"input_ids": ids, "attention_mask": attention, "content_mask": content,
                "special_tokens_mask": special, "labels": labels,
                "position_ids": np.maximum(np.cumsum(attention, axis=1)-1, 0),
                "user_keys": tuple(row.user_key for row in inputs),
                "partitions": tuple(row.partition for row in inputs),
                "dataset_ids": tuple(row.dataset_id for row in inputs), "provenance": inputs}


def batch_signature(batch):
    """Validate row/source alignment and fingerprint the actual bounded Batch.

    Used to bind separately prepared supervision to a model output. This is a
    runtime input identity.
    """
    if not isinstance(batch, Mapping):
        raise DataContractError("Expected a public Batch mapping")
    ids = batch.get("input_ids")
    if not isinstance(ids, np.ndarray) or ids.ndim != 2 or not ids.size or ids.dtype.kind not in "iu":
        raise DataContractError("Expected integer input_ids [B,T]")
    count = len(ids)
    for key in ("dataset_ids", "user_keys", "partitions", "provenance"):
        if not isinstance(batch.get(key), tuple) or len(batch[key]) != count:
            raise DataContractError(f"Missing or misaligned Batch {key}")
    if len(set(zip(batch["dataset_ids"], batch["user_keys"]))) != count:
        raise DataContractError("Duplicate Batch identity")
    if len(set(batch["partitions"])) != 1:
        raise DataContractError("Batch mixes partitions")
    labels = batch.get("labels")
    if labels is not None and (not isinstance(labels, np.ndarray) or labels.shape != (count, 4)
            or labels.dtype.kind not in "biu" or not np.isin(labels, (0, 1)).all()):
        raise DataContractError("Expected canonical binary labels [B,4] or None")
    digest = hashlib.sha256()
    for key in ("input_ids", "attention_mask", "content_mask", "special_tokens_mask", "position_ids"):
        array = batch.get(key)
        if (not isinstance(array, np.ndarray) or array.shape != ids.shape or array.dtype.kind not in "biu"
                or np.any(array < 0) or np.any(array > np.iinfo(np.int64).max)):
            raise DataContractError(f"Invalid Batch {key}")
        if key.endswith("mask") and not np.isin(array, (0, 1)).all():
            raise DataContractError(f"Nonbinary Batch {key}")
        digest.update(np.asarray(array, dtype="<i8").tobytes())
    for i, row in enumerate(batch["provenance"]):
        if not isinstance(row, TokenInput):
            raise DataContractError("Expected TokenInput provenance")
        TokenizerAdapter.validate_user(row.source)
        identity = (batch["dataset_ids"][i], batch["user_keys"][i], batch["partitions"][i])
        if (identity != (row.dataset_id, row.user_key, row.partition)
                or identity != (row.source.dataset_id, row.source.user_key, row.source.partition)
                or row.labels != row.source.bits
                or row.labels != (None if labels is None else tuple(labels[i]))
                or row.text_sha256 != text_sha(row.source.text)):
            raise DataContractError("Batch identity, labels or source text differs from provenance")
        active = batch["attention_mask"][i].astype(bool)
        for key in ("input_ids", "attention_mask", "content_mask", "special_tokens_mask"):
            if not np.array_equal(batch[key][i, active], getattr(row, key)):
                raise DataContractError("Actual Batch tokens/masks differ from provenance")
        if not row.token_eligible:
            raise DataContractError("Ineligible TokenInput")
        record = [identity, row.labels, row.text_sha256, row.encoding_fingerprint,
                  row.retained_characters, row.content_character_spans,
                  [(p.user_text_start, p.user_text_end) for p in row.source.posts]]
        digest.update(json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode())
    return digest.hexdigest()
