"""DeepSeek-V3 plain inputs, with runtime compatibility and source-boundary checks."""

import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

from .data import DataContractError
from .text_data import find_explicit_type, text_sha
from .token_inputs import TokenInput, TokenizerAdapter, _integer, _vector, user_text

REPO_ID = "deepseek-ai/DeepSeek-V3"
TOKEN_POLICY = "deepseek-v3-plain-prefix-v2"


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer resource selection and input encoding settings.

    tokenizer_name_or_path may name an official/mirrored DeepSeek-V3 tokenizer
    repository or a local directory. None uses model_name_or_path. revision=None
    follows the selected source's default; pin a commit for reproducible studies.
    """
    model_name_or_path: str = REPO_ID
    revision: str | None = None
    tokenizer_name_or_path: str | None = None
    local_files_only: bool = False
    max_length: int = 2048
    padding_side: str = "right"

    def __post_init__(self):
        for value in (self.model_name_or_path, self.tokenizer_name_or_path):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Model/tokenizer sources must be nonempty strings")
        if self.model_name_or_path is None:
            raise ValueError("A DeepSeek-V3 model source is required")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("revision must be None or a nonempty string")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be boolean")
        if not _integer(self.max_length) or not 1 <= self.max_length <= 2048:
            raise ValueError("max_length must be in [1,2048], including BOS")
        if self.padding_side not in ("left", "right"):
            raise ValueError("padding_side must be left or right")


def load_tokenizer(config=None):
    """Public loader. Only tokenizer files are loaded; remote code is forbidden."""
    return DeepSeekTokenizerAdapter(config=TokenizerConfig() if config is None else config)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _signature(tokenizer):
    return {"class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id, "unk_token_id": tokenizer.unk_token_id,
            "add_bos_token": tokenizer.add_bos_token, "add_eos_token": tokenizer.add_eos_token,
            "vocab_size": tokenizer.vocab_size, "total_tokenizer_entries": len(tokenizer),
            "all_special_ids": list(tokenizer.all_special_ids),
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side}


class DeepSeekTokenizerAdapter(TokenizerAdapter):
    """Actual joined-text BPE offsets, right prefix budget and safe final boundary.

    Full encoding is performed once, including official BOS. Prefix IDs exactly
    follow official right truncation, except retreat needed to retain a complete
    Unicode character or avoid exposing an explicit MBTI type at the cut. No new
    text is substituted, no chat template is used, and no post is repacked.
    """

    def __init__(self, *, config=None):
        config = TokenizerConfig() if config is None else config
        if not isinstance(config, TokenizerConfig):
            raise TypeError("Expected TokenizerConfig")
        try:
            self._init_public(config)
        except DataContractError:
            raise
        except (ImportError, AttributeError, KeyError, TypeError) as exc:
            raise DataContractError("Incompatible tokenizer API: fast BPE, registered specials, offsets and reversible decoding are required") from exc

    def _init_public(self, config):
        from transformers import AutoTokenizer
        source = config.tokenizer_name_or_path or config.model_name_or_path
        backend = AutoTokenizer.from_pretrained(source, revision=config.revision,
            use_fast=True, local_files_only=config.local_files_only, trust_remote_code=False)
        if (not backend.is_fast
                or json.loads(backend.backend_tokenizer.to_str())["model"]["type"] != "BPE"):
            raise DataContractError("Expected a DeepSeek-V3 compatible fast BPE tokenizer")
        if not backend.add_bos_token or backend.add_eos_token:
            raise DataContractError("Plain DeepSeek-V3 inputs require one automatic BOS and no automatic EOS")
        valid_ids = frozenset(backend.get_vocab().values())
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(backend, name, None)
            if not _integer(value) or value not in valid_ids or value not in backend.all_special_ids:
                raise DataContractError(f"Invalid or unregistered {name}")
        if backend.bos_token_id in (backend.eos_token_id, backend.pad_token_id):
            raise DataContractError("BOS must be distinct from EOS and PAD")
        if _integer(backend.model_max_length) and backend.model_max_length < config.max_length:
            raise DataContractError("Configured budget exceeds tokenizer model_max_length")
        resolved = backend.init_kwargs.get("_commit_hash")
        super().__init__(backend, tokenizer_id=source, revision=resolved or config.revision or "unversioned-local",
                         max_length=config.max_length, padding_side=config.padding_side)
        self.special_ids |= frozenset(k for k, v in backend.added_tokens_decoder.items() if v.special)
        self.valid_token_ids, self.bos_token_id = valid_ids, backend.bos_token_id
        self.config = config
        self._runtime = _signature(backend)
        self._adapter_settings = (self.tokenizer_id, self.revision, self.max_length, self.padding_side)
        self._public_special_ids = self.special_ids
        self.encoding_fingerprint = fingerprint({"backend": text_sha(backend.backend_tokenizer.to_str()),
            "runtime": self._runtime, "max_length": self.max_length, "padding_side": self.padding_side,
            "policy": TOKEN_POLICY})
        deps = {}
        for name in ("transformers", "tokenizers", "huggingface-hub", "numpy"):
            try:
                deps[name] = version(name)
            except PackageNotFoundError:
                deps[name] = "unavailable"
        self.compatibility = {"config": asdict(config), "resolved_revision": resolved,
            "runtime": self._runtime, "backend_special_ids": sorted(self.special_ids),
            "maximum_token_id": max(valid_ids), "dependencies": deps,
            "encoding_fingerprint": self.encoding_fingerprint,
            "chat_template_used": False, "trust_remote_code": False, "backbone_verified": False}
        # Probe the actual postprocessor, IDs, offsets and reversible decoding,
        # rather than trusting configuration flags or package version strings.
        for text in ("", "Plain input 🙂 中文!"):
            self.encode_user(user_text("compatibility-probe", text))

    def check_model_compatibility(self, model_config):
        """Validate Backbone configuration against the tokenizer's vocabulary and context.

        A missing model PAD is allowed because attention_mask defines padding.
        """
        required = ("model_type", "vocab_size", "bos_token_id", "eos_token_id", "max_position_embeddings")
        if not isinstance(model_config, dict) or any(k not in model_config for k in required):
            raise DataContractError("Incomplete Backbone compatibility metadata")
        if (model_config["model_type"] != "deepseek_v3"
                or not _integer(model_config["vocab_size"])
                or model_config["vocab_size"] <= max(self.valid_token_ids)
                or not _integer(model_config["bos_token_id"]) or not _integer(model_config["eos_token_id"])
                or model_config["bos_token_id"] != self.tokenizer.bos_token_id
                or model_config["eos_token_id"] != self.tokenizer.eos_token_id
                or model_config.get("pad_token_id") is not None and (
                    not _integer(model_config["pad_token_id"]) or model_config["pad_token_id"] != self.pad_token_id)
                or not _integer(model_config["max_position_embeddings"])
                or model_config["max_position_embeddings"] < self.max_length):
            raise DataContractError("Tokenizer and declared DeepSeek-V3 Backbone configuration are incompatible")
        return {"metadata_compatible": True, "planned_pad_token_id": self.pad_token_id,
                "backbone_forward_verified": False}

    def _validate_runtime(self):
        if ((self.tokenizer_id, self.revision, self.max_length, self.padding_side) != self._adapter_settings
                or self.special_ids != self._public_special_ids or _signature(self.tokenizer) != self._runtime
                or self.pad_token_id != self.tokenizer.pad_token_id or self.bos_token_id != self.tokenizer.bos_token_id):
            raise DataContractError("Tokenizer runtime settings changed after loading")

    def encode_user(self, sample):
        self.validate_user(sample)
        self._validate_runtime()
        try:
            result = self.tokenizer(sample.text, add_special_tokens=True, padding=False, truncation=False,
                                    return_attention_mask=True, return_special_tokens_mask=True,
                                    return_offsets_mapping=True)
            ids = _vector(result["input_ids"], "input_ids")
            attention = _vector(result["attention_mask"], "attention_mask", binary=True)
            auto = _vector(result["special_tokens_mask"], "special_tokens_mask", binary=True)
            offsets = tuple(tuple(pair) for pair in result["offset_mapping"])
        except DataContractError:
            raise
        except (AttributeError, KeyError, TypeError, NotImplementedError) as exc:
            raise DataContractError("Tokenizer must return flat IDs, masks and character offsets") from exc
        if not (len(ids) == len(attention) == len(auto) == len(offsets)) or any(a != 1 for a in attention):
            raise DataContractError("Official unpadded encoding/masks are inconsistent")
        if not ids or ids[0] != self.bos_token_id or offsets[0] != (0, 0) or auto[0] != 1 or sum(auto) != 1:
            raise DataContractError("Official BOS/template behavior changed")
        if any(token not in self.valid_token_ids for token in ids):
            raise DataContractError("Tokenizer returned an ID outside its vocabulary")
        if any(len(pair) != 2 or not 0 <= pair[0] <= pair[1] <= len(sample.text) for pair in offsets):
            raise DataContractError("Invalid official character offsets")
        n = min(len(ids), self.max_length)
        unicode_removed = label_removed = 0
        # Several byte-level pieces may share a character offset. Never retain
        # only some bytes of a character at the right edge.
        while 1 < n < len(ids) and offsets[n][0] < offsets[n-1][1]:
            n -= 1
            unicode_removed += 1

        def prefix(count):
            return self.tokenizer.decode([ids[i] for i in range(count) if not auto[i]],
                                         skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded = prefix(n)
        original_has_label = bool(find_explicit_type(sample.text, sample.posts))
        while not original_has_label:
            match = find_explicit_type(decoded, sample.posts)
            if match is None:
                break
            previous = n
            while n > 1 and offsets[n-1][1] > match[0]:
                n -= 1
            label_removed += previous-n
            decoded = prefix(n)
        char_end = max((end for start, end in offsets[:n]), default=0)
        if decoded != sample.text[:char_end]:
            raise DataContractError("Tokenizer decoding is not an exact source-text prefix")
        special = tuple(int(token in self.special_ids) for token in ids[:n])
        content = tuple(1-s for s in special)
        reasons = []
        if not sample.text_eligible:
            reasons.append("source_text_ineligible")
        if original_has_label:
            reasons.append("explicit_type_in_source_text")
        if not any(content):
            reasons.append("no_content_within_token_budget")
        eligible = not reasons
        starts = [offsets[i][0] for i in range(n) if content[i]]
        ends = [offsets[i][1] for i in range(n) if content[i]]
        # Actual joined-encoding overlaps; a cross-boundary token can count for
        # both posts, so these counts must never be summed as a sequence length.
        coverage = tuple(max(0, bisect_left(starts, p.user_text_end)-bisect_right(ends, p.user_text_start))
                         for p in sample.posts)
        spans = []
        for start, end in zip(starts, ends):
            if start == end:
                continue
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], end))
            else:
                spans.append((start, end))
        return TokenInput(sample.dataset_id, sample.user_key, sample.partition, sample.bits,
            ids[:n], attention[:n], content, special, self.tokenizer_id, self.revision, self.max_length,
            len(ids), len(ids)>n, eligible, eligible and sample.label_eligible, text_sha(sample.text), sample,
            "actual joined-token character offsets; counts are per-post overlaps, not additive",
            char_end, sum(token not in self.special_ids for token in ids), coverage,
            unicode_removed, label_removed, tuple(reasons), getattr(self, "encoding_fingerprint", None),
            content_character_spans=tuple(spans))

    def batch(self, inputs, **kwargs):
        self._validate_runtime()
        return super().batch(inputs, **kwargs)
