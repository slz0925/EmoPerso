"""Frozen native DeepSeek-V3 features, shared projection and attention pooling."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from numbers import Integral

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("Model features require the optional emoperso[backbone] dependencies") from exc

from .data import DataContractError
from .deepseek_tokenizer import REPO_ID, DeepSeekTokenizerAdapter
from .token_inputs import validate_identifier


class BackboneLoadError(RuntimeError):
    """The selected native model/configuration/weights could not be loaded safely."""


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class BackboneConfig:
    """Load native DeepSeek-V3 weights, using local resources by default.

    The backend uses the Transformers implementation and rejects remote code,
    including repository code referenced by auto_map.
    device places an unsharded model; device_map/max_memory/offload_folder are
    delegated to Transformers/Accelerate instead of moving a dispatched model.
    """
    model_name_or_path: str = REPO_ID
    revision: str | None = None
    local_files_only: bool = True
    dtype: str = "auto"
    device: str | None = None
    device_map: str | dict | None = None
    max_memory: dict | None = None
    offload_folder: str | None = None
    attn_implementation: str = "eager"
    trust_remote_code: bool = False

    def __post_init__(self):
        for name in ("model_name_or_path", "revision", "device", "offload_folder"):
            value = getattr(self, name)
            if (name == "model_name_or_path" or value is not None) and (
                    not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.local_files_only, bool) or not isinstance(self.trust_remote_code, bool):
            raise ValueError("Loading policy flags must be boolean")
        if self.trust_remote_code:
            raise ValueError("The native DeepSeek-V3 backend does not execute remote code; trust_remote_code=True is unsupported")
        if self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")
        if self.device is not None and self.device_map is not None:
            raise ValueError("Choose device or device_map, not both")
        if self.device_map is not None and not isinstance(self.device_map, (str, dict)):
            raise ValueError("device_map must be a string or mapping")
        if self.max_memory is not None and not isinstance(self.max_memory, dict):
            raise ValueError("max_memory must be a mapping")
        if self.device_map is None and (self.max_memory is not None or self.offload_folder is not None):
            raise ValueError("max_memory/offload_folder require device_map")
        if self.attn_implementation not in ("eager", "sdpa", "flash_attention_2"):
            raise ValueError("Unsupported attention implementation")


def _model_dimensions(config):
    if getattr(config, "model_type", None) != "deepseek_v3":
        raise DataContractError("Only deepseek_v3 Backbone configurations are supported")
    return (_positive(getattr(config, "hidden_size", None), "hidden_size"),
            _positive(getattr(config, "vocab_size", None), "vocab_size"))


def _check_loading_info(info, config):
    """Validate encoder tensors and permitted CausalLM checkpoint extras.

    CausalLM checkpoints have an unused LM head and may have MTP layers after the
    configured decoder layers. Those are not the encoder's final hidden states.
    """
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise BackboneLoadError("Missing, mismatched or invalid Backbone weights")
    extras = info.get("unexpected_keys", [])
    causal = "DeepseekV3ForCausalLM" in (getattr(config, "architectures", None) or [])
    start = config.num_hidden_layers
    count = getattr(config, "num_nextn_predict_layers", 0)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise BackboneLoadError("Invalid declared MTP layer count")
    allowed_prefixes = tuple(f"{prefix}layers.{i}." for i in range(start, start+count)
                             for prefix in ("", "model."))
    rejected = [key for key in extras if not (causal and (
        key == "lm_head.weight" or allowed_prefixes and key.startswith(allowed_prefixes)))]
    if rejected:
        raise BackboneLoadError(f"Unexpected non-head Backbone weights ({len(rejected)} keys); incompatible checkpoint")


def load_backbone(config=None, *, tokenizer):
    """Load pretrained safetensors using the native bare DeepseekV3Model.

    AutoConfig selects metadata only, with remote code disabled. Loading the bare
    model from a CausalLM checkpoint avoids constructing vocab logits/LM-head
    activations and preserves the checkpoint's native FP8 quantization settings.
    """
    config = BackboneConfig() if config is None else config
    if not isinstance(config, BackboneConfig):
        raise TypeError("Expected BackboneConfig")
    if not isinstance(tokenizer, DeepSeekTokenizerAdapter):
        raise TypeError("Pretrained weight loading requires DeepSeekTokenizerAdapter")
    try:
        from packaging.version import Version
        if not Version("4.51.3") <= Version(version("transformers")) < Version("5"):
            raise BackboneLoadError("Native Backbone support requires transformers>=4.51.3,<5; tokenizer-only support is separate")
        from transformers import AutoConfig, DeepseekV3Model
        model_config = AutoConfig.from_pretrained(config.model_name_or_path, revision=config.revision,
            local_files_only=config.local_files_only, trust_remote_code=False)
        _model_dimensions(model_config)
        tokenizer.check_model_compatibility(model_config.to_dict())
        quantization = getattr(model_config, "quantization_config", {})
        if isinstance(quantization, dict) and quantization.get("quant_method") == "fp8" and not torch.cuda.is_available():
            raise BackboneLoadError("This native FP8 path needs a supported CUDA GPU; supply a properly converted BF16 checkpoint for CPU. Changing dtype or deleting quantization metadata does not convert FP8 weights")
        options = dict(config=model_config,
            revision=getattr(model_config, "_commit_hash", None) or config.revision,
            local_files_only=config.local_files_only, trust_remote_code=False,
            torch_dtype="auto" if config.dtype == "auto" else getattr(torch, config.dtype),
            attn_implementation=config.attn_implementation, use_safetensors=True,
            output_loading_info=True, ignore_mismatched_sizes=False)
        for key in ("device_map", "max_memory", "offload_folder"):
            if getattr(config, key) is not None:
                options[key] = getattr(config, key)
        model, info = DeepseekV3Model.from_pretrained(config.model_name_or_path, **options)
        _check_loading_info(info, model_config)
        if config.device is not None:
            model.to(config.device)
        result = FrozenBackbone(model, tokenizer=tokenizer)
        result.loading_info = info
        result.loading_config = config
        return result
    except (BackboneLoadError, DataContractError):
        raise
    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
        raise BackboneLoadError("Cannot load native DeepSeek-V3 weights: check local files/revision, optional dependencies, dtype/quantization and device resources") from exc


def _hook_device(hook):
    if hook is None:
        return None
    value = getattr(hook, "execution_device", None)
    if value is not None:
        return torch.device(value)
    for child in getattr(hook, "hooks", ()):
        value = _hook_device(child)
        if value is not None:
            return value
    return None


def _embedding_device(model):
    embedding = model.get_input_embeddings()
    # Accelerate may keep offloaded weights on meta/CPU until its execution hook.
    device = _hook_device(getattr(embedding, "_hf_hook", None))
    if device is None:
        device = _hook_device(getattr(model, "_hf_hook", None))
    if device is None:
        device = embedding.weight.device
    if device.type == "meta":
        raise DataContractError("Cannot determine the dispatched embedding execution device; an executable model is required")
    return device


def _numpy_matrix(batch, key, shape=None, binary=False):
    value = batch.get(key)
    if (not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype.kind not in ("biu" if binary else "iu")
            or value.size == 0 or shape is not None and value.shape != shape):
        raise DataContractError(f"{key} must be a nonempty integer/binary NumPy [B,T] array with matching shape")
    if binary and not np.all((value == 0) | (value == 1)):
        raise DataContractError(f"{key} must be binary")
    if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
        raise DataContractError(f"{key} exceeds int64 range")
    return value


class FrozenBackbone(nn.Module):
    """An explicitly supplied compatible bare model, always frozen and in eval.

    Accepts caller-owned models compatible with the bound tokenizer. Feature
    extraction uses no_grad so outputs can feed the trainable projection.
    """
    def __init__(self, model, *, tokenizer):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("Backbone must be a torch.nn.Module")
        self.hidden_size, self.vocab_size = _model_dimensions(model.config)
        tokenizer.check_model_compatibility(model.config.to_dict())
        embedding = model.get_input_embeddings()
        if embedding.weight.shape != (self.vocab_size, self.hidden_size):
            raise DataContractError("Backbone embedding and configuration dimensions disagree")
        self.model, self.tokenizer = model, tokenizer
        self.model.requires_grad_(False)
        # Existing grads on an injected module must not look like new gradients.
        for parameter in self.model.parameters():
            parameter.grad = None
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def _inputs(self, batch):
        from .token_inputs import batch_signature
        batch_signature(batch)
        if not isinstance(batch, Mapping):
            raise DataContractError("Expected the public NumPy Batch mapping")
        if hasattr(self.tokenizer, "_validate_runtime"):
            self.tokenizer._validate_runtime()
        ids = _numpy_matrix(batch, "input_ids")
        attention = _numpy_matrix(batch, "attention_mask", ids.shape, True)
        positions = _numpy_matrix(batch, "position_ids", ids.shape)
        content = _numpy_matrix(batch, "content_mask", ids.shape, True).astype(bool)
        special = _numpy_matrix(batch, "special_tokens_mask", ids.shape, True).astype(bool)
        if ids.shape[1] > min(self.tokenizer.max_length, self.model.config.max_position_embeddings):
            raise DataContractError("Batch exceeds model/tokenizer context budget")
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise DataContractError("Token IDs outside the loaded Backbone vocabulary")
        valid = getattr(self.tokenizer, "valid_token_ids", None)
        if valid is not None and any(int(i) not in valid for i in np.unique(ids)):
            raise DataContractError("Token ID not registered in the bound tokenizer")
        registered = np.isin(ids, tuple(self.tokenizer.special_ids))
        if not np.array_equal(special, registered) or not np.array_equal(content, attention.astype(bool) & ~special):
            raise DataContractError("Content/special masks disagree with attention or registered tokens")
        if not np.all(content.any(axis=1)):
            raise DataContractError("Every user must have at least one valid content token")
        expected_positions = np.maximum(np.cumsum(attention, axis=1)-1, 0)
        if not np.array_equal(positions, expected_positions):
            raise DataContractError("position_ids must count nonpadding tokens from zero")
        for row, marks in zip(ids, attention):
            n = int(marks.sum())
            start = 0 if self.tokenizer.padding_side == "right" else len(row)-n
            if not np.all(marks[start:start+n] == 1) or np.any(row[marks == 0] != self.tokenizer.pad_token_id):
                raise DataContractError("Padding must match the tokenizer side, PAD ID and contiguous attention mask")
        metadata = {}
        for key in ("dataset_ids", "user_keys", "partitions", "provenance"):
            value = batch.get(key)
            if not isinstance(value, tuple) or len(value) != ids.shape[0]:
                raise DataContractError(f"{key} must preserve one metadata item per user")
            metadata[key] = value
        for dataset, user in zip(metadata["dataset_ids"], metadata["user_keys"]):
            validate_identifier(dataset, "dataset_id")
            validate_identifier(user, "user_key")
        if len(set(metadata["partitions"])) != 1 or metadata["partitions"][0] not in (None, "train", "validation", "test"):
            raise DataContractError("Batch mixes or misdeclares partitions")
        if len(set(zip(metadata["dataset_ids"], metadata["user_keys"]))) != ids.shape[0]:
            raise DataContractError("Duplicate user in Batch")
        labels = batch.get("labels")
        if labels is not None and (not isinstance(labels, np.ndarray) or labels.shape != (ids.shape[0], 4)
                                   or labels.dtype.kind not in "biu" or not np.all((labels == 0) | (labels == 1))):
            raise DataContractError("labels must be None or canonical binary NumPy [B,4]")
        metadata["labels"] = labels
        for i, row in enumerate(metadata["provenance"]):
            if ((getattr(row, "dataset_id", None), getattr(row, "user_key", None), getattr(row, "partition", None)) !=
                    (metadata["dataset_ids"][i], metadata["user_keys"][i], metadata["partitions"][i])):
                raise DataContractError("Batch metadata differs from token provenance")
            if getattr(row, "labels", None) != (tuple(labels[i]) if labels is not None else None):
                raise DataContractError("Batch labels differ from token provenance")
            if getattr(row, "encoding_fingerprint", None) != getattr(self.tokenizer, "encoding_fingerprint", None):
                raise DataContractError("Batch was encoded with a different tokenizer configuration")
        device = _embedding_device(self.model)
        # Copy bounded arrays: never mutate caller arrays, including read-only ones.
        model_inputs = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in
                        (("input_ids", ids), ("attention_mask", attention), ("position_ids", positions))}
        return model_inputs, torch.tensor(content, dtype=torch.bool, device=device), metadata

    def forward(self, batch):
        inputs, mask, metadata = self._inputs(batch)
        self.model.eval()
        with torch.no_grad():
            result = self.model(**inputs, use_cache=False, return_dict=True,
                                output_hidden_states=False, output_attentions=False)
        hidden = getattr(result, "last_hidden_state", None)
        if (not isinstance(hidden, torch.Tensor) or not hidden.is_floating_point() or
                hidden.shape != (*inputs["input_ids"].shape, self.hidden_size)):
            raise DataContractError("Expected Backbone last_hidden_state [B,T,config.hidden_size]")
        return hidden.detach(), mask.to(hidden.device), metadata


class AttentionPooling(nn.Module):
    """Additive attention: v^T tanh(Wh+b), masked softmax, weighted sum.

    Noncontent values are zeroed BEFORE scoring and reduction. Accumulation is
    float32 for float16/bfloat16/float32 inputs, float64 for float64 inputs/params;
    this also applies under external autocast. Returned weights/pooled use that
    accumulation dtype. Invalid rows/nonfinite valid content fail explicitly.
    """
    def __init__(self, hidden_size, *, device=None, dtype=None):
        super().__init__()
        self.hidden_size = _positive(hidden_size, "hidden_size")
        self.transform = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.score_vector = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        nn.init.normal_(self.score_vector, std=hidden_size**-0.5)

    def forward(self, hidden, content_mask, *, return_attention_weights=False):
        if (not isinstance(hidden, torch.Tensor) or not hidden.is_floating_point() or hidden.ndim != 3
                or hidden.shape[-1] != self.hidden_size or not hidden.shape[0] or not hidden.shape[1]):
            raise DataContractError("Pooling requires floating [B,T,shared_hidden_size]")
        if (not isinstance(content_mask, torch.Tensor) or content_mask.dtype != torch.bool
                or content_mask.shape != hidden.shape[:2] or content_mask.device != hidden.device
                or self.transform.weight.device != hidden.device):
            raise DataContractError("Pooling mask must be bool [B,T] on the feature/parameter device")
        if not content_mask.any(dim=1).all():
            raise DataContractError("Cannot pool an all-masked user")
        dtype = torch.float64 if torch.float64 in (hidden.dtype, self.transform.weight.dtype) else torch.float32
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            safe = hidden.masked_fill(~content_mask.unsqueeze(-1), 0).to(dtype)
            if not torch.isfinite(safe).all():
                raise DataContractError("Nonfinite valid pooling content")
            transformed = torch.tanh(F.linear(safe, self.transform.weight.to(dtype), self.transform.bias.to(dtype)))
            scores = F.linear(transformed, self.score_vector.to(dtype).unsqueeze(0)).squeeze(-1)
            if not torch.isfinite(scores).all():
                raise DataContractError("Nonfinite attention scores")
            weights = torch.softmax(scores.masked_fill(~content_mask, -torch.inf), dim=1)
            weights = weights.masked_fill(~content_mask, 0)
            pooled = torch.sum(weights.unsqueeze(-1)*safe, dim=1)
            if not torch.isfinite(pooled).all():
                raise DataContractError("Nonfinite pooled features")
        return pooled, weights if return_attention_weights else None


@dataclass(frozen=True)
class EncoderOutput:
    token_features: torch.Tensor
    pooled_features: torch.Tensor
    content_mask: torch.Tensor
    attention_weights: torch.Tensor | None
    metadata: dict = field(repr=False)


class EmoPersoEncoder(nn.Module):
    """NumPy batch -> frozen native states -> trainable projection -> pooling.

    Project the model's native hidden width to shared_hidden_size (default 4096).
    Masked positions are zero; source metadata remains on CPU.
    """
    def __init__(self, backbone, *, shared_hidden_size=4096, shared_device=None, shared_dtype=torch.float32):
        super().__init__()
        if not isinstance(backbone, FrozenBackbone):
            raise TypeError("Expected FrozenBackbone")
        shared_hidden_size = _positive(shared_hidden_size, "shared_hidden_size")
        if shared_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("Expected a floating shared_dtype")
        device = _embedding_device(backbone.model) if shared_device is None else torch.device(shared_device)
        self.backbone = backbone
        self.projection = nn.Linear(backbone.hidden_size, shared_hidden_size, device=device, dtype=shared_dtype)
        self.pooling = AttentionPooling(shared_hidden_size, device=device, dtype=shared_dtype)

    @classmethod
    def from_pretrained(cls, config=None, *, tokenizer, **shared_options):
        return cls(load_backbone(config, tokenizer=tokenizer), **shared_options)

    def forward(self, batch, *, return_attention_weights=False):
        native, mask, metadata = self.backbone(batch)
        target = self.projection.weight
        mask = mask.to(target.device)
        # Sanitize before projection, not merely after a NaN/Inf multiply by zero.
        native = native.masked_fill(~mask.to(native.device).unsqueeze(-1), 0)
        if not torch.isfinite(native).all():
            raise DataContractError("Nonfinite valid Backbone hidden states")
        tokens = self.projection(native.to(device=target.device, dtype=target.dtype))
        tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0)
        pooled, weights = self.pooling(tokens, mask, return_attention_weights=return_attention_weights)
        return EncoderOutput(tokens, pooled, mask, weights, metadata)
