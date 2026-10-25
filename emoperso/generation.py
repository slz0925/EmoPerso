"""Reusable frozen CausalLM generation with configurable backends.

Custom backends implement ``generate(prompt, *, seed=None) -> GenerationResult``
and expose a JSON-serializable ``identity`` describing their model/settings.
"""
from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real
from pathlib import Path
import threading

from .deepseek_tokenizer import fingerprint, _resolve_tokenizer_revision
from .model_resources import local_model_fingerprint, check_weight_scales

_RNG_LOCK = threading.RLock()


class GenerationLoadError(RuntimeError):
    """The configured generation model or tokenizer could not be loaded."""


class GenerationExecutionError(RuntimeError):
    """Generation execution failed; the request has no cacheable outcome."""


def generate_checked(generator, prompt, *, seed=None):
    """Call a backend outside content validation, without exposing provider errors.

    Content rejections return GenerationResult. Backends must raise on execution
    failure, or return an error status (error, *_error, optionally :details).
    Invalid return objects and tensor-shape failures are protocol errors.
    """
    try:
        result = generator.generate(prompt, seed=seed)
    except GenerationLoadError:
        raise GenerationLoadError("Generation resources could not be loaded") from None
    except Exception:
        raise GenerationExecutionError("Generation backend execution failed") from None
    if not isinstance(result, GenerationResult) or not isinstance(result.status, str):
        raise GenerationExecutionError("Generation backend returned an invalid result")
    status = result.status.split(':', 1)[0].casefold()
    if (status == 'error' or status.endswith('_error') or
            status in ('oom', 'cuda_oom', 'invalid_generation_shape', 'invalid_generator_result')):
        raise GenerationExecutionError("Generation backend reported an execution failure")
    if not isinstance(result.text, str) or not isinstance(result.metadata, dict):
        raise GenerationExecutionError("Generation backend returned an invalid payload")
    return result


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_causal_loading_info(info, model):
    """Require all CausalLM tensors, allowing only declared native V3 MTP extras.

    Transformers' autoregressive V3 model does not construct the checkpoint's
    multi-token prediction modules. Its LM head is required, never an extra.
    """
    if any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Incomplete or incompatible CausalLM weights')
    extras = set(info.get('unexpected_keys', ()))
    if not extras:
        return
    from transformers import DeepseekV3ForCausalLM
    config = model.config
    if not isinstance(model, DeepseekV3ForCausalLM) or config.model_type != 'deepseek_v3':
        raise ValueError('Unexpected CausalLM weights')
    count = getattr(config, 'num_nextn_predict_layers', 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Invalid declared MTP layer count')
    # The MTP decoder has the ordinary final-layer tensors plus these six
    # components in the official V3 weight index. Quantized decoder buffers are
    # obtained from the actually constructed model rather than ignored by suffix.
    suffixes = set(model.model.layers[-1].state_dict()) | {
        'embed_tokens.weight', 'enorm.weight', 'hnorm.weight', 'eh_proj.weight',
        'shared_head.norm.weight', 'shared_head.head.weight'}
    allowed = {f'model.layers.{i}.{key}' for i in
               range(config.num_hidden_layers, config.num_hidden_layers + count) for key in suffixes}
    if extras - allowed:
        raise ValueError('Unexpected non-MTP CausalLM weights')


@dataclass(frozen=True)
class GenerationConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    revision: str | None = None
    local_files_only: bool = True
    dtype: str = "auto"
    device: str | None = None
    device_map: str | dict | None = None
    temperature: float = 1.
    top_p: float = .9
    max_new_tokens: int = 512
    max_prompt_tokens: int = 2048
    do_sample: bool = True
    seed: int = 42
    use_chat_template: bool = True

    def __post_init__(self):
        for name in ("model_name_or_path", "tokenizer_name_or_path", "revision", "device"):
            value = getattr(self, name)
            if (name == "model_name_or_path" or value is not None) and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string")
        if self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("Unsupported generation dtype")
        if self.device is not None and self.device_map is not None:
            raise ValueError("Choose device or device_map")
        if self.device_map is not None and not isinstance(self.device_map, (str, dict)):
            raise ValueError("device_map must be a string or mapping")
        for name in ("local_files_only", "do_sample", "use_chat_template"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("temperature", "top_p"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.top_p > 1:
            raise ValueError("top_p must be <=1")
        for name in ("max_prompt_tokens", "max_new_tokens"):
            _positive_int(getattr(self, name), name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")


@dataclass(frozen=True)
class GenerationResult:
    text: str = field(default="", repr=False)
    status: str = "ok"
    metadata: dict = field(default_factory=dict)

    @property
    def successful(self):
        return self.status == "ok" and isinstance(self.text, str) and bool(self.text.strip())


class HFCausalLMGenerator:
    """Lazy AutoModelForCausalLM + its own tokenizer; sequential bounded requests.

    Cache identity resolves metadata and tokenizer files without loading weights.
    Loads safetensors without executing repository code. Resource
    loading failures raise GenerationLoadError; execution failures raise
    GenerationExecutionError. Over-budget prompts and
    unfinished responses return unsuccessful GenerationResult values.
    """
    def __init__(self, config):
        if not isinstance(config, GenerationConfig):
            raise TypeError("Expected GenerationConfig")
        self.config = config
        self.model = self.tokenizer = None
        self._model_config = self._resource_identity = None

    @property
    def identity(self):
        self._resolve_resources()
        return {"backend": "hf-causal-lm-v2", **asdict(self.config), **self._resource_identity}

    @classmethod
    def from_components(cls, model, tokenizer, config):
        """Explicit injection of an executable compatible CausalLM/tokenizer."""
        obj = cls(config)
        obj._attach(model, tokenizer)
        obj._model_config = model.config
        obj._resource_identity = obj._describe_resources(model.config, tokenizer)
        return obj

    def _describe_resources(self, model_config, tokenizer):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        return {"model_revision": getattr(model_config, "_commit_hash", None),
            "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer_fingerprint": fingerprint({
                "class": type(tokenizer).__name__,
                "encoding": backend.to_str() if backend is not None else tokenizer.get_vocab(),
                "special_tokens": tokenizer.special_tokens_map,
                "chat_template": getattr(tokenizer, "chat_template", None),
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
                "model_max_length": tokenizer.model_max_length,
                "clean_up_tokenization_spaces": tokenizer.clean_up_tokenization_spaces})}

    def _resolve_resources(self):
        """Resolve metadata and tokenizer before binding caches; defer weights."""
        if self._resource_identity is not None:
            return
        try:
            from transformers import AutoConfig, AutoTokenizer
            c = self.config
            options = dict(revision=c.revision, local_files_only=c.local_files_only, trust_remote_code=False)
            model_config = AutoConfig.from_pretrained(c.model_name_or_path, **options)
            resolved = None if Path(c.model_name_or_path).is_dir() else getattr(model_config, "_commit_hash", None)
            if not Path(c.model_name_or_path).is_dir() and not resolved:
                raise ValueError("Hub model revision could not be resolved")
            source = c.tokenizer_name_or_path or c.model_name_or_path
            tokenizer_revision = (resolved if source == c.model_name_or_path else
                _resolve_tokenizer_revision(source, c.revision, c.local_files_only))
            if tokenizer_revision:
                options["revision"] = tokenizer_revision
            tokenizer = AutoTokenizer.from_pretrained(source, **options)
            identity = self._describe_resources(model_config, tokenizer)
            identity["model_revision"] = resolved
            identity["tokenizer_revision"] = tokenizer_revision or identity["tokenizer_revision"]
            if Path(c.model_name_or_path).is_dir():
                identity["local_model_fingerprint"] = local_model_fingerprint(c.model_name_or_path, include_generation=True)
            self._model_config, self.tokenizer = model_config, tokenizer
            self._resource_identity = identity
        except (OSError, RuntimeError, ValueError, TypeError, ImportError, AttributeError):
            raise GenerationLoadError("Cannot resolve generation model/tokenizer resources") from None

    def _attach(self, model, tokenizer):
        import torch
        if (not isinstance(model, torch.nn.Module) or not callable(getattr(model, "generate", None))
                or getattr(model.config, "is_encoder_decoder", False)):
            raise ValueError("Expected a decoder-only CausalLM with generate()")
        vocab = tokenizer.get_vocab()
        if not vocab or min(vocab.values()) < 0 or max(vocab.values()) >= model.get_input_embeddings().weight.shape[0]:
            raise ValueError("Generation tokenizer IDs exceed model embeddings")
        if model.get_output_embeddings() is None or max(vocab.values()) >= model.get_output_embeddings().weight.shape[0]:
            raise ValueError("Generation tokenizer IDs exceed LM head")
        eos = tokenizer.eos_token_id
        if eos is None or eos not in vocab.values():
            raise ValueError("Generation tokenizer requires a valid EOS")
        model.requires_grad_(False)
        for parameter in model.parameters():
            parameter.grad = None
        model.eval()
        self.model, self.tokenizer = model, tokenizer

    def load(self):
        if self.model is not None:
            return self
        try:
            return self._load_resources()
        except (OSError, RuntimeError, ValueError, TypeError, ImportError, AttributeError) as exc:
            raise GenerationLoadError(
                "Cannot load generation resources: check model/tokenizer paths, revision, "
                "complete safetensors weights, dependencies and device settings"
            ) from None

    def _load_resources(self):
        import torch
        from transformers import AutoModelForCausalLM
        c = self.config
        self._resolve_resources()
        local_identity = self._resource_identity.get('local_model_fingerprint')
        if local_identity is not None and local_model_fingerprint(c.model_name_or_path, include_generation=True) != local_identity:
            raise ValueError('Local generation model files changed after resource resolution')
        options = dict(config=self._model_config,
            revision=self._resource_identity["model_revision"] or c.revision,
            local_files_only=c.local_files_only, trust_remote_code=False)
        model, info = AutoModelForCausalLM.from_pretrained(c.model_name_or_path, **options,
            torch_dtype="auto" if c.dtype == "auto" else getattr(torch, c.dtype),
            device_map=c.device_map, use_safetensors=True, output_loading_info=True,
            ignore_mismatched_sizes=False)
        _check_causal_loading_info(info, model)
        check_weight_scales(model, c.model_name_or_path, options['revision'])
        if c.device is not None:
            model.to(c.device)
        self._attach(model, self.tokenizer)
        return self

    def generate(self, prompt, *, seed=None):
        if not isinstance(prompt, str) or not prompt.strip():
            return GenerationResult(status="empty_prompt")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63):
            raise ValueError("seed must be an integer in [0,2**63)")
        self.load()
        stage = "prompt encoding"
        try:
            import torch
            from .backbone import _embedding_device
            c, tokenizer, model = self.config, self.tokenizer, self.model
            if c.use_chat_template and getattr(tokenizer, "chat_template", None):
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
                add_special_tokens = False
            else:
                add_special_tokens = True
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens,
                                truncation=False, return_token_type_ids=False)
            n = encoded["input_ids"].shape[1]
            meta = {"prompt_tokens": n, "seed": c.seed if seed is None else seed}
            if not n or n > c.max_prompt_tokens:
                return GenerationResult(status="prompt_budget_exceeded", metadata=meta)
            context = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
            if isinstance(context, int) and n+c.max_new_tokens > context:
                return GenerationResult(status="context_budget_exceeded", metadata=meta)
            stage = "input placement"
            inputs = {k: v.to(_embedding_device(model)) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
            options = dict(max_new_tokens=c.max_new_tokens, do_sample=c.do_sample,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, forced_eos_token_id=None,
                return_dict_in_generate=True, output_scores=False)
            if c.do_sample:
                options.update(temperature=c.temperature, top_p=c.top_p, top_k=0)
            # Restore caller RNG state, including CUDA states; serialize seeded
            # calls in this backend rather than perturbing training randomness.
            stage = "model execution"
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with _RNG_LOCK, torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(meta["seed"])
                model.eval()
                sequence = model.generate(**inputs, **options).sequences
            if sequence.ndim != 2 or sequence.shape[0] != 1 or not torch.equal(sequence[0, :n], inputs["input_ids"][0]):
                raise GenerationExecutionError("Invalid generation tensor shape or prompt prefix")
            stage = "response decoding"
            new = sequence[0, n:].tolist()
            meta["generated_tokens"] = len(new)
            text = tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            if not new or not text:
                return GenerationResult(status="empty_output", metadata=meta)
            if len(new) >= c.max_new_tokens and new[-1] != tokenizer.eos_token_id:
                return GenerationResult(status="length_limit", metadata=meta)
            return GenerationResult(text, metadata=meta)
        except Exception:
            # Stage is controlled here; never include provider text, paths or a
            # chained exception that could expose the private prompt.
            raise GenerationExecutionError("Generation failed during " + stage) from None

    def generate_many(self, prompts):
        """Streaming results; no padded prompt matrix or unbounded request list."""
        for prompt in prompts:
            yield self.generate(prompt)
