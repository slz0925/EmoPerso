"""Safetensors model identity and FP8 scale validation."""
import hashlib
import json
from pathlib import Path


def _weight_files(source, revision=None):
    """Locate selected safetensors weight files."""
    local = Path(source).is_dir()
    def locate(name, required=True):
        if local:
            path = Path(source)/name
            if path.is_file():
                return path
        else:
            from transformers.utils.hub import cached_file
            path = cached_file(source, name, revision=revision, local_files_only=True,
                _raise_exceptions_for_missing_entries=False)
            if path is not None:
                return Path(path)
        if required:
            raise OSError("Required safetensors resource is missing")
        return None
    single = locate('model.safetensors', required=False)
    if single is not None:
        return (single,), None
    index = locate('model.safetensors.index.json')
    mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
    return tuple(locate(name) for name in sorted(set(mapping.values()))), index


def local_model_fingerprint(source, *, include_generation=False):
    """Return a content fingerprint for local model files, or None for Hub sources."""
    if not Path(source).is_dir():
        return None
    weights, index = _weight_files(source)
    files = [Path(source)/'config.json', *weights]
    if index is not None:
        files.append(index)
    defaults = Path(source)/'generation_config.json'
    if include_generation and defaults.is_file():
        files.append(defaults)
    identity = []
    for path in sorted(files):
        before = path.stat()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError("Local model files changed during fingerprinting")
        identity.append((path.relative_to(source).as_posix(), digest))
    return hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()


def check_weight_scales(model, source, revision=None):
    """Validate required FP8 scale tensors in the selected Safetensors files."""
    required = {name for name, _ in (*model.named_parameters(), *model.named_buffers())
                if name.endswith('.weight_scale_inv')}
    if not required:
        return
    from safetensors import safe_open
    files, _ = _weight_files(source, revision)
    keys = set()
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as stream:
            keys.update(stream.keys())
    missing = [name for name in required if name not in keys and
               ('model.'+name if not name.startswith('model.') else name[6:]) not in keys]
    if missing:
        raise ValueError("Missing required FP8 scale tensors")
