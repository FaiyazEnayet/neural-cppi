"""Fail-closed experiment provenance: manifests, hashes, sealed-evaluation guard.

A frozen experiment is a manifest plus the bytes it names. Nothing here
trusts a declaration: loading validates the schema, verifying recomputes
hashes, and any mismatch or missing file raises instead of proceeding.
No vendor API, no network, and no private data is required to use this
module: a fresh checkout either reproduces the recorded hashes or fails
with an explicit error naming what is missing or changed.
"""

import hashlib
import json
import os

# field name -> accepted type(s). Every required field must be present and
# of an accepted type; anything else fails validation instead of passing
# silently.
REQUIRED_FIELDS = {
    "data": list,
    "code_commit": str,
    "code_dirty": bool,
    "dependencies": (list, dict),
    "train_seed": int,
    "hyperparameters": dict,
    "selected_windows": list,
    "training_exclusions": list,
    "timestamp": str,
}

_SHA256_HEX_LENGTH = 64


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_strings(v)


def _is_home_anchored(path):
    home = os.path.expanduser("~")
    return (
        path == home
        or path.startswith(home + os.sep)
        or (home + os.sep) in path
        or "/home/" in path
        or "/Users/" in path
        or "C:\\Users\\" in path
        or "C:/Users/" in path
    )


def _validate_data_entries(manifest, errors):
    data = manifest.get("data")
    if not isinstance(data, list):
        return
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            errors.append(f"data[{i}] is not a mapping")
            continue
        path = entry.get("path")
        sha = entry.get("sha256")
        if not isinstance(path, str) or not path:
            errors.append(f"data[{i}] has no path string")
        elif os.path.isabs(path):
            errors.append(
                f"data[{i}] path must be relative, got absolute: {path}")
        if (not isinstance(sha, str) or len(sha) != _SHA256_HEX_LENGTH
                or any(c not in "0123456789abcdef" for c in sha.lower())):
            errors.append(f"data[{i}] sha256 is not a 64-hex digest")


def validate_manifest(manifest):
    """Return a list of schema errors; empty means valid. Never raises."""
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest is not a mapping"]
    for field, types in REQUIRED_FIELDS.items():
        if field not in manifest:
            errors.append(f"missing required field: {field}")
        elif not isinstance(manifest[field], types):
            errors.append(
                f"field {field} has wrong type: expected "
                f"{getattr(types, '__name__', types)}, "
                f"got {type(manifest[field]).__name__}")
    # bool is a subclass of int: a bool train_seed must not pass as int.
    if isinstance(manifest.get("train_seed"), bool):
        errors.append("field train_seed has wrong type: expected int, got bool")
    _validate_data_entries(manifest, errors)
    checkpoint = manifest.get("model_checkpoint")
    if checkpoint is not None:
        if not isinstance(checkpoint, dict):
            errors.append("model_checkpoint is not a mapping")
        else:
            path = checkpoint.get("path")
            sha = checkpoint.get("sha256")
            if not isinstance(path, str) or not path:
                errors.append("model_checkpoint has no path string")
            elif os.path.isabs(path):
                errors.append(
                    "model_checkpoint path must be relative, "
                    f"got absolute: {path}")
            if (not isinstance(sha, str)
                    or len(sha) != _SHA256_HEX_LENGTH):
                errors.append("model_checkpoint sha256 is not a 64-hex digest")
    for s in _walk_strings(manifest):
        if _is_home_anchored(s):
            errors.append(
                f"absolute user-specific path in manifest: {s}")
            break
    return errors


def assert_valid_manifest(manifest):
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(
            "invalid experiment manifest: " + "; ".join(errors))
    return manifest


def write_manifest(path, manifest):
    """Validate, then write. An invalid manifest is never written."""
    assert_valid_manifest(manifest)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return path


def load_manifest(path):
    """Load and validate. Missing files and bad JSON fail explicitly."""
    with open(path) as f:
        try:
            manifest = json.load(f)
        except ValueError as exc:
            raise ValueError(f"manifest {path} is not valid JSON: {exc}")
    return assert_valid_manifest(manifest)


def verify_data_hashes(manifest, base_dir):
    """Recompute every recorded data hash. Returns {path: sha256}.

    A missing file raises FileNotFoundError naming the file; a changed
    file raises ValueError naming the file and both hashes.
    """
    assert_valid_manifest(manifest)
    hashes = {}
    for entry in manifest["data"]:
        path = entry["path"]
        full = os.path.join(base_dir, path)
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"provenance data missing: {path} (expected sha256 "
                f"{entry['sha256']})")
        actual = sha256_file(full)
        if actual != entry["sha256"]:
            raise ValueError(
                f"provenance data hash mismatch: {path}: expected "
                f"{entry['sha256']}, got {actual}")
        hashes[path] = actual
    return hashes


def verify_checkpoint(path, expected_sha256):
    """Verify one checkpoint file against its recorded hash."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"provenance checkpoint missing: {path} (expected sha256 "
            f"{expected_sha256})")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"provenance checkpoint hash mismatch: {path}: expected "
            f"{expected_sha256}, got {actual}")
    return actual


def load_checkpoint_tensor_only(path):
    """Load a checkpoint with tensor-only deserialization.

    Refuses arbitrary pickle payloads: weights_only=True permits tensors
    and plain containers, nothing else. Import torch lazily so manifest
    work never requires it.
    """
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def sealed_eval_guard(manifest_path, base_dir, checkpoint_path=None,
                      expected_checkpoint_sha256=None):
    """Fail-closed gate for a sealed evaluation.

    Loads and validates the manifest, verifies every recorded data hash,
    and verifies the checkpoint hash when one is supplied or recorded.
    Returns a report dict. Any failure raises; nothing proceeds past it.
    """
    manifest = load_manifest(manifest_path)
    data_hashes = verify_data_hashes(manifest, base_dir)
    checkpoint_sha = None
    recorded = manifest.get("model_checkpoint") or {}
    if checkpoint_path is not None:
        expected = expected_checkpoint_sha256 or recorded.get("sha256")
        if expected is None:
            raise ValueError(
                "sealed evaluation needs a checkpoint hash: pass "
                "expected_checkpoint_sha256 or record model_checkpoint "
                "in the manifest")
        checkpoint_sha = verify_checkpoint(checkpoint_path, expected)
    elif recorded.get("sha256") and recorded.get("path"):
        checkpoint_sha = verify_checkpoint(
            os.path.join(base_dir, recorded["path"]), recorded["sha256"])
    return {"manifest": manifest, "data_hashes": data_hashes,
            "checkpoint_sha256": checkpoint_sha}
