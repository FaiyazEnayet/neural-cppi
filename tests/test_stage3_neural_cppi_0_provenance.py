"""Stage 3 neural-cppi P0-C: data and freeze provenance, fail-closed.

A frozen experiment must be verifiable from a fresh checkout: the manifest
schema validates, every named data file hashes to its recorded value, a
mismatched checkpoint aborts sealed evaluation, and no absolute
user-specific path is allowed. Nothing here needs the vendor API.
"""

import importlib.util
import json
import os
import sys

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

from dhl.eval.provenance import (  # noqa: E402
    assert_valid_manifest,
    load_manifest,
    sealed_eval_guard,
    sha256_file,
    verify_data_hashes,
    write_manifest,
)


def _manifest(tmp_path, data_files):
    return {
        "data": [{"path": name, "sha256": sha256_file(str(path))}
                 for name, path in data_files],
        "code_commit": "0" * 40,
        "code_dirty": False,
        "dependencies": ["torch==2.14.1+cpu"],
        "train_seed": 7,
        "hyperparameters": {"lr": 0.001, "epochs": 10},
        "selected_windows": ["2020-03-01:2020-03-31"],
        "training_exclusions": ["2020-02-01:2020-02-29"],
        "timestamp": "2026-10-28T00:00:00Z",
    }


def _two_files(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("date,close\n2020-01-01,100\n")
    b.write_text("date,close\n2020-01-02,101\n")
    return [("a.csv", a), ("b.csv", b)]


def test_stage3_neural_cppi_0_valid_manifest_round_trips(tmp_path):
    manifest = _manifest(tmp_path, _two_files(tmp_path))
    assert_valid_manifest(manifest)
    path = tmp_path / "manifest.json"
    write_manifest(str(path), manifest)
    assert load_manifest(str(path)) == manifest


def test_stage3_neural_cppi_0_missing_field_rejected(tmp_path):
    manifest = _manifest(tmp_path, _two_files(tmp_path))
    del manifest["train_seed"]
    with pytest.raises(ValueError, match="train_seed"):
        assert_valid_manifest(manifest)


def test_stage3_neural_cppi_0_wrong_type_rejected(tmp_path):
    manifest = _manifest(tmp_path, _two_files(tmp_path))
    manifest["train_seed"] = "7"
    with pytest.raises(ValueError, match="train_seed"):
        assert_valid_manifest(manifest)


def test_stage3_neural_cppi_0_write_refuses_invalid_manifest(tmp_path):
    manifest = _manifest(tmp_path, _two_files(tmp_path))
    del manifest["code_commit"]
    with pytest.raises(ValueError, match="code_commit"):
        write_manifest(str(tmp_path / "bad.json"), manifest)
    assert not (tmp_path / "bad.json").exists()


def test_stage3_neural_cppi_0_data_hashes_verified(tmp_path):
    files = _two_files(tmp_path)
    manifest = _manifest(tmp_path, files)
    hashes = verify_data_hashes(manifest, str(tmp_path))
    assert hashes["a.csv"] == sha256_file(str(files[0][1]))
    assert hashes["b.csv"] == sha256_file(str(files[1][1]))


def test_stage3_neural_cppi_0_tampered_data_aborts(tmp_path):
    files = _two_files(tmp_path)
    manifest = _manifest(tmp_path, files)
    files[0][1].write_text("date,close\n2020-01-01,999\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_data_hashes(manifest, str(tmp_path))


def test_stage3_neural_cppi_0_missing_data_aborts(tmp_path):
    files = _two_files(tmp_path)
    manifest = _manifest(tmp_path, files)
    files[1][1].unlink()
    with pytest.raises(FileNotFoundError, match="b.csv"):
        verify_data_hashes(manifest, str(tmp_path))


def test_stage3_neural_cppi_0_checkpoint_mismatch_aborts(tmp_path):
    files = _two_files(tmp_path)
    manifest = _manifest(tmp_path, files)
    manifest["model_checkpoint"] = {
        "path": "ckpt.pt", "sha256": "0" * 64}
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    write_manifest(str(tmp_path / "m.json"), manifest)
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        sealed_eval_guard(str(tmp_path / "m.json"), str(tmp_path),
                          checkpoint_path=str(ckpt),
                          expected_checkpoint_sha256="0" * 64)


def test_stage3_neural_cppi_0_absolute_user_path_rejected(tmp_path):
    manifest = _manifest(tmp_path, _two_files(tmp_path))
    manifest["notes"] = "trained on /home/someone/private/data.csv"
    with pytest.raises(ValueError, match="absolute"):
        assert_valid_manifest(manifest)
    manifest = _manifest(tmp_path, _two_files(tmp_path))
    manifest["data"] = [{"path": "/tmp/abs.csv", "sha256": "0" * 64}]
    with pytest.raises(ValueError, match="relative"):
        assert_valid_manifest(manifest)


def _load_reproduce_script():
    spec = importlib.util.spec_from_file_location(
        "stage3_reproduce",
        os.path.join(REPO_ROOT, "scripts", "reproduce_historical.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage3_neural_cppi_0_reproduce_script(tmp_path, capsys):
    script = _load_reproduce_script()
    files = _two_files(tmp_path)
    manifest = _manifest(tmp_path, files)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), manifest)
    assert script.main(["--manifest", str(manifest_path),
                        "--base-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "a.csv" in out and "code_commit" in out
    # Tampered data: explicit failure, nonzero exit.
    files[0][1].write_text("date,close\n2020-01-01,999\n")
    assert script.main(["--manifest", str(manifest_path),
                        "--base-dir", str(tmp_path)]) == 1


def test_stage3_neural_cppi_0_tensor_only_loading(tmp_path):
    torch = pytest.importorskip("torch")
    from dhl.eval.provenance import load_checkpoint_tensor_only
    state = {"w": torch.tensor([1.0, 2.0])}
    path = tmp_path / "state.pt"
    torch.save(state, str(path))
    loaded = load_checkpoint_tensor_only(str(path))
    assert torch.equal(loaded["w"], state["w"])
    # A custom class instance pickles fine but weights-only loading must
    # refuse it: arbitrary code must never execute on checkpoint load.
    evil = tmp_path / "evil.pt"
    torch.save({"obj": _Evil()}, str(evil))
    with pytest.raises(Exception):
        load_checkpoint_tensor_only(str(evil))


class _Evil:
    def __init__(self):
        self.x = 1
