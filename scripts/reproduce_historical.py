"""Fresh-checkout reproduction check for a frozen experiment.

Given an experiment manifest, verify from the current checkout that every
recorded data file hashes to its recorded value (and the checkpoint, when
given) or fail with an explicit error naming what is missing or changed.
No vendor API and no private data is required: a manifest that names data
absent from this checkout fails with a missing-data error, which is the
specified outcome, not a crash.

Usage:
    python scripts/reproduce_historical.py --manifest manifest.json
        [--base-dir .] [--checkpoint ckpt.pt]
        [--expected-checkpoint-sha256 HEX]

Exit code 0 when everything reproduces, 1 on any provenance failure.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dhl.eval.provenance import sealed_eval_guard  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify a frozen experiment manifest from this checkout.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    args = parser.parse_args(argv)
    try:
        report = sealed_eval_guard(
            args.manifest, args.base_dir,
            checkpoint_path=args.checkpoint,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    except (FileNotFoundError, ValueError) as exc:
        print(f"PROVENANCE ERROR: {exc}", file=sys.stderr)
        return 1
    manifest = report["manifest"]
    print("provenance OK")
    print(f"  manifest: {args.manifest}")
    print(f"  code_commit: {manifest['code_commit']} "
          f"(dirty: {manifest['code_dirty']})")
    print(f"  train_seed: {manifest['train_seed']}")
    for path, sha in report["data_hashes"].items():
        print(f"  data {path}: {sha}")
    if report["checkpoint_sha256"] is not None:
        print(f"  checkpoint sha256: {report['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
