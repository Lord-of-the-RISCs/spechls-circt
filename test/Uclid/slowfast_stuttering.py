#!/usr/bin/env python3
"""Generate and check a source-versus-speculation Uclid stuttering harness.

This test utility is the integration boundary between native SpecHLS MLIR and
Uclid.  It parses the non-speculated source task and the FSM/delay/rollback
lowered task, emits their two transition-system modules plus a third harness,
then invokes Uclid on the combined compilation unit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def repository_root() -> Path:
    """Return the SpecHLS repository root from this script's stable location."""
    return Path(__file__).resolve().parents[2]


def configure_xdsl_import(root: Path) -> None:
    """Make the in-tree xDSL SpecHLS package importable for standalone execution."""
    xdsl_root = root / "xdsl"
    if str(xdsl_root) not in sys.path:
        sys.path.insert(0, str(xdsl_root))


def generate_bundle(source: Path, speculated: Path, output: Path):
    """Parse both MLIR inputs and write the reference, transformed, and driver modules."""
    from spechls.native_mlir import parse_native_mlir
    from spechls.uclid import emit_stuttering_bisimulation_driver

    bundle = emit_stuttering_bisimulation_driver(
        parse_native_mlir(source.read_text(encoding="ascii")),
        parse_native_mlir(speculated.read_text(encoding="ascii")),
    )
    bundle.write(output)
    return bundle


def run_uclid(source: Path) -> int:
    """Run Uclid and return failure when it refutes a generated invariant."""
    executable = os.environ.get("UCLID") or shutil.which("uclid")
    if executable is None:
        raise RuntimeError("Uclid executable not found; set UCLID or add uclid to PATH")
    result = subprocess.run([executable, str(source)], text=True, capture_output=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        return result.returncode
    return 0 if "0 assertions failed" in result.stdout else 1


def main() -> int:
    """Generate the slowfast proof harness and delegate verification to Uclid."""
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "slowfast-end-to-end.mlir",
    )
    parser.add_argument(
        "--speculated",
        type=Path,
        default=Path(__file__).resolve().parent / "slowfast-end-to-end.speculated.mlir",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    arguments = parser.parse_args()
    configure_xdsl_import(root)
    bundle = generate_bundle(arguments.source, arguments.speculated, arguments.output)
    with tempfile.TemporaryDirectory(prefix="slowfast-uclid-") as temporary:
        combined = Path(temporary) / "slowfast-stuttering-bisimulation.ucl"
        combined.write_text(bundle.combined_source(), encoding="ascii")
        return run_uclid(combined)


if __name__ == "__main__":
    raise SystemExit(main())
