#!/usr/bin/env python3
"""Check the platform support record against what actually resolves.

Reads ``platform-support.toml``, resolves every entry in it with ``uv pip compile``,
and reports each cell's actual outcome beside the recorded expectation. Exits non-zero
when any cell diverges — in either direction. An entry recorded as an expected failure
that starts succeeding fails the run too: the record has become wrong, and saying so is
the point of this check.

Run it the same way CI does::

    python scripts/check_platform_support.py

Requires ``uv`` on PATH and network access to PyPI. Nothing else; the record is read
with ``tomllib`` from the standard library.

Two subtleties this script exists to get right, both of which produce a plausible
answer rather than an error when got wrong:

* **Resolve per interpreter, not per range.** ``uv pip compile`` honours the project's
  declared ``requires-python`` and yields one resolution valid across that whole range.
  With ``>=3.11`` every cell is dragged back to the oldest compatible stack, which is
  not what a user receives — ``pip`` resolves against the running interpreter alone. So
  each cell is resolved against a copy of ``pyproject.toml`` whose ``requires-python``
  is pinned to the single version under test.

* **Give every cell its own output path.** ``uv pip compile`` treats an existing ``-o``
  file as a preference source, in pip-tools fashion. Reusing one path makes each cell
  inherit the previous cell's pins, and the grid comes out uniform and wrong.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORD_PATH = REPO_ROOT / "platform-support.toml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

WHEEL_COMPLETE = "wheel-complete"
REQUIRES_BUILD = "requires-build"
DOES_NOT_RESOLVE = "does-not-resolve"

# uv reports an unreachable or failing index differently from an unsatisfiable
# requirement. A network failure must never be reported as a platform that stopped
# working, so it is retried and then raised rather than recorded as an outcome.
NETWORK_MARKERS = (
    "failed to fetch",
    "error sending request",
    "connection reset",
    "temporary failure in name resolution",
    "timed out",
    "could not connect",
    "503 service unavailable",
    "502 bad gateway",
)

NETWORK_ATTEMPTS = 3


class NetworkError(RuntimeError):
    """Raised when uv could not reach the index, as distinct from a real conflict."""


@dataclass(frozen=True)
class Cell:
    """One platform x Python x install-shape combination and its expectation."""

    platform: str
    platform_name: str
    python: str
    shape: str
    expected: str
    blame: str | None

    @property
    def label(self) -> str:
        """Return the aligned one-line label for this cell's output row."""
        return f"{self.platform_name:<15} py{self.python}  {self.shape:<5}"


@dataclass(frozen=True)
class Outcome:
    """What actually happened for a cell."""

    result: str
    blame: str | None


def load_record(path: Path) -> tuple[list[Cell], list[str]]:
    """Read the record, returning its cells and the pure-Python sdist exemptions."""
    with path.open("rb") as handle:
        record = tomllib.load(handle)

    exemptions = record.get("pure-python-sdists", [])
    cells = [
        Cell(
            platform=platform["tag"],
            platform_name=platform["name"],
            python=entry["python"],
            shape=entry["shape"],
            expected=entry["expected"],
            blame=entry.get("blame"),
        )
        for platform in record["platform"]
        for entry in platform["entries"]
    ]
    return cells, exemptions


def next_minor(python: str) -> str:
    """Return the version that upper-bounds ``python``: 3.13 -> 3.14."""
    major, minor = python.split(".")
    return f"{major}.{int(minor) + 1}"


def pinned_pyproject(python: str, into: Path) -> Path:
    """Copy pyproject.toml with requires-python pinned to a single interpreter.

    This is what makes a cell's resolution match what ``pip`` gives a user running
    that exact Python, rather than a resolution valid across the declared range.
    """
    source = PYPROJECT_PATH.read_text()
    pinned, count = re.subn(
        r'^requires-python = "[^"]*"',
        f'requires-python = ">={python},<{next_minor(python)}"',
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"could not pin requires-python in {PYPROJECT_PATH}: {count} matches")

    directory = into / f"py{python}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "pyproject.toml"
    target.write_text(pinned)
    return target


def run_uv(
    project: Path, cell: Cell, output: Path, *, wheels_only: bool, exemptions: list[str]
) -> subprocess.CompletedProcess[str]:
    """Invoke uv for one cell, retrying when the index is unreachable."""
    command = [
        "uv",
        "pip",
        "compile",
        str(project),
        "--python-platform",
        cell.platform,
        "--python-version",
        cell.python,
        "--output-file",
        str(output),
        "--quiet",
    ]
    if cell.shape == "trees":
        command += ["--extra", "trees"]
    if wheels_only:
        command += ["--only-binary", ":all:"]
        for package in exemptions:
            command += ["--no-binary", package]

    for attempt in range(1, NETWORK_ATTEMPTS + 1):
        completed = subprocess.run(command, capture_output=True, text=True, cwd=project.parent)
        if completed.returncode == 0:
            return completed
        combined = (completed.stdout + completed.stderr).lower()
        if not any(marker in combined for marker in NETWORK_MARKERS):
            return completed
        if attempt == NETWORK_ATTEMPTS:
            raise NetworkError(f"{cell.label}: index unreachable after {NETWORK_ATTEMPTS} attempts\n{completed.stderr}")
    raise AssertionError("unreachable")


def blamed_package(uv_output: str) -> str | None:
    """Pull the responsible package out of uv's explanation, which names it directly."""
    patterns = (
        r"wheels for `([A-Za-z0-9._-]+)`",
        r"Because ([A-Za-z0-9._-]+)==\S+ has no usable wheels",
        r"Because ([A-Za-z0-9._-]+)\S* has no wheels with a matching platform tag",
        r"Because ([A-Za-z0-9._-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, uv_output)
        if match:
            return match.group(1)
    return None


def evaluate(cell: Cell, workdir: Path, exemptions: list[str]) -> Outcome:
    """Resolve one cell twice — once for a solution, once for wheels — and classify it."""
    project = pinned_pyproject(cell.python, workdir)

    # Every cell writes to its own file: uv would otherwise read a shared output file
    # as a preference source and carry the previous cell's pins into this one.
    stem = f"{cell.python}_{cell.platform}_{cell.shape}"

    resolved = run_uv(project, cell, workdir / f"{stem}.txt", wheels_only=False, exemptions=exemptions)
    if resolved.returncode != 0:
        return Outcome(DOES_NOT_RESOLVE, blamed_package(resolved.stderr))

    wheels = run_uv(project, cell, workdir / f"{stem}.wheels.txt", wheels_only=True, exemptions=exemptions)
    if wheels.returncode != 0:
        return Outcome(REQUIRES_BUILD, blamed_package(wheels.stderr))

    return Outcome(WHEEL_COMPLETE, None)


def divergence(cell: Cell, outcome: Outcome) -> str | None:
    """Describe how a cell differs from its record, or None when it matches."""
    if outcome.result == cell.expected:
        return None

    ranking = {WHEEL_COMPLETE: 2, REQUIRES_BUILD: 1, DOES_NOT_RESOLVE: 0}
    got_better = ranking[outcome.result] > ranking[cell.expected]

    if got_better:
        return (
            f"records {cell.expected}, but it {describe(outcome.result)}. "
            f"The record and the documentation written from it are now wrong in the "
            f"package's favour — update them rather than leaving the claim understated."
        )
    culprit = f" ({outcome.blame})" if outcome.blame else ""
    return f"records {cell.expected}, but it {describe(outcome.result)}{culprit}."


def describe(result: str) -> str:
    """Return a human-readable phrase for an outcome value."""
    return {
        WHEEL_COMPLETE: "installs from wheels alone",
        REQUIRES_BUILD: "needs a build toolchain",
        DOES_NOT_RESOLVE: "does not resolve",
    }[result]


def main() -> int:
    """Resolve every recorded cell, report divergences, and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=RECORD_PATH, help="path to platform-support.toml")
    arguments = parser.parse_args()

    if shutil.which("uv") is None:
        print("error: uv is not on PATH; see https://docs.astral.sh/uv/", file=sys.stderr)
        return 2

    cells, exemptions = load_record(arguments.record)
    print(f"Checking {len(cells)} entries from {arguments.record.name}")
    print(f"Pure-Python sdists permitted without wheels: {', '.join(exemptions)}\n")

    divergences: list[tuple[Cell, Outcome, str]] = []

    with tempfile.TemporaryDirectory(prefix="atlas-platform-support-") as temporary:
        workdir = Path(temporary)
        for cell in cells:
            outcome = evaluate(cell, workdir, exemptions)
            difference = divergence(cell, outcome)
            mark = "ok  " if difference is None else "FAIL"
            detail = outcome.result if difference is None else f"{cell.expected} -> {outcome.result}"
            print(f"  {mark}  {cell.label}  {detail}")
            if difference is not None:
                divergences.append((cell, outcome, difference))

    # The whole grid is reported before the exit status is decided: a support set
    # containing a known limitation can never be expressed as one pass or fail.
    print()
    if not divergences:
        print(f"All {len(cells)} entries match the record.")
        return 0

    print(f"{len(divergences)} of {len(cells)} entries diverge from the record:\n")
    for cell, _outcome, difference in divergences:
        print(f"  {cell.platform} / Python {cell.python} / {cell.shape}")
        print(f"      {difference}\n")
    print(f"Update {arguments.record.name}, and the documentation written from it, to match.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NetworkError as error:
        print(f"error: {error}", file=sys.stderr)
        print("The index was unreachable. This is not a platform support failure.", file=sys.stderr)
        sys.exit(2)
