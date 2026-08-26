#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RELEASE_VERSION = re.compile(r"\d+(\.\d+)*(\.post\d+)?")

WRONG_VERSION = """refusing to publish version {version!r}.

This is not a release version. ATLAS derives its version from git tags through
hatch-vcs, so a development or local version means the tag was not visible to the
build — most often because the GitHub release was created before the tag reached
the mirror.

Check the tag, not this gate. The release sequence is in docs/contributing.md.
"""


def version_of(distribution: Path) -> str:
    """Retrieve version"""
    name = distribution.name
    for suffix in (".tar.gz", ".zip"):
        name = name.removesuffix(suffix)
    return name.split("-")[1]


def main() -> int:
    """Check that the built distribution has a release version, not a dev/local one."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dist", type=Path, default=Path("dist"), help="directory holding the built distributions")
    arguments = parser.parse_args()

    wheels = sorted(arguments.dist.glob("*.whl"))
    if len(wheels) != 1:
        print(f"error: expected exactly one wheel in {arguments.dist}/, found {len(wheels)}", file=sys.stderr)
        return 2

    version = version_of(wheels[0])

    # The sdist is built from the same source and must agree. A disagreement means the
    # two were not produced by the same build, which is worth stopping for on its own.
    sdists = sorted(arguments.dist.glob("*.tar.gz"))
    for sdist in sdists:
        if version_of(sdist) != version:
            print(
                f"error: the wheel is version {version!r} but {sdist.name} is "
                f"{version_of(sdist)!r}; these were not built together",
                file=sys.stderr,
            )
            return 2

    if not RELEASE_VERSION.fullmatch(version):
        print(WRONG_VERSION.format(version=version), file=sys.stderr)
        return 1

    print(f"version {version} is a release version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
