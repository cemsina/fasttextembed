#!/usr/bin/env bash
# Copy the C engine sources into ./csrc so the Python package is self-contained
# (builds under pip's build isolation and from an sdist, on any platform).
# csrc/ is gitignored and regenerated; run this before `python -m build`.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
rm -rf "$here/csrc"
mkdir -p "$here/csrc"
cp -R "$root/src" "$here/csrc/src"
cp -R "$root/include" "$here/csrc/include"
echo "vendored engine sources -> bindings/python/csrc/"
