#!/usr/bin/env bash
# Fetch the `cuvs` Rust crate at VERSION into ../vendor/cuvs-VERSION (a sibling of
# this checkout, like `lance`) and make `Resources::handle()` public.
#
# cuvs >= 26.8 no longer exposes the raw cuvsResources_t (the old `Resources.0`),
# and the backend's FFI calls (cuvsIvfPqBuild, cuvsRMMAlloc, ...) need it.
# backends/cuvs_26_02/Cargo.toml points its `cuvs` dependency at this directory.
#
#   tools/vendor_cuvs.sh 26.8.1
set -euo pipefail
v="${1:?usage: vendor_cuvs.sh VERSION}"
dest="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/vendor/cuvs-$v"
mkdir -p "$(dirname "$dest")"
rm -rf "$dest"
tmp="$(mktemp -d)"
curl -fsSL -A vendor-cuvs "https://crates.io/api/v1/crates/cuvs/$v/download" | tar xz -C "$tmp"
mv "$tmp/cuvs-$v" "$dest"
sed -i 's/pub(crate) fn handle(&self)/pub fn handle(\&self)/' "$dest/src/resources.rs"
grep -q 'pub fn handle(&self)' "$dest/src/resources.rs" || { echo "patch failed" >&2; exit 1; }
echo "vendored $dest"
