#!/usr/bin/env bash
#
# Build a self-contained tarball of the grasp-perception module for the Viam
# REGISTRY (`viam module upload`). This is the deployment path when you have no
# filesystem access to the arm: you upload from anywhere with internet (e.g.
# westeros), and viam-server on the arm auto-downloads and runs it.
#
# The module imports the repo's components/ package, so we bundle BOTH under one
# tarball root. Layout produced:
#   <root>/meta.json                   (entrypoint: grasp_perception_module/run.sh)
#   <root>/grasp_perception_module/    (the module; tests/.venv/__pycache__ stripped)
#   <root>/components/                 (the python deps it imports)
# run.sh bootstraps a .venv and pip-installs grasp_perception_module/requirements.txt
# on first launch; main.py puts the tarball root on sys.path so `import components.*`
# and `import grasp_perception_module.*` both resolve.
#
# Usage (from anywhere in the repo):
#   scripts/package_grasp_module.sh
#   OUT=/tmp/build scripts/package_grasp_module.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${OUT:-$ROOT/dist}"
TARBALL="$OUT/grasp-perception-module.tar.gz"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging module + components/ deps ..."
# Copy the two dirs with junk stripped, preserving structure, using a tar pipe
# (portable; no rsync dependency).
( cd "$ROOT" && tar \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='grasp_perception_module/tests' \
    --exclude='grasp_perception_module/dist' \
    -cf - grasp_perception_module components ) | tar -C "$STAGE" -xf -

# Registry meta.json at the tarball ROOT, with the entrypoint path relative to it.
python3 - "$ROOT/grasp_perception_module/meta.json" "$STAGE/meta.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
m = json.load(open(src))
m["entrypoint"] = "grasp_perception_module/run.sh"
json.dump(m, open(dst, "w"), indent=2)
print(f"  meta.json -> entrypoint={m['entrypoint']!r}, module_id={m.get('module_id')!r}")
PY

mkdir -p "$OUT"
tar -C "$STAGE" -czf "$TARBALL" .
# Also drop the registry meta.json next to the tarball, so you can
# `cd dist && viam module upload ...` (the CLI reads meta.json from cwd).
cp "$STAGE/meta.json" "$OUT/meta.json"
echo "Built $TARBALL"
echo "Contents (top level):"
tar -tzf "$TARBALL" | sed 's#^\./##' | awk -F/ 'NF<=2' | sort -u | sed 's/^/  /'

cat <<EOF

Next steps (run from a box with internet + the viam CLI, e.g. westeros):
  1. Install + login (one time):
       curl -o viam 'https://storage.googleapis.com/packages.viam.com/apps/viam-cli/viam-cli-latest-linux-amd64' && chmod +x viam
       ./viam login
  2. Ensure the module exists in YOUR org's registry. meta.json's module_id is
     'hackathons:grasp-perception' -- the 'hackathons' namespace must be YOUR org's public
     namespace. If it isn't, edit grasp_perception_module/meta.json's module_id
     to '<your-namespace>:grasp-perception' and re-run this script, then:
       ./viam module create --name grasp-perception
  3. Upload:
       ./viam module upload --version 0.0.1 --platform any "$TARBALL"
  4. In the machine config, reference it as a REGISTRY module (not local):
       "modules":  [{ "type": "registry", "name": "grasp-perception",
                      "module_id": "<namespace>:grasp-perception", "version": "0.0.1" }]
       "services": [{ "name": "grasp-service", "api": "rdk:service:generic",
                      "model": "<namespace>:grasp-perception:grasp-service",
                      "attributes": { "camera": "cam", "segmenter": "vision-segment",
                                      "detector": "shape-detector", "floor_z": 179.75673 },
                      "depends_on": ["cam", "vision-segment", "shape-detector"] }]
  viam-server on the arm then downloads and runs it; no arm filesystem access needed.
EOF
