#!/usr/bin/env bash
# One-command rollback of the calibrated-pour patch to the pre-patch commit.
#   scripts/rollback_pour.sh            # asks for confirmation
#   scripts/rollback_pour.sh --yes
# Runtime-only rollback needs no code change: leave ENABLE_CALIBRATED_POUR unset.
# Calibration evidence under config/calibration/ is left in place.
set -euo pipefail
BASE="c8b7d041d5dc4ce38ce157228563aee49ebf6a90"
cd "$(dirname "$0")/.."

MODIFIED=(components/policy.py components/shapes.py components/skills.py components/voice.py
          services/orchestrator_service.py voice/voice.py scripts/pour_can.py README.md .gitignore)
NEW=(components/transforms.py components/calibration.py components/rgbd.py components/object_pose.py
     components/pour_planner.py components/pouring.py components/pour_evidence.py
     components/pour_calibration.py scripts/calibrate_pour_setup.py docs/CALIBRATED_POUR.md tests
     scripts/rollback_pour.sh)

echo "Restores ${MODIFIED[*]} from ${BASE:0:7} and deletes: ${NEW[*]}"
if [[ "${1:-}" != "--yes" ]]; then
  read -r -p "Type ROLLBACK to continue: " ans
  [[ "$ans" == "ROLLBACK" ]] || { echo "cancelled"; exit 1; }
fi
git checkout "$BASE" -- "${MODIFIED[@]}"
git rm -r -q --cached --ignore-unmatch "${NEW[@]}" >/dev/null 2>&1 || true
rm -rf "${NEW[@]}"
echo "rolled back to the ${BASE:0:7} pour experiment (uncalibrated; do not run it with --go)."
