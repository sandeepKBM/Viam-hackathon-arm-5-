#!/usr/bin/env bash
#
# Reverse SSH tunnel: expose the arm's viam-server at localhost:<VM_LISTEN_PORT>
# ON WESTEROS (the VM), so code running there can connect DIRECTLY over the LAN
# path (VIAM_LOCAL=1), bypassing Viam cloud signaling.
#
# ►► RUN THIS ON THE MACHINE THAT IS ON THE ARM'S WiFi (192.168.1.x) ◄◄
#    e.g. your Mac. Do NOT run it from the Windows laptop (10.66.x) or from
#    westeros -- neither can reach the arm, so the tunnel would forward to a
#    dead endpoint. The dial to the arm happens FROM THIS machine's network.
#
# Flow:  westeros:localhost:<VM_LISTEN_PORT>  --(ssh -R)-->  THIS machine  -->  <ARM_IP>:<ARM_PORT>
#
# Usage (on the Mac):
#   VM_HOST=ss5772@<westeros-host> ./scripts/tunnel_to_arm.sh
#   ARM_IP=10.1.9.212 VM_HOST=ss5772@<westeros-host> ./scripts/tunnel_to_arm.sh   # other arm NIC
#
# Then on westeros:
#   export VIAM_ALLOW_LIVE=1 VIAM_LOCAL=1
#   export VIAM_MACHINE_LOCAL_ADDRESS=localhost:<VM_LISTEN_PORT>
#   python scripts/local_connect_smoke_test.py
#
set -euo pipefail

ARM_IP="${ARM_IP:-192.168.1.150}"          # arm LAN IP (also 10.1.9.212). Must be reachable from THIS machine.
ARM_PORT="${ARM_PORT:-8080}"               # viam-server port
VM_HOST="${VM_HOST:-}"                      # ssh target for westeros, e.g. ss5772@westeros.cs.rutgers.edu
VM_LISTEN_PORT="${VM_LISTEN_PORT:-8080}"   # port opened on westeros -> VIAM_MACHINE_LOCAL_ADDRESS=localhost:<this>

if [[ -z "$VM_HOST" ]]; then
  echo "ERROR: set VM_HOST to your westeros ssh target." >&2
  echo "  e.g.  VM_HOST=ss5772@westeros $0" >&2
  exit 1
fi

# --- Preflight: can THIS machine actually reach the arm? ---------------------
# This is the check that was missing before. If it fails, you are NOT on the
# arm's network and the tunnel cannot work -- run this from the Mac on the
# arm's WiFi instead.
echo "Preflight: can this machine reach ${ARM_IP}:${ARM_PORT}? ..."
reachable=""
if command -v nc >/dev/null 2>&1; then
  if nc -z -w 4 "$ARM_IP" "$ARM_PORT" >/dev/null 2>&1; then reachable="yes"; fi
elif command -v python3 >/dev/null 2>&1; then
  if python3 - "$ARM_IP" "$ARM_PORT" <<'PY' >/dev/null 2>&1
import socket,sys
s=socket.socket(); s.settimeout(4)
s.connect((sys.argv[1], int(sys.argv[2]))); s.close()
PY
  then reachable="yes"; fi
else
  # Last resort: bash /dev/tcp (no clean timeout on old bash; may hang ~TCP default)
  if (exec 3<>"/dev/tcp/${ARM_IP}/${ARM_PORT}") >/dev/null 2>&1; then reachable="yes"; fi
fi

if [[ "$reachable" != "yes" ]]; then
  echo "✗ Cannot reach ${ARM_IP}:${ARM_PORT} from this machine." >&2
  echo "  You are NOT on the arm's LAN, OR viam-server is on a different port." >&2
  echo "  Run this script from the machine on the arm's WiFi (192.168.1.x)," >&2
  echo "  or try the other NIC:  ARM_IP=10.1.9.212 $0" >&2
  echo "  (Check the arm's port in the Viam app CONNECT tab or 'ss -ltn' on the arm.)" >&2
  exit 2
fi
echo "✓ Arm reachable from here. Opening reverse tunnel..."

echo "  westeros($VM_HOST) localhost:${VM_LISTEN_PORT}  ->  ${ARM_IP}:${ARM_PORT}  (via this machine)"
echo "  On westeros set:  VIAM_LOCAL=1  VIAM_MACHINE_LOCAL_ADDRESS=localhost:${VM_LISTEN_PORT}"
echo "  Ctrl-C here tears the tunnel down."
echo

# -N: forwarding only. ExitOnForwardFailure: fail if the westeros port is taken.
# Keepalives so the tunnel survives idle periods. autossh auto-reconnects.
SSH_OPTS=(
  -N
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -o ExitOnForwardFailure=yes
  -R "${VM_LISTEN_PORT}:${ARM_IP}:${ARM_PORT}"
)

if command -v autossh >/dev/null 2>&1; then
  exec autossh -M 0 "${SSH_OPTS[@]}" "$VM_HOST"
else
  echo "(tip: 'brew install autossh' for auto-reconnect; using plain ssh)"
  exec ssh "${SSH_OPTS[@]}" "$VM_HOST"
fi
