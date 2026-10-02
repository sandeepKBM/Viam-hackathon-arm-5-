# services/

Shared plumbing for the voice-uq multi-service architecture: four small
FastAPI services (`perception`, `uq`, `cognition`, `orchestrator`) that
talk to each other over plain HTTP, the same way color-sort's
`moondream/server.py` + `components/moondream_client.py` already work.

```
services/
  services.yaml   # where each service lives (host:port)
  config.py       # loads services.yaml + env overrides -> service_url()/service_addr()
  base.py         # make_service(name) FastAPI factory + call_service() HTTP client
  README.md       # this file
  tests/          # offline tests (no network, no hardware)
```

This directory only provides the plumbing. Each actual service
(`perception`, `uq`, `cognition`, `orchestrator`) is expected to live in
its own module and build its app with `services.base.make_service`, e.g.:

```python
# services/perception_server.py (example, not included here)
from services.base import make_service

app = make_service("perception")

@app.post("/detect")
async def detect(...): ...
```

and be run with uvicorn, matching how `moondream/server.py` and
`voice/voice.py` are already run in this repo:

```sh
python -m uvicorn services.perception_server:app --host 127.0.0.1 --port 8801
python -m uvicorn services.uq_server:app           --host 127.0.0.1 --port 8802
python -m uvicorn services.cognition_server:app    --host 127.0.0.1 --port 8803
python -m uvicorn services.orchestrator_server:app --host 127.0.0.1 --port 8804
```

(Ports come from `services.yaml`; pass `--port` explicitly if you override
it via env vars instead, so uvicorn's bind matches what `config.py` tells
callers to dial.)

## Single-Mac setup (default)

Nothing to configure: `services.yaml` ships with all four services on
`127.0.0.1`. Run all four uvicorn processes locally; `call_service(name,
path, ...)` resolves each one to `http://127.0.0.1:<port>`.

## Two-Mac setup (Thunderbolt bridge)

The goal is to use both machines' compute -- e.g. run the heavy
perception model (SAM / the detector) on whichever Mac has the better
GPU/NPU, and keep `cognition` + `orchestrator` (lighter, more
latency-sensitive) on the other.

1. **Wire the Macs together** with a Thunderbolt cable and turn on a
   Thunderbolt Bridge network (System Settings > Network > add the
   Thunderbolt interface, or `Bridge`). Each Mac gets an IP on that link,
   e.g. Mac A = `169.254.10.1`, Mac B = `169.254.10.2`. Confirm with
   `ifconfig` / `ipconfig getifaddr en<N>` on each side and a quick `ping`
   across the bridge.

2. **Decide the split.** Example: `perception` (and maybe `uq`, since it
   also touches the detector) on Mac A; `cognition` and `orchestrator` on
   Mac B, which is the one actually talking to the arm/voice UI.

3. **Edit `services/services.yaml`** (or point `SERVICES_CONFIG` at a
   host-specific copy) so each service's `host` is the Thunderbolt IP of
   the machine that *runs* it -- not `127.0.0.1`, even for services that
   happen to be co-located with the caller you're editing this for:

   ```yaml
   perception:  {host: 169.254.10.1, port: 8801}   # Mac A
   uq:          {host: 169.254.10.1, port: 8802}   # Mac A
   cognition:   {host: 169.254.10.2, port: 8803}   # Mac B
   orchestrator:{host: 169.254.10.2, port: 8804}   # Mac B
   ```

   Both Macs should use the *same* `services.yaml` (or the same
   `SERVICES_CONFIG` file, e.g. synced/copied to both), so every process
   agrees on where everything lives.

4. **Start each service on the Mac that owns it**, binding to the bridge
   IP (or `0.0.0.0`) instead of loopback so the other Mac can reach it:

   ```sh
   # on Mac A
   python -m uvicorn services.perception_server:app --host 169.254.10.1 --port 8801

   # on Mac B
   python -m uvicorn services.cognition_server:app --host 169.254.10.2 --port 8803
   python -m uvicorn services.orchestrator_server:app --host 169.254.10.2 --port 8804
   ```

5. Everything else is unchanged: `call_service("perception", "/detect",
   ...)` on Mac B now goes out over the Thunderbolt bridge to Mac A
   instead of loopback, because `config.service_url("perception")` reads
   the `169.254.10.1` host from `services.yaml`.

### Overriding one service without editing the yaml

For a quick one-off (e.g. testing against a service running on a
teammate's laptop), skip the yaml edit and set an env var for just that
process:

```sh
VOICEUQ_PERCEPTION_URL=http://169.254.10.1:8801 python -m uvicorn services.orchestrator_server:app ...
# or, equivalently:
VOICEUQ_PERCEPTION_HOST=169.254.10.1 VOICEUQ_PERCEPTION_PORT=8801 python -m uvicorn services.orchestrator_server:app ...
```

Precedence (highest wins): `VOICEUQ_<NAME>_URL` > `VOICEUQ_<NAME>_HOST`
/ `VOICEUQ_<NAME>_PORT` > `services.yaml` (or whatever `SERVICES_CONFIG`
points at) > the hard-coded `127.0.0.1` defaults in `services/config.py`.
`<NAME>` is the service name upper-cased (`PERCEPTION`, `UQ`,
`COGNITION`, `ORCHESTRATOR`).

## Testing

```sh
cd /common/users/ss5772/viam_5-voice-uq
PYTHONPATH=. /common/users/ss5772/viam_5/.venv/bin/python -m pytest services/tests/ -q
```

All tests are offline: `call_service`'s HTTP transport
(`urllib.request.urlopen`) is monkeypatched, so no real network calls
(and no FastAPI/uvicorn install) are needed just to exercise the config
and client logic.
