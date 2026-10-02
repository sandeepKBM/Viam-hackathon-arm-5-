"""Shared plumbing for the voice-uq multi-service stack.

Each service (perception, uq, cognition, orchestrator) is a small FastAPI
process. ``services.config`` resolves where each one lives (single Mac by
default, or split across two Thunderbolt-linked Macs) and
``services.base`` provides a tiny app factory + HTTP client helper so every
service follows the same conventions as color-sort's moondream/server.py
and components/moondream_client.py.
"""
