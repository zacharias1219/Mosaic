"""Load mosaic.yaml and candidates.yaml; build clients per candidate backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from mosaic.types import Candidate

ROOT = Path(__file__).resolve().parent.parent


def resolve(v: Any) -> Any:
    """'env:NAME|default' -> os.environ[NAME] or default."""
    if isinstance(v, str) and v.startswith("env:"):
        name, _, default = v[4:].partition("|")
        return os.environ.get(name, default)
    return v


def load_config(path: Optional[str | Path] = None) -> dict[str, Any]:
    p = Path(path) if path else ROOT / "mosaic.yaml"
    cfg = yaml.safe_load(p.read_text())
    for prov in cfg.get("providers", {}).values():
        for k in list(prov):
            prov[k] = resolve(prov[k])
    return cfg


def load_candidates(path: Optional[str | Path] = None) -> dict[str, Candidate]:
    p = Path(path) if path else ROOT / "candidates.yaml"
    rows = yaml.safe_load(p.read_text())["candidates"]
    return {r["id"]: Candidate(**r) for r in rows}


def save_candidates(cands: dict[str, Candidate], path: Optional[str | Path] = None, header: str = "") -> None:
    p = Path(path) if path else ROOT / "candidates.yaml"
    old = p.read_text() if p.exists() else ""
    head = header or "".join(ln + "\n" for ln in old.splitlines() if ln.startswith("#"))
    rows = [c.model_dump(exclude_none=True) for c in cands.values()]
    p.write_text(head + yaml.safe_dump({"candidates": rows}, sort_keys=False, width=110))


class ReplayOnlyClient:
    """Stands in for a live provider when replaying from cache: any cache miss is an error,
    so an offline replay can never silently call the network."""
    replay_only = True

    async def complete(self, cand, messages, **kw):
        from mosaic.providers.base import ProviderError
        raise ProviderError(f"{cand.id}: not in cache (offline replay)")


def build_clients(cands: dict[str, Candidate], cfg: dict[str, Any], backends: dict[str, str], world=None,
                  seed: int = 42) -> dict[str, Any]:
    """backends: candidate id -> 'live' | 'sim'."""
    from mosaic.providers.cerebras import CerebrasClient
    from mosaic.providers.groq import GroqClient
    from mosaic.providers.local import LocalClient
    from mosaic.providers.openrouter import OpenRouterClient
    from mosaic.providers.sim import SimClient

    sim = SimClient(world, seed=seed) if world is not None else None
    kinds = {"groq": GroqClient, "cerebras": CerebrasClient, "local": LocalClient, "openrouter": OpenRouterClient}
    live: dict[str, Any] = {}
    out = {}
    for cid, c in cands.items():
        mode = backends.get(cid, backends.get("default", "sim"))
        if mode == "replay":
            out[cid] = ReplayOnlyClient()
        elif mode == "sim":
            if sim is None:
                raise ValueError(f"{cid} is on sim but no SimWorld was given")
            out[cid] = sim
        else:
            if c.provider not in live:
                pc = cfg["providers"].get(c.provider, {})
                cls = kinds[c.provider]
                live[c.provider] = cls(pc.get("base_url") or cls.default_base_url)
            out[cid] = live[c.provider]
    return out
