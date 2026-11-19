"""Wire config, data, candidates, clients, router and gateway for one pipeline run."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from mosaic.config import ROOT, build_clients, load_candidates, load_config
from mosaic.data.store import load_corpus, load_questions
from mosaic.gateway import Gateway
from mosaic.providers.cache import ResponseCache
from mosaic.providers.ratelimit import RealClock, VirtualClock
from mosaic.providers.sim import SimWorld
from mosaic.routing.predictor import Predictor
from mosaic.routing.router import Router
from mosaic.trace import Tracer
from mosaic.types import Candidate, Profile


def load_profiles(path: Path) -> dict[str, Profile]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    df = df[df["concurrency"] == 1] if "concurrency" in df else df
    df = df.assign(_live=df["source"].str.startswith("live")).sort_values(["candidate_id", "_live"])
    # one row per candidate; a live measurement wins over a simulated one
    return {r["candidate_id"]: Profile(**{k: r[k] for k in Profile.model_fields if k in r}) for _, r in df.iterrows()}


def make_world(data_dir: Path, splits=("eval", "profiling"), sizes=None) -> SimWorld:
    w = SimWorld()
    for split in splits:
        qp = data_dir / f"questions-{split}.jsonl"
        if not qp.exists():
            continue
        qs = load_questions(data_dir, split)
        w.add_questions(qs)
        for size in sorted({q.size for q in qs} if sizes is None else sizes):
            w.add_corpus(load_corpus(data_dir, split, size))
    return w


class Session:
    def __init__(self, cfg: Optional[dict] = None, cands: Optional[dict[str, Candidate]] = None,
                 root: Path = ROOT, world: Optional[SimWorld] = None):
        self.cfg = cfg or load_config(root / "mosaic.yaml")
        self.cands = cands or load_candidates(root / "candidates.yaml")
        self.root = root
        self.data_dir = root / self.cfg.get("data_dir", "data")
        self.runs_dir = root / self.cfg.get("runs_dir", "runs")
        self._world = world

    @property
    def world(self) -> SimWorld:
        if self._world is None:
            self._world = make_world(self.data_dir)
        return self._world

    def router(self, pipeline: str, pool: Optional[list[str]] = None, explore: Optional[float] = None,
               predictor_path: Optional[Path] = None) -> Router:
        cfg = self.cfg
        pool = pool or cfg["main_pool"]
        common = dict(thresholds=cfg["quality_threshold"], seed=cfg["seed"], latency_budget_ms=cfg["latency_budget_ms"],
                      max_queue_ms=cfg["max_queue_ms"], pool=pool)
        if pipeline in ("P0", "P1"):  # R0: every call on one candidate (failures retry it)
            fixed = pool[0] if len(pool) == 1 else cfg["baseline"]
            return Router("fixed", fixed=fixed, **{**common, "pool": [fixed]})
        if pipeline == "P2":
            return Router("rules", **common)
        pp = predictor_path or self.root / "router" / "model.json"
        pred = Predictor.load(pp) if pp.exists() else None
        return Router("learned" if pred else "rules", predictor=pred,
                      explore_rate=cfg["explore_rate"] if explore is None else explore, **common)

    def gateway(self, router: Router, backends: dict[str, str], tracer: Tracer, live: bool,
                use_cache: bool = True, read_cache: bool = True, spend_cap: Optional[float] = None) -> Gateway:
        clients = build_clients(self.cands, self.cfg, backends, world=self.world if "sim" in backends.values()
                                or backends.get("default") == "sim" else None, seed=self.cfg["seed"])
        cache = ResponseCache(self.root / self.cfg["cache_path"], read=read_cache, write=True) if use_cache else None
        clock: Any = RealClock() if live else VirtualClock()
        profiles = load_profiles(self.root / "profiles" / "profiles.parquet")
        lc = self.cfg["providers"]["local"]
        return Gateway(self.cands, clients, router, tracer, clock, cache=cache, profiles=profiles,
                       local_cost={"watts": lc.get("watts", 28), "usd_per_kwh": lc.get("usd_per_kwh", 0.1),
                                   "amortisation_usd_per_hour": lc.get("amortisation_usd_per_hour", 0.05)},
                       spend_cap_usd=spend_cap)
