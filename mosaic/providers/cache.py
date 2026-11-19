"""Response cache: every live call is replayable offline (PRD 9.5).

Key = hash(model id, provider, messages, parameters, schema). Value = completion with its
original timings, including queue wait, so a replay reproduces the run that produced it.
API keys are never part of the key or the value.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from mosaic.providers.base import Completion


def cache_key(provider: str, model: str, messages: list[dict], params: dict[str, Any],
              schema: Optional[dict] = None) -> str:
    blob = json.dumps({"provider": provider, "model": model, "messages": messages, "params": params,
                       "schema": schema}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path, read: bool = True, write: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.read, self.write = read, write
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, provider TEXT, "
                        "model TEXT, value TEXT)")
        self.db.commit()
        self.hits = self.misses = 0

    def get(self, key: str) -> Optional[Completion]:
        if not self.read:
            return None
        row = self.db.execute("SELECT value FROM responses WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return Completion(**{**json.loads(row[0]), "replayed": True})

    def put(self, key: str, provider: str, model: str, comp: Completion) -> None:
        if not self.write:
            return
        value = comp.model_dump(exclude={"replayed"})
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
                        (key, provider, model, json.dumps(value, sort_keys=True)))
        self.db.commit()

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self.db.close()
