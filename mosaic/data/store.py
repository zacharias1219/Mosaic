"""On-disk layout for corpora and question sets, with a manifest of hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mosaic.data.gen import SIZES, dumps, generate, loads
from mosaic.data.questions import build_questions
from mosaic.types import Question, Ticket

SPLIT_SEED_OFFSET = {"eval": 0, "profiling": 1}  # splits use different seeds and never mix


def corpus_path(data_dir: Path, split: str, size: str) -> Path:
    return data_dir / f"corpus-{split}-{size}.jsonl"


def generate_all(data_dir: Path, sizes: list[str], seed: int, n_per_class: int, n_profiling: int) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": seed, "files": {}}
    for split in ("eval", "profiling"):
        qs: list[Question] = []
        s_seed = seed + SPLIT_SEED_OFFSET[split]
        n = n_per_class if split == "eval" else max(1, n_profiling // (3 * len(sizes)))
        for size in sizes:
            tickets = generate(SIZES[size], s_seed)
            p = corpus_path(data_dir, split, size)
            p.write_text(dumps(tickets))
            for cls in "LAP":
                qs += build_questions(tickets, cls, n, s_seed, size, split=split)
        qp = data_dir / f"questions-{split}.jsonl"
        qp.write_text("".join(json.dumps(q.model_dump(), sort_keys=True) + "\n" for q in qs))
    for p in sorted(data_dir.glob("*.jsonl")):
        manifest["files"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["data_hash"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()[:16]
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def load_corpus(data_dir: Path, split: str, size: str) -> list[Ticket]:
    return loads(corpus_path(data_dir, split, size).read_text())


def load_questions(data_dir: Path, split: str) -> list[Question]:
    p = data_dir / f"questions-{split}.jsonl"
    return [Question(**json.loads(ln)) for ln in p.read_text().splitlines() if ln.strip()]


def data_hash(data_dir: Path) -> str:
    p = data_dir / "manifest.json"
    return json.loads(p.read_text()).get("data_hash", "") if p.exists() else ""
