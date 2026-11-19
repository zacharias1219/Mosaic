"""Deterministic SVG charts (fixed hash salt, no dates) so reports regenerate byte-identically."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "mosaic", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
PALETTE = {"P0": "#6b7280", "P1": "#2563eb", "P2": "#d97706", "P3": "#059669"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)


def non_dominated(points: list[tuple[str, float, float, float]]) -> set[str]:
    """points: (name, cost_per_correct, p95_latency, accuracy). Lower cost and latency, higher accuracy."""
    out = set()
    for n, c, l, a in points:
        dominated = any((c2 <= c and l2 <= l and a2 >= a) and (c2 < c or l2 < l or a2 > a)
                        for n2, c2, l2, a2 in points if n2 != n)
        if not dominated:
            out.add(n)
    return out


def pareto(points: list[tuple[str, float, float, float]], baseline: str, path: Path, title: str) -> set[str]:
    nd = non_dominated(points)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n, c, l, a in points:
        pipe = n.split()[0]
        ax.scatter(c * 1000, l / 1000, s=40 + 260 * a, color=PALETTE.get(pipe, "#111"), alpha=0.85,
                   edgecolor="black" if n in nd else "none", linewidth=1.4, marker="s" if n == baseline else "o", zorder=3)
        ax.annotate(f"{n}\n{a:.0%}", (c * 1000, l / 1000), textcoords="offset points", xytext=(7, 4), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("cost per correct answer (milli-dollars, log scale)")
    ax.set_ylabel("p95 latency per question (s)")
    ax.set_title(title, fontsize=9.5)
    ax.text(0.01, 0.01, "square = baseline · outlined = non-dominated · area ∝ accuracy", transform=ax.transAxes,
            fontsize=7, color="#555")
    ax.grid(alpha=0.25, zorder=0)
    _save(fig, path)
    return nd


def accuracy_by_size(series: dict[str, list[tuple[str, float, float, float]]], path: Path, title: str) -> None:
    """series: pipeline -> [(size, acc, lo, hi)]"""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for pipe, pts in series.items():
        xs = list(range(len(pts)))
        ax.errorbar([x + 0.06 * list(series).index(pipe) for x in xs], [p[1] * 100 for p in pts],
                    yerr=[[max(0.0, 100 * (p[1] - p[2])) for p in pts], [max(0.0, 100 * (p[3] - p[1])) for p in pts]],
                    label=pipe, color=PALETTE.get(pipe), marker="o", capsize=3, lw=1.5)
        ax.set_xticks(xs, [p[0] for p in pts])
    ax.set_ylim(0, 105)
    ax.set_ylabel("accuracy (%), Wilson 95% CI")
    ax.set_xlabel("corpus size")
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, path)


def reliability(curves: dict[str, list[dict]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    for name, bins in curves.items():
        if bins:
            ax.plot([b["mean_pred"] for b in bins], [b["observed"] for b in bins], marker="o", label=name, lw=1.4)
    ax.set_xlabel("predicted P(correct)")
    ax.set_ylabel("observed share correct")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path)


def device_latency(rows: list[tuple[str, float, float, str]], path: Path, title: str) -> None:
    """rows: (label, p50_ms, p95_ms, source)"""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ys = list(range(len(rows)))
    for y, (lab, p50, p95, src) in zip(ys, rows):
        col = "#2563eb" if src == "live" else "#d97706"
        ax.plot([p50, p95], [y, y], color=col, lw=2)
        ax.scatter([p50], [y], color=col, zorder=3)
        ax.scatter([p95], [y], color=col, marker="|", s=120, zorder=3)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("map-call service time, ms (dot = p50, bar = p95; blue = live, orange = simulated)")
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.25, axis="x")
    _save(fig, path)
