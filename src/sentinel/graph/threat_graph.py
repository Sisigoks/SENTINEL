"""Dynamic threat graph (design doc §3.3).

A typed property graph (NetworkX) augmented with a signature vector store for
similarity search. It is the agent's structured threat memory and the source of
the threat-behavior science: recurrence, migration, and defense-reuse all derive
from queries here.

Nodes:  Threat, Defense, Module, Outcome, AttackClass
Edges:  defeated_by, resembles, migrated_to, bypassed, evolved_into

Backend is in-process for reproducibility; ``snapshot``/``load`` persist to JSON.
A persistent graph DB (Kùzu/Neo4j) can be slotted behind the same query methods.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from ..core.types import BehavioralSignature, DefenseStrategy, Outcome, ThreatClass, ThreatEvent


class ThreatGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self._sig_ids: list[str] = []
        self._sig_matrix: list[np.ndarray] = []
        # ensure class nodes exist
        for c in ThreatClass:
            self.g.add_node(f"class:{c.value}", ntype="AttackClass", owasp=c.value)

    # ---------------------------------------------------------------- ingest
    def add_event(self, event: ThreatEvent, cycle: int) -> None:
        if not event.is_threat or event.threat_class is None:
            return
        tid = f"threat:{event.event_id}"
        self.g.add_node(
            tid,
            ntype="Threat",
            owasp=event.threat_class.value,
            confidence=event.confidence,
            cycle=cycle,
            timestamp=event.timestamp,
            channel=event.channel.value,
        )
        self.g.add_edge(tid, f"class:{event.threat_class.value}", etype="instance_of")
        if event.signature is not None:
            self._sig_ids.append(tid)
            self._sig_matrix.append(event.signature.vector())
            # link to k nearest prior threats by signature
            for nbr, sim in self.similar(event.signature, k=3):
                if nbr != tid:
                    self.g.add_edge(tid, nbr, etype="resembles", weight=float(sim))

    def record_outcome(
        self, event: ThreatEvent, plan_strategies: list[DefenseStrategy], outcome: Outcome
    ) -> None:
        tid = f"threat:{event.event_id}"
        if tid not in self.g:
            return
        for strat in plan_strategies:
            did = f"defense:{strat.value}"
            if did not in self.g:
                self.g.add_node(did, ntype="Defense", strategy=strat.value)
            etype = "bypassed" if outcome.attack_succeeded else "defeated_by"
            self.g.add_edge(tid, did, etype=etype, cycle=self.g.nodes[tid].get("cycle", 0))

    def record_migration(self, src: ThreatClass, dst: ThreatClass, cycle: int) -> None:
        self.g.add_edge(
            f"class:{src.value}", f"class:{dst.value}", etype="migrated_to", cycle=cycle
        )

    def record_evolution(self, parent_genome: str, child_genome: str) -> None:
        for gid in (parent_genome, child_genome):
            if f"genome:{gid}" not in self.g:
                self.g.add_node(f"genome:{gid}", ntype="Module")
        self.g.add_edge(f"genome:{parent_genome}", f"genome:{child_genome}", etype="evolved_into")

    # ------------------------------------------------------------- queries
    def similar(self, signature: BehavioralSignature, k: int = 5) -> list[tuple[str, float]]:
        """Cosine-similarity ANN over stored signatures (brute force at this scale)."""
        if not self._sig_matrix:
            return []
        q = signature.vector()
        mat = np.vstack(self._sig_matrix)
        # align dims (residual length can vary before fit); truncate to min
        d = min(q.shape[0], mat.shape[1])
        qd, md = q[:d], mat[:, :d]
        sims = md @ qd / (np.linalg.norm(md, axis=1) * np.linalg.norm(qd) + 1e-9)
        order = np.argsort(-sims)[:k]
        return [(self._sig_ids[i], float(sims[i])) for i in order]

    def recurrence(self, window: int | None = None) -> dict[str, list[int]]:
        """Per-class occurrence cycles (frequency over time) for recurrence curves."""
        out: dict[str, list[int]] = defaultdict(list)
        for _, data in self.g.nodes(data=True):
            if data.get("ntype") == "Threat":
                out[data["owasp"]].append(int(data.get("cycle", 0)))
        return {k: sorted(v) for k, v in out.items()}

    def class_distribution(self, cycle_lo: int, cycle_hi: int) -> dict[str, int]:
        c: Counter[str] = Counter()
        for _, data in self.g.nodes(data=True):
            if data.get("ntype") == "Threat" and cycle_lo <= data.get("cycle", -1) < cycle_hi:
                c[data["owasp"]] += 1
        return dict(c)

    def signature_drift_series(self, n_epochs: int = 4) -> dict[str, list[float]]:
        """Per-class behavioral drift over time: cosine distance between the mean
        signature of consecutive epochs (cycle-ordered). Feeds the drift figure."""
        from ..metrics.catalog import signature_drift

        if not self._sig_matrix:
            return {}
        d = min(v.shape[0] for v in self._sig_matrix)
        by_class: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for tid, vec in zip(self._sig_ids, self._sig_matrix, strict=True):
            data = self.g.nodes[tid]
            by_class[data["owasp"]].append((int(data.get("cycle", 0)), vec[:d]))
        out: dict[str, list[float]] = {}
        for cls, items in by_class.items():
            items.sort(key=lambda t: t[0])
            vecs = [v for _, v in items]
            if len(vecs) < 2 * 2:  # need >= 2 epochs of >= 2 samples
                continue
            k = min(n_epochs, len(vecs) // 2)
            splits = np.array_split(np.vstack(vecs), k)
            means = [s.mean(axis=0) for s in splits if len(s)]
            drifts = [signature_drift(a, b)["cosine_distance"]
                      for a, b in zip(means[:-1], means[1:], strict=False)]
            if drifts:
                out[cls] = drifts
        return out

    def defense_reuse(self) -> dict[str, int]:
        """How many distinct threat classes each defense has defeated (reuse ratio input)."""
        reuse: dict[str, set[str]] = defaultdict(set)
        for u, v, data in self.g.edges(data=True):
            if data.get("etype") == "defeated_by":
                owasp = self.g.nodes[u].get("owasp")
                strat = self.g.nodes[v].get("strategy")
                if owasp and strat:
                    reuse[strat].add(owasp)
        return {k: len(v) for k, v in reuse.items()}

    def bypass_counts(self) -> dict[str, int]:
        """Per-defense bypass counts — drives evolution's 'persistently bypassed' trigger."""
        c: Counter[str] = Counter()
        for _, v, data in self.g.edges(data=True):
            if data.get("etype") == "bypassed":
                strat = self.g.nodes[v].get("strategy")
                if strat:
                    c[strat] += 1
        return dict(c)

    # ------------------------------------------------------------- persist
    def snapshot(self, path: str | Path) -> None:
        data = nx.node_link_data(self.g)
        Path(path).write_text(json.dumps({"graph": data, "ts": time.time()}, default=str))

    @property
    def n_threats(self) -> int:
        return sum(1 for _, d in self.g.nodes(data=True) if d.get("ntype") == "Threat")
