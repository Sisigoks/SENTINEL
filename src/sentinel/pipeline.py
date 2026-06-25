"""System assembly — build a fitted SENTINEL stack from a config.

Fits the Sentinel cascade (anomaly screen, classifier, signature residual) on the
benign reference set + the *seen* portion of the probe corpus, then returns a ready
:class:`SentinelAgent`. This is the one place model + encoder + corpus come together.
"""

from __future__ import annotations

import numpy as np

from .agent import SentinelAgent
from .core.logging import get_logger
from .corpora.loaders import ProbeCorpus, benign_corpus
from .graph.threat_graph import ThreatGraph
from .models.backend import ModelBackend
from .models.encoder import FrozenEncoder
from .sentinel_layer.anomaly_screen import AnomalyScreen
from .sentinel_layer.cascade import SentinelCascade
from .sentinel_layer.classifier import NeuralThreatClassifier
from .sentinel_layer.rule_screen import RuleScreen
from .sentinel_layer.signature import SignatureExtractor

log = get_logger(__name__)


def fit_cascade(
    encoder: FrozenEncoder, corpus: ProbeCorpus, *, n_benign: int = 60, seed: int = 0
) -> SentinelCascade:
    """Fit all learned cascade components on benign refs + seen probes."""
    benign_texts = benign_corpus(n_benign)
    seen = corpus.seen()
    threat_texts = [p.text for p in seen]
    threat_labels = [p.threat_class.value for p in seen]

    benign_emb = encoder.encode(benign_texts)
    threat_emb = encoder.encode(threat_texts) if threat_texts else benign_emb[:1]

    # anomaly screen on benign manifold
    anomaly = AnomalyScreen(random_state=seed)
    anomaly.fit(benign_emb)

    # classifier on benign + seen threats (BENIGN vs OWASP classes)
    clf = NeuralThreatClassifier(random_state=seed)
    X = np.vstack([benign_emb, threat_emb])
    y = ["BENIGN"] * len(benign_emb) + threat_labels
    clf.fit(X, y)

    # signature residual PCA
    sig = SignatureExtractor(residual_dim=min(16, threat_emb.shape[0] - 1 if threat_emb.shape[0] > 1 else 1))
    sig.fit(threat_emb, benign_emb)

    log.info("cascade fitted", n_benign=len(benign_emb), n_threat=len(threat_emb),
             metrics=clf.macro_metrics(threat_emb, threat_labels) if threat_texts else {})
    return SentinelCascade(encoder, RuleScreen(), anomaly, clf, sig)


def build_agent(
    backend: ModelBackend, encoder: FrozenEncoder, corpus: ProbeCorpus, *, seed: int = 0
) -> SentinelAgent:
    from .substrate.fgae import ReferenceFGAE

    cascade = fit_cascade(encoder, corpus, seed=seed)
    graph = ThreatGraph()
    return SentinelAgent(ReferenceFGAE(backend), cascade, graph)
