"""Publication-quality figures."""

from .figures import (
    plot_ablation,
    plot_asr_curves,
    plot_cross_transfer_matrix,
    plot_defense_evolution,
    plot_migration_heatmap,
    plot_recurrence,
    plot_security_utility_pareto,
    plot_signature_drift,
    plot_stability,
    plot_transfer_bars,
    set_paper_style,
)

__all__ = [
    "set_paper_style",
    "plot_asr_curves",
    "plot_migration_heatmap",
    "plot_defense_evolution",
    "plot_recurrence",
    "plot_security_utility_pareto",
    "plot_cross_transfer_matrix",
    "plot_signature_drift",
    "plot_stability",
    "plot_ablation",
    "plot_transfer_bars",
]
