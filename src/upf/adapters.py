from __future__ import annotations

from typing import Dict, Protocol, Tuple

import numpy as np

from .core import clip


class DomainAdapter(Protocol):
    name: str

    def normalize_inputs(self, raw: Dict[str, float]) -> Tuple[float, float]: ...
    def doc_from_raw(self, raw: Dict[str, float]) -> float: ...
    def interpret(self, state: Dict[str, np.ndarray], S_layers: np.ndarray) -> Dict[str, float]: ...


class FusionAdapter:
    name = "fusion"

    def normalize_inputs(self, raw):
        return clip(raw.get("confinement", 0.6), 0, 1), clip(raw.get("instability", 0.4), 0, 1)

    def doc_from_raw(self, raw):
        return clip(raw.get("impurities", 0.35), 0, 1)

    def interpret(self, state, S):
        return {
            "pellet_injection_rate": clip(100 * (state["E"][1] - 0.30), 0, 100),
            "rf_power_boost": clip(100 * (0.55 - state["C"][1]), 0, 100),
            "efficiency_score": clip(100 * (S.sum() / 2.6), 0, 100),
        }


class QuantumAdapter:
    name = "quantum"

    def normalize_inputs(self, raw):
        return clip(raw.get("pump_rate", 0.7), 0, 1), clip(raw.get("decoherence", 0.3), 0, 1)

    def doc_from_raw(self, raw):
        return clip(raw.get("temperature_noise", 0.25), 0, 1)

    def interpret(self, state, S):
        return {
            "coherence_time_ms": clip(500 * state["C"][1], 10, 2000),
            "stability_index": clip(100 * (1.0 - state["E"][1] / 2.0), 0, 100),
            "orch_or_score": clip(100 * state["C"][0], 0, 100),
        }
