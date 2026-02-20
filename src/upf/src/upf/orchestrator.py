from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .core import PureUniversalCore, tanh, clip
from .adapters import DomainAdapter


@dataclass
class HomeostasisConfig:
    S_target: float = 1.92
    eta: float = 0.065
    zeta: float = 0.038
    eta_neg: float = 0.052
    S_band: float = 0.22


class UniversalOrchestrator:
    def __init__(self, core: PureUniversalCore, adapter: DomainAdapter, homeo: HomeostasisConfig):
        self.core = core
        self.adapter = adapter
        self.hcfg = homeo
        self.u_pos = 0.5
        self.u_neg = 0.5

    def _homeostasis(self, S_sum: float):
        err = self.hcfg.S_target - S_sum
        band = self.hcfg.S_band

        du_pos = self.hcfg.eta * err if abs(err) > band else 0.5 * self.hcfg.eta * err
        du_pos -= self.hcfg.zeta * (self.u_pos - 0.5)

        du_neg = -self.hcfg.eta_neg * max(0.0, err) + 0.5 * self.hcfg.zeta * (0.5 - self.u_neg)

        self.u_pos = clip(self.u_pos + du_pos, 0.0, 1.0)
        self.u_neg = clip(self.u_neg + du_neg, 0.0, 1.0)

    def step(self, raw: Dict[str, float]) -> Dict[str, Any]:
        u_pos, u_neg = self.adapter.normalize_inputs(raw)
        DOC = self.adapter.doc_from_raw(raw)

        blend = 0.42
        u_pos_used = (1 - blend) * self.u_pos + blend * u_pos
        u_neg_used = (1 - blend) * self.u_neg + blend * u_neg

        drive = tanh(2.0 * (u_pos_used - self.core.cfg.alpha * u_neg_used))

        S_layers = self.core.step(drive, DOC)
        S_sum = float(np.sum(S_layers))

        self._homeostasis(S_sum)

        state = self.core.state_dict()
        actions = self.adapter.interpret(state, S_layers)

        return {
            "domain": self.adapter.name,
            "DOC": float(DOC),
            "S_layers": S_layers,
            "S_sum": S_sum,
            "u_pos_used": float(u_pos_used),
            "u_neg_used": float(u_neg_used),
            "u_pos_policy": float(self.u_pos),
            "u_neg_policy": float(self.u_neg),
            "actions": actions,
            "state": state,
        }
