from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def tanh(x: float) -> float:
    return float(np.tanh(x))


def clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class UPFConfig:
    levels: int = 3
    phi: float = (1 + 5 ** 0.5) / 2

    gE: float = 0.01
    gC: float = 0.02
    gR: float = 0.01

    leak_E: float = 0.02
    leak_C: float = 0.02
    leak_R: float = 0.02

    k: float = 0.55
    beta: float = 0.18
    gamma: float = 0.12

    alpha: float = 1.0
    kappa: float = 0.04

    E_base: float = 0.45
    scale_direction: str = "up"  # "up": layer 0 strongest

    process_noise_sigma: float = 0.008

    E_min: float = 0.01
    E_max: float = 2.0
    C_min: float = 0.0
    C_max: float = 1.0
    rho_min: float = 0.1
    rho_max: float = 5.0


class PureUniversalCore:
    """
    Stable, bounded, N-layer projection field core.

    States per layer:
      rho: capacity/flow
      C:   coherence/coupling
      E:   entropy/noise proxy

    Inputs:
      drive in [-1,1]
      DOC in [0,1] (degradation / throughput loss)
    """

    def __init__(self, cfg: UPFConfig | None = None, seed: int = 137):
        self.cfg = cfg or UPFConfig()
        self.rng = np.random.default_rng(seed)
        L = self.cfg.levels

        self.rho = np.ones(L) * 1.0
        self.C = np.ones(L) * 0.62
        self.E = np.ones(L) * 0.38

        self.rho0 = self.rho.copy()
        self.C0 = self.C.copy()
        self.E0 = self.E.copy()

        self.last_S = np.zeros(L)

    @staticmethod
    def noise_assist_boost(E: float, E_opt: float) -> float:
        x = np.clip(E / max(E_opt, 1e-9), 1e-6, 50.0)
        return float(x * np.exp(1 - x))

    def _scale_factor(self, ell: int) -> float:
        if self.cfg.scale_direction == "up":
            return float(self.cfg.phi ** (self.cfg.levels - 1 - ell))
        return float(self.cfg.phi ** ell)

    def _E_opt_level(self, ell: int, DOC: float) -> float:
        base = self.cfg.E_base / (self.cfg.phi ** ell)
        return float(max(0.02, base * (1.0 - 0.25 * DOC)))

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"rho": self.rho.copy(), "C": self.C.copy(), "E": self.E.copy(), "S": self.last_S.copy()}

    def step(self, drive: float, DOC: float = 0.0) -> np.ndarray:
        DOC = clip(DOC, 0.0, 1.0)
        M_eff = 1.0 - DOC

        S_prev = np.array(
            [
                (self.rho[i] * self.C[i] / (1.0 + self.E[i])) * self._scale_factor(i) * M_eff
                for i in range(self.cfg.levels)
            ],
            dtype=float,
        )

        S_layers = np.zeros(self.cfg.levels, dtype=float)

        for ell in range(self.cfg.levels):
            boost = self.noise_assist_boost(self.E[ell], self._E_opt_level(ell, DOC))

            # diffusive coupling
            coupling = 0.0
            if self.cfg.kappa != 0.0:
                if ell > 0:
                    coupling += (S_prev[ell - 1] - S_prev[ell])
                if ell < self.cfg.levels - 1:
                    coupling += (S_prev[ell + 1] - S_prev[ell])
                coupling *= self.cfg.kappa

            delta_S = (
                self.cfg.k * drive * self.rho[ell] * self.C[ell]
                + self.cfg.beta * boost
                + self.cfg.gamma * M_eff
                + coupling
                + self.rng.normal(0.0, self.cfg.process_noise_sigma)
            )

            self.E[ell] = clip(
                self.E[ell] - self.cfg.gE * delta_S + self.cfg.leak_E * (self.E0[ell] - self.E[ell]),
                self.cfg.E_min,
                self.cfg.E_max,
            )
            self.C[ell] = clip(
                self.C[ell] + self.cfg.gC * delta_S + self.cfg.leak_C * (self.C0[ell] - self.C[ell]),
                self.cfg.C_min,
                self.cfg.C_max,
            )
            self.rho[ell] = clip(
                self.rho[ell] + self.cfg.gR * delta_S + self.cfg.leak_R * (self.rho0[ell] - self.rho[ell]),
                self.cfg.rho_min,
                self.cfg.rho_max,
            )

            S_layers[ell] = (self.rho[ell] * self.C[ell] / (1.0 + self.E[ell])) * self._scale_factor(ell) * M_eff

        self.last_S = S_layers.copy()
        return S_layers
