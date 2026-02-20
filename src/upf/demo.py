from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from .core import UPFConfig, PureUniversalCore, clip
from .orchestrator import HomeostasisConfig, UniversalOrchestrator
from .adapters import FusionAdapter, QuantumAdapter


def _raw_sample(domain: str, rng: np.random.Generator) -> Dict[str, float]:
    if domain == "fusion":
        return {
            "confinement": clip(0.55 + 0.15 * rng.normal(), 0, 1),
            "instability": clip(0.45 + 0.18 * rng.normal(), 0, 1),
            "impurities": clip(0.32 + 0.12 * rng.normal(), 0, 1),
        }
    if domain == "quantum":
        return {
            "pump_rate": clip(0.68 + 0.12 * rng.normal(), 0, 1),
            "decoherence": clip(0.28 + 0.14 * rng.normal(), 0, 1),
            "temperature_noise": clip(0.22 + 0.10 * rng.normal(), 0, 1),
        }
    return {"pos": 0.6, "neg": 0.4, "doc": 0.3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["fusion", "quantum"], default="quantum")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cfg = UPFConfig()
    homeo = HomeostasisConfig()
    core = PureUniversalCore(cfg, seed=args.seed)

    adapter = FusionAdapter() if args.domain == "fusion" else QuantumAdapter()
    orch = UniversalOrchestrator(core, adapter, homeo)

    rng = np.random.default_rng(args.seed)
    history: List[dict] = []

    for t in range(args.steps):
        raw = _raw_sample(args.domain, rng)
        out = orch.step(raw)
        out["t"] = t
        out["raw"] = raw
        history.append(out)

    last = history[-1]
    print(f"domain={last['domain']} DOC={last['DOC']:.3f} S_sum={last['S_sum']:.3f} S_layers={last['S_layers']}")
    print("actions:", last["actions"])

    if args.plot:
        import matplotlib.pyplot as plt

        t = np.arange(args.steps)
        S_sum = np.array([h["S_sum"] for h in history], dtype=float)
        S_layers = np.stack([h["S_layers"] for h in history], axis=0)

        plt.figure(figsize=(12, 4))
        plt.plot(t, S_sum)
        plt.title(f"S_sum over time – {args.domain}")
        plt.xlabel("t")
        plt.ylabel("S_sum")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.figure(figsize=(12, 5))
        for ell in range(S_layers.shape[1]):
            plt.plot(t, S_layers[:, ell], label=f"Layer {ell}")
        plt.title(f"S_layers over time – {args.domain}")
        plt.xlabel("t")
        plt.ylabel("S_layer")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
