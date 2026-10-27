"""Benchmark local do motor Barnes-Hut.

Execute a partir da raiz do repositorio:

    python scripts/benchmark.py

O aquecimento do Numba e separado da medicao. Por padrao, o programa mede
25 Myr (50 passos) com N=10000 e estima o custo de uma simulacao de 500 Myr.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from minigalaxy.bh_engine_v1 import (  # noqa: E402
    ENGINE_VERSION,
    NUMBA_AVAILABLE,
    NUMBA_VERSION,
    SimulationConfig,
    benchmark_config,
    run_simulation,
    warmup_numba,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mede o desempenho local do Barnes-Hut sem gerar GIF."
    )
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--benchmark-myr", type=float, default=25.0)
    parser.add_argument("--full-myr", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.50)
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats deve ser pelo menos 1")

    print("=" * 68)
    print("GalaxyDynamicsLab - benchmark Barnes-Hut")
    print("=" * 68)
    print(f"Motor:       {ENGINE_VERSION}")
    print(f"Python:      {platform.python_version()}")
    print(f"NumPy:       {np.__version__}")
    print(f"Numba:       {NUMBA_VERSION}")
    print(f"CPU:         {platform.processor() or platform.machine()}")
    print(f"CPU logicas: {os.cpu_count()}")
    print()

    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba nao esta instalado; execute pip install -r requirements.txt")

    print("Aquecendo/compilando o Numba...")
    warmup_time = warmup_numba()
    print(f"Aquecimento concluido em {warmup_time:.2f} s (fora da medicao).")

    base = SimulationConfig(
        n=args.n,
        epsilon=args.epsilon,
        theta=args.theta,
        dt=args.dt,
        t_end=args.full_myr,
    )
    cfg = benchmark_config(base, args.benchmark_myr)

    print()
    print(
        f"Teste: N={cfg.n}, theta={cfg.theta}, epsilon={cfg.epsilon} kpc, "
        f"dt={cfg.dt} Myr, T={cfg.t_end} Myr, passos={cfg.n_steps}"
    )

    runtimes: list[float] = []
    for repeat in range(1, args.repeats + 1):
        result = run_simulation(cfg, method="tree", progress_every=0)
        runtime = float(result["runtime"])
        runtimes.append(runtime)
        print(
            f"Repeticao {repeat}/{args.repeats}: {runtime:.3f} s "
            f"({runtime / cfg.n_steps:.4f} s/passo)"
        )

    median_runtime = statistics.median(runtimes)
    estimated_runtime = median_runtime * args.full_myr / cfg.t_end

    print()
    print("-" * 68)
    print(f"Mediana do teste:             {median_runtime:.3f} s")
    print(f"Estimativa para {args.full_myr:g} Myr:       {estimated_runtime:.1f} s")
    print(f"Estimativa em minutos:        {estimated_runtime / 60.0:.2f} min")
    print("-" * 68)
    print(
        "A estimativa supoe custo aproximadamente linear no numero de passos; "
        "a simulacao completa pode variar com a evolucao da arvore e a temperatura da CPU."
    )


if __name__ == "__main__":
    main()

