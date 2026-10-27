"""Motor N-body 2D usado nos experimentos Barnes-Hut do projeto.

Esta versao preserva a formulacao validada no notebook 02:

- disco exponencial truncado de particulas de mesma massa;
- halo de Plummer fixo;
- autogravidade newtoniana com softening de Plummer;
- soma direta ou quadtree Barnes-Hut monopolar;
- integrador Leapfrog kick-drift-kick.

O modulo nao contem graficos ou animacoes. Ele recebe uma configuracao e
devolve arrays NumPy, mantendo a simulacao independente da apresentacao.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import numba
    from numba import njit

    NUMBA_AVAILABLE = True
    NUMBA_VERSION = numba.__version__
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_VERSION = "unavailable"

    def njit(*args: Any, **kwargs: Any):
        """Fallback que mantem o modulo importavel sem Numba."""

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


ENGINE_VERSION = "1.0.0"
G = 4.498502151575286e-12  # kpc^3 / (Msun Myr^2)
KPC_MYR_TO_KM_S = 977.79222168


@dataclass(frozen=True)
class SimulationConfig:
    """Parametros fisicos, numericos e de amostragem de uma simulacao."""

    n: int = 10_000
    m_halo: float = 2.0e11
    a_halo: float = 6.0
    m_disk: float = 4.0e10
    r_disk: float = 3.0
    r_min: float = 0.35
    r_max: float = 15.0
    velocity_dispersion_fraction: float = 0.025
    epsilon: float = 0.20
    dt: float = 0.50
    t_end: float = 500.0
    record_every: int = 10
    seed: int = 20261027
    theta: float = 0.50
    bucket_capacity: int = 8
    max_depth: int = 48

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n deve ser maior ou igual a 2")
        if self.m_halo <= 0.0 or self.m_disk <= 0.0:
            raise ValueError("m_halo e m_disk devem ser positivos")
        if self.a_halo <= 0.0 or self.r_disk <= 0.0:
            raise ValueError("a_halo e r_disk devem ser positivos")
        if not 0.0 <= self.r_min < self.r_max:
            raise ValueError("use 0 <= r_min < r_max")
        if self.velocity_dispersion_fraction < 0.0:
            raise ValueError("velocity_dispersion_fraction nao pode ser negativa")
        if self.epsilon <= 0.0 or self.dt <= 0.0 or self.t_end <= 0.0:
            raise ValueError("epsilon, dt e t_end devem ser positivos")
        if self.record_every < 1:
            raise ValueError("record_every deve ser maior ou igual a 1")
        if self.theta <= 0.0:
            raise ValueError("theta deve ser positivo")
        if self.bucket_capacity < 1 or self.max_depth < 1:
            raise ValueError("bucket_capacity e max_depth devem ser positivos")

    @property
    def n_steps(self) -> int:
        return int(round(self.t_end / self.dt))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def config_id(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:12]


def sample_exponential_disk(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Gera posicoes e velocidades iniciais do disco exponencial frio."""

    rng = np.random.default_rng(cfg.seed)
    radii: list[float] = []
    while len(radii) < cfg.n:
        draw = rng.gamma(shape=2.0, scale=cfg.r_disk, size=2 * cfg.n)
        accepted = draw[(draw >= cfg.r_min) & (draw <= cfg.r_max)]
        radii.extend(accepted.tolist())

    radius = np.asarray(radii[: cfg.n], dtype=np.float64)
    angle = rng.uniform(0.0, 2.0 * np.pi, cfg.n)
    pos = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))

    er = pos / radius[:, None]
    et = np.column_stack((-er[:, 1], er[:, 0]))

    vc2_halo = (
        G
        * cfg.m_halo
        * radius**2
        / (radius**2 + cfg.a_halo**2) ** 1.5
    )
    scaled_radius = radius / cfg.r_disk
    enclosed_disk = cfg.m_disk * (
        1.0 - np.exp(-scaled_radius) * (1.0 + scaled_radius)
    )
    vc = np.sqrt(np.maximum(vc2_halo + G * enclosed_disk / radius, 0.0))

    sigma = cfg.velocity_dispersion_fraction * vc
    v_radial = rng.normal(0.0, sigma)
    v_tangent = vc + rng.normal(0.0, sigma)
    vel = v_radial[:, None] * er + v_tangent[:, None] * et
    vel -= np.mean(vel, axis=0, keepdims=True)
    return pos, vel


def halo_acceleration_and_potential(
    pos: np.ndarray, m_halo: float, a_halo: float
) -> tuple[np.ndarray, np.ndarray]:
    """Aceleracao e potencial por unidade de massa do halo de Plummer."""

    r2 = np.sum(pos * pos, axis=1)
    denom = np.sqrt(r2 + a_halo * a_halo)
    acc = -G * m_halo * pos / denom[:, None] ** 3
    phi = -G * m_halo / denom
    return acc, phi


@njit(cache=True)
def _direct_disk_force_loop(pos, particle_mass, epsilon):
    n = pos.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)
    phi = np.zeros(n, dtype=np.float64)
    eps2 = epsilon * epsilon

    for i in range(n):
        ax = 0.0
        ay = 0.0
        potential = 0.0
        xi = pos[i, 0]
        yi = pos[i, 1]
        for j in range(n):
            if i == j:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            d2 = dx * dx + dy * dy + eps2
            distance = np.sqrt(d2)
            inv_r3 = 1.0 / (d2 * distance)
            ax += G * particle_mass * dx * inv_r3
            ay += G * particle_mass * dy * inv_r3
            potential -= G * particle_mass / distance
        acc[i, 0] = ax
        acc[i, 1] = ay
        phi[i] = potential
    return acc, phi


def _direct_disk_force_numpy(pos, particle_mass, epsilon):
    delta = pos[None, :, :] - pos[:, None, :]
    dist2 = np.sum(delta * delta, axis=2) + epsilon * epsilon
    inv_r3 = dist2**-1.5
    inv_r = dist2**-0.5
    np.fill_diagonal(inv_r3, 0.0)
    np.fill_diagonal(inv_r, 0.0)
    acc = G * particle_mass * np.sum(delta * inv_r3[:, :, None], axis=1)
    phi = -G * particle_mass * np.sum(inv_r, axis=1)
    return acc, phi


def direct_disk_force(pos, particle_mass, epsilon):
    if NUMBA_AVAILABLE:
        return _direct_disk_force_loop(pos, particle_mass, epsilon)
    return _direct_disk_force_numpy(pos, particle_mass, epsilon)


@njit(cache=True)
def _quadrant(x, y, center_x, center_y):
    quadrant = 0
    if x >= center_x:
        quadrant += 1
    if y >= center_y:
        quadrant += 2
    return quadrant


@njit(cache=True)
def _build_quadtree(pos, particle_mass, bucket_capacity, max_depth):
    n = pos.shape[0]
    max_nodes = max(32, 8 * n + 1)

    children = np.full((max_nodes, 4), -1, dtype=np.int64)
    is_leaf = np.ones(max_nodes, dtype=np.uint8)
    head = np.full(max_nodes, -1, dtype=np.int64)
    count = np.zeros(max_nodes, dtype=np.int64)
    next_particle = np.full(n, -1, dtype=np.int64)

    center_x = np.zeros(max_nodes, dtype=np.float64)
    center_y = np.zeros(max_nodes, dtype=np.float64)
    half_size = np.zeros(max_nodes, dtype=np.float64)
    mass = np.zeros(max_nodes, dtype=np.float64)
    com_x = np.zeros(max_nodes, dtype=np.float64)
    com_y = np.zeros(max_nodes, dtype=np.float64)

    xmin = np.min(pos[:, 0])
    xmax = np.max(pos[:, 0])
    ymin = np.min(pos[:, 1])
    ymax = np.max(pos[:, 1])
    span = max(xmax - xmin, ymax - ymin)
    center_x[0] = 0.5 * (xmin + xmax)
    center_y[0] = 0.5 * (ymin + ymax)
    half_size[0] = 0.5 * span * (1.0 + 1.0e-12) + 1.0e-12
    node_count = 1

    for particle in range(n):
        node = 0
        depth = 0
        while True:
            if is_leaf[node] == 1:
                if count[node] < bucket_capacity or depth >= max_depth:
                    next_particle[particle] = head[node]
                    head[node] = particle
                    count[node] += 1
                    break

                if node_count + 4 > max_nodes:
                    raise RuntimeError("Quadtree excedeu o numero maximo de nos")

                is_leaf[node] = 0
                child_half = 0.5 * half_size[node]
                for q in range(4):
                    child = node_count
                    node_count += 1
                    children[node, q] = child
                    half_size[child] = child_half
                    center_x[child] = center_x[node] + (
                        child_half if (q & 1) else -child_half
                    )
                    center_y[child] = center_y[node] + (
                        child_half if (q & 2) else -child_half
                    )

                old_particle = head[node]
                head[node] = -1
                count[node] = 0
                while old_particle != -1:
                    following = next_particle[old_particle]
                    q_old = _quadrant(
                        pos[old_particle, 0],
                        pos[old_particle, 1],
                        center_x[node],
                        center_y[node],
                    )
                    child = children[node, q_old]
                    next_particle[old_particle] = head[child]
                    head[child] = old_particle
                    count[child] += 1
                    old_particle = following
            else:
                q = _quadrant(
                    pos[particle, 0],
                    pos[particle, 1],
                    center_x[node],
                    center_y[node],
                )
                node = children[node, q]
                depth += 1

    for node in range(node_count - 1, -1, -1):
        if is_leaf[node] == 1:
            particle = head[node]
            total_mass = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            while particle != -1:
                total_mass += particle_mass
                weighted_x += particle_mass * pos[particle, 0]
                weighted_y += particle_mass * pos[particle, 1]
                particle = next_particle[particle]
            mass[node] = total_mass
            if total_mass > 0.0:
                com_x[node] = weighted_x / total_mass
                com_y[node] = weighted_y / total_mass
        else:
            total_mass = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            for q in range(4):
                child = children[node, q]
                child_mass = mass[child]
                total_mass += child_mass
                weighted_x += child_mass * com_x[child]
                weighted_y += child_mass * com_y[child]
            mass[node] = total_mass
            if total_mass > 0.0:
                com_x[node] = weighted_x / total_mass
                com_y[node] = weighted_y / total_mass

    return (
        children,
        is_leaf,
        head,
        next_particle,
        center_x,
        center_y,
        half_size,
        mass,
        com_x,
        com_y,
        node_count,
    )


@njit(cache=True)
def _tree_disk_force_core(
    pos, particle_mass, epsilon, theta, bucket_capacity, max_depth
):
    (
        children,
        is_leaf,
        head,
        next_particle,
        center_x,
        center_y,
        half_size,
        mass,
        com_x,
        com_y,
        node_count,
    ) = _build_quadtree(pos, particle_mass, bucket_capacity, max_depth)

    n = pos.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)
    phi = np.zeros(n, dtype=np.float64)
    eps2 = epsilon * epsilon
    stack = np.empty(node_count, dtype=np.int64)

    for i in range(n):
        stack_size = 1
        stack[0] = 0
        ax = 0.0
        ay = 0.0
        potential = 0.0
        xi = pos[i, 0]
        yi = pos[i, 1]

        while stack_size > 0:
            stack_size -= 1
            node = stack[stack_size]
            if mass[node] <= 0.0:
                continue

            if is_leaf[node] == 1:
                j = head[node]
                while j != -1:
                    if j != i:
                        dx = pos[j, 0] - xi
                        dy = pos[j, 1] - yi
                        d2 = dx * dx + dy * dy + eps2
                        distance = np.sqrt(d2)
                        inv_r3 = 1.0 / (d2 * distance)
                        ax += G * particle_mass * dx * inv_r3
                        ay += G * particle_mass * dy * inv_r3
                        potential -= G * particle_mass / distance
                    j = next_particle[j]
                continue

            dx = com_x[node] - xi
            dy = com_y[node] - yi
            distance_geom = np.sqrt(dx * dx + dy * dy) + 1.0e-30
            contains_target = (
                abs(xi - center_x[node]) <= half_size[node]
                and abs(yi - center_y[node]) <= half_size[node]
            )
            accept = (
                not contains_target
                and 2.0 * half_size[node] / distance_geom < theta
            )

            if accept:
                d2 = dx * dx + dy * dy + eps2
                distance = np.sqrt(d2)
                inv_r3 = 1.0 / (d2 * distance)
                ax += G * mass[node] * dx * inv_r3
                ay += G * mass[node] * dy * inv_r3
                potential -= G * mass[node] / distance
            else:
                for q in range(4):
                    child = children[node, q]
                    if child >= 0 and mass[child] > 0.0:
                        stack[stack_size] = child
                        stack_size += 1

        acc[i, 0] = ax
        acc[i, 1] = ay
        phi[i] = potential

    return acc, phi, node_count


def tree_disk_force(
    pos: np.ndarray,
    particle_mass: float,
    epsilon: float,
    theta: float,
    bucket_capacity: int = 8,
    max_depth: int = 48,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not NUMBA_AVAILABLE and pos.shape[0] > 2000:
        raise RuntimeError("Para N>2000, instale o Numba para executar o Tree Code")
    return _tree_disk_force_core(
        pos, particle_mass, epsilon, theta, bucket_capacity, max_depth
    )


def total_force(pos: np.ndarray, cfg: SimulationConfig, method: str):
    particle_mass = cfg.m_disk / cfg.n
    halo_acc, halo_phi = halo_acceleration_and_potential(
        pos, cfg.m_halo, cfg.a_halo
    )
    if method == "direct":
        disk_acc, disk_phi = direct_disk_force(pos, particle_mass, cfg.epsilon)
        node_count = 0
    elif method == "tree":
        disk_acc, disk_phi, node_count = tree_disk_force(
            pos,
            particle_mass,
            cfg.epsilon,
            cfg.theta,
            cfg.bucket_capacity,
            cfg.max_depth,
        )
    else:
        raise ValueError("method deve ser 'direct' ou 'tree'")
    return halo_acc + disk_acc, halo_phi, disk_phi, node_count


def compute_diagnostics(
    pos: np.ndarray,
    vel: np.ndarray,
    halo_phi: np.ndarray,
    disk_phi: np.ndarray,
    cfg: SimulationConfig,
) -> tuple[float, float, np.ndarray, float, float, float, float]:
    particle_mass = cfg.m_disk / cfg.n
    kinetic = 0.5 * particle_mass * np.sum(vel * vel)
    potential_halo = particle_mass * np.sum(halo_phi)
    potential_disk = 0.5 * particle_mass * np.sum(disk_phi)
    energy = kinetic + potential_halo + potential_disk
    lz = particle_mass * np.sum(
        pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    )

    radius = np.linalg.norm(pos, axis=1)
    angle = np.arctan2(pos[:, 1], pos[:, 0])
    a1 = abs(np.mean(np.exp(1j * angle)))
    a2 = abs(np.mean(np.exp(2j * angle)))
    return (
        float(energy),
        float(lz),
        np.quantile(radius, [0.1, 0.5, 0.9]),
        float(np.mean(radius > 15.0)),
        float(np.mean(radius > 20.0)),
        float(a1),
        float(a2),
    )


def run_simulation(
    cfg: SimulationConfig,
    method: str = "tree",
    progress_every: int = 100,
) -> dict[str, np.ndarray]:
    """Executa uma simulacao e devolve snapshots e diagnosticos."""

    pos, vel = sample_exponential_disk(cfg)
    records: dict[str, list[Any]] = {
        "time": [],
        "positions": [],
        "velocities": [],
        "energy": [],
        "lz": [],
        "r_quantiles": [],
        "frac_gt15": [],
        "frac_gt20": [],
        "a1": [],
        "a2": [],
        "node_count": [],
    }

    def record(step, halo_phi, disk_phi, node_count):
        values = compute_diagnostics(pos, vel, halo_phi, disk_phi, cfg)
        records["time"].append(step * cfg.dt)
        records["positions"].append(pos.astype(np.float32))
        records["velocities"].append(vel.astype(np.float32))
        records["energy"].append(values[0])
        records["lz"].append(values[1])
        records["r_quantiles"].append(values[2])
        records["frac_gt15"].append(values[3])
        records["frac_gt20"].append(values[4])
        records["a1"].append(values[5])
        records["a2"].append(values[6])
        records["node_count"].append(node_count)

    start = time.perf_counter()
    acc, halo_phi, disk_phi, node_count = total_force(pos, cfg, method)
    record(0, halo_phi, disk_phi, node_count)

    for step in range(1, cfg.n_steps + 1):
        vel_half = vel + 0.5 * cfg.dt * acc
        pos = pos + cfg.dt * vel_half
        acc_new, halo_phi, disk_phi, node_count = total_force(pos, cfg, method)
        vel = vel_half + 0.5 * cfg.dt * acc_new
        acc = acc_new

        if step % cfg.record_every == 0 or step == cfg.n_steps:
            record(step, halo_phi, disk_phi, node_count)
        if progress_every and step % progress_every == 0:
            print(f"{method}: passo {step}/{cfg.n_steps}")

    result = {key: np.asarray(value) for key, value in records.items()}
    result["runtime"] = np.asarray(time.perf_counter() - start)
    result["method"] = np.asarray(method)
    result["config_json"] = np.asarray(cfg.to_json())
    result["config_id"] = np.asarray(cfg.config_id)
    result["engine_version"] = np.asarray(ENGINE_VERSION)
    result["created_utc"] = np.asarray(datetime.now(timezone.utc).isoformat())
    result["python_version"] = np.asarray(platform.python_version())
    result["numpy_version"] = np.asarray(np.__version__)
    result["numba_version"] = np.asarray(NUMBA_VERSION)
    return result


def result_filename(method: str, cfg: SimulationConfig) -> str:
    theta_part = f"_theta{cfg.theta:.2f}" if method == "tree" else ""
    sigma_part = f"_sig{cfg.velocity_dispersion_fraction:.3f}"
    return (
        f"resultado_{method}_N{cfg.n}{theta_part}_eps{cfg.epsilon:.2f}"
        f"_dt{cfg.dt:.2f}_T{cfg.t_end:.0f}{sigma_part}_seed{cfg.seed}"
        f"_{cfg.config_id}.npz"
    )


def save_result(result: dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return path


def load_result(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def force_comparison(
    cfg: SimulationConfig,
    theta_values: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0),
) -> list[dict[str, float | int]]:
    """Compara uma avaliacao direta com diferentes aberturas da arvore."""

    pos, _ = sample_exponential_disk(cfg)
    particle_mass = cfg.m_disk / cfg.n

    start = time.perf_counter()
    acc_direct, _ = direct_disk_force(pos, particle_mass, cfg.epsilon)
    direct_time = time.perf_counter() - start

    rows: list[dict[str, float | int]] = []
    direct_norm = np.linalg.norm(acc_direct, axis=1)
    scale = max(np.median(direct_norm) * 1.0e-12, 1.0e-30)
    for theta in theta_values:
        start = time.perf_counter()
        acc_tree, _, node_count = tree_disk_force(
            pos,
            particle_mass,
            cfg.epsilon,
            theta,
            cfg.bucket_capacity,
            cfg.max_depth,
        )
        tree_time = time.perf_counter() - start
        relative = np.linalg.norm(acc_tree - acc_direct, axis=1) / (
            direct_norm + scale
        )
        rows.append(
            {
                "theta": theta,
                "median": float(np.median(relative)),
                "p95": float(np.quantile(relative, 0.95)),
                "maximum": float(np.max(relative)),
                "direct_time": direct_time,
                "tree_time": tree_time,
                "speedup": direct_time / tree_time,
                "node_count": int(node_count),
            }
        )
    return rows


def warmup_numba() -> float:
    """Compila as rotinas Numba sem contaminar a medicao de desempenho."""

    cfg = SimulationConfig(
        n=128,
        epsilon=0.50,
        t_end=1.0,
        dt=0.50,
        record_every=2,
    )
    start = time.perf_counter()
    run_simulation(cfg, method="tree", progress_every=0)
    return time.perf_counter() - start


def benchmark_config(
    base: SimulationConfig, benchmark_myr: float
) -> SimulationConfig:
    """Cria uma configuracao curta, registrando apenas inicio e fim."""

    n_steps = int(round(benchmark_myr / base.dt))
    if n_steps < 1:
        raise ValueError("benchmark_myr deve conter pelo menos um passo")
    return replace(base, t_end=n_steps * base.dt, record_every=n_steps)

