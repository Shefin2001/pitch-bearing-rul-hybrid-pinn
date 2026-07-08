"""propagation.py -- failure-mode propagation forecast (severity + graph).

Reuses the tribology-validated PROGRESSION_GRAPH DAG (common/rul_labels.py,
Bhandare et al. 2024) and gives every edge a PHYSICS-DERIVED sojourn time:
the Paris-law time for the crack to grow from the source class anchor to the
destination class anchor under the destination's stress concentration K_t.

Semi-Markov outputs per window:
    next_stages : immediate successors with risk (faster edge = higher risk,
                  weighted by how much class-posterior mass sits on the node)
                  and an ETA distribution from the C-scatter + damage sigma
    terminal    : shortest-time path + cumulative ETA to the terminal IORW

The graph is DATA (editable without retraining) and must pass expert review
-- see validate.py's report section. Torch-free by design.
"""
from __future__ import annotations

import heapq
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.rul_labels import (  # noqa: E402
    FAULT_INDEX,
    INDEX_FAULT,
    PROGRESSION_GRAPH,
)
from Hybrid_PINN_ParisRUL.common.paris_labels import (  # noqa: E402
    A_MAP_M,
    CYCLE_SECONDS,
)
from Hybrid_PINN_ParisRUL.v3.paris_engine import (  # noqa: E402
    DELTA_SIGMA_MPA,
    SIGMA_LN_C_DEFAULT,
    paris_cycles_closed_form,
)

_Z90 = 1.6449


def edge_sojourn_seconds(src: str, dst: str,
                         a_start_m: Optional[float] = None) -> float:
    """Median Paris time to grow from src anchor (or a_start_m) to the dst
    anchor under the dst stress concentration."""
    a0 = A_MAP_M[src] if a_start_m is None else a_start_m
    a1 = A_MAP_M[dst]
    a0 = min(max(a0, 1e-6), a1 * 0.999)
    dsig = DELTA_SIGMA_MPA[FAULT_INDEX[dst]]
    n = float(paris_cycles_closed_form(a0, dsig, a_fail_m=a1))
    return n * CYCLE_SECONDS


def build_sojourn_table() -> Dict[str, Dict[str, float]]:
    """{src: {dst: median sojourn seconds}} for every DAG edge."""
    return {src: {dst: edge_sojourn_seconds(src, dst) for dst in succ}
            for src, succ in PROGRESSION_GRAPH.items()}


def shortest_time_to_terminal(start: str,
                              a_start_m: Optional[float] = None,
                              terminal: str = "IORW"
                              ) -> Tuple[float, List[str]]:
    """Dijkstra over the DAG with sojourn seconds as edge weights.
    Returns (seconds, path). (inf, []) if unreachable (start == terminal -> 0)."""
    dist = {start: 0.0}
    prev: Dict[str, str] = {}
    heap = [(0.0, start)]
    first = True
    while heap:
        d, node = heapq.heappop(heap)
        if node == terminal:
            path = [node]
            while node in prev:
                node = prev[node]
                path.append(node)
            return d, path[::-1]
        if d > dist.get(node, math.inf):
            continue
        for succ in PROGRESSION_GRAPH.get(node, []):
            w = edge_sojourn_seconds(node, succ,
                                     a_start_m if first and node == start else None)
            nd = d + w
            if nd < dist.get(succ, math.inf):
                dist[succ] = nd
                prev[succ] = node
                heapq.heappush(heap, (nd, succ))
        first = False
    return math.inf, []


def _eta_ci(median_s: float, log_a_sigma: float) -> Dict[str, float]:
    """ETA CI from the lognormal C scatter plus the damage-position sigma.

    Both act multiplicatively on the Paris time (C directly; a0 via the
    integrand ~ a^(1-m/2), locally exp(|1-m/2| * sigma_ln_a) for m=3 -> 0.5x).
    Combined log-sd: sqrt(sigma_lnC^2 + (0.5 * sigma_ln_a)^2)."""
    s = math.sqrt(SIGMA_LN_C_DEFAULT ** 2 + (0.5 * log_a_sigma) ** 2)
    return {"p5_h": median_s * math.exp(-_Z90 * s) / 3600.0,
            "p50_h": median_s / 3600.0,
            "p95_h": median_s * math.exp(_Z90 * s) / 3600.0}


def propagation_forecast(class_probs: np.ndarray,
                         log_a_mu: float,
                         log_a_sigma: float,
                         top_k: int = 3) -> Dict:
    """Full propagation intimation for one damage estimate.

    Risk over immediate successors: posterior-weighted over plausible current
    nodes (>= 5% mass), rate-weighted within a node (1/sojourn normalised).
    """
    p = np.asarray(class_probs, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    a_hat = float(np.exp(log_a_mu))
    current = INDEX_FAULT[int(p.argmax())]

    risk: Dict[str, float] = {}
    eta_s: Dict[str, float] = {}
    for ci, mass in enumerate(p):
        if mass < 0.05:
            continue
        node = INDEX_FAULT[ci]
        succ = PROGRESSION_GRAPH.get(node, [])
        if not succ:
            continue
        rates = {}
        for dst in succ:
            t = edge_sojourn_seconds(node, dst, a_start_m=a_hat)
            rates[dst] = 1.0 / max(t, 1.0)
            eta_s[dst] = min(eta_s.get(dst, math.inf), t)
        z = sum(rates.values())
        for dst, r in rates.items():
            risk[dst] = risk.get(dst, 0.0) + mass * r / z

    total = sum(risk.values())
    if total > 0:
        risk = {k: v / total for k, v in risk.items()}
    ranked = sorted(risk.items(), key=lambda kv: -kv[1])[:top_k]

    next_stages = [{"class": dst, "risk": float(round(float(r), 4)),
                    "eta_hours": _eta_ci(eta_s[dst], log_a_sigma)}
                   for dst, r in ranked]

    term_s, term_path = shortest_time_to_terminal(current, a_start_m=a_hat)
    terminal = ({"path": term_path, "eta_hours": _eta_ci(term_s, log_a_sigma)}
                if math.isfinite(term_s) else
                {"path": [current], "eta_hours": _eta_ci(0.0, log_a_sigma)})

    return {"current": current, "next_stages": next_stages,
            "terminal": terminal}


# ---------------------------------------------------------------------------
# Stage 13 CLI -- materialise the table for expert review
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser()
    default_dir = os.environ.get(
        "V3_RESULTS_DIR", str(ROOT / "Hybrid_PINN_ParisRUL" / "results" / "v3"))
    parser.add_argument("--out", default=str(Path(default_dir)
                                             / "propagation_table.json"))
    args = parser.parse_args()

    table = build_sojourn_table()
    doc = {"edges": [], "terminal_paths": {}}
    for src, dsts in table.items():
        for dst, sec in dsts.items():
            doc["edges"].append({
                "src": src, "dst": dst,
                "a_from_mm": A_MAP_M[src] * 1e3, "a_to_mm": A_MAP_M[dst] * 1e3,
                "sojourn_hours_p50": sec / 3600.0,
            })
    for cond in FAULT_INDEX:
        sec, path = shortest_time_to_terminal(cond)
        doc["terminal_paths"][cond] = {
            "path": path, "hours_p50": (sec / 3600.0 if math.isfinite(sec)
                                        else None)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    print(f"[v3:propagation] {len(doc['edges'])} edges -> {out}")

    print(f"{'src':<7}{'dst':<7}{'sojourn(h)':>12}")
    for e in sorted(doc["edges"], key=lambda e: e["sojourn_hours_p50"]):
        print(f"{e['src']:<7}{e['dst']:<7}{e['sojourn_hours_p50']:>12.1f}")


if __name__ == "__main__":
    # self-test then CLI
    t_h, path_h = shortest_time_to_terminal("Health")
    t_i, path_i = shortest_time_to_terminal("IORS")
    assert math.isfinite(t_h) and math.isfinite(t_i) and t_i < t_h
    assert path_h[0] == "Health" and path_h[-1] == "IORW"

    probs = np.zeros(len(FAULT_INDEX))
    probs[FAULT_INDEX["IRS"]] = 0.85
    probs[FAULT_INDEX["ORS"]] = 0.15
    fc = propagation_forecast(probs, math.log(1.6e-3), 0.3)
    assert fc["current"] == "IRS"
    assert 0.99 < sum(s["risk"] for s in fc["next_stages"]) <= 1.001
    assert all(s["eta_hours"]["p5_h"] <= s["eta_hours"]["p50_h"]
               <= s["eta_hours"]["p95_h"] for s in fc["next_stages"])
    print(f"[self-test] IRS forecast: next={[(s['class'], s['risk']) for s in fc['next_stages']]}")
    print(f"[self-test] terminal: {' -> '.join(fc['terminal']['path'])} "
          f"p50={fc['terminal']['eta_hours']['p50_h']:.0f}h")
    print("[OK] propagation self-test passed")
    main()
