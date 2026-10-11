"""
VoltRoute — Optimal Delivery and Charging Planning for EV-Based Delivery Partners
Flask backend: Exact Branch & Bound route optimizer with Dijkstra road-network
routing, battery-threshold-aware charging decisions, and verifiable global optimality.
"""

from flask import Flask, render_template, request, jsonify
from math import hypot, ceil, factorial
from itertools import count, permutations
import heapq

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Illustrative city network & graph definition
# ---------------------------------------------------------------------------
KM_PER_PIXEL = 0.075

NODES = {
    "Depot":    {"x":  70, "y": 265, "kind": "depot",    "name": "Depot"},
    "Pharmacy": {"x": 145, "y":  85, "kind": "delivery", "name": "Pharmacy", "order_id": "A"},
    "Office":   {"x": 225, "y": 185, "kind": "delivery", "name": "Office",   "order_id": "B"},
    "Grocery":  {"x": 340, "y": 275, "kind": "delivery", "name": "Grocery",  "order_id": "C"},
    "CH-1":     {"x": 445, "y": 165, "kind": "charger",  "name": "CH-1", "rate_kw": 22},
    "CH-2":     {"x": 300, "y":  70, "kind": "charger",  "name": "CH-2", "rate_kw": 22},
    "CH-3":     {"x": 510, "y":  75, "kind": "charger",  "name": "CH-3", "rate_kw": 50},
    "Hub":      {"x": 635, "y": 245, "kind": "hub",       "name": "Hub"},
}

ORDERS = {
    "A": {"id": "A", "name": "Pharmacy", "destination": "Pharmacy",
          "weight_kg": 2.0, "priority": 5, "service_min": 4, "deadline_min": 35},
    "B": {"id": "B", "name": "Office",   "destination": "Office",
          "weight_kg": 1.0, "priority": 3, "service_min": 4, "deadline_min": 50},
    "C": {"id": "C", "name": "Grocery",  "destination": "Grocery",
          "weight_kg": 4.0, "priority": 4, "service_min": 4, "deadline_min": 45},
}

CHARGERS = ["CH-1", "CH-2", "CH-3"]

# Road network edges corresponding to the simulated street map
ROAD_EDGES = [
    ("Depot", "Pharmacy"),
    ("Pharmacy", "Office"),
    ("Office", "Grocery"),
    ("Grocery", "CH-1"),
    ("CH-1", "Hub"),
    ("Depot", "Office"),
    ("Office", "CH-2"),
    ("CH-2", "Pharmacy"),
    ("CH-2", "CH-3"),
    ("CH-3", "Hub"),
    ("Grocery", "CH-3"),
    ("CH-3", "CH-1"),
    ("Pharmacy", "Grocery"),
    ("Office", "CH-1"),
    ("Depot", "CH-2"),
]


def edge_euclidean_distance(u: str, v: str) -> float:
    """Straight-line road distance (km) between two adjacent nodes."""
    pu, pv = NODES[u], NODES[v]
    return round(hypot(pu["x"] - pv["x"], pu["y"] - pv["y"]) * KM_PER_PIXEL, 3)


# Build adjacency map with edge weights
ROAD_ADJ = {k: {} for k in NODES}
for _u, _v in ROAD_EDGES:
    _d = edge_euclidean_distance(_u, _v)
    ROAD_ADJ[_u][_v] = _d
    ROAD_ADJ[_v][_u] = _d


# ---------------------------------------------------------------------------
# Dijkstra's Algorithm for Shortest Paths on Road Network
# ---------------------------------------------------------------------------

def dijkstra(start: str, end: str) -> tuple[float, list[str]]:
    """
    Dijkstra's shortest path algorithm for non-negative road edge distances.
    Returns (shortest_distance_km, list_of_vertex_names).
    """
    if start == end:
        return 0.0, [start]

    dist = {k: float("inf") for k in NODES}
    prev = {k: None for k in NODES}
    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            break
        for v, w in ROAD_ADJ[u].items():
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    path = []
    curr = end
    while curr:
        path.append(curr)
        curr = prev[curr]
    path.reverse()
    return round(dist[end], 3), path


# Precompute all-pairs shortest paths for fast lookups
_ALL_SHORTEST_DIST = {}
_ALL_SHORTEST_PATH = {}
for _n1 in NODES:
    for _n2 in NODES:
        _d, _p = dijkstra(_n1, _n2)
        _ALL_SHORTEST_DIST[(_n1, _n2)] = _d
        _ALL_SHORTEST_PATH[(_n1, _n2)] = _p


def shortest_distance(a: str, b: str) -> float:
    """Shortest road distance (km) between nodes a and b using Dijkstra."""
    return _ALL_SHORTEST_DIST.get((a, b), dijkstra(a, b)[0])


def shortest_path(a: str, b: str) -> list[str]:
    """Sequence of road vertices on the Dijkstra shortest path between a and b."""
    return _ALL_SHORTEST_PATH.get((a, b), dijkstra(a, b)[1])


def distance(a: str, b: str) -> float:
    """Road distance between two named nodes (computed via Dijkstra)."""
    return shortest_distance(a, b)


def route_distance(route: list) -> float:
    """Total road distance for Depot → *route → Hub."""
    seq = ["Depot"] + route + ["Hub"]
    return round(sum(distance(seq[i], seq[i + 1]) for i in range(len(seq) - 1)), 3)


# ---------------------------------------------------------------------------
# Minimum Spanning Tree (MST) Lower Bound (Prim's Algorithm)
# ---------------------------------------------------------------------------

def mst_cost(nodes: list) -> float:
    """Minimum spanning tree cost over a set of node names using Prim's algorithm."""
    if len(nodes) <= 1:
        return 0.0
    remaining = set(nodes)
    first = next(iter(remaining))
    remaining.remove(first)
    best = {node: distance(first, node) for node in remaining}
    total = 0.0
    while remaining:
        node = min(remaining, key=lambda n: best[n])
        total += best[node]
        remaining.remove(node)
        for other in remaining:
            best[other] = min(best[other], distance(node, other))
    return round(total, 3)


def lower_bound(current: str, remaining: list, end: str = "Hub") -> float:
    """
    Admissible B&B lower bound for remaining deliveries.
    LB = min(current→remaining) + MST(remaining) + min(remaining→end)
    Never overestimates the true completion distance.
    """
    if not remaining:
        return distance(current, end)
    return round(
        min(distance(current, node) for node in remaining)
        + mst_cost(remaining)
        + min(distance(node, end) for node in remaining),
        3
    )


# ---------------------------------------------------------------------------
# Energy & Battery Decision Logic
# ---------------------------------------------------------------------------

def energy_for_leg(dist_km: float, efficiency_km_per_kwh: float) -> float:
    """Energy (kWh) required to travel dist_km at the given efficiency."""
    return dist_km / efficiency_km_per_kwh


def soc_pct(battery_kwh: float, capacity_kwh: float) -> float:
    """Battery state of charge percentage from absolute kWh."""
    return round(battery_kwh / capacity_kwh * 100.0, 1)


def evaluate_battery_decision(battery_kwh: float, capacity_kwh: float,
                              threshold_pct: float, reserve_pct: float,
                              cur: str, nxt: str, eff: float) -> dict:
    """
    Evaluates battery feasibility for traveling from `cur` to `nxt`.
    Distinguishes:
      - 'above_threshold': strictly above threshold T and satisfying reserve R
      - 'threshold_equal': projected battery exactly equal to threshold T
      - 'below_threshold': projected battery falls below threshold T (but >= reserve R)
      - 'insufficient': projected battery falls below safety reserve R
    """
    threshold_kwh = capacity_kwh * threshold_pct / 100.0
    reserve_kwh = capacity_kwh * reserve_pct / 100.0
    leg_dist = distance(cur, nxt)
    needed_kwh = energy_for_leg(leg_dist, eff)
    projected = battery_kwh - needed_kwh
    projected_soc = soc_pct(projected, capacity_kwh)

    # 1. Safety reserve check
    if projected < reserve_kwh - 1e-9:
        status = "insufficient"
        status_label = "Insufficient battery to reach next destination"
        can_proceed = False
        reason = (
            f"Battery {soc_pct(battery_kwh, capacity_kwh):.1f}% ({battery_kwh:.2f} kWh) is insufficient "
            f"to reach {nxt} ({leg_dist:.2f} km, needs {needed_kwh:.2f} kWh) while preserving the "
            f"{reserve_pct:.0f}% ({reserve_kwh:.2f} kWh) safety reserve. Projected: {projected_soc:.1f}%."
        )
    # 2. Threshold equality check
    elif abs(projected - threshold_kwh) < 1e-9:
        status = "threshold_equal"
        status_label = "Battery exactly at threshold"
        can_proceed = False
        reason = (
            f"Projected battery after reaching {nxt} is exactly equal to the configured threshold of "
            f"{threshold_pct:.0f}% ({threshold_kwh:.2f} kWh). Per the threshold policy, charging "
            f"evaluation is required before proceeding."
        )
    # 3. Below threshold check
    elif projected < threshold_kwh - 1e-9:
        status = "below_threshold"
        status_label = "Battery below threshold"
        can_proceed = False
        reason = (
            f"Projected battery after reaching {nxt} would be {projected_soc:.1f}% ({projected:.2f} kWh), "
            f"falling below the configured threshold of {threshold_pct:.0f}% ({threshold_kwh:.2f} kWh). "
            f"Charging evaluation is required."
        )
    # 4. Strictly above threshold
    else:
        status = "above_threshold"
        status_label = "Battery above threshold"
        can_proceed = True
        reason = (
            f"Battery {soc_pct(battery_kwh, capacity_kwh):.1f}% ({battery_kwh:.2f} kWh) safely supports "
            f"direct travel to {nxt}. Projected battery {projected_soc:.1f}% ({projected:.2f} kWh) "
            f"remains strictly above threshold {threshold_pct:.0f}% ({threshold_kwh:.2f} kWh) "
            f"and satisfies reserve {reserve_pct:.0f}%."
        )

    # Evaluate candidate chargers
    charger_evaluations = []
    for ch in CHARGERS:
        if ch == cur:
            continue
        dist_to_ch = distance(cur, ch)
        e_to_ch = energy_for_leg(dist_to_ch, eff)
        bat_at_ch = battery_kwh - e_to_ch
        reachable = (bat_at_ch >= reserve_kwh - 1e-9)

        dist_from_ch = distance(ch, nxt)
        e_from_ch = energy_for_leg(dist_from_ch, eff)
        # Charging up to capacity at station
        add_kwh = max(0.0, capacity_kwh - bat_at_ch) if reachable else 0.0
        bat_at_nxt = capacity_kwh - e_from_ch
        continuation_feasible = reachable and (bat_at_nxt >= reserve_kwh - 1e-9)

        total_via = dist_to_ch + dist_from_ch
        detour_diff = total_via - leg_dist
        charger_evaluations.append({
            "name": ch,
            "dist_to_charger_km": round(dist_to_ch, 3),
            "energy_to_charger_kwh": round(e_to_ch, 3),
            "battery_at_charger_kwh": round(bat_at_ch, 4),
            "battery_at_charger_pct": soc_pct(bat_at_ch, capacity_kwh),
            "reachable": reachable,
            "energy_added_kwh": round(add_kwh, 3),
            "dist_to_next_km": round(dist_from_ch, 3),
            "energy_from_charger_kwh": round(e_from_ch, 3),
            "battery_at_next_kwh": round(bat_at_nxt, 4) if reachable else 0.0,
            "battery_at_next_pct": soc_pct(bat_at_nxt, capacity_kwh) if reachable else 0.0,
            "total_detour_dist_km": round(total_via, 3),
            "detour_increase_km": round(detour_diff, 3),
            "continuation_feasible": continuation_feasible,
            "status_detail": (
                f"Reachable (arrives with {soc_pct(bat_at_ch, capacity_kwh):.1f}% > {reserve_pct:.0f}% reserve); "
                f"recharge enables reaching {nxt} with {soc_pct(bat_at_nxt, capacity_kwh):.1f}%; "
                f"detour effect: +{detour_diff:.2f} km."
                if continuation_feasible else
                (f"Unreachable: arrival battery at {ch} ({soc_pct(bat_at_ch, capacity_kwh):.1f}%) violates {reserve_pct:.0f}% reserve constraint."
                 if not reachable else
                 f"Cannot reach {nxt} from {ch} even after 100% recharge ({soc_pct(bat_at_nxt, capacity_kwh):.1f}% < {reserve_pct:.0f}% reserve).")
            ),
        })

    return {
        "cur": cur,
        "nxt": nxt,
        "can_proceed_direct": can_proceed,
        "current_battery_kwh": round(battery_kwh, 4),
        "current_soc_pct": soc_pct(battery_kwh, capacity_kwh),
        "projected_battery_kwh": round(projected, 4),
        "projected_soc_pct": projected_soc,
        "threshold_kwh": round(threshold_kwh, 4),
        "threshold_pct": threshold_pct,
        "reserve_kwh": round(reserve_kwh, 4),
        "reserve_pct": reserve_pct,
        "needed_kwh": round(needed_kwh, 4),
        "leg_dist_km": round(leg_dist, 3),
        "status": status,
        "status_label": status_label,
        "reason": reason,
        "charger_candidates": charger_evaluations,
    }


def battery_decision(battery_kwh: float, capacity_kwh: float,
                     threshold_pct: float, next_loc: str, current_loc: str,
                     efficiency: float, reserve_kwh: float) -> dict:
    """Backwards-compatible wrapper for evaluate_battery_decision."""
    reserve_pct = (reserve_kwh / capacity_kwh) * 100.0 if capacity_kwh > 0 else 0.0
    return evaluate_battery_decision(
        battery_kwh=battery_kwh,
        capacity_kwh=capacity_kwh,
        threshold_pct=threshold_pct,
        reserve_pct=reserve_pct,
        cur=current_loc,
        nxt=next_loc,
        eff=efficiency
    )


# ---------------------------------------------------------------------------
# Branch & Bound — Joint Delivery Order & Charging Route Optimization
# ---------------------------------------------------------------------------

def branch_and_bound_with_charging(order_nodes: list, cfg: dict) -> dict:
    """
    Branch and Bound algorithm searching for the globally minimum-distance
    feasible route visiting all delivery locations and ending at Hub,
    considering battery capacity, safety reserve, and threshold constraints.

    State representation:
        (lower_bound, serial, current_loc, delivered_orders_mask, cost_so_far,
         battery_kwh, path, events, tree_node_id)
    """
    capacity = cfg["capacity_kwh"]
    battery0 = capacity * cfg["soc_pct"] / 100.0
    reserve_pct = cfg["reserve_pct"]
    reserve_kwh = capacity * reserve_pct / 100.0
    threshold_pct = cfg["threshold_pct"]
    threshold_kwh = capacity * threshold_pct / 100.0
    eff = cfg["km_per_kwh"]
    rate_kw = cfg["charger_kw"]

    full_orders = frozenset(order_nodes)

    # Bookkeeping for search tree visualization
    tree = []
    node_id_gen = count(0)
    serial_gen = count(0)

    def new_tree_node(parent_id, label, bound, cost, depth):
        nid = next(node_id_gen)
        tree.append({
            "id": nid, "parent": parent_id, "label": label,
            "bound": round(bound, 3), "cost": round(cost, 3),
            "status": "open", "depth": depth
        })
        return nid

    init_lb = lower_bound("Depot", order_nodes, "Hub")
    root_id = new_tree_node(None, "Start at Depot", init_lb, 0.0, 0)

    # Priority queue: min-heap ordered by (lower_bound, serial, ...)
    # Tuple: (lb, serial, cur, delivered, cost_so_far, bat, path, planned, events, tid)
    heap = []
    heapq.heappush(heap, (
        init_lb, next(serial_gen),
        "Depot", frozenset(), 0.0, battery0, [], ["Depot"], [], root_id
    ))

    best_cost = float("inf")
    best_route = []
    best_planned = []
    best_events = []
    explored = 0
    pruned = 0

    while heap:
        (lb, _, cur, delivered, cost_so_far,
         bat, path, planned, events, tid) = heapq.heappop(heap)

        # Admissible pruning: bound cannot beat best known complete solution
        if lb >= best_cost - 1e-9:
            tree[tid]["status"] = "pruned"
            pruned += 1
            continue

        explored += 1
        tree[tid]["status"] = "expanded"

        remaining = [nd for nd in order_nodes if nd not in delivered]

        # -------------------------------------------------------------------
        # Terminal condition: all deliveries completed, return to Hub
        # -------------------------------------------------------------------
        if not remaining:
            leg_to_hub = distance(cur, "Hub")
            needed_hub = energy_for_leg(leg_to_hub, eff)
            proj_hub = bat - needed_hub
            dec_hub = evaluate_battery_decision(
                battery_kwh=bat,
                capacity_kwh=capacity,
                threshold_pct=threshold_pct,
                reserve_pct=reserve_pct,
                cur=cur,
                nxt="Hub",
                eff=eff
            )

            # Option A: Direct return to Hub if projected battery remains strictly above threshold
            if dec_hub["can_proceed_direct"]:
                total_direct = cost_so_far + leg_to_hub
                hub_events = events + [
                    {
                        "type": "battery_decision",
                        "from": cur, "to": "Hub",
                        "battery_decision": dec_hub,
                        "charger_candidates": dec_hub["charger_candidates"],
                        "selected_charger": None,
                        "battery_before": round(bat, 4),
                        "battery_after": round(proj_hub, 4),
                        "soc_before": soc_pct(bat, capacity),
                        "soc_after": soc_pct(proj_hub, capacity),
                        "message": (
                            f"At {cur} before Hub: Battery {soc_pct(bat, capacity):.1f}% ({bat:.2f} kWh). "
                            f"Direct travel to Hub ({leg_to_hub:.2f} km) requires {needed_hub:.2f} kWh. "
                            f"Projected battery at Hub is {soc_pct(proj_hub, capacity):.1f}% ({proj_hub:.2f} kWh), "
                            f"strictly above {threshold_pct:.0f}% threshold ({threshold_kwh:.2f} kWh). "
                            f"Charging required? No. Vehicle proceeds directly to Hub."
                        )
                    },
                    {
                        "type": "finish",
                        "from": cur,
                        "to": "Hub",
                        "distance_km": round(leg_to_hub, 3),
                        "energy_kwh": round(needed_hub, 3),
                        "battery_before": round(bat, 4),
                        "battery_after": round(proj_hub, 4),
                        "soc_before": soc_pct(bat, capacity),
                        "soc_after": soc_pct(proj_hub, capacity),
                        "battery_decision": dec_hub,
                        "message": (
                            f"Travel {cur} → Hub: {leg_to_hub:.2f} km, use {needed_hub:.2f} kWh. "
                            f"Battery remaining: {soc_pct(proj_hub, capacity):.1f}%."
                        )
                    }
                ]
                if total_direct < best_cost:
                    best_cost = total_direct
                    best_route = list(path)
                    best_planned = list(planned) + ["Hub"]
                    best_events = hub_events
                    lid = new_tree_node(
                        tid, f"Finish Hub · {total_direct:.2f} km",
                        total_direct, total_direct, len(path) + 1
                    )
                    tree[lid]["status"] = "best"
                else:
                    lid = new_tree_node(
                        tid, f"Finish Hub · {total_direct:.2f} km",
                        total_direct, total_direct, len(path) + 1
                    )
                    tree[lid]["status"] = "complete"

            # Option B: Projected battery <= threshold or insufficient: evaluate chargers before Hub
            else:
                for ch in CHARGERS:
                    if ch == cur:
                        continue
                    to_ch = distance(cur, ch)
                    e_to_ch = energy_for_leg(to_ch, eff)
                    if bat - e_to_ch < reserve_kwh - 1e-9:
                        continue  # Cannot reach charger

                    bat_at_ch = bat - e_to_ch
                    add_kwh = capacity - bat_at_ch
                    ch_to_hub = distance(ch, "Hub")
                    e_ch_to_hub = energy_for_leg(ch_to_hub, eff)
                    bat_at_hub = capacity - e_ch_to_hub

                    if bat_at_hub < reserve_kwh - 1e-9:
                        continue  # Cannot reach Hub even after charging

                    total_via = cost_so_far + to_ch + ch_to_hub
                    if total_via < best_cost:
                        charge_min = add_kwh / rate_kw * 60.0
                        detour_diff = (to_ch + ch_to_hub) - leg_to_hub
                        sel_reason = {
                            "charger": ch,
                            "current_location": cur,
                            "next_destination": "Hub",
                            "battery_before_travel_kwh": round(bat, 4),
                            "battery_before_travel_pct": soc_pct(bat, capacity),
                            "dist_to_charger_km": round(to_ch, 3),
                            "energy_to_charger_kwh": round(e_to_ch, 3),
                            "battery_at_charger_kwh": round(bat_at_ch, 4),
                            "battery_at_charger_pct": soc_pct(bat_at_ch, capacity),
                            "energy_added_kwh": round(add_kwh, 3),
                            "battery_after_charge_kwh": round(capacity, 4),
                            "battery_after_charge_pct": 100.0,
                            "dist_from_charger_km": round(ch_to_hub, 3),
                            "energy_from_charger_kwh": round(e_ch_to_hub, 3),
                            "battery_at_destination_kwh": round(bat_at_hub, 4),
                            "battery_at_destination_pct": soc_pct(bat_at_hub, capacity),
                            "threshold_kwh": round(threshold_kwh, 4),
                            "threshold_pct": threshold_pct,
                            "reserve_kwh": round(reserve_kwh, 4),
                            "reserve_pct": reserve_pct,
                            "projected_direct_kwh": round(dec_hub["projected_battery_kwh"], 4),
                            "projected_direct_pct": dec_hub["projected_soc_pct"],
                            "direct_dist_km": round(leg_to_hub, 3),
                            "total_via_km": round(to_ch + ch_to_hub, 3),
                            "detour_increase_km": round(detour_diff, 3),
                            "explanation": (
                                f"Charging at {ch} selected: Direct travel {cur} → Hub ({leg_to_hub:.2f} km) "
                                f"would leave {dec_hub['projected_soc_pct']:.1f}% battery ({dec_hub['projected_battery_kwh']:.2f} kWh), "
                                f"failing the {threshold_pct:.0f}% threshold ({threshold_kwh:.2f} kWh). "
                                f"Charger {ch} is reachable at {to_ch:.2f} km ({e_to_ch:.2f} kWh), arriving with "
                                f"{soc_pct(bat_at_ch, capacity):.1f}% battery (satisfies {reserve_pct:.0f}% reserve). "
                                f"Recharging to 100% (+{add_kwh:.2f} kWh in {charge_min:.1f} min) enables reaching Hub "
                                f"with {soc_pct(bat_at_hub, capacity):.1f}% battery. "
                                f"Selected because it minimizes total route distance (detour effect: +{detour_diff:.2f} km) "
                                f"among all feasible alternatives."
                            )
                        }
                        fin_events = events + [
                            {
                                "type": "battery_decision",
                                "from": cur, "to": "Hub",
                                "battery_decision": dec_hub,
                                "charger_candidates": dec_hub["charger_candidates"],
                                "selected_charger": ch,
                                "selection_reason": sel_reason,
                                "message": (
                                    f"Battery {soc_pct(bat, capacity):.1f}% — {dec_hub['status_label']}. "
                                    f"Charging evaluation selected {ch} for lowest-cost feasible return to Hub."
                                )
                            },
                            {
                                "type": "travel_to_charger",
                                "from": cur, "to": ch,
                                "distance_km": round(to_ch, 3),
                                "energy_kwh": round(e_to_ch, 3),
                                "battery_before": round(bat, 4),
                                "battery_after": round(bat_at_ch, 4),
                                "soc_before": soc_pct(bat, capacity),
                                "soc_after": soc_pct(bat_at_ch, capacity),
                                "message": f"Travel {cur} → {ch}: {to_ch:.2f} km, use {e_to_ch:.2f} kWh."
                            },
                            {
                                "type": "charge",
                                "station": ch,
                                "added_kwh": round(add_kwh, 3),
                                "charge_minutes": round(charge_min, 1),
                                "battery_before": round(bat_at_ch, 4),
                                "battery_after": round(capacity, 4),
                                "soc_before": soc_pct(bat_at_ch, capacity),
                                "soc_after": 100.0,
                                "message": f"Charge {add_kwh:.2f} kWh at {ch} ({charge_min:.1f} min). SOC: {soc_pct(bat_at_ch, capacity):.1f}% → 100.0%."
                            },
                            {
                                "type": "finish",
                                "from": ch, "to": "Hub",
                                "distance_km": round(ch_to_hub, 3),
                                "energy_kwh": round(e_ch_to_hub, 3),
                                "battery_before": round(capacity, 4),
                                "battery_after": round(bat_at_hub, 4),
                                "soc_before": 100.0,
                                "soc_after": soc_pct(bat_at_hub, capacity),
                                "message": f"Travel {ch} → Hub: {ch_to_hub:.2f} km, use {e_ch_to_hub:.2f} kWh."
                            }
                        ]
                        best_cost = total_via
                        best_route = list(path)
                        best_planned = list(planned) + [ch, "Hub"]
                        best_events = fin_events
                        lid = new_tree_node(
                            tid, f"Hub via {ch} · {total_via:.2f} km",
                            total_via, total_via, len(path) + 2
                        )
                        tree[lid]["status"] = "best"
            continue

        # -------------------------------------------------------------------
        # Expand each remaining delivery
        # -------------------------------------------------------------------
        for nxt in remaining:
            leg = distance(cur, nxt)
            needed = energy_for_leg(leg, eff)
            dec = evaluate_battery_decision(
                battery_kwh=bat,
                capacity_kwh=capacity,
                threshold_pct=threshold_pct,
                reserve_pct=reserve_pct,
                cur=cur,
                nxt=nxt,
                eff=eff
            )

            # Option A: Direct delivery (if projected battery remains strictly above threshold)
            if dec["can_proceed_direct"]:
                new_bat = bat - needed
                new_cost = cost_so_far + leg
                new_del = delivered | {nxt}
                new_path = path + [nxt]
                new_planned = planned + [nxt]
                new_rem = [nd for nd in order_nodes if nd not in new_del]
                lb_direct = round(new_cost + lower_bound(nxt, new_rem, "Hub"), 3)

                new_ev = events + [
                    {
                        "type": "battery_decision",
                        "from": cur, "to": nxt,
                        "battery_decision": dec,
                        "charger_candidates": dec["charger_candidates"],
                        "selected_charger": None,
                        "battery_before": round(bat, 4),
                        "battery_after": round(new_bat, 4),
                        "soc_before": soc_pct(bat, capacity),
                        "soc_after": soc_pct(new_bat, capacity),
                        "message": (
                            f"At {cur} before {nxt}: Battery {soc_pct(bat, capacity):.1f}% ({bat:.2f} kWh). "
                            f"Direct travel to {nxt} ({leg:.2f} km) requires {needed:.2f} kWh. "
                            f"Projected battery after arrival: {soc_pct(new_bat, capacity):.1f}% ({new_bat:.2f} kWh), "
                            f"strictly above {threshold_pct:.0f}% threshold ({threshold_kwh:.2f} kWh). "
                            f"Charging required? No. Direct delivery approved."
                        )
                    },
                    {
                        "type": "delivery",
                        "from": cur, "to": nxt,
                        "distance_km": round(leg, 3),
                        "energy_kwh": round(needed, 3),
                        "battery_before": round(bat, 4),
                        "battery_after": round(new_bat, 4),
                        "soc_before": soc_pct(bat, capacity),
                        "soc_after": soc_pct(new_bat, capacity),
                        "battery_decision": dec,
                        "message": (
                            f"Travel {cur} → {nxt}: {leg:.2f} km, use {needed:.2f} kWh. "
                            f"Battery: {soc_pct(new_bat, capacity):.1f}%."
                        )
                    }
                ]

                status = "open" if lb_direct < best_cost - 1e-9 else "pruned"
                child_id = new_tree_node(tid, f"Visit {nxt}", lb_direct, new_cost, len(new_path))
                if status == "pruned":
                    tree[child_id]["status"] = "pruned"
                    pruned += 1
                else:
                    heapq.heappush(heap, (
                        lb_direct, next(serial_gen),
                        nxt, new_del, new_cost, new_bat,
                        new_path, new_planned, new_ev, child_id
                    ))

            # Option B: Projected battery <= threshold or insufficient: evaluate chargers
            else:
                any_charger_pushed = False
                for ch in CHARGERS:
                    if ch == cur:
                        continue
                    to_ch = distance(cur, ch)
                    e_to_ch = energy_for_leg(to_ch, eff)
                    if bat - e_to_ch < reserve_kwh - 1e-9:
                        continue  # Cannot reach charger

                    bat_at_ch = bat - e_to_ch
                    ch_to_nxt = distance(ch, nxt)
                    e_ch_to_nxt = energy_for_leg(ch_to_nxt, eff)

                    # Charge to full capacity at station
                    add_kwh = capacity - bat_at_ch
                    bat_at_nxt = capacity - e_ch_to_nxt
                    if bat_at_nxt < reserve_kwh - 1e-9:
                        continue  # Cannot reach next delivery even after full charge

                    new_bat = bat_at_nxt
                    new_cost = cost_so_far + to_ch + ch_to_nxt
                    new_del = delivered | {nxt}
                    new_path = path + [nxt]
                    new_planned = planned + [ch, nxt]
                    new_rem = [nd for nd in order_nodes if nd not in new_del]
                    lb_via = round(new_cost + lower_bound(nxt, new_rem, "Hub"), 3)
                    charge_min = add_kwh / rate_kw * 60.0
                    detour_diff = (to_ch + ch_to_nxt) - leg

                    sel_reason = {
                        "charger": ch,
                        "current_location": cur,
                        "next_destination": nxt,
                        "battery_before_travel_kwh": round(bat, 4),
                        "battery_before_travel_pct": soc_pct(bat, capacity),
                        "dist_to_charger_km": round(to_ch, 3),
                        "energy_to_charger_kwh": round(e_to_ch, 3),
                        "battery_at_charger_kwh": round(bat_at_ch, 4),
                        "battery_at_charger_pct": soc_pct(bat_at_ch, capacity),
                        "energy_added_kwh": round(add_kwh, 3),
                        "battery_after_charge_kwh": round(capacity, 4),
                        "battery_after_charge_pct": 100.0,
                        "dist_from_charger_km": round(ch_to_nxt, 3),
                        "energy_from_charger_kwh": round(e_ch_to_nxt, 3),
                        "battery_at_destination_kwh": round(new_bat, 4),
                        "battery_at_destination_pct": soc_pct(new_bat, capacity),
                        "threshold_kwh": round(threshold_kwh, 4),
                        "threshold_pct": threshold_pct,
                        "reserve_kwh": round(reserve_kwh, 4),
                        "reserve_pct": reserve_pct,
                        "projected_direct_kwh": round(dec["projected_battery_kwh"], 4),
                        "projected_direct_pct": dec["projected_soc_pct"],
                        "direct_dist_km": round(leg, 3),
                        "total_via_km": round(to_ch + ch_to_nxt, 3),
                        "detour_increase_km": round(detour_diff, 3),
                        "explanation": (
                            f"Charging at {ch} selected: Direct travel {cur} → {nxt} ({leg:.2f} km) "
                            f"would leave {dec['projected_soc_pct']:.1f}% battery ({dec['projected_battery_kwh']:.2f} kWh), "
                            f"failing the {threshold_pct:.0f}% threshold ({threshold_kwh:.2f} kWh). "
                            f"Charger {ch} is reachable at {to_ch:.2f} km ({e_to_ch:.2f} kWh), arriving with "
                            f"{soc_pct(bat_at_ch, capacity):.1f}% battery (satisfies {reserve_pct:.0f}% reserve). "
                            f"Recharging to 100% (+{add_kwh:.2f} kWh in {charge_min:.1f} min) enables reaching {nxt} "
                            f"with {soc_pct(new_bat, capacity):.1f}% battery. "
                            f"Selected because it minimizes total route distance (detour effect: +{detour_diff:.2f} km) "
                            f"among all feasible alternatives."
                        )
                    }

                    new_ev = events + [
                        {
                            "type": "battery_decision",
                            "from": cur, "to": nxt,
                            "battery_decision": dec,
                            "charger_candidates": dec["charger_candidates"],
                            "selected_charger": ch,
                            "selection_reason": sel_reason,
                            "message": (
                                f"Battery {soc_pct(bat, capacity):.1f}% — {dec['status_label']}. "
                                f"Charging evaluation selected {ch} for lowest-cost feasible continuation."
                            )
                        },
                        {
                            "type": "travel_to_charger",
                            "from": cur, "to": ch,
                            "distance_km": round(to_ch, 3),
                            "energy_kwh": round(e_to_ch, 3),
                            "battery_before": round(bat, 4),
                            "battery_after": round(bat_at_ch, 4),
                            "soc_before": soc_pct(bat, capacity),
                            "soc_after": soc_pct(bat_at_ch, capacity),
                            "message": f"Travel {cur} → {ch}: {to_ch:.2f} km, use {e_to_ch:.2f} kWh."
                        },
                        {
                            "type": "charge",
                            "station": ch,
                            "added_kwh": round(add_kwh, 3),
                            "charge_minutes": round(charge_min, 1),
                            "battery_before": round(bat_at_ch, 4),
                            "battery_after": round(capacity, 4),
                            "soc_before": soc_pct(bat_at_ch, capacity),
                            "soc_after": 100.0,
                            "message": (
                                f"Charge {add_kwh:.2f} kWh at {ch} ({charge_min:.1f} min). "
                                f"SOC: {soc_pct(bat_at_ch, capacity):.1f}% → 100.0%."
                            )
                        },
                        {
                            "type": "delivery",
                            "from": ch, "to": nxt,
                            "distance_km": round(ch_to_nxt, 3),
                            "energy_kwh": round(e_ch_to_nxt, 3),
                            "battery_before": round(capacity, 4),
                            "battery_after": round(new_bat, 4),
                            "soc_before": 100.0,
                            "soc_after": soc_pct(new_bat, capacity),
                            "message": (
                                f"Travel {ch} → {nxt}: {ch_to_nxt:.2f} km, use {e_ch_to_nxt:.2f} kWh. "
                                f"Battery: {soc_pct(new_bat, capacity):.1f}%."
                            )
                        }
                    ]

                    status = "open" if lb_via < best_cost - 1e-9 else "pruned"
                    child_id = new_tree_node(
                        tid, f"Via {ch} → {nxt}", lb_via, new_cost, len(new_path)
                    )
                    if status == "pruned":
                        tree[child_id]["status"] = "pruned"
                        pruned += 1
                    else:
                        heapq.heappush(heap, (
                            lb_via, next(serial_gen),
                            nxt, new_del, new_cost, new_bat,
                            new_path, new_planned, new_ev, child_id
                        ))
                        any_charger_pushed = True

                if not any_charger_pushed:
                    child_id = new_tree_node(
                        tid, f"✗ Infeasible for {nxt}", float("inf"), cost_so_far, len(path) + 1
                    )
                    tree[child_id]["status"] = "pruned"
                    pruned += 1

    return {
        "best_route": best_route,
        "best_cost_km": round(best_cost, 3) if best_cost < float("inf") else None,
        "search_tree": tree,
        "explored": explored,
        "pruned": pruned,
        "best_events": best_events,
        "best_planned": best_planned,
        "feasible": best_cost < float("inf"),
    }


def branch_and_bound(order_nodes: list, cfg: dict = None) -> tuple:
    """
    Backwards-compatible interface for unconstrained or default B&B.
    Returns (best_route, best_cost, search_tree, explored, pruned).
    """
    if cfg is None:
        cfg = {
            "capacity_kwh": 40.0,
            "soc_pct": 100.0,
            "km_per_kwh": 6.0,
            "average_speed_kmh": 30.0,
            "charger_kw": 22.0,
            "reserve_pct": 0.0,
            "threshold_pct": 0.0,
            "route_budget_km": 1000.0,
        }
    res = branch_and_bound_with_charging(order_nodes, cfg)
    return res["best_route"], res["best_cost_km"], res["search_tree"], res["explored"], res["pruned"]


# ---------------------------------------------------------------------------
# Independent Exhaustive Brute-Force Search for Optimality Verification
# ---------------------------------------------------------------------------

def exhaustive_search(order_nodes: list, cfg: dict) -> dict:
    """
    Exhaustive search over all permutations and feasible charger choices
    to independently verify the global optimality of Branch & Bound.
    """
    capacity = cfg["capacity_kwh"]
    bat0 = capacity * cfg["soc_pct"] / 100.0
    reserve_kwh = capacity * cfg["reserve_pct"] / 100.0
    threshold_kwh = capacity * cfg["threshold_pct"] / 100.0
    eff = cfg["km_per_kwh"]

    def eval_perm(perm):
        def solve(idx, cur, bat, dist_so_far, planned_seq):
            if idx == len(perm):
                req = energy_for_leg(distance(cur, "Hub"), eff)
                best_via = float("inf")
                best_seq = None
                # Direct to Hub (if projected battery remains strictly above threshold and reserve)
                if bat - req > threshold_kwh + 1e-9 and bat - req >= reserve_kwh - 1e-9:
                    best_via = dist_so_far + distance(cur, "Hub")
                    best_seq = planned_seq + ["Hub"]
                # Via charger to Hub
                for ch in CHARGERS:
                    if ch == cur:
                        continue
                    to_ch = energy_for_leg(distance(cur, ch), eff)
                    if bat - to_ch >= reserve_kwh - 1e-9:
                        from_ch = energy_for_leg(distance(ch, "Hub"), eff)
                        if capacity - from_ch >= reserve_kwh - 1e-9:
                            cand_cost = dist_so_far + distance(cur, ch) + distance(ch, "Hub")
                            if cand_cost < best_via:
                                best_via = cand_cost
                                best_seq = planned_seq + [ch, "Hub"]
                return best_via, best_seq

            nxt = perm[idx]
            req = energy_for_leg(distance(cur, nxt), eff)
            proj = bat - req

            best_via = float("inf")
            best_seq = None

            # Direct move if strictly above threshold
            if proj > threshold_kwh + 1e-9 and proj >= reserve_kwh - 1e-9:
                c, s = solve(idx + 1, nxt, proj, dist_so_far + distance(cur, nxt), planned_seq + [nxt])
                if c < best_via:
                    best_via, best_seq = c, s
            else:
                # Charger moves
                for ch in CHARGERS:
                    if ch == cur:
                        continue
                    to_ch = energy_for_leg(distance(cur, ch), eff)
                    if bat - to_ch >= reserve_kwh - 1e-9:
                        from_ch = energy_for_leg(distance(ch, nxt), eff)
                        if capacity - from_ch >= reserve_kwh - 1e-9:
                            c, s = solve(
                                idx + 1, nxt, capacity - from_ch,
                                dist_so_far + distance(cur, ch) + distance(ch, nxt),
                                planned_seq + [ch, nxt]
                            )
                            if c < best_via:
                                best_via, best_seq = c, s
            return best_via, best_seq

        return solve(0, "Depot", bat0, 0.0, ["Depot"])

    best_dist = float("inf")
    best_perm = None
    best_planned = None

    for p in permutations(order_nodes):
        cost, planned = eval_perm(list(p))
        if cost < best_dist:
            best_dist = cost
            best_perm = list(p)
            best_planned = planned

    return {
        "best_route": best_perm,
        "best_cost_km": round(best_dist, 3) if best_dist < float("inf") else None,
        "best_planned": best_planned,
        "feasible": best_dist < float("inf"),
    }


# ---------------------------------------------------------------------------
# Road-Segment-by-Road-Segment Simulation Steps Builder
# ---------------------------------------------------------------------------

def build_simulation_steps(bnb_result: dict, cfg: dict, selected: list) -> list:
    """
    Decomposes the planned route into road segments along Dijkstra shortest paths.
    Generates granular simulation steps with gradual battery depletion,
    explicit charging steps, and synchronized explanations.
    """
    capacity = cfg["capacity_kwh"]
    steps = []

    steps.append({
        "type": "start",
        "from": "Depot",
        "to": "Depot",
        "title": "Start at Depot",
        "detail": (
            f"Delivery partner departs from Depot with starting battery "
            f"{cfg['soc_pct']:.0f}% ({capacity * cfg['soc_pct'] / 100:.2f} kWh). "
            f"Orders to deliver: {', '.join(o['name'] for o in selected)}."
        ),
        "formula": (
            f"Available energy = {capacity:.1f} × {cfg['soc_pct']:.0f}/100 "
            f"= {capacity * cfg['soc_pct'] / 100:.2f} kWh | "
            f"Safety reserve ({cfg['reserve_pct']:.0f}%) = {capacity * cfg['reserve_pct'] / 100:.2f} kWh | "
            f"Threshold ({cfg['threshold_pct']:.0f}%) = {capacity * cfg['threshold_pct'] / 100:.2f} kWh"
        ),
        "battery_kwh": round(capacity * cfg["soc_pct"] / 100, 4),
        "battery_pct": cfg["soc_pct"],
        "distance_so_far_km": 0.0,
        "energy_so_far_kwh": 0.0,
        "dwell_ms": 2000,
    })

    dist_acc = 0.0
    energy_acc = 0.0

    for ev in bnb_result.get("best_events", []):
        t = ev.get("type", "")

        # -------------------------------------------------------------------
        # Battery decision step (pause at current node before continuing)
        # -------------------------------------------------------------------
        if t == "battery_decision":
            dec = ev.get("battery_decision", {})
            candidates = ev.get("charger_candidates", [])
            cand_str = " | ".join(
                f"{c['name']}: {c['dist_to_charger_km']:.2f} km ({'reachable' if c['reachable'] else 'unreachable'})"
                for c in candidates
            )
            steps.append({
                "type": "battery_decision",
                "from": ev.get("from"),
                "to": ev.get("from"),
                "title": f"Battery Decision — At {ev.get('from')} before {ev.get('to')}",
                "detail": (
                    f"Current location: {ev.get('from')} | "
                    f"Battery: {dec.get('current_soc_pct', 0):.1f}% ({dec.get('current_battery_kwh', 0):.2f} kWh) | "
                    f"Threshold: {cfg['threshold_pct']:.0f}% ({capacity * cfg['threshold_pct'] / 100:.2f} kWh) | "
                    f"Energy needed: {dec.get('needed_kwh', 0):.2f} kWh | "
                    f"Decision: {dec.get('status_label', '')} | "
                    f"Candidate chargers: {cand_str} | "
                    f"Selected charger: {ev.get('selected_charger', '—')} (minimizes total feasible route distance)."
                ),
                "formula": (
                    f"Projected battery = {dec.get('projected_battery_kwh', 0):.2f} kWh "
                    f"({dec.get('projected_soc_pct', 0):.1f}%) | "
                    f"Threshold = {dec.get('threshold_kwh', 0):.2f} kWh | "
                    f"Status: {dec.get('status_label', '')}"
                ),
                "battery_kwh": ev.get("battery_before", dec.get("current_battery_kwh", 0)),
                "battery_pct": dec.get("current_soc_pct", 0),
                "distance_so_far_km": round(dist_acc, 3),
                "energy_so_far_kwh": round(energy_acc, 3),
                "battery_decision": dec,
                "selected_charger": ev.get("selected_charger"),
                "selection_reason": ev.get("selection_reason"),
                "target_destination": ev.get("to"),
                "dwell_ms": 2500,
            })

        # -------------------------------------------------------------------
        # Travel leg (broken into Dijkstra road segments)
        # -------------------------------------------------------------------
        elif t in ("travel_to_charger", "delivery", "finish"):
            u_from = ev["from"]
            u_to = ev["to"]
            path_vertices = shortest_path(u_from, u_to)
            n_segments = max(1, len(path_vertices) - 1)

            bat_start = ev.get("battery_before", 0)
            bat_end = ev.get("battery_after", 0)
            total_leg_energy = ev.get("energy_kwh", 0)
            total_leg_dist = ev.get("distance_km", 0)

            cur_bat = bat_start

            for s_idx in range(n_segments):
                seg_u = path_vertices[s_idx]
                seg_v = path_vertices[s_idx + 1]
                seg_dist = distance(seg_u, seg_v)
                seg_energy = energy_for_leg(seg_dist, cfg["km_per_kwh"])
                next_bat = max(0.0, cur_bat - seg_energy)

                dist_acc += seg_dist
                energy_acc += seg_energy

                step_type = "travel_segment"
                is_last_segment = (s_idx == n_segments - 1)
                is_transit = not is_last_segment
                is_charger_node = seg_v in CHARGERS

                if is_last_segment and t == "delivery":
                    step_type = "delivery"
                    title = f"Deliver to {u_to} (Arrival)"
                    detail = (
                        f"Delivered order at {u_to}. Leg distance: {total_leg_dist:.2f} km. "
                        f"Energy consumed: {total_leg_energy:.2f} kWh. "
                        f"Remaining battery: {soc_pct(next_bat, capacity):.1f}% ({next_bat:.2f} kWh)."
                    )
                elif is_last_segment and t == "travel_to_charger":
                    step_type = "travel_to_charger"
                    title = f"Arrive at Charger {u_to} (Scheduled Charging Stop)"
                    detail = (
                        f"Arrived at charging station {u_to}. Leg distance: {total_leg_dist:.2f} km. "
                        f"Energy consumed: {total_leg_energy:.2f} kWh. "
                        f"Battery on arrival: {soc_pct(next_bat, capacity):.1f}% ({next_bat:.2f} kWh). "
                        f"Ready to recharge."
                    )
                elif is_last_segment and t == "finish":
                    step_type = "finish"
                    title = f"Arrive at Hub (Complete)"
                    detail = (
                        f"Returned to ending {u_to}. All deliveries complete! "
                        f"Total distance: {dist_acc:.2f} km. "
                        f"Battery remaining: {soc_pct(next_bat, capacity):.1f}% ({next_bat:.2f} kWh)."
                    )
                elif is_charger_node and is_transit:
                    title = f"En route to {u_to} · Road waypoint {seg_v} (Transit only — No charging needed)"
                    detail = (
                        f"Driving road segment {seg_u} → {seg_v} ({seg_dist:.2f} km). "
                        f"{seg_v} is an intermediate road-network intersection on the shortest path to {u_to}. "
                        f"Current battery is {soc_pct(next_bat, capacity):.1f}% ({next_bat:.2f} kWh), "
                        f"strictly above the {cfg['threshold_pct']:.0f}% threshold ({capacity * cfg['threshold_pct'] / 100:.2f} kWh) "
                        f"and safety reserve ({cfg['reserve_pct']:.0f}%), so no charging is performed at {seg_v}."
                    )
                else:
                    title = f"Road Segment: {seg_u} → {seg_v} ({s_idx + 1}/{n_segments})"
                    detail = (
                        f"Driving road segment {seg_u} → {seg_v} ({seg_dist:.2f} km). "
                        f"Segment energy: {seg_energy:.2f} kWh. Heading toward {u_to}."
                    )

                steps.append({
                    "type": step_type,
                    "from": seg_u,
                    "to": seg_v,
                    "final_destination": u_to,
                    "target_destination": u_to,
                    "is_transit_waypoint": is_transit,
                    "is_charging_stop": (t == "travel_to_charger" and is_last_segment),
                    "segment": [seg_u, seg_v],
                    "segment_index": s_idx + 1,
                    "segment_count": n_segments,
                    "title": title,
                    "detail": detail,
                    "formula": (
                        f"Segment energy = {seg_dist:.2f} km / {cfg['km_per_kwh']:.1f} km/kWh = {seg_energy:.2f} kWh | "
                        f"Battery: {soc_pct(cur_bat, capacity):.1f}% → {soc_pct(next_bat, capacity):.1f}%"
                    ),
                    "distance_km": round(seg_dist, 3),
                    "energy_kwh": round(seg_energy, 3),
                    "battery_before_pct": soc_pct(cur_bat, capacity),
                    "battery_after_pct": soc_pct(next_bat, capacity),
                    "battery_kwh": round(next_bat, 4),
                    "battery_pct": soc_pct(next_bat, capacity),
                    "distance_so_far_km": round(dist_acc, 3),
                    "energy_so_far_kwh": round(energy_acc, 3),
                    "dwell_ms": 2000 if is_last_segment else 300,
                })
                cur_bat = next_bat

        # -------------------------------------------------------------------
        # Charging step
        # -------------------------------------------------------------------
        elif t == "charge":
            ch_station = ev["station"]
            added = ev.get("added_kwh", 0)
            charge_min = ev.get("charge_minutes", 0)
            bat_before = ev.get("battery_before", 0)
            bat_after = ev.get("battery_after", capacity)
            soc_before = ev.get("soc_before", soc_pct(bat_before, capacity))
            soc_after = ev.get("soc_after", 100.0)

            steps.append({
                "type": "charge",
                "from": ch_station,
                "to": ch_station,
                "title": f"Charging at {ch_station}",
                "detail": (
                    f"Charging at {ch_station} ({NODES[ch_station].get('rate_kw', cfg['charger_kw'])} kW charger). "
                    f"Energy added: {added:.2f} kWh. Charging duration: {charge_min:.1f} min. "
                    f"Battery: {soc_before:.1f}% ({bat_before:.2f} kWh) → {soc_after:.1f}% ({bat_after:.2f} kWh)."
                ),
                "formula": (
                    f"Charge time = {added:.2f} kWh / {cfg['charger_kw']:.0f} kW × 60 = {charge_min:.1f} min | "
                    f"Battery replenished to 100.0% ({bat_after:.2f} kWh)"
                ),
                "station": ch_station,
                "added_kwh": round(added, 3),
                "charge_minutes": round(charge_min, 1),
                "battery_before_pct": soc_before,
                "battery_after_pct": soc_after,
                "battery_kwh": round(bat_after, 4),
                "battery_pct": soc_after,
                "distance_so_far_km": round(dist_acc, 3),
                "energy_so_far_kwh": round(energy_acc, 3),
                "dwell_ms": 3000,
            })

    return steps


# ---------------------------------------------------------------------------
# Main Optimization Entrypoint
# ---------------------------------------------------------------------------

def optimize(payload: dict) -> dict:
    """Validate inputs, run Branch & Bound, and construct complete response."""
    selected_ids = payload.get("order_ids", ["A", "B", "C"])
    selected_ids = [x for x in selected_ids if x in ORDERS]
    if not selected_ids:
        raise ValueError("Select at least one delivery order.")

    cfg = {
        "capacity_kwh":      float(payload.get("capacity_kwh", 40)),
        "soc_pct":           float(payload.get("soc_pct", 75)),
        "km_per_kwh":        float(payload.get("km_per_kwh", 6)),
        "average_speed_kmh": float(payload.get("average_speed_kmh", 30)),
        "charger_kw":        float(payload.get("charger_kw", 22)),
        "reserve_pct":       float(payload.get("reserve_pct", 15)),
        "threshold_pct":     float(payload.get("threshold_pct", 25)),
        "route_budget_km":   float(payload.get("route_budget_km", 100)),
    }

    # Strict input validation
    if not (5 <= cfg["capacity_kwh"] <= 120):
        raise ValueError("Battery capacity must be between 5 and 120 kWh.")
    if not (10 <= cfg["soc_pct"] <= 100):
        raise ValueError("Starting charge must be between 10% and 100%.")
    if not (2 <= cfg["km_per_kwh"] <= 12):
        raise ValueError("Efficiency must be between 2 and 12 km/kWh.")
    if not (10 <= cfg["average_speed_kmh"] <= 80):
        raise ValueError("Average speed must be between 10 and 80 km/h.")
    if not (1 <= cfg["charger_kw"] <= 150):
        raise ValueError("Charger rate must be between 1 and 150 kW.")
    if not (0 <= cfg["reserve_pct"] <= 40):
        raise ValueError("Reserve must be between 0% and 40%.")
    if not (0 <= cfg["threshold_pct"] <= 80):
        raise ValueError("Battery threshold must be between 0% and 80%.")
    if cfg["threshold_pct"] < cfg["reserve_pct"]:
        raise ValueError("Battery threshold must be ≥ reserve percentage.")
    if cfg["route_budget_km"] <= 0:
        raise ValueError("Route distance budget must be positive.")

    selected = [ORDERS[x] for x in selected_ids]
    order_nodes = [o["destination"] for o in selected]

    # Run Branch & Bound
    bnb = branch_and_bound_with_charging(order_nodes, cfg)

    if not bnb["feasible"]:
        reason = (
            "No feasible route exists: the EV cannot complete the selected deliveries "
            "and reach the Hub without violating battery capacity or reserve constraints. "
            "Consider increasing starting SOC, raising battery capacity, or lowering threshold."
        )
        return {
            "nodes": NODES, "orders": selected, "chargers": CHARGERS, "config": cfg,
            "feasible": False,
            "infeasibility_reason": reason,
            "search_tree": bnb["search_tree"],
            "explored_states": bnb["explored"],
            "pruned_states": bnb["pruned"],
            "metrics": {
                "distance_km": None, "energy_kwh": None,
                "orders_done": len(selected), "orders_total": len(ORDERS),
                "time_min": None, "charge_min": None,
                "priority_score": sum(o["priority"] for o in selected),
                "remaining_kwh": None, "feasible": False, "within_budget": False,
            },
            "steps": [],
            "simulation_steps": [],
            "charging_events": [],
            "planned_route": ["Depot"],
            "route_order": [],
            "bnb_distance_km": None,
            "road_edges": ROAD_EDGES,
            "assumptions": _assumptions(),
        }

    planned_route = bnb["best_planned"]
    charging_events = bnb["best_events"]
    total_dist = bnb["best_cost_km"]
    total_energy = round(sum(e.get("energy_kwh", 0) for e in charging_events), 3)
    total_charge_min = round(
        sum(e.get("charge_minutes", 0) for e in charging_events if e["type"] == "charge"), 1
    )
    remaining_kwh = charging_events[-1].get("battery_after", 0) if charging_events else 0
    service_time = sum(o["service_min"] for o in selected)
    travel_time = total_dist / cfg["average_speed_kmh"] * 60.0
    total_time = round(travel_time + total_charge_min + service_time, 1)
    within_budget = total_dist <= cfg["route_budget_km"]
    priority_score = sum(o["priority"] for o in selected)

    sim_steps = build_simulation_steps(bnb, cfg, selected)

    n_charges = sum(1 for e in charging_events if e["type"] == "charge")
    explanation_steps = [
        {
            "title": "Validate inputs & Build Road Graph",
            "detail": (
                f"{len(selected)} order(s) selected: {', '.join(o['name'] for o in selected)}. "
                f"Battery {cfg['capacity_kwh']:.1f} kWh at {cfg['soc_pct']:.0f}% SOC. "
                f"Threshold: {cfg['threshold_pct']:.0f}% ({cfg['capacity_kwh'] * cfg['threshold_pct'] / 100:.2f} kWh). "
                f"Reserve: {cfg['reserve_pct']:.0f}% ({cfg['capacity_kwh'] * cfg['reserve_pct'] / 100:.2f} kWh). "
                f"Shortest paths between all road vertices precomputed using Dijkstra's algorithm."
            ),
            "formula": (
                f"Available = {cfg['capacity_kwh'] * cfg['soc_pct'] / 100:.2f} kWh | "
                f"Reserve = {cfg['capacity_kwh'] * cfg['reserve_pct'] / 100:.2f} kWh | "
                f"Threshold = {cfg['capacity_kwh'] * cfg['threshold_pct'] / 100:.2f} kWh"
            )
        },
        {
            "title": "Branch & Bound State Space",
            "detail": (
                f"State = (current_loc, delivered_mask, battery_kwh). "
                f"Transitions evaluate direct delivery if projected battery > threshold, "
                f"or evaluate all reachable charging stations if projected battery ≤ threshold. "
                f"Total candidate orderings: {len(selected)}! = {factorial(len(selected))}."
            ),
            "formula": "State = (u, delivered, bat) | Admissible LB = cost + min(u→rem) + MST(rem) + min(rem→Hub)"
        },
        {
            "title": "Search Tree Pruning",
            "detail": (
                f"Explored {bnb['explored']} states, pruned {bnb['pruned']} states. "
                f"Branches whose lower-bound estimate equals or exceeds the best complete route cost "
                f"are safely pruned without discarding any optimal solution."
            ),
            "formula": "Prune branch when: LB(node) ≥ best_cost"
        },
        {
            "title": "Battery Threshold & Charging Decisions",
            "detail": (
                f"At each stop, the optimizer checks: projected battery after reaching the next delivery "
                f"must remain strictly above {cfg['threshold_pct']:.0f}% SOC. "
                f"Whenever projected battery is equal to or below threshold, reachable chargers are evaluated. "
                f"The station providing the globally lowest-cost continuation is chosen."
            ),
            "formula": "Direct if: battery − E(cur, nxt) > threshold; else evaluate min-cost charger continuation"
        },
        {
            "title": "Optimal Route Discovered",
            "detail": (
                f"Global minimum-distance route: Depot → {' → '.join(bnb['best_route'])} → Hub. "
                f"Total travel distance: {total_dist:.2f} km with {n_charges} charging stop(s)."
            ),
            "formula": f"Objective: min total travel distance = {total_dist:.2f} km"
        },
        {
            "title": "Final Schedule & Metrics",
            "detail": (
                f"Route distance: {total_dist:.2f} km | Total time: {total_time:.1f} min "
                f"(travel {travel_time:.1f} min + service {service_time} min + charging {total_charge_min:.1f} min). "
                f"Battery remaining at Hub: {soc_pct(remaining_kwh, cfg['capacity_kwh']):.1f}%. "
                f"Budget status: {'Within budget ✓' if within_budget else '⚠ Exceeds budget'}."
            ),
            "formula": f"Time = ({total_dist:.2f} / {cfg['average_speed_kmh']:.0f}) × 60 + {service_time} + {total_charge_min:.1f} = {total_time:.1f} min"
        }
    ]

    return {
        "nodes": NODES,
        "orders": selected,
        "chargers": CHARGERS,
        "config": cfg,
        "feasible": True,
        "route_order": bnb["best_route"],
        "bnb_distance_km": total_dist,
        "planned_route": planned_route,
        "charging_events": charging_events,
        "charging_stops": [e for e in charging_events if e.get("type") == "charge"],
        "search_tree": bnb["search_tree"],
        "explored_states": bnb["explored"],
        "pruned_states": bnb["pruned"],
        "metrics": {
            "distance_km": total_dist,
            "energy_kwh": total_energy,
            "orders_done": len(selected),
            "orders_total": len(ORDERS),
            "time_min": total_time,
            "charge_min": total_charge_min,
            "priority_score": priority_score,
            "remaining_kwh": round(remaining_kwh, 3),
            "feasible": True,
            "within_budget": within_budget,
        },
        "steps": explanation_steps,
        "simulation_steps": sim_steps,
        "road_edges": ROAD_EDGES,
        "assumptions": _assumptions(),
    }


def _assumptions() -> list:
    return [
        "Road network is modeled as a connected graph G=(V,E) matching the simulated city map; all distances use Dijkstra shortest paths.",
        "Branch & Bound jointly optimizes delivery sequencing and charging station visits in a single exact search.",
        "Battery threshold decision rule strictly prevents direct moves when projected battery ≤ threshold, triggering charger evaluations.",
        "Charging station recharge restores the vehicle to full capacity (100% SOC) at rated charger power, maximizing range for remaining deliveries.",
        "Vehicle speed and charger power are assumed constant during travel and charging phases.",
    ]


# ---------------------------------------------------------------------------
# Flask HTTP Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        nodes=NODES,
        orders=list(ORDERS.values()),
        chargers=CHARGERS
    )


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(optimize(payload))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "VoltRoute optimizer"})


if __name__ == "__main__":
    app.run(debug=True)
