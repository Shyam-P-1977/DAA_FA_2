"""
VoltRoute — Optimal Delivery and Charging Planning for EV-Based Delivery Partners
Flask backend: User-Configurable Road Network, Dijkstra Shortest Paths, Two-Stage Simulation
(Stage 1: Unoptimized Baseline Route; Stage 2: Branch & Bound Battery-Aware Route),
and Verifiable Route Comparison.
"""

from flask import Flask, render_template, request, jsonify
from math import hypot, ceil, factorial, cos, sin, pi
from itertools import count, permutations
import heapq

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Default Sample Network Definition
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_NETWORK = {
    "start_node": {"id": "Depot", "name": "Main Depot", "x": 70, "y": 265},
    "end_node": {"id": "Hub", "name": "Return Hub", "x": 635, "y": 245},
    "delivery_nodes": [
        {"id": "Pharmacy", "name": "Pharmacy", "desc": "Medical Supplies", "x": 145, "y": 85},
        {"id": "Office", "name": "Office", "desc": "Corporate Documents", "x": 225, "y": 185},
        {"id": "Grocery", "name": "Grocery", "desc": "Fresh Produce", "x": 340, "y": 275},
    ],
    "charging_stations": [
        {"id": "CH-1", "name": "CH-1 Station", "rate_kw": 22.0, "cost_per_kwh": 0.20, "x": 445, "y": 165},
        {"id": "CH-2", "name": "CH-2 Station", "rate_kw": 22.0, "cost_per_kwh": 0.20, "x": 300, "y": 70},
        {"id": "CH-3", "name": "CH-3 Fast", "rate_kw": 50.0, "cost_per_kwh": 0.35, "x": 510, "y": 75},
    ],
    "roads": [
        {"from": "Depot", "to": "Pharmacy", "distance_km": 14.825, "bidirectional": True},
        {"from": "Pharmacy", "to": "Office", "distance_km": 9.618, "bidirectional": True},
        {"from": "Office", "to": "Grocery", "distance_km": 11.026, "bidirectional": True},
        {"from": "Grocery", "to": "CH-1", "distance_km": 12.042, "bidirectional": True},
        {"from": "CH-1", "to": "Hub", "distance_km": 14.825, "bidirectional": True},
        {"from": "Depot", "to": "Office", "distance_km": 13.064, "bidirectional": True},
        {"from": "Office", "to": "CH-2", "distance_km": 10.308, "bidirectional": True},
        {"from": "CH-2", "to": "Pharmacy", "distance_km": 11.673, "bidirectional": True},
        {"from": "CH-2", "to": "CH-3", "distance_km": 15.754, "bidirectional": True},
        {"from": "CH-3", "to": "Hub", "distance_km": 15.828, "bidirectional": True},
        {"from": "Grocery", "to": "CH-3", "distance_km": 19.686, "bidirectional": True},
        {"from": "CH-3", "to": "CH-1", "distance_km": 8.246, "bidirectional": True},
        {"from": "Pharmacy", "to": "Grocery", "distance_km": 20.408, "bidirectional": True},
        {"from": "Office", "to": "CH-1", "distance_km": 16.576, "bidirectional": True},
        {"from": "Depot", "to": "CH-2", "distance_km": 22.569, "bidirectional": True},
    ]
}

# Backward compatibility dictionaries & structures for existing tests
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


# ---------------------------------------------------------------------------
# Network Graph Builder & Dijkstra Engine
# ---------------------------------------------------------------------------

def compute_schematic_layout(start_id: str, end_id: str, delivery_ids: list, charger_ids: list) -> dict:
    """
    Computes an aesthetic, collision-free coordinate layout on a 740x360 SVG canvas
    for any user-configured network.
    """
    coords = {}
    is_round_trip = (start_id == end_id)

    # Start position
    coords[start_id] = {"x": 75, "y": 255}

    # End position
    if is_round_trip:
        pass
    else:
        coords[end_id] = {"x": 665, "y": 255}

    # Deliveries positioned in central columns
    n_del = len(delivery_ids)
    if n_del == 1:
        coords[delivery_ids[0]] = {"x": 370, "y": 170}
    else:
        left_x = 180
        right_x = 560
        span_x = (right_x - left_x) / max(1, n_del - 1)
        for i, d_id in enumerate(delivery_ids):
            x = int(left_x + i * span_x)
            # Alternate vertical bands
            if i % 3 == 0:
                y = 90
            elif i % 3 == 1:
                y = 200
            else:
                y = 290
            coords[d_id] = {"x": x, "y": y}

    # Chargers positioned in upper and middle strategic corridors
    n_ch = len(charger_ids)
    if n_ch == 1:
        coords[charger_ids[0]] = {"x": 370, "y": 80}
    elif n_ch == 2:
        coords[charger_ids[0]] = {"x": 290, "y": 70}
        coords[charger_ids[1]] = {"x": 480, "y": 80}
    else:
        ch_xs = [300, 480, 400, 240, 530]
        ch_ys = [70, 75, 170, 270, 180]
        for i, c_id in enumerate(charger_ids):
            coords[c_id] = {
                "x": ch_xs[i % len(ch_xs)],
                "y": ch_ys[i % len(ch_ys)]
            }

    return coords


def validate_network_spec(spec: dict) -> dict:
    """
    Validates a complete user-defined network specification.
    Returns: {"valid": bool, "errors": list[str], "warnings": list[str], "normalized": dict}
    """
    errors = []
    warnings = []

    if not isinstance(spec, dict):
        return {"valid": False, "errors": ["Network specification must be a JSON object."], "warnings": []}

    start_node = spec.get("start_node") or {}
    end_node = spec.get("end_node") or {}
    delivery_nodes = spec.get("delivery_nodes") or []
    charging_stations = spec.get("charging_stations") or []
    roads = spec.get("roads") or []

    start_id = str(start_node.get("id", "Depot")).strip()
    end_id = str(end_node.get("id", "Hub")).strip()

    if not start_id:
        errors.append("Start node ID cannot be empty.")
    if not end_id:
        errors.append("End node ID cannot be empty.")

    if not delivery_nodes or len(delivery_nodes) < 1:
        errors.append("At least 1 delivery location is required.")
    elif len(delivery_nodes) > 10:
        errors.append("Maximum 10 delivery locations allowed for exact optimization.")

    all_node_ids = set()
    node_id_counts = {}

    def track_id(nid, category):
        if not nid:
            errors.append(f"A {category} has an empty ID.")
            return
        node_id_counts[nid] = node_id_counts.get(nid, 0) + 1
        all_node_ids.add(nid)

    track_id(start_id, "start node")
    if end_id != start_id:
        track_id(end_id, "end node")

    del_ids = []
    for idx, d in enumerate(delivery_nodes):
        did = str(d.get("id", f"D{idx+1}")).strip()
        if not did:
            errors.append(f"Delivery node #{idx+1} has an empty ID.")
            continue
        del_ids.append(did)
        track_id(did, "delivery node")

    ch_ids = []
    for idx, c in enumerate(charging_stations):
        cid = str(c.get("id", f"C{idx+1}")).strip()
        if not cid:
            errors.append(f"Charging station #{idx+1} has an empty ID.")
            continue
        ch_ids.append(cid)
        track_id(cid, "charging station")
        rate = float(c.get("rate_kw", 22.0))
        if rate <= 0 or rate > 350:
            errors.append(f"Charging station '{cid}' has an invalid rate ({rate} kW). Must be 1–350 kW.")

    # Check for duplicate IDs
    for nid, count_val in node_id_counts.items():
        if count_val > 1:
            errors.append(f"Duplicate node ID detected: '{nid}'. All node IDs must be unique.")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Validate roads
    if not roads or len(roads) < 1:
        errors.append("The network must have at least one road connection.")

    adj = {nid: {} for nid in all_node_ids}
    normalized_roads = []
    for idx, r in enumerate(roads):
        u = str(r.get("from", "")).strip()
        v = str(r.get("to", "")).strip()
        try:
            dist = float(r.get("distance_km", 0.0))
        except (ValueError, TypeError):
            dist = 0.0

        bidi = bool(r.get("bidirectional", True))

        if u not in all_node_ids:
            errors.append(f"Road #{idx+1} refers to unknown source node '{u}'.")
        if v not in all_node_ids:
            errors.append(f"Road #{idx+1} refers to unknown destination node '{v}'.")
        if u == v:
            errors.append(f"Road #{idx+1} has identical source and destination '{u}' (self-loop not permitted).")
        if dist <= 0:
            errors.append(f"Road #{idx+1} ({u} → {v}) must have positive distance > 0 km (received {dist}).")

        if u in all_node_ids and v in all_node_ids and dist > 0 and u != v:
            # Add edge
            adj[u][v] = min(adj[u].get(v, float("inf")), round(dist, 3))
            if bidi:
                adj[v][u] = min(adj[v].get(u, float("inf")), round(dist, 3))
            normalized_roads.append({
                "from": u, "to": v, "distance_km": round(dist, 3), "bidirectional": bidi
            })

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Graph reachability check using Dijkstra
    for d_id in del_ids:
        dist_val, _ = dijkstra_on_adj(adj, start_id, d_id)
        if dist_val == float("inf"):
            errors.append(f"Delivery node '{d_id}' is unreachable from start node '{start_id}'.")
        dist_to_end, _ = dijkstra_on_adj(adj, d_id, end_id)
        if dist_to_end == float("inf"):
            errors.append(f"End node '{end_id}' is unreachable from delivery node '{d_id}'.")

    # Layout coordinates
    layout_defaults = compute_schematic_layout(start_id, end_id, del_ids, ch_ids)

    norm_start = {
        "id": start_id,
        "name": str(start_node.get("name", "Main Depot")),
        "kind": "depot",
        "x": int(start_node.get("x", layout_defaults[start_id]["x"])),
        "y": int(start_node.get("y", layout_defaults[start_id]["y"])),
    }
    norm_end = {
        "id": end_id,
        "name": str(end_node.get("name", "Return Hub")),
        "kind": "hub",
        "x": int(end_node.get("x", layout_defaults[end_id]["x"] if end_id in layout_defaults else 665)),
        "y": int(end_node.get("y", layout_defaults[end_id]["y"] if end_id in layout_defaults else 255)),
    }
    norm_del = []
    for d in delivery_nodes:
        did = str(d.get("id")).strip()
        norm_del.append({
            "id": did,
            "name": str(d.get("name", did)),
            "desc": str(d.get("desc", "")),
            "kind": "delivery",
            "x": int(d.get("x", layout_defaults.get(did, {}).get("x", 370))),
            "y": int(d.get("y", layout_defaults.get(did, {}).get("y", 180))),
        })
    norm_ch = []
    for c in charging_stations:
        cid = str(c.get("id")).strip()
        norm_ch.append({
            "id": cid,
            "name": str(c.get("name", cid)),
            "rate_kw": float(c.get("rate_kw", 22.0)),
            "cost_per_kwh": float(c.get("cost_per_kwh", 0.20)),
            "kind": "charger",
            "x": int(c.get("x", layout_defaults.get(cid, {}).get("x", 370))),
            "y": int(c.get("y", layout_defaults.get(cid, {}).get("y", 80))),
        })

    normalized = {
        "start_node": norm_start,
        "end_node": norm_end,
        "delivery_nodes": norm_del,
        "charging_stations": norm_ch,
        "roads": normalized_roads,
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized": normalized
    }


def dijkstra_on_adj(adj: dict, start: str, end: str) -> tuple[float, list[str]]:
    """Dijkstra shortest path on an arbitrary adjacency dict."""
    if start == end:
        return 0.0, [start]
    if start not in adj or end not in adj:
        return float("inf"), []

    dist = {k: float("inf") for k in adj}
    prev = {k: None for k in adj}
    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            break
        for v, w in adj[u].items():
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    if dist[end] == float("inf"):
        return float("inf"), []

    path = []
    curr = end
    while curr:
        path.append(curr)
        curr = prev[curr]
    path.reverse()
    return round(dist[end], 3), path


class NetworkGraph:
    """Holds a validated network specification, adjacency graph, and all-pairs shortest paths."""
    def __init__(self, spec: dict):
        val = validate_network_spec(spec)
        if not val["valid"]:
            raise ValueError("; ".join(val["errors"]))
        self.spec = val["normalized"]
        self.start_id = self.spec["start_node"]["id"]
        self.end_id = self.spec["end_node"]["id"]
        self.delivery_ids = [d["id"] for d in self.spec["delivery_nodes"]]
        self.charger_ids = [c["id"] for c in self.spec["charging_stations"]]
        self.chargers_dict = {c["id"]: c for c in self.spec["charging_stations"]}
        self.deliveries_dict = {d["id"]: d for d in self.spec["delivery_nodes"]}

        # All node IDs
        self.all_nodes = {
            self.start_id: self.spec["start_node"],
            self.end_id: self.spec["end_node"],
            **{d["id"]: d for d in self.spec["delivery_nodes"]},
            **{c["id"]: c for c in self.spec["charging_stations"]},
        }

        # Build adjacency
        self.adj = {nid: {} for nid in self.all_nodes}
        for road in self.spec["roads"]:
            u, v = road["from"], road["to"]
            d = road["distance_km"]
            self.adj[u][v] = min(self.adj[u].get(v, float("inf")), d)
            if road["bidirectional"]:
                self.adj[v][u] = min(self.adj[v].get(u, float("inf")), d)

        # Precompute all-pairs shortest paths
        self.dist_matrix = {}
        self.path_matrix = {}
        for u in self.all_nodes:
            for v in self.all_nodes:
                d, p = dijkstra_on_adj(self.adj, u, v)
                self.dist_matrix[(u, v)] = d
                self.path_matrix[(u, v)] = p

    def distance(self, u: str, v: str) -> float:
        return self.dist_matrix.get((u, v), float("inf"))

    def shortest_path(self, u: str, v: str) -> list[str]:
        return self.path_matrix.get((u, v), [u, v])


# Global default graph instance for backward compatibility
_DEFAULT_GRAPH = NetworkGraph(DEFAULT_SAMPLE_NETWORK)

def dijkstra(start: str, end: str) -> tuple[float, list[str]]:
    return _DEFAULT_GRAPH.distance(start, end), _DEFAULT_GRAPH.shortest_path(start, end)

def shortest_distance(a: str, b: str) -> float:
    return _DEFAULT_GRAPH.distance(a, b)

def shortest_path(a: str, b: str) -> list[str]:
    return _DEFAULT_GRAPH.shortest_path(a, b)

def distance(a: str, b: str) -> float:
    return _DEFAULT_GRAPH.distance(a, b)


# ---------------------------------------------------------------------------
# Minimum Spanning Tree (MST) Lower Bound (Prim's Algorithm)
# ---------------------------------------------------------------------------

def mst_cost(nodes: list, dist_fn=None) -> float:
    """Minimum spanning tree cost over a set of node names using Prim's algorithm."""
    if dist_fn is None:
        dist_fn = distance
    if len(nodes) <= 1:
        return 0.0
    remaining = set(nodes)
    first = next(iter(remaining))
    remaining.remove(first)
    best = {node: dist_fn(first, node) for node in remaining}
    total = 0.0
    while remaining:
        node = min(remaining, key=lambda n: best[n])
        total += best[node]
        remaining.remove(node)
        for other in remaining:
            best[other] = min(best[other], dist_fn(node, other))
    return round(total, 3)


def lower_bound(current: str, remaining: list, end: str = "Hub", dist_fn=None) -> float:
    """
    Admissible B&B lower bound for remaining deliveries.
    LB = min(current→remaining) + MST(remaining) + min(remaining→end)
    Never overestimates the true completion distance.
    """
    if dist_fn is None:
        dist_fn = distance
    if not remaining:
        return dist_fn(current, end)
    return round(
        min(dist_fn(current, node) for node in remaining)
        + mst_cost(remaining, dist_fn)
        + min(dist_fn(node, end) for node in remaining),
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


def evaluate_battery_decision(
    battery_kwh: float,
    capacity_kwh: float,
    threshold_pct: float,
    reserve_pct: float,
    cur: str,
    nxt: str,
    eff: float,
    chargers: list = None,
    dist_fn = None
) -> dict:
    """
    Evaluates battery feasibility for traveling from `cur` to `nxt`.
    Distinguishes:
      - 'above_threshold': strictly above threshold T and satisfying reserve R
      - 'threshold_equal': projected battery exactly equal to threshold T
      - 'below_threshold': projected battery falls below threshold T (but >= reserve R)
      - 'insufficient': projected battery falls below safety reserve R
    """
    if chargers is None:
        chargers = CHARGERS
    if dist_fn is None:
        dist_fn = distance

    threshold_kwh = capacity_kwh * threshold_pct / 100.0
    reserve_kwh = capacity_kwh * reserve_pct / 100.0
    leg_dist = dist_fn(cur, nxt)
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
    for ch in chargers:
        if ch == cur:
            continue
        dist_to_ch = dist_fn(cur, ch)
        if dist_to_ch == float("inf"):
            continue
        e_to_ch = energy_for_leg(dist_to_ch, eff)
        bat_at_ch = battery_kwh - e_to_ch
        reachable = (bat_at_ch >= reserve_kwh - 1e-9)

        dist_from_ch = dist_fn(ch, nxt)
        e_from_ch = energy_for_leg(dist_from_ch, eff) if dist_from_ch != float("inf") else float("inf")
        add_kwh = max(0.0, capacity_kwh - bat_at_ch) if reachable else 0.0
        bat_at_nxt = capacity_kwh - e_from_ch if reachable and dist_from_ch != float("inf") else 0.0
        continuation_feasible = reachable and (dist_from_ch != float("inf")) and (bat_at_nxt >= reserve_kwh - 1e-9)

        total_via = dist_to_ch + dist_from_ch if dist_from_ch != float("inf") else float("inf")
        detour_diff = total_via - leg_dist if total_via != float("inf") and leg_dist != float("inf") else 0.0

        charger_evaluations.append({
            "name": ch,
            "dist_to_charger_km": round(dist_to_ch, 3),
            "energy_to_charger_kwh": round(e_to_ch, 3),
            "battery_at_charger_kwh": round(bat_at_ch, 4),
            "battery_at_charger_pct": soc_pct(bat_at_ch, capacity_kwh),
            "energy_added_kwh": round(add_kwh, 3),
            "dist_from_charger_km": round(dist_from_ch, 3) if dist_from_ch != float("inf") else None,
            "energy_from_charger_kwh": round(e_from_ch, 3) if e_from_ch != float("inf") else None,
            "battery_at_next_kwh": round(bat_at_nxt, 4) if continuation_feasible else None,
            "battery_at_next_pct": soc_pct(bat_at_nxt, capacity_kwh) if continuation_feasible else None,
            "total_route_via_km": round(total_via, 3) if total_via != float("inf") else None,
            "detour_increase_km": round(detour_diff, 3) if total_via != float("inf") else None,
            "reachable": reachable,
            "continuation_feasible": continuation_feasible,
            "status_detail": (
                "Reachable & feasible continuation" if continuation_feasible
                else "Unreachable (violates safety reserve)" if not reachable
                else "Cannot reach next destination from charger"
            )
        })

    return {
        "status": status,
        "status_label": status_label,
        "can_proceed_direct": can_proceed,
        "reason": reason,
        "current_battery_kwh": round(battery_kwh, 4),
        "current_soc_pct": soc_pct(battery_kwh, capacity_kwh),
        "needed_kwh": round(needed_kwh, 4),
        "projected_battery_kwh": round(projected, 4),
        "projected_soc_pct": projected_soc,
        "threshold_kwh": round(threshold_kwh, 4),
        "threshold_pct": threshold_pct,
        "reserve_kwh": round(reserve_kwh, 4),
        "reserve_pct": reserve_pct,
        "charger_candidates": charger_evaluations,
    }


# ---------------------------------------------------------------------------
# Stage 1: Unoptimized Baseline Delivery Route
# ---------------------------------------------------------------------------

def solve_baseline_route(graph: NetworkGraph, cfg: dict, order_sequence: list = None) -> dict:
    """
    Computes Stage 1: Unoptimized Baseline Delivery Route.
    Strategy:
      1. Deliver orders in user-entered sequence D1 → D2 → ... → Dn.
      2. Follow Dijkstra shortest path between consecutive locations.
      3. Complete tour at configured end node (Hub).
      4. If battery falls <= threshold or < reserve, insert the geographically
         nearest reachable charging station that permits a feasible continuation.
    """
    capacity = cfg["capacity_kwh"]
    soc0 = cfg["soc_pct"]
    eff = cfg["km_per_kwh"]
    speed = cfg["average_speed_kmh"]
    ch_kw = cfg["charger_kw"]
    reserve_pct = cfg["reserve_pct"]
    threshold_pct = cfg["threshold_pct"]

    battery0 = capacity * soc0 / 100.0

    if order_sequence is None:
        order_sequence = list(graph.delivery_ids)

    cur = graph.start_id
    bat = battery0
    cost_acc = 0.0
    energy_acc = 0.0
    charge_min_acc = 0.0

    events = []
    planned = [cur]
    explanation_steps = []

    # Target sequence: delivery orders then ending hub
    targets = [(d_id, "delivery") for d_id in order_sequence] + [(graph.end_id, "finish")]

    is_feasible = True
    infeasibility_reason = ""

    for target_idx, (tgt, target_type) in enumerate(targets):
        leg_dist = graph.distance(cur, tgt)
        if leg_dist == float("inf"):
            is_feasible = False
            infeasibility_reason = f"No road path exists between {cur} and {tgt} in the configured network."
            break

        dec = evaluate_battery_decision(
            battery_kwh=bat,
            capacity_kwh=capacity,
            threshold_pct=threshold_pct,
            reserve_pct=reserve_pct,
            cur=cur,
            nxt=tgt,
            eff=eff,
            chargers=graph.charger_ids,
            dist_fn=graph.distance
        )

        # Baseline Strategy: If can proceed direct without violating threshold, proceed direct
        if dec["can_proceed_direct"]:
            e_leg = energy_for_leg(leg_dist, eff)
            new_bat = bat - e_leg
            cost_acc += leg_dist
            energy_acc += e_leg

            events.append({
                "type": "battery_decision",
                "from": cur, "to": tgt,
                "battery_decision": dec,
                "charger_candidates": dec["charger_candidates"],
                "selected_charger": None,
                "battery_before": round(bat, 4),
                "battery_after": round(new_bat, 4),
                "soc_before": soc_pct(bat, capacity),
                "soc_after": soc_pct(new_bat, capacity),
                "message": (
                    f"Baseline rule: Direct travel from {cur} to {tgt} ({leg_dist:.2f} km) approved. "
                    f"Projected battery ({soc_pct(new_bat, capacity):.1f}%) is strictly above {threshold_pct:.0f}% threshold."
                )
            })

            ev_type = "delivery" if target_type == "delivery" else "finish"
            events.append({
                "type": ev_type,
                "from": cur, "to": tgt,
                "distance_km": round(leg_dist, 3),
                "energy_kwh": round(e_leg, 3),
                "battery_before": round(bat, 4),
                "battery_after": round(new_bat, 4),
                "soc_before": soc_pct(bat, capacity),
                "soc_after": soc_pct(new_bat, capacity),
                "message": f"Traveled {cur} → {tgt} ({leg_dist:.2f} km, consumed {e_leg:.2f} kWh)."
            })

            planned.append(tgt)
            bat = new_bat
            cur = tgt

        else:
            # Baseline Strategy: evaluate reachable chargers, pick nearest reachable charger with feasible continuation
            candidates = [c for c in dec["charger_candidates"] if c["continuation_feasible"]]
            if not candidates:
                # No feasible charger
                is_feasible = False
                infeasibility_reason = (
                    f"Baseline route failed at {cur}: Battery {soc_pct(bat, capacity):.1f}% is insufficient "
                    f"to reach {tgt} safely, and no reachable charging station permits a feasible continuation."
                )
                events.append({
                    "type": "battery_decision",
                    "from": cur, "to": tgt,
                    "battery_decision": dec,
                    "charger_candidates": dec["charger_candidates"],
                    "selected_charger": None,
                    "battery_before": round(bat, 4),
                    "battery_after": round(bat, 4),
                    "soc_before": soc_pct(bat, capacity),
                    "soc_after": soc_pct(bat, capacity),
                    "message": infeasibility_reason
                })
                break

            # Baseline picks geographically nearest charger
            chosen_cand = min(candidates, key=lambda c: c["dist_to_charger_km"])
            ch_name = chosen_cand["name"]
            dist_to_ch = chosen_cand["dist_to_charger_km"]
            e_to_ch = chosen_cand["energy_to_charger_kwh"]
            bat_at_ch = chosen_cand["battery_at_charger_kwh"]
            add_kwh = chosen_cand["energy_added_kwh"]
            ch_rate = graph.chargers_dict.get(ch_name, {}).get("rate_kw", ch_kw)
            ch_minutes = add_kwh / ch_rate * 60.0

            dist_from_ch = chosen_cand["dist_from_charger_km"]
            e_from_ch = chosen_cand["energy_from_charger_kwh"]
            bat_at_tgt = chosen_cand["battery_at_next_kwh"]

            sel_reason = {
                "charger": ch_name,
                "strategy": "Baseline: Geographically nearest reachable charger",
                "current_location": cur,
                "next_destination": tgt,
                "battery_before_travel_kwh": round(bat, 4),
                "battery_before_travel_pct": soc_pct(bat, capacity),
                "dist_to_charger_km": round(dist_to_ch, 3),
                "energy_to_charger_kwh": round(e_to_ch, 3),
                "battery_at_charger_kwh": round(bat_at_ch, 4),
                "battery_at_charger_pct": soc_pct(bat_at_ch, capacity),
                "energy_added_kwh": round(add_kwh, 3),
                "battery_after_charge_kwh": round(capacity, 4),
                "battery_after_charge_pct": 100.0,
                "dist_from_charger_km": round(dist_from_ch, 3),
                "energy_from_charger_kwh": round(e_from_ch, 3),
                "battery_at_destination_kwh": round(bat_at_tgt, 4),
                "battery_at_destination_pct": soc_pct(bat_at_tgt, capacity),
                "threshold_kwh": round(dec["threshold_kwh"], 4),
                "threshold_pct": threshold_pct,
                "reserve_kwh": round(dec["reserve_kwh"], 4),
                "reserve_pct": reserve_pct,
                "projected_direct_kwh": round(dec["projected_battery_kwh"], 4),
                "projected_direct_pct": dec["projected_soc_pct"],
                "direct_dist_km": round(leg_dist, 3),
                "total_via_km": round(chosen_cand["total_route_via_km"], 3),
                "detour_increase_km": round(chosen_cand["detour_increase_km"], 3),
                "explanation": (
                    f"Baseline decision: Selected nearest reachable charger {ch_name} ({dist_to_ch:.2f} km). "
                    f"Direct travel would leave {dec['projected_soc_pct']:.1f}% battery, failing threshold. "
                    f"Recharging to 100% (+{add_kwh:.2f} kWh in {ch_minutes:.1f} min) enables reaching {tgt} safely. "
                    f"Detour increases route distance by +{chosen_cand['detour_increase_km']:.2f} km."
                )
            }

            events.append({
                "type": "battery_decision",
                "from": cur, "to": tgt,
                "battery_decision": dec,
                "charger_candidates": dec["charger_candidates"],
                "selected_charger": ch_name,
                "selection_reason": sel_reason,
                "battery_before": round(bat, 4),
                "battery_after": round(bat, 4),
                "soc_before": soc_pct(bat, capacity),
                "soc_after": soc_pct(bat, capacity),
                "message": (
                    f"Battery at {cur} ({soc_pct(bat, capacity):.1f}%) requires charging. "
                    f"Baseline selected nearest reachable station {ch_name}."
                )
            })

            # Travel to charger
            cost_acc += dist_to_ch
            energy_acc += e_to_ch
            events.append({
                "type": "travel_to_charger",
                "from": cur, "to": ch_name,
                "distance_km": round(dist_to_ch, 3),
                "energy_kwh": round(e_to_ch, 3),
                "battery_before": round(bat, 4),
                "battery_after": round(bat_at_ch, 4),
                "soc_before": soc_pct(bat, capacity),
                "soc_after": soc_pct(bat_at_ch, capacity),
                "message": f"Baseline travel {cur} → {ch_name}: {dist_to_ch:.2f} km, use {e_to_ch:.2f} kWh."
            })

            # Charge
            charge_min_acc += ch_minutes
            events.append({
                "type": "charge",
                "station": ch_name,
                "added_kwh": round(add_kwh, 3),
                "charge_minutes": round(ch_minutes, 1),
                "battery_before": round(bat_at_ch, 4),
                "battery_after": round(capacity, 4),
                "soc_before": soc_pct(bat_at_ch, capacity),
                "soc_after": 100.0,
                "message": f"Recharged +{add_kwh:.2f} kWh at {ch_name} ({ch_minutes:.1f} min)."
            })

            # Continue to destination
            cost_acc += dist_from_ch
            energy_acc += e_from_ch
            ev_type = "delivery" if target_type == "delivery" else "finish"
            events.append({
                "type": ev_type,
                "from": ch_name, "to": tgt,
                "distance_km": round(dist_from_ch, 3),
                "energy_kwh": round(e_from_ch, 3),
                "battery_before": round(capacity, 4),
                "battery_after": round(bat_at_tgt, 4),
                "soc_before": 100.0,
                "soc_after": soc_pct(bat_at_tgt, capacity),
                "message": f"Travel {ch_name} → {tgt}: {dist_from_ch:.2f} km, use {e_from_ch:.2f} kWh."
            })

            planned.extend([ch_name, tgt])
            bat = bat_at_tgt
            cur = tgt

    # Metrics
    service_time = len(order_sequence) * 4.0  # 4 min standard service per delivery
    travel_time = (cost_acc / speed * 60.0) if speed > 0 else 0.0
    total_time = round(travel_time + charge_min_acc + service_time, 1)
    within_budget = cost_acc <= cfg.get("route_budget_km", 9999.0)

    # Simulation steps
    sim_steps = build_graph_simulation_steps(events, graph, cfg)

    return {
        "stage": "baseline",
        "feasible": is_feasible,
        "infeasibility_reason": infeasibility_reason if not is_feasible else None,
        "planned_route": planned,
        "delivery_order": order_sequence,
        "charging_events": events,
        "charging_stops": [e for e in events if e.get("type") == "charge"],
        "metrics": {
            "distance_km": round(cost_acc, 3) if is_feasible else None,
            "energy_kwh": round(energy_acc, 3) if is_feasible else None,
            "orders_done": len(order_sequence) if is_feasible else 0,
            "orders_total": len(order_sequence),
            "time_min": total_time if is_feasible else None,
            "charge_min": round(charge_min_acc, 1) if is_feasible else None,
            "remaining_kwh": round(bat, 3) if is_feasible else None,
            "remaining_pct": soc_pct(bat, capacity) if is_feasible else None,
            "feasible": is_feasible,
            "within_budget": within_budget,
        },
        "simulation_steps": sim_steps,
    }


# ---------------------------------------------------------------------------
# Stage 2: Branch & Bound Optimization Engine
# ---------------------------------------------------------------------------

def branch_and_bound_with_charging(order_nodes: list, cfg: dict, graph: NetworkGraph = None) -> dict:
    """
    Branch and Bound solver with joint charging station decisions.
    Works seamlessly on both default sample network and custom user graphs.
    """
    if graph is None:
        graph = _DEFAULT_GRAPH

    capacity = cfg["capacity_kwh"]
    soc0 = cfg["soc_pct"]
    eff = cfg["km_per_kwh"]
    speed = cfg.get("average_speed_kmh", 30.0)
    ch_kw = cfg.get("charger_kw", 22.0)
    reserve_pct = cfg["reserve_pct"]
    threshold_pct = cfg["threshold_pct"]
    budget_km = cfg.get("route_budget_km", 9999.0)

    battery0 = capacity * soc0 / 100.0
    threshold_kwh = capacity * threshold_pct / 100.0
    reserve_kwh = capacity * reserve_pct / 100.0

    # Search tree container
    tree = {}
    node_id_counter = count(1)
    serial_gen = count(1)

    def new_tree_node(parent_id, label, cost, lb, depth):
        nid = next(node_id_counter)
        tree[nid] = {
            "id": nid, "parent": parent_id, "label": label,
            "cost": round(cost, 2), "lb": round(lb, 2),
            "depth": depth, "status": "candidate"
        }
        return nid

    init_lb = lower_bound(graph.start_id, order_nodes, graph.end_id, graph.distance)
    root_id = new_tree_node(None, f"Start: {graph.start_id}", 0.0, init_lb, 0)
    tree[root_id]["status"] = "root"

    heap = []
    heapq.heappush(heap, (
        init_lb, next(serial_gen),
        graph.start_id, frozenset(), 0.0, battery0, [], [graph.start_id], [], root_id
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

        if lb >= best_cost - 1e-9:
            tree[tid]["status"] = "pruned"
            pruned += 1
            continue

        explored += 1
        tree[tid]["status"] = "expanded"

        remaining = [nd for nd in order_nodes if nd not in delivered]

        # Terminal state: all deliveries visited, travel to configured end node
        if not remaining:
            leg_to_end = graph.distance(cur, graph.end_id)
            if leg_to_end == float("inf"):
                continue

            needed_end = energy_for_leg(leg_to_end, eff)
            proj_end = bat - needed_end

            dec_end = evaluate_battery_decision(
                battery_kwh=bat,
                capacity_kwh=capacity,
                threshold_pct=threshold_pct,
                reserve_pct=reserve_pct,
                cur=cur,
                nxt=graph.end_id,
                eff=eff,
                chargers=graph.charger_ids,
                dist_fn=graph.distance
            )

            # Option A: Direct finish
            if dec_end["can_proceed_direct"]:
                total_direct = cost_so_far + leg_to_end
                end_events = events + [
                    {
                        "type": "battery_decision",
                        "from": cur, "to": graph.end_id,
                        "battery_decision": dec_end,
                        "charger_candidates": dec_end["charger_candidates"],
                        "selected_charger": None,
                        "battery_before": round(bat, 4),
                        "battery_after": round(proj_end, 4),
                        "soc_before": soc_pct(bat, capacity),
                        "soc_after": soc_pct(proj_end, capacity),
                        "message": (
                            f"At {cur} before {graph.end_id}: Battery {soc_pct(bat, capacity):.1f}% ({bat:.2f} kWh). "
                            f"Direct travel to {graph.end_id} ({leg_to_end:.2f} km) requires {needed_end:.2f} kWh. "
                            f"Projected battery at {graph.end_id} is {soc_pct(proj_end, capacity):.1f}% ({proj_end:.2f} kWh), "
                            f"strictly above {threshold_pct:.0f}% threshold. Direct arrival approved."
                        )
                    },
                    {
                        "type": "finish",
                        "from": cur, "to": graph.end_id,
                        "distance_km": round(leg_to_end, 3),
                        "energy_kwh": round(needed_end, 3),
                        "battery_before": round(bat, 4),
                        "battery_after": round(proj_end, 4),
                        "soc_before": soc_pct(bat, capacity),
                        "soc_after": soc_pct(proj_end, capacity),
                        "message": f"Travel {cur} → {graph.end_id}: {leg_to_end:.2f} km, use {needed_end:.2f} kWh."
                    }
                ]
                if total_direct < best_cost:
                    best_cost = total_direct
                    best_route = list(path)
                    best_planned = list(planned) + [graph.end_id]
                    best_events = end_events
                    lid = new_tree_node(
                        tid, f"{graph.end_id} · {total_direct:.2f} km",
                        total_direct, total_direct, len(path) + 1
                    )
                    tree[lid]["status"] = "best"

            # Option B: Charger detour before returning to End node
            else:
                for cand in dec_end["charger_candidates"]:
                    if not cand["continuation_feasible"]:
                        continue
                    ch = cand["name"]
                    to_ch = cand["dist_to_charger_km"]
                    e_to_ch = cand["energy_to_charger_kwh"]
                    bat_at_ch = cand["battery_at_charger_kwh"]
                    add_kwh = cand["energy_added_kwh"]
                    rate_kw = graph.chargers_dict.get(ch, {}).get("rate_kw", ch_kw)
                    charge_min = add_kwh / rate_kw * 60.0

                    ch_to_end = cand["dist_from_charger_km"]
                    e_ch_to_end = cand["energy_from_charger_kwh"]
                    bat_at_end = cand["battery_at_next_kwh"]

                    total_via = cost_so_far + to_ch + ch_to_end
                    if total_via < best_cost:
                        detour_diff = (to_ch + ch_to_end) - leg_to_end
                        sel_reason = {
                            "charger": ch,
                            "current_location": cur,
                            "next_destination": graph.end_id,
                            "battery_before_travel_kwh": round(bat, 4),
                            "battery_before_travel_pct": soc_pct(bat, capacity),
                            "dist_to_charger_km": round(to_ch, 3),
                            "energy_to_charger_kwh": round(e_to_ch, 3),
                            "battery_at_charger_kwh": round(bat_at_ch, 4),
                            "battery_at_charger_pct": soc_pct(bat_at_ch, capacity),
                            "energy_added_kwh": round(add_kwh, 3),
                            "battery_after_charge_kwh": round(capacity, 4),
                            "battery_after_charge_pct": 100.0,
                            "dist_from_charger_km": round(ch_to_end, 3),
                            "energy_from_charger_kwh": round(e_ch_to_end, 3),
                            "battery_at_destination_kwh": round(bat_at_end, 4),
                            "battery_at_destination_pct": soc_pct(bat_at_end, capacity),
                            "threshold_kwh": round(threshold_kwh, 4),
                            "threshold_pct": threshold_pct,
                            "reserve_kwh": round(reserve_kwh, 4),
                            "reserve_pct": reserve_pct,
                            "projected_direct_kwh": round(dec_end["projected_battery_kwh"], 4),
                            "projected_direct_pct": dec_end["projected_soc_pct"],
                            "direct_dist_km": round(leg_to_end, 3),
                            "total_via_km": round(to_ch + ch_to_end, 3),
                            "detour_increase_km": round(detour_diff, 3),
                            "explanation": (
                                f"Charging at {ch} selected: Direct travel {cur} → {graph.end_id} ({leg_to_end:.2f} km) "
                                f"would leave {dec_end['projected_soc_pct']:.1f}% battery, failing threshold. "
                                f"Recharging at {ch} allows reaching {graph.end_id} with {soc_pct(bat_at_end, capacity):.1f}% battery. "
                                f"Selected because it minimizes total route distance among all feasible alternatives."
                            )
                        }
                        fin_events = events + [
                            {
                                "type": "battery_decision",
                                "from": cur, "to": graph.end_id,
                                "battery_decision": dec_end,
                                "charger_candidates": dec_end["charger_candidates"],
                                "selected_charger": ch,
                                "selection_reason": sel_reason,
                                "message": (
                                    f"Battery {soc_pct(bat, capacity):.1f}% — {dec_end['status_label']}. "
                                    f"Selected {ch} for lowest-cost feasible return to {graph.end_id}."
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
                                "message": f"Charge {add_kwh:.2f} kWh at {ch} ({charge_min:.1f} min)."
                            },
                            {
                                "type": "finish",
                                "from": ch, "to": graph.end_id,
                                "distance_km": round(ch_to_end, 3),
                                "energy_kwh": round(e_ch_to_end, 3),
                                "battery_before": round(capacity, 4),
                                "battery_after": round(bat_at_end, 4),
                                "soc_before": 100.0,
                                "soc_after": soc_pct(bat_at_end, capacity),
                                "message": f"Travel {ch} → {graph.end_id}: {ch_to_end:.2f} km, use {e_ch_to_end:.2f} kWh."
                            }
                        ]
                        best_cost = total_via
                        best_route = list(path)
                        best_planned = list(planned) + [ch, graph.end_id]
                        best_events = fin_events
                        lid = new_tree_node(
                            tid, f"{graph.end_id} via {ch} · {total_via:.2f} km",
                            total_via, total_via, len(path) + 2
                        )
                        tree[lid]["status"] = "best"
            continue

        # Expand each remaining delivery
        for nxt in remaining:
            leg = graph.distance(cur, nxt)
            if leg == float("inf"):
                continue

            needed = energy_for_leg(leg, eff)
            dec = evaluate_battery_decision(
                battery_kwh=bat,
                capacity_kwh=capacity,
                threshold_pct=threshold_pct,
                reserve_pct=reserve_pct,
                cur=cur,
                nxt=nxt,
                eff=eff,
                chargers=graph.charger_ids,
                dist_fn=graph.distance
            )

            # Option A: Direct delivery (if projected battery remains strictly above threshold)
            if dec["can_proceed_direct"]:
                new_bat = bat - needed
                new_cost = cost_so_far + leg
                new_del = delivered | {nxt}
                new_path = path + [nxt]
                new_planned = planned + [nxt]
                new_rem = [nd for nd in order_nodes if nd not in new_del]
                lb_direct = round(new_cost + lower_bound(nxt, new_rem, graph.end_id, graph.distance), 3)

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
                            f"strictly above {threshold_pct:.0f}% threshold. Direct delivery approved."
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
                        "message": f"Travel {cur} → {nxt}: {leg:.2f} km, use {needed:.2f} kWh."
                    }
                ]

                nid = new_tree_node(tid, f"{nxt} (direct)", new_cost, lb_direct, len(new_path))
                if lb_direct < best_cost:
                    heapq.heappush(heap, (
                        lb_direct, next(serial_gen),
                        nxt, new_del, new_cost, new_bat, new_path, new_planned, new_ev, nid
                    ))
                else:
                    tree[nid]["status"] = "pruned"
                    pruned += 1

            # Option B: Charging detour required before reaching next delivery
            else:
                for cand in dec["charger_candidates"]:
                    if not cand["continuation_feasible"]:
                        continue
                    ch = cand["name"]
                    to_ch = cand["dist_to_charger_km"]
                    e_to_ch = cand["energy_to_charger_kwh"]
                    bat_at_ch = cand["battery_at_charger_kwh"]
                    add_kwh = cand["energy_added_kwh"]
                    rate_kw = graph.chargers_dict.get(ch, {}).get("rate_kw", ch_kw)
                    charge_min = add_kwh / rate_kw * 60.0

                    ch_to_nxt = cand["dist_from_charger_km"]
                    e_ch_to_nxt = cand["energy_from_charger_kwh"]
                    bat_after_delivery = cand["battery_at_next_kwh"]

                    detour_cost = cost_so_far + to_ch + ch_to_nxt
                    new_del = delivered | {nxt}
                    new_path = path + [nxt]
                    new_planned = planned + [ch, nxt]
                    new_rem = [nd for nd in order_nodes if nd not in new_del]
                    lb_ch = round(detour_cost + lower_bound(nxt, new_rem, graph.end_id, graph.distance), 3)

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
                        "battery_at_destination_kwh": round(bat_after_delivery, 4),
                        "battery_at_destination_pct": soc_pct(bat_after_delivery, capacity),
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
                            f"Charging at {ch} evaluated: Direct travel {cur} → {nxt} ({leg:.2f} km) would leave "
                            f"{dec['projected_soc_pct']:.1f}% battery, failing threshold. "
                            f"Recharging at {ch} to 100% (+{add_kwh:.2f} kWh in {charge_min:.1f} min) enables reaching {nxt} "
                            f"with {soc_pct(bat_after_delivery, capacity):.1f}% battery. Detour effect: +{detour_diff:.2f} km."
                        )
                    }

                    ev_ch = events + [
                        {
                            "type": "battery_decision",
                            "from": cur, "to": nxt,
                            "battery_decision": dec,
                            "charger_candidates": dec["charger_candidates"],
                            "selected_charger": ch,
                            "selection_reason": sel_reason,
                            "message": (
                                f"Battery {soc_pct(bat, capacity):.1f}% — {dec['status_label']}. "
                                f"Branch evaluates detour via {ch} to reach {nxt} safely."
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
                            "message": f"Detour {cur} → {ch}: {to_ch:.2f} km, use {e_to_ch:.2f} kWh."
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
                            "message": f"Charge {add_kwh:.2f} kWh at {ch} ({charge_min:.1f} min)."
                        },
                        {
                            "type": "delivery",
                            "from": ch, "to": nxt,
                            "distance_km": round(ch_to_nxt, 3),
                            "energy_kwh": round(e_ch_to_nxt, 3),
                            "battery_before": round(capacity, 4),
                            "battery_after": round(bat_after_delivery, 4),
                            "soc_before": 100.0,
                            "soc_after": soc_pct(bat_after_delivery, capacity),
                            "message": f"Deliver {ch} → {nxt}: {ch_to_nxt:.2f} km, use {e_ch_to_nxt:.2f} kWh."
                        }
                    ]

                    nid = new_tree_node(tid, f"{nxt} via {ch}", detour_cost, lb_ch, len(new_path))
                    if lb_ch < best_cost:
                        heapq.heappush(heap, (
                            lb_ch, next(serial_gen),
                            nxt, new_del, detour_cost, bat_after_delivery, new_path, new_planned, ev_ch, nid
                        ))
                    else:
                        tree[nid]["status"] = "pruned"
                        pruned += 1

    feasible = best_cost < float("inf")
    return {
        "feasible": feasible,
        "best_route": best_route,
        "best_planned": best_planned,
        "best_cost_km": round(best_cost, 3) if feasible else None,
        "best_events": best_events,
        "explored": explored,
        "pruned": pruned,
        "search_tree": list(tree.values()),
    }


def branch_and_bound(order_nodes: list, cfg: dict = None) -> tuple[float, list[str]]:
    """Classic Branch & Bound without charging (returns (best_distance_km, best_route))."""
    if cfg is None:
        cfg = {"capacity_kwh": 40, "soc_pct": 100, "km_per_kwh": 6, "average_speed_kmh": 30,
               "charger_kw": 22, "reserve_pct": 10, "threshold_pct": 20, "route_budget_km": 150}
    res = branch_and_bound_with_charging(order_nodes, cfg)
    return res["best_cost_km"], res["best_route"]


def exhaustive_search(order_nodes: list, cfg: dict, graph: NetworkGraph = None) -> tuple[float, list[str]]:
    """Brute-force verification over all order permutations."""
    if graph is None:
        graph = _DEFAULT_GRAPH

    capacity = cfg["capacity_kwh"]
    soc0 = cfg["soc_pct"]
    eff = cfg["km_per_kwh"]
    reserve_pct = cfg["reserve_pct"]
    threshold_pct = cfg["threshold_pct"]

    battery0 = capacity * soc0 / 100.0
    reserve_kwh = capacity * reserve_pct / 100.0
    threshold_kwh = capacity * threshold_pct / 100.0

    best_cost = float("inf")
    best_perm = None

    for perm in permutations(order_nodes):
        cur = graph.start_id
        bat = battery0
        cost = 0.0
        feasible_perm = True

        for nxt in perm:
            d_direct = graph.distance(cur, nxt)
            if d_direct == float("inf"):
                feasible_perm = False
                break
            e_direct = energy_for_leg(d_direct, eff)
            proj_bat = bat - e_direct

            if proj_bat > threshold_kwh and proj_bat >= reserve_kwh - 1e-9:
                cost += d_direct
                bat = proj_bat
                cur = nxt
            else:
                best_detour = float("inf")
                best_ch = None
                for ch in graph.charger_ids:
                    d1 = graph.distance(cur, ch)
                    d2 = graph.distance(ch, nxt)
                    if d1 == float("inf") or d2 == float("inf"):
                        continue
                    if bat - energy_for_leg(d1, eff) < reserve_kwh - 1e-9:
                        continue
                    if capacity - energy_for_leg(d2, eff) < reserve_kwh - 1e-9:
                        continue
                    if d1 + d2 < best_detour:
                        best_detour = d1 + d2
                        best_ch = ch

                if best_ch is None:
                    feasible_perm = False
                    break
                cost += best_detour
                bat = capacity - energy_for_leg(graph.distance(best_ch, nxt), eff)
                cur = nxt

        if not feasible_perm:
            continue

        # Leg to end node
        d_end = graph.distance(cur, graph.end_id)
        if d_end == float("inf"):
            continue
        e_end = energy_for_leg(d_end, eff)
        proj_end = bat - e_end

        if proj_end > threshold_kwh and proj_end >= reserve_kwh - 1e-9:
            cost += d_end
        else:
            best_detour = float("inf")
            for ch in graph.charger_ids:
                d1 = graph.distance(cur, ch)
                d2 = graph.distance(ch, graph.end_id)
                if d1 == float("inf") or d2 == float("inf"):
                    continue
                if bat - energy_for_leg(d1, eff) < reserve_kwh - 1e-9:
                    continue
                if capacity - energy_for_leg(d2, eff) < reserve_kwh - 1e-9:
                    continue
                if d1 + d2 < best_detour:
                    best_detour = d1 + d2
            if best_detour == float("inf"):
                continue
            cost += best_detour

        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)

    feasible = (best_cost < float("inf"))
    return {
        "feasible": feasible,
        "best_cost_km": round(best_cost, 3) if feasible else None,
        "best_route": best_perm,
    }


# ---------------------------------------------------------------------------
# Granular Simulation Steps Generator
# ---------------------------------------------------------------------------

def build_graph_simulation_steps(events: list, graph: NetworkGraph, cfg: dict) -> list:
    """
    Decomposes route events into granular road-segment simulation steps
    along Dijkstra shortest paths.
    """
    capacity = cfg["capacity_kwh"]
    soc0 = cfg["soc_pct"]
    eff = cfg["km_per_kwh"]

    steps = [{
        "type": "start",
        "from": graph.start_id,
        "to": graph.start_id,
        "title": f"Start at {graph.start_id}",
        "detail": (
            f"EV departing from {graph.start_id}. Battery: {soc0:.1f}% "
            f"({capacity * soc0 / 100:.2f} kWh). Threshold: {cfg['threshold_pct']:.0f}% "
            f"({capacity * cfg['threshold_pct'] / 100:.2f} kWh). Reserve: {cfg['reserve_pct']:.0f}% "
            f"({capacity * cfg['reserve_pct'] / 100:.2f} kWh)."
        ),
        "formula": (
            f"Initial Energy = {capacity:.1f} kWh × {soc0:.0f}% = {capacity * soc0 / 100:.2f} kWh | "
            f"Threshold = {capacity * cfg['threshold_pct'] / 100:.2f} kWh"
        ),
        "battery_kwh": round(capacity * soc0 / 100, 4),
        "battery_pct": soc0,
        "distance_so_far_km": 0.0,
        "energy_so_far_kwh": 0.0,
        "dwell_ms": 1500,
    }]

    dist_acc = 0.0
    energy_acc = 0.0

    for ev in events:
        t = ev.get("type", "")

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
                    f"Selected charger: {ev.get('selected_charger', '—')}."
                ),
                "formula": (
                    f"Projected battery = {dec.get('projected_battery_kwh', 0):.2f} kWh "
                    f"({dec.get('projected_soc_pct', 0):.1f}%) | "
                    f"Threshold = {dec.get('threshold_kwh', 0):.2f} kWh"
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

        elif t in ("travel_to_charger", "delivery", "finish"):
            u_from = ev["from"]
            u_to = ev["to"]
            path_vertices = graph.shortest_path(u_from, u_to)
            n_segments = max(1, len(path_vertices) - 1)

            bat_start = ev.get("battery_before", 0)
            total_leg_energy = ev.get("energy_kwh", 0)
            total_leg_dist = ev.get("distance_km", 0)

            cur_bat = bat_start

            for s_idx in range(n_segments):
                seg_u = path_vertices[s_idx]
                seg_v = path_vertices[s_idx + 1]
                seg_dist = graph.distance(seg_u, seg_v)
                seg_energy = energy_for_leg(seg_dist, eff)
                next_bat = max(0.0, cur_bat - seg_energy)

                dist_acc += seg_dist
                energy_acc += seg_energy

                step_type = "travel_segment"
                is_last_segment = (s_idx == n_segments - 1)
                is_transit = not is_last_segment
                is_charger_node = seg_v in graph.charger_ids

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
                        f"Battery on arrival: {soc_pct(next_bat, capacity):.1f}% ({next_bat:.2f} kWh). Ready to charge."
                    )
                elif is_last_segment and t == "finish":
                    step_type = "finish"
                    title = f"Arrive at {u_to} (Complete)"
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
                        f"Current battery is {soc_pct(next_bat, capacity):.1f}%, strictly above threshold; "
                        f"no charging is performed at {seg_v}."
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
                    "is_charging_stop": (is_last_segment and t == "travel_to_charger"),
                    "title": title,
                    "detail": detail,
                    "formula": f"Energy = {seg_dist:.2f} km / {eff:.1f} km/kWh = {seg_energy:.2f} kWh | Battery: {soc_pct(cur_bat, capacity):.1f}% → {soc_pct(next_bat, capacity):.1f}%",
                    "segment_distance_km": round(seg_dist, 3),
                    "segment_energy_kwh": round(seg_energy, 3),
                    "battery_before_pct": soc_pct(cur_bat, capacity),
                    "battery_after_pct": soc_pct(next_bat, capacity),
                    "battery_kwh": round(next_bat, 4),
                    "battery_pct": soc_pct(next_bat, capacity),
                    "distance_so_far_km": round(dist_acc, 3),
                    "energy_so_far_kwh": round(energy_acc, 3),
                    "dwell_ms": 2000,
                })
                cur_bat = next_bat

        elif t == "charge":
            ch_station = ev.get("station", "Charger")
            added = ev.get("added_kwh", 0)
            charge_min = ev.get("charge_minutes", 0)
            soc_before = ev.get("soc_before", 0)
            soc_after = ev.get("soc_after", 100.0)
            bat_after = ev.get("battery_after", capacity)

            steps.append({
                "type": "charge",
                "from": ch_station,
                "to": ch_station,
                "title": f"Recharging at {ch_station} (+{added:.2f} kWh · {charge_min:.1f} min)",
                "detail": (
                    f"Charging at {ch_station} replenished {added:.2f} kWh in {charge_min:.1f} min. "
                    f"Battery: {soc_before:.1f}% → {soc_after:.1f}% ({bat_after:.2f} kWh). "
                    f"EV fully charged to 100% capacity."
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


def build_simulation_steps(bnb_result: dict, cfg: dict, selected_orders: list = None) -> list:
    """Wrapper for backward compatibility with existing tests."""
    return build_graph_simulation_steps(bnb_result.get("best_events", []), _DEFAULT_GRAPH, cfg)


# ---------------------------------------------------------------------------
# Route Comparison Logic
# ---------------------------------------------------------------------------

def compare_routes(baseline_res: dict, optimized_res: dict) -> dict:
    """
    Computes a rigorous, side-by-side comparison between Stage 1 (Baseline)
    and Stage 2 (Optimized Branch & Bound).
    """
    b_met = baseline_res.get("metrics", {})
    o_met = optimized_res.get("metrics", {})

    b_dist = b_met.get("distance_km")
    o_dist = o_met.get("distance_km")

    if b_dist is not None and o_dist is not None:
        dist_saved = round(b_dist - o_dist, 3)
        pct_improvement = round((dist_saved / b_dist * 100.0), 2) if b_dist > 0 else 0.0
    else:
        dist_saved = None
        pct_improvement = None

    b_energy = b_met.get("energy_kwh")
    o_energy = o_met.get("energy_kwh")
    energy_saved = round(b_energy - o_energy, 3) if (b_energy is not None and o_energy is not None) else None

    b_time = b_met.get("time_min")
    o_time = o_met.get("time_min")
    time_saved = round(b_time - o_time, 1) if (b_time is not None and o_time is not None) else None

    return {
        "baseline_distance_km": b_dist,
        "optimized_distance_km": o_dist,
        "distance_saved_km": dist_saved,
        "percentage_improvement": pct_improvement,
        "energy_saved_kwh": energy_saved,
        "time_saved_min": time_saved,
        "baseline_charges": len(baseline_res.get("charging_stops", [])),
        "optimized_charges": len(optimized_res.get("charging_stops", [])),
        "baseline_feasible": baseline_res.get("feasible", False),
        "optimized_feasible": optimized_res.get("feasible", False),
        "explored_states": optimized_res.get("explored_states"),
        "pruned_states": optimized_res.get("pruned_states"),
    }


# ---------------------------------------------------------------------------
# Main Optimization Entrypoint
# ---------------------------------------------------------------------------

def optimize(payload: dict) -> dict:
    """Validate inputs, run Branch & Bound, and construct complete response."""
    # Check if a custom network is provided
    custom_net = payload.get("network")
    if custom_net:
        graph = NetworkGraph(custom_net)
        delivery_nodes = graph.delivery_ids
    else:
        graph = _DEFAULT_GRAPH
        selected_ids = payload.get("order_ids", ["A", "B", "C"])
        selected_ids = [x for x in selected_ids if x in ORDERS]
        if not selected_ids:
            raise ValueError("Select at least one delivery order.")
        delivery_nodes = [ORDERS[x]["destination"] for x in selected_ids]

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

    # Run Branch & Bound
    bnb = branch_and_bound_with_charging(delivery_nodes, cfg, graph)

    if not bnb["feasible"]:
        reason = (
            "No feasible route exists: the EV cannot complete the selected deliveries "
            f"and reach {graph.end_id} without violating battery capacity or reserve constraints. "
            "Consider increasing starting SOC, raising battery capacity, or lowering threshold."
        )
        return {
            "network": graph.spec,
            "nodes": graph.all_nodes,
            "chargers": graph.charger_ids,
            "config": cfg,
            "feasible": False,
            "infeasibility_reason": reason,
            "search_tree": bnb["search_tree"],
            "explored_states": bnb["explored"],
            "pruned_states": bnb["pruned"],
            "metrics": {
                "distance_km": None, "energy_kwh": None,
                "orders_done": len(delivery_nodes), "orders_total": len(delivery_nodes),
                "time_min": None, "charge_min": None,
                "priority_score": 10,
                "remaining_kwh": None, "feasible": False, "within_budget": False,
            },
            "steps": [],
            "simulation_steps": [],
            "charging_events": [],
            "charging_stops": [],
            "planned_route": [graph.start_id],
            "route_order": [],
            "bnb_distance_km": None,
            "road_edges": [(r["from"], r["to"]) for r in graph.spec["roads"]],
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
    service_time = len(delivery_nodes) * 4.0
    travel_time = total_dist / cfg["average_speed_kmh"] * 60.0
    total_time = round(travel_time + total_charge_min + service_time, 1)
    within_budget = total_dist <= cfg["route_budget_km"]

    sim_steps = build_graph_simulation_steps(charging_events, graph, cfg)

    explanation_steps = [
        {
            "title": "Validate inputs & Build Road Graph",
            "detail": (
                f"{len(delivery_nodes)} delivery location(s): {', '.join(delivery_nodes)}. "
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
                f"Total candidate orderings: {len(delivery_nodes)}! = {factorial(len(delivery_nodes))}."
            ),
            "formula": f"State = (u, delivered, bat) | Admissible LB = cost + min(u→rem) + MST(rem) + min(rem→{graph.end_id})"
        },
        {
            "title": "Search Summary & Incumbent Pruning",
            "detail": (
                f"Expanded {bnb['explored']} state(s); pruned {bnb['pruned']} sub-optimal or infeasible branch(es). "
                f"Global minimum-cost feasible complete route: {' → '.join(planned_route)} "
                f"covering {total_dist:.2f} km."
            ),
            "formula": f"Objective: min total travel distance = {total_dist:.2f} km"
        },
        {
            "title": "Final Schedule & Metrics",
            "detail": (
                f"Route distance: {total_dist:.2f} km | Total time: {total_time:.1f} min "
                f"(travel {travel_time:.1f} min + service {service_time:.0f} min + charging {total_charge_min:.1f} min). "
                f"Battery remaining at {graph.end_id}: {soc_pct(remaining_kwh, cfg['capacity_kwh']):.1f}%. "
                f"Budget status: {'Within budget ✓' if within_budget else '⚠ Exceeds budget'}."
            ),
            "formula": f"Time = ({total_dist:.2f} / {cfg['average_speed_kmh']:.0f}) × 60 + {service_time} + {total_charge_min:.1f} = {total_time:.1f} min"
        }
    ]

    return {
        "network": graph.spec,
        "nodes": graph.all_nodes,
        "orders": [{"id": d, "name": d, "destination": d} for d in delivery_nodes],
        "chargers": graph.charger_ids,
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
            "orders_done": len(delivery_nodes),
            "orders_total": len(delivery_nodes),
            "time_min": total_time,
            "charge_min": total_charge_min,
            "priority_score": 10,
            "remaining_kwh": round(remaining_kwh, 3),
            "feasible": True,
            "within_budget": within_budget,
        },
        "steps": explanation_steps,
        "simulation_steps": sim_steps,
        "road_edges": [(r["from"], r["to"]) for r in graph.spec["roads"]],
        "assumptions": _assumptions(),
    }


def _assumptions() -> list:
    return [
        "Road network is modeled as a weighted graph G=(V,E); all travel distances follow Dijkstra shortest paths.",
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
        sample_network=DEFAULT_SAMPLE_NETWORK,
        nodes=NODES,
        orders=list(ORDERS.values()),
        chargers=CHARGERS
    )


@app.route("/api/sample_network", methods=["GET"])
def api_sample_network():
    return jsonify({
        "network": DEFAULT_SAMPLE_NETWORK,
        "config": {
            "capacity_kwh": 40.0,
            "soc_pct": 75.0,
            "km_per_kwh": 6.0,
            "average_speed_kmh": 30.0,
            "charger_kw": 22.0,
            "reserve_pct": 15.0,
            "threshold_pct": 25.0,
            "route_budget_km": 100.0,
        }
    })


@app.route("/api/validate_network", methods=["POST"])
def api_validate_network():
    payload = request.get_json(force=True) or {}
    spec = payload.get("network", payload)
    res = validate_network_spec(spec)
    return jsonify(res), (200 if res["valid"] else 400)


@app.route("/api/baseline", methods=["POST"])
def api_baseline():
    try:
        payload = request.get_json(force=True) or {}
        custom_net = payload.get("network")
        graph = NetworkGraph(custom_net) if custom_net else _DEFAULT_GRAPH

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

        order_seq = payload.get("order_sequence")
        res = solve_baseline_route(graph, cfg, order_seq)
        return jsonify(res)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(optimize(payload))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/compare", methods=["POST"])
def api_compare():
    try:
        payload = request.get_json(force=True) or {}
        baseline_res = payload.get("baseline")
        optimized_res = payload.get("optimized")
        if not baseline_res or not optimized_res:
            return jsonify({"error": "Both baseline and optimized results are required for comparison."}), 400
        comp = compare_routes(baseline_res, optimized_res)
        return jsonify(comp)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "VoltRoute optimizer"})


if __name__ == "__main__":
    app.run(debug=True)
