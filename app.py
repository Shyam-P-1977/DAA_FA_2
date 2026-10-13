"""
VoltRoute — Optimal Delivery and Charging Planning for EV-Based Delivery Partners.

Flask backend. Road distances come from Dijkstra on a city graph (which the
user can extend with their own stops and chargers, or block by closing
roads); the delivery order and
charging stops are chosen jointly by a best-first Branch & Bound search with
an MST lower bound. A brute-force search runs alongside to verify that B&B
returns the true optimum.
"""

import heapq
import re
from itertools import count
from math import hypot

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

EPS = 1e-9
KM_PER_PIXEL = 0.04       # map is drawn in pixels; 1 px = 40 m
MAP_W, MAP_H = 1000, 600
SERVICE_MIN = 4           # minutes spent at each delivery
CHARGER_RATES = (22, 50, 120)
MAX_CUSTOM_NODES = 8
MAX_CLOSED_ROADS = 15
MAX_ORDERS = 10
BRUTE_FORCE_LIMIT = 120_000   # moves; past this, brute force is skipped
NEW_NODE_LINKS = 3            # roads built from a user-added node to its nearest neighbours
RIVER_CLEARANCE = 40          # px; nothing can be built this close to the river

# ---------------------------------------------------------------------------
# Default city: two districts split by a river with only two bridges, so the
# road distance between places is often much longer than the straight line.
# Junctions are plain road intersections.
# ---------------------------------------------------------------------------
RIVER = [(505, 0), (500, 150), (515, 300), (505, 470), (515, 600)]  # centre line

BASE_NODES = {
    "Depot":    {"x":  70, "y": 520, "kind": "depot"},
    "Hub":      {"x": 930, "y":  70, "kind": "hub"},
    "Pharmacy": {"x": 110, "y": 220, "kind": "delivery", "order_id": "A"},
    "Office":   {"x": 255, "y": 215, "kind": "delivery", "order_id": "B"},
    "Market":   {"x": 255, "y": 545, "kind": "delivery", "order_id": "C"},
    "School":   {"x": 395, "y":  55, "kind": "delivery", "order_id": "D"},
    "Hospital": {"x": 645, "y": 215, "kind": "delivery", "order_id": "E"},
    "Library":  {"x": 770, "y": 225, "kind": "delivery", "order_id": "F"},
    "Bakery":   {"x": 770, "y": 545, "kind": "delivery", "order_id": "G"},
    "Grocery":  {"x": 905, "y": 385, "kind": "delivery", "order_id": "H"},
    "CH-1":     {"x":  60, "y": 380, "kind": "charger", "rate_kw": 22},
    "CH-2":     {"x": 430, "y": 300, "kind": "charger", "rate_kw": 22},
    "CH-3":     {"x": 590, "y": 300, "kind": "charger", "rate_kw": 50},
    "CH-4":     {"x": 945, "y": 555, "kind": "charger", "rate_kw": 120},
    "CH-5":     {"x": 700, "y":  45, "kind": "charger", "rate_kw": 50},
    "J1":  {"x": 180, "y": 470, "kind": "junction"},
    "J2":  {"x": 180, "y": 300, "kind": "junction"},
    "J3":  {"x": 180, "y": 130, "kind": "junction"},
    "J4":  {"x": 330, "y": 470, "kind": "junction"},
    "J5":  {"x": 330, "y": 300, "kind": "junction"},
    "J6":  {"x": 330, "y": 130, "kind": "junction"},
    "J7":  {"x": 440, "y": 150, "kind": "junction"},
    "J8":  {"x": 440, "y": 470, "kind": "junction"},
    "J9":  {"x": 580, "y": 150, "kind": "junction"},
    "J10": {"x": 580, "y": 470, "kind": "junction"},
    "J11": {"x": 700, "y": 300, "kind": "junction"},
    "J12": {"x": 700, "y": 150, "kind": "junction"},
    "J13": {"x": 700, "y": 470, "kind": "junction"},
    "J14": {"x": 840, "y": 300, "kind": "junction"},
    "J15": {"x": 840, "y": 150, "kind": "junction"},
}

BASE_EDGES = [
    # west district
    ("Depot", "J1"), ("Depot", "CH-1"), ("Depot", "Market"),
    ("J1", "Market"), ("J1", "J2"), ("J1", "J4"),
    ("CH-1", "J2"), ("J2", "Pharmacy"), ("J2", "J5"), ("J2", "Office"),
    ("Pharmacy", "J3"), ("J3", "J6"),
    ("Office", "J6"), ("Office", "J5"),
    ("J4", "Market"), ("J4", "J5"), ("J4", "J8"),
    ("J5", "J6"), ("J5", "CH-2"),
    ("J6", "School"), ("J6", "J7"), ("School", "J7"),
    ("J7", "CH-2"), ("CH-2", "J8"),
    # the two bridges
    ("J7", "J9"), ("J8", "J10"),
    # east district
    ("J9", "Hospital"), ("J9", "CH-3"), ("J9", "J12"),
    ("CH-3", "J10"), ("CH-3", "J11"), ("J10", "J13"),
    ("Hospital", "J12"), ("Hospital", "J11"),
    ("J11", "J12"), ("J11", "J13"), ("J11", "J14"),
    ("J12", "CH-5"), ("J12", "J15"), ("CH-5", "Hub"),
    ("J13", "Bakery"), ("J13", "J14"), ("Bakery", "CH-4"),
    ("CH-4", "Grocery"), ("Grocery", "J14"),
    ("J14", "Library"), ("Library", "J15"), ("J14", "J15"), ("J15", "Hub"),
]


def river_x(y: float) -> float:
    """x-coordinate of the river's centre line at height y."""
    for (x0, y0), (x1, y1) in zip(RIVER, RIVER[1:]):
        if y0 <= y <= y1:
            return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    return RIVER[-1][0]


def bank(x: float, y: float) -> str:
    return "west" if x < river_x(y) else "east"


def straight_km(nodes: dict, a: str, b: str) -> float:
    """As-the-crow-flies distance, ignoring roads."""
    return hypot(nodes[a]["x"] - nodes[b]["x"], nodes[a]["y"] - nodes[b]["y"]) * KM_PER_PIXEL


def road_key(a: str, b: str) -> tuple:
    return tuple(sorted((a, b)))


class City:
    """A road graph plus all-pairs shortest paths (Dijkstra from every vertex)."""

    def __init__(self, nodes: dict, edges: list, closed: list = ()):
        self.nodes = nodes
        self.edges = edges              # open roads
        self.closed = list(closed)      # roads closed by the user (drawn, but not usable)
        self.chargers = [n for n, v in nodes.items() if v["kind"] == "charger"]
        self.deliveries = [n for n, v in nodes.items() if v["kind"] == "delivery"]
        self.adj = {n: {} for n in nodes}
        for u, v in edges:
            self.adj[u][v] = self.adj[v][u] = self.edge_length(u, v)
        self._shortest = {}
        for a in nodes:
            dist, prev = self.dijkstra(a)
            for b in nodes:
                path, cur = [], b
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                self._shortest[(a, b)] = (round(dist[b], 3), path[::-1])

    def edge_length(self, u: str, v: str) -> float:
        """Length (km) of the straight road between two adjacent nodes."""
        return round(straight_km(self.nodes, u, v), 3)

    def dijkstra(self, start: str) -> tuple[dict, dict]:
        """Single-source shortest paths  —  O((V + E) log V) with a binary heap."""
        dist = {n: float("inf") for n in self.nodes}
        prev = {n: None for n in self.nodes}
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj[u].items():
                if d + w < dist[v]:
                    dist[v] = d + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
        return dist, prev

    def dijkstra_trace(self, start: str, targets: list) -> dict:
        """
        The same Dijkstra as above, but it records every step so the page can
        replay it, and stops as soon as every target's distance is final.
          start — source gets distance 0 and goes into the queue
          pop   — smallest entry taken from the queue; its distance is now final
          stale — an outdated queue entry for an already-final place is skipped
          relax — a road u → v is tried: is dist[u] + w better than dist[v]?
          done  — all targets final, or the queue ran empty
        """
        dist = {start: 0.0}
        prev = {}
        final = set()
        waiting = set(targets)
        steps = [{"t": "start", "node": start}]
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in final:
                steps.append({"t": "stale", "node": u, "d": round(d, 3)})
                continue
            final.add(u)
            waiting.discard(u)
            steps.append({"t": "pop", "node": u, "d": round(d, 3), "target": u in targets})
            if targets and not waiting:
                break
            for v, w in sorted(self.adj[u].items(), key=lambda kv: kv[0]):
                if v in final:
                    continue
                old = dist.get(v)
                new = d + w
                better = old is None or new < old - EPS
                steps.append({"t": "relax", "from": u, "to": v, "w": w,
                              "old": None if old is None else round(old, 3),
                              "new": round(new, 3), "better": better})
                if better:
                    dist[v] = new
                    prev[v] = u
                    heapq.heappush(pq, (new, v))

        results = {}
        for t in targets:
            if t not in final:
                results[t] = None
                continue
            path, cur = [t], t
            while cur != start:
                cur = prev[cur]
                path.append(cur)
            results[t] = {"km": round(dist[t], 2), "path": path[::-1],
                          "straight_km": round(straight_km(self.nodes, start, t), 2)}
        steps.append({"t": "done", "found": sum(r is not None for r in results.values())})
        return {"source": start, "targets": targets, "steps": steps, "results": results}

    def distance(self, a: str, b: str) -> float:
        return self._shortest[(a, b)][0]

    def path(self, a: str, b: str) -> list[str]:
        return self._shortest[(a, b)][1]

    def to_json(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges, "closed": self.closed, "river": RIVER}


DEFAULT_CITY = City(BASE_NODES, BASE_EDGES)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]{0,19}$")


def build_city(custom: list, closed_roads: list = ()) -> City:
    """
    Default city plus user-added places and closed roads. Each new place gets
    a road to its NEW_NODE_LINKS nearest places on the same side of the river
    (only the bridges cross it). Closed roads can't be driven on.
    """
    if not custom and not closed_roads:
        return DEFAULT_CITY
    if not isinstance(custom, list) or len(custom) > MAX_CUSTOM_NODES:
        raise ValueError(f"You can add at most {MAX_CUSTOM_NODES} places.")
    if not isinstance(closed_roads, (list, tuple)) or len(closed_roads) > MAX_CLOSED_ROADS:
        raise ValueError(f"You can close at most {MAX_CLOSED_ROADS} roads.")

    nodes = {k: dict(v) for k, v in BASE_NODES.items()}
    edges = list(BASE_EDGES)
    used_ids = {v["order_id"] for v in nodes.values() if "order_id" in v}
    for item in custom:
        try:
            name = str(item["name"]).strip()
            kind = item["kind"]
            x, y = float(item["x"]), float(item["y"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Each added place needs a name, kind and position.")
        if not _NAME_RE.match(name) or name in nodes:
            raise ValueError(f"'{name}' is not a valid, unique place name.")
        if kind not in ("delivery", "charger"):
            raise ValueError("Added places must be a delivery stop or a charger.")
        if not (0 <= x <= MAP_W and 0 <= y <= MAP_H):
            raise ValueError(f"{name} is outside the map.")
        if abs(x - river_x(y)) < RIVER_CLEARANCE:
            raise ValueError(f"{name} is in the river.")

        node = {"x": round(x), "y": round(y), "kind": kind, "custom": True}
        if kind == "charger":
            rate = item.get("rate_kw", 50)
            if rate not in CHARGER_RATES:
                raise ValueError(f"Charger power must be one of {CHARGER_RATES} kW.")
            node["rate_kw"] = rate
        else:
            node["order_id"] = next(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used_ids)
            used_ids.add(node["order_id"])

        side = bank(x, y)
        same_side = [n for n in nodes if bank(nodes[n]["x"], nodes[n]["y"]) == side]
        nearest = sorted(same_side, key=lambda n: hypot(nodes[n]["x"] - x, nodes[n]["y"] - y))
        edges += [(name, n) for n in nearest[:NEW_NODE_LINKS]]
        nodes[name] = node

    # Unknown roads are ignored rather than rejected: a closed road disappears
    # when the place it led to is removed.
    existing = {road_key(*e) for e in edges}
    closed = []
    for pair in closed_roads:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and road_key(*map(str, pair)) in existing:
            closed.append(road_key(*map(str, pair)))
    closed = sorted(set(closed))
    open_edges = [e for e in edges if road_key(*e) not in set(closed)]
    return City(nodes, open_edges, closed)


# ---------------------------------------------------------------------------
# Lower bound: min(cur → R) + MST(R) + min(R → Hub)
# ---------------------------------------------------------------------------
def mst_cost(city: City, nodes: list) -> float:
    """Prim's algorithm over the shortest-path metric  —  O(n²)."""
    if len(nodes) <= 1:
        return 0.0
    first, *rest = nodes
    best = {n: city.distance(first, n) for n in rest}
    total = 0.0
    while best:
        n = min(best, key=best.get)
        total += best.pop(n)
        for m in best:
            best[m] = min(best[m], city.distance(n, m))
    return total


def lower_bound(city: City, cur: str, remaining: list) -> float:
    """
    Admissible bound on the distance still to drive. Any completion must leave
    `cur`, connect every remaining stop, and finally enter the Hub, so it costs
    at least this much. Charging detours only add distance, so they can't
    break the bound.
    """
    if not remaining:
        return city.distance(cur, "Hub")
    return (min(city.distance(cur, n) for n in remaining)
            + mst_cost(city, remaining)
            + min(city.distance(n, "Hub") for n in remaining))


# ---------------------------------------------------------------------------
# Battery model
# ---------------------------------------------------------------------------
def battery_limits(cfg: dict) -> tuple[float, float, float]:
    """(capacity, threshold, reserve) in kWh."""
    cap = cfg["capacity_kwh"]
    return cap, cap * cfg["threshold_pct"] / 100, cap * cfg["reserve_pct"] / 100


def energy(km: float, cfg: dict) -> float:
    return km / cfg["km_per_kwh"]


def soc(kwh: float, cfg: dict) -> float:
    return round(kwh / cfg["capacity_kwh"] * 100, 1)


def charger_options(city: City, cur: str, nxt: str, bat: float, cfg: dict) -> list[dict]:
    """
    Every way to reach `nxt` through one charging stop. A charger is usable if
    we arrive with at least the reserve and, after charging to 100%, can still
    reach `nxt` with at least the reserve.
    """
    cap, _, reserve = battery_limits(cfg)
    options = []
    for ch in city.chargers:
        to_ch, from_ch = city.distance(cur, ch), city.distance(ch, nxt)
        if to_ch == float("inf") or from_ch == float("inf"):
            continue  # cut off by closed roads
        at_ch = bat - energy(to_ch, cfg)
        at_nxt = cap - energy(from_ch, cfg)
        reachable = at_ch >= reserve - EPS
        options.append({
            "charger": ch,
            "to_charger_km": to_ch,
            "straight_km": straight_km(city.nodes, cur, ch),
            "total_km": to_ch + from_ch,
            "arrive_charger_kwh": at_ch,
            "arrive_charger_pct": soc(at_ch, cfg),
            "arrive_next_kwh": at_nxt,
            "reachable": reachable,
            "feasible": reachable and at_nxt >= reserve - EPS,
        })
    return options


def evaluate_battery_decision(city: City, bat: float, cur: str, nxt: str, cfg: dict) -> dict:
    """
    Threshold rule: drive directly only if the battery left on arrival is
    strictly above the threshold (and therefore above the reserve). Otherwise
    the EV must detour through a charger.
    """
    _, threshold, reserve = battery_limits(cfg)
    leg = city.distance(cur, nxt)
    projected = bat - energy(leg, cfg)

    if projected < reserve - EPS:
        status, label = "insufficient", "Below safety reserve"
    elif abs(projected - threshold) < EPS:
        status, label = "threshold_equal", "Exactly at threshold"
    elif projected < threshold:
        status, label = "below_threshold", "Below threshold"
    else:
        status, label = "above_threshold", "Above threshold"

    direct = status == "above_threshold"
    return {
        "from": cur,
        "to": nxt,
        "status": status,
        "label": label,
        "can_proceed_direct": direct,
        "leg_km": round(leg, 2),
        "battery_kwh": bat,
        "battery_pct": soc(bat, cfg),
        "projected_kwh": projected,
        "projected_pct": soc(projected, cfg),
        "threshold_kwh": threshold,
        "charger_options": [] if direct else charger_options(city, cur, nxt, bat, cfg),
    }


def transitions(city: City, cur: str, nxt: str, bat: float, cfg: dict):
    """Yield (charger or None, km driven, battery on arrival) for each feasible move."""
    dec = evaluate_battery_decision(city, bat, cur, nxt, cfg)
    if dec["can_proceed_direct"]:
        yield None, city.distance(cur, nxt), dec["projected_kwh"]
        return
    for opt in dec["charger_options"]:
        if opt["feasible"]:
            yield opt["charger"], opt["total_km"], opt["arrive_next_kwh"]


# ---------------------------------------------------------------------------
# Branch & Bound — joint delivery order + charging stops
# ---------------------------------------------------------------------------
def branch_and_bound(city: City, order_nodes: list, cfg: dict) -> dict:
    """
    Best-first B&B over states (location, delivered set, battery).

    Each branch is "go to delivery X next" (directly, or via one of the
    chargers when the threshold rule forbids going direct). When every order
    is delivered, the only branch left is "go to Hub". Nodes are popped in
    order of cost-so-far + lower_bound, and any node whose bound is no better
    than the best complete route found so far is pruned.
    """
    cap = cfg["capacity_kwh"]
    bat0 = cap * cfg["soc_pct"] / 100
    serial = count()

    heap = [(lower_bound(city, "Depot", order_nodes), next(serial),
             "Depot", frozenset(), 0.0, bat0, ())]
    best_cost, best_legs = float("inf"), None
    expanded = pruned = 0

    while heap:
        bound, _, cur, done, cost, bat, legs = heapq.heappop(heap)
        if bound >= best_cost - EPS:
            pruned += 1
            continue
        expanded += 1

        if cur == "Hub":
            best_cost, best_legs = cost, legs
            continue

        remaining = [n for n in order_nodes if n not in done]
        for nxt in remaining or ["Hub"]:
            rest = [n for n in remaining if n != nxt]
            for via, km, new_bat in transitions(city, cur, nxt, bat, cfg):
                new_cost = cost + km
                new_bound = new_cost if nxt == "Hub" else new_cost + lower_bound(city, nxt, rest)
                if new_bound >= best_cost - EPS:
                    pruned += 1
                    continue
                heapq.heappush(heap, (new_bound, next(serial), nxt, done | {nxt},
                                      new_cost, new_bat, legs + ((cur, nxt, via),)))

    feasible = best_legs is not None
    return {
        "feasible": feasible,
        "cost_km": round(best_cost, 3) if feasible else None,
        "legs": list(best_legs) if feasible else [],
        "delivery_order": [nxt for _, nxt, _ in best_legs if nxt != "Hub"] if feasible else [],
        "expanded": expanded,
        "pruned": pruned,
    }


class _TooLarge(Exception):
    pass


def exhaustive_search(city: City, order_nodes: list, cfg: dict, limit: int | None = None) -> dict:
    """
    Brute force: try every delivery order (n!) and every feasible charger
    choice on every leg, with no pruning. Used to check that Branch & Bound is
    optimal. Gives up (skipped=True) after `limit` moves.
    """
    bat0 = cfg["capacity_kwh"] * cfg["soc_pct"] / 100
    best = {"cost": float("inf"), "legs": None}
    routes = moves = 0

    def walk(cur, remaining, bat, cost, legs):
        nonlocal routes, moves
        for nxt in remaining or ["Hub"]:
            rest = [n for n in remaining if n != nxt]
            for via, km, new_bat in transitions(city, cur, nxt, bat, cfg):
                moves += 1
                if limit and moves > limit:
                    raise _TooLarge
                step = legs + [(cur, nxt, via)]
                if nxt == "Hub":
                    routes += 1
                    if cost + km < best["cost"]:
                        best["cost"], best["legs"] = cost + km, step
                else:
                    walk(nxt, rest, new_bat, cost + km, step)

    try:
        walk("Depot", list(order_nodes), bat0, 0.0, [])
    except _TooLarge:
        return {"skipped": True, "feasible": None, "cost_km": None, "legs": [], "routes_checked": routes}

    feasible = best["legs"] is not None
    return {
        "skipped": False,
        "feasible": feasible,
        "cost_km": round(best["cost"], 3) if feasible else None,
        "legs": best["legs"] or [],
        "routes_checked": routes,
    }


# ---------------------------------------------------------------------------
# Turn the optimal legs into a step-by-step plan
# ---------------------------------------------------------------------------
def build_plan(city: City, legs: list, cfg: dict) -> list[dict]:
    """
    Replays the chosen legs and emits three kinds of events, in order:
      decision — battery check before each leg (and which charger, if any)
      travel   — one drive along a Dijkstra road path
      charge   — recharge to 100% at a station
    """
    cap = cfg["capacity_kwh"]
    bat = cap * cfg["soc_pct"] / 100
    events = []

    def travel(a, b, purpose):
        nonlocal bat
        km = city.distance(a, b)
        used = energy(km, cfg)
        events.append({
            "type": "travel", "from": a, "to": b, "purpose": purpose,
            "path": city.path(a, b),
            "distance_km": round(km, 2), "energy_kwh": round(used, 2),
            "soc_before": soc(bat, cfg), "soc_after": soc(bat - used, cfg),
        })
        bat -= used

    for cur, nxt, via in legs:
        dec = evaluate_battery_decision(city, bat, cur, nxt, cfg)
        events.append({
            "type": "decision", "from": cur, "to": nxt, "charger": via,
            "status": dec["status"], "label": dec["label"],
            "leg_km": dec["leg_km"],
            "battery_pct": dec["battery_pct"], "projected_pct": dec["projected_pct"],
            "options": [{
                "charger": o["charger"],
                "to_charger_km": round(o["to_charger_km"], 2),
                "straight_km": round(o["straight_km"], 2),
                "total_km": round(o["total_km"], 2),
                "arrive_charger_pct": o["arrive_charger_pct"],
                "reachable": o["reachable"],
                "feasible": o["feasible"],
            } for o in dec["charger_options"]],
        })
        purpose = "finish" if nxt == "Hub" else "deliver"
        if via:
            travel(cur, via, "charge")
            rate = city.nodes[via]["rate_kw"]
            added = cap - bat
            events.append({
                "type": "charge", "at": via, "rate_kw": rate,
                "added_kwh": round(added, 2), "minutes": round(added / rate * 60, 1),
                "soc_before": soc(bat, cfg), "soc_after": 100.0,
            })
            bat = cap
            travel(via, nxt, purpose)
        else:
            travel(cur, nxt, purpose)
    return events


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
LIMITS = {
    "capacity_kwh":      (5, 120, 40, "Battery capacity", "kWh"),
    "soc_pct":           (10, 100, 75, "Starting charge", "%"),
    "km_per_kwh":        (2, 12, 6, "Efficiency", "km/kWh"),
    "average_speed_kmh": (10, 80, 30, "Average speed", "km/h"),
    "threshold_pct":     (0, 80, 25, "Charging threshold", "%"),
    "reserve_pct":       (0, 40, 15, "Safety reserve", "%"),
}


def parse_config(payload: dict) -> dict:
    cfg = {}
    for key, (lo, hi, default, name, unit) in LIMITS.items():
        try:
            value = float(payload.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number.")
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi} {unit}.")
        cfg[key] = value
    if cfg["threshold_pct"] < cfg["reserve_pct"]:
        raise ValueError("Charging threshold can't be lower than the safety reserve.")
    return cfg


def optimize(payload: dict) -> dict:
    city = build_city(payload.get("custom_nodes") or [], payload.get("closed_roads") or [])
    stops = [s for s in dict.fromkeys(payload.get("stops", city.deliveries)) if s in city.deliveries]
    if not stops:
        raise ValueError("Select at least one delivery stop.")
    if len(stops) > MAX_ORDERS:
        raise ValueError(f"Select at most {MAX_ORDERS} delivery stops.")
    cfg = parse_config(payload)

    bnb = branch_and_bound(city, stops, cfg)
    brute = exhaustive_search(city, stops, cfg, limit=BRUTE_FORCE_LIMIT)
    if brute["skipped"]:
        matches = None
    elif None in (bnb["cost_km"], brute["cost_km"]):
        matches = bnb["cost_km"] == brute["cost_km"]
    else:
        matches = abs(bnb["cost_km"] - brute["cost_km"]) < 1e-6
    search = {
        "orders": len(stops),
        "expanded": bnb["expanded"],
        "pruned": bnb["pruned"],
        "brute_force_skipped": brute["skipped"],
        "brute_force_routes": brute["routes_checked"],
        "brute_force_km": brute["cost_km"],
        "matches_brute_force": matches,
    }
    base = {"config": cfg, "search": search, "graph": city.to_json(), "stops": stops}

    if not bnb["feasible"]:
        cut_off = [s for s in stops + ["Hub"] if city.distance("Depot", s) == float("inf")]
        reason = (f"{', '.join(cut_off)} can't be reached from the Depot by road. "
                  "Reopen one of the closed roads." if cut_off else
                  "No route can deliver every selected order and reach the Hub "
                  "without dropping below the safety reserve. Try a higher starting "
                  "charge, a bigger battery, or a lower reserve.")
        return {**base, "feasible": False, "reason": reason}

    events = build_plan(city, bnb["legs"], cfg)
    charges = [e for e in events if e["type"] == "charge"]
    travels = [e for e in events if e["type"] == "travel"]
    travel_min = bnb["cost_km"] / cfg["average_speed_kmh"] * 60
    charge_min = sum(e["minutes"] for e in charges)
    service_min = SERVICE_MIN * len(stops)

    return {
        **base,
        "feasible": True,
        "delivery_order": bnb["delivery_order"],
        "route": ["Depot"] + [e["to"] for e in travels],
        "events": events,
        "metrics": {
            "distance_km": round(bnb["cost_km"], 2),
            "energy_kwh": round(sum(e["energy_kwh"] for e in travels), 2),
            "charging_stops": len(charges),
            "charge_min": round(charge_min, 1),
            "drive_min": round(travel_min, 1),
            "total_min": round(travel_min + charge_min + service_min, 1),
            "final_soc_pct": travels[-1]["soc_after"],
        },
    }


@app.route("/")
def index():
    return render_template(
        "index.html", graph=DEFAULT_CITY.to_json(), limits=LIMITS,
        settings={"map_w": MAP_W, "map_h": MAP_H, "max_custom": MAX_CUSTOM_NODES,
                  "max_closed": MAX_CLOSED_ROADS, "max_orders": MAX_ORDERS,
                  "rates": CHARGER_RATES, "river_clearance": RIVER_CLEARANCE,
                  "km_per_px": KM_PER_PIXEL})


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    try:
        return jsonify(optimize(request.get_json(silent=True) or {}))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/dijkstra", methods=["POST"])
def api_dijkstra():
    """Step-by-step Dijkstra trace for the lab view."""
    try:
        payload = request.get_json(silent=True) or {}
        city = build_city(payload.get("custom_nodes") or [], payload.get("closed_roads") or [])
        source = payload.get("source")
        targets = [t for t in dict.fromkeys(payload.get("targets") or []) if t in city.nodes]
        if source not in city.nodes:
            raise ValueError("Pick a starting place.")
        return jsonify(city.dijkstra_trace(source, targets))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
