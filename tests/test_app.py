"""
Tests for VoltRoute.

  1  Enough battery          — drives directly, no charging
  2  Threshold equality      — projected == threshold is rejected, chargers evaluated
  3  Low battery             — a charging stop is planned
  4  Nearest charger trap    — the optimizer picks the better charger, not the nearest
  5  All deliveries          — each order visited exactly once, route ends at Hub
  6  Infeasible              — reported cleanly instead of crashing
  7  Transit through charger — passing a charger on the road is not a charging stop
  8  Optimality              — Branch & Bound matches brute force on many configs
  9  Dijkstra lab trace      — the recorded steps really are Dijkstra's
 10  Road closures           — closing a bridge forces a detour; cutting a place off is reported
Plus graph, lower-bound, custom-place and API validation tests.
"""

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (  # noqa: E402
    DEFAULT_CITY as CITY, MAX_CUSTOM_NODES,
    app, bank, branch_and_bound, build_city, evaluate_battery_decision,
    exhaustive_search, lower_bound, parse_config, straight_km,
)

ALL_STOPS = CITY.deliveries


def config(**overrides):
    return parse_config(overrides)


def charges(events):
    return [e["at"] for e in events if e["type"] == "charge"]


class GraphTests(unittest.TestCase):
    def test_dijkstra_paths_use_real_roads(self):
        for u in CITY.nodes:
            for v in CITY.nodes:
                path = CITY.path(u, v)
                self.assertEqual((path[0], path[-1]), (u, v))
                self.assertAlmostEqual(
                    CITY.distance(u, v),
                    sum(CITY.adj[a][b] for a, b in zip(path, path[1:])), places=3)

    def test_graph_is_connected(self):
        for u in CITY.nodes:
            for v in CITY.nodes:
                self.assertLess(CITY.distance(u, v), float("inf"))

    def test_triangle_inequality(self):
        for a, b, c in itertools.permutations(CITY.nodes, 3):
            self.assertLessEqual(CITY.distance(a, c), CITY.distance(a, b) + CITY.distance(b, c) + 1e-9)

    def test_lower_bound_is_admissible(self):
        # The bound from Depot can never exceed the best unconstrained tour.
        for k in range(1, 5):
            for stops in itertools.combinations(ALL_STOPS, k):
                best = min(
                    sum(CITY.distance(a, b) for a, b in zip(("Depot",) + p, p + ("Hub",)))
                    for p in itertools.permutations(stops))
                self.assertLessEqual(lower_bound(CITY, "Depot", list(stops)), best + 1e-9)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, **payload):
        return self.client.post("/api/optimize", json=payload)

    def test_1_enough_battery(self):
        res = self.post(capacity_kwh=60, soc_pct=95, reserve_pct=10, threshold_pct=20).json
        self.assertTrue(res["feasible"])
        self.assertEqual(res["metrics"]["charging_stops"], 0)
        dec = evaluate_battery_decision(CITY, 57.0, "Depot", "Pharmacy",
                                        config(capacity_kwh=60, reserve_pct=10, threshold_pct=20))
        self.assertEqual(dec["status"], "above_threshold")
        self.assertTrue(dec["can_proceed_direct"])

    def test_2_threshold_equality(self):
        cfg = config(capacity_kwh=40, threshold_pct=25, reserve_pct=10)
        battery = 10.0 + CITY.distance("Depot", "Pharmacy") / cfg["km_per_kwh"]
        dec = evaluate_battery_decision(CITY, battery, "Depot", "Pharmacy", cfg)
        self.assertEqual(dec["status"], "threshold_equal")
        self.assertFalse(dec["can_proceed_direct"])
        self.assertAlmostEqual(dec["projected_kwh"], 10.0, places=6)
        self.assertEqual(len(dec["charger_options"]), len(CITY.chargers))

    def test_3_low_battery(self):
        res = self.post(soc_pct=45).json
        self.assertTrue(res["feasible"])
        self.assertGreaterEqual(res["metrics"]["charging_stops"], 1)
        cfg = config(soc_pct=45)
        dec = evaluate_battery_decision(CITY, 10.5, "Depot", "Library", cfg)
        self.assertEqual(dec["status"], "insufficient")

    def test_4_nearest_charger_is_not_best(self):
        # Hospital → Grocery: CH-3 looks closest on the map, CH-5 is closest by
        # road, but going via CH-3 gives the shortest route overall.
        res = self.post(stops=["Office", "Hospital", "Grocery"], soc_pct=45, reserve_pct=10).json
        self.assertTrue(res["feasible"])
        dec = next(e for e in res["events"] if e["type"] == "decision" and e["charger"])
        self.assertEqual((dec["from"], dec["to"], dec["charger"]), ("Hospital", "Grocery", "CH-3"))
        by_air = min(dec["options"], key=lambda o: o["straight_km"])["charger"]
        by_road = min((o for o in dec["options"] if o["reachable"]), key=lambda o: o["to_charger_km"])["charger"]
        self.assertEqual((by_air, by_road), ("CH-3", "CH-5"))

    def test_5_every_delivery_once_and_ends_at_hub(self):
        res = self.post(capacity_kwh=50, soc_pct=80).json
        self.assertEqual(sorted(res["delivery_order"]), sorted(ALL_STOPS))
        self.assertEqual(res["route"][0], "Depot")
        self.assertEqual(res["route"][-1], "Hub")
        for stop in ALL_STOPS:
            self.assertEqual(res["route"].count(stop), 1)

    def test_6_infeasible(self):
        r = self.post(soc_pct=15, threshold_pct=25, reserve_pct=15)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json["feasible"])
        self.assertIn("No route", r.json["reason"])

    def test_7_passing_a_charger_is_not_a_charging_stop(self):
        res = self.post(stops=["Office", "Hospital"], soc_pct=100).json
        self.assertEqual(charges(res["events"]), [])
        passed = {n for e in res["events"] if e["type"] == "travel" for n in e["path"][1:-1]}
        self.assertTrue(passed & set(CITY.chargers), "route should drive through a charger")

    def test_8_branch_and_bound_matches_brute_force(self):
        checked = 0
        for soc, cap, thr, res in itertools.product((20, 35, 50, 75, 100), (20, 40), (15, 25, 40), (5, 15)):
            cfg = config(soc_pct=soc, capacity_kwh=cap, threshold_pct=thr, reserve_pct=res)
            for k in (1, 2, 3, 4):
                for stops in itertools.combinations(ALL_STOPS[:6], k):
                    bnb = branch_and_bound(CITY, list(stops), cfg)
                    brute = exhaustive_search(CITY, list(stops), cfg)
                    self.assertEqual(bnb["feasible"], brute["feasible"], (stops, cfg))
                    if bnb["feasible"]:
                        self.assertAlmostEqual(bnb["cost_km"], brute["cost_km"], places=6)
                    checked += 1
        self.assertEqual(checked, 60 * 56)

    def test_8b_all_eight_stops_match_brute_force(self):
        for soc in (45, 100):
            cfg = config(soc_pct=soc)
            bnb = branch_and_bound(CITY, ALL_STOPS, cfg)
            brute = exhaustive_search(CITY, ALL_STOPS, cfg)
            self.assertAlmostEqual(bnb["cost_km"], brute["cost_km"], places=6)
            self.assertLess(bnb["expanded"], brute["routes_checked"])

    def test_charging_respects_reserve(self):
        res = self.post(soc_pct=45).json
        for e in res["events"]:
            if e["type"] in ("travel", "charge"):
                self.assertGreaterEqual(e["soc_after"], res["config"]["reserve_pct"] - 0.1)

    def test_unreachable_charger_flagged(self):
        cfg = config(capacity_kwh=40, threshold_pct=35, reserve_pct=30)
        dec = evaluate_battery_decision(CITY, 14.0, "Depot", "Pharmacy", cfg)
        far = next(o for o in dec["charger_options"] if o["charger"] == "CH-5")
        self.assertFalse(far["reachable"])
        self.assertFalse(far["feasible"])

    def test_brute_force_is_skipped_when_too_large(self):
        res = self.post(soc_pct=75).json
        self.assertTrue(res["search"]["brute_force_skipped"])
        self.assertIsNone(res["search"]["matches_brute_force"])
        self.assertTrue(res["feasible"])


class CustomPlaceTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_added_places_are_connected_and_used(self):
        custom = [{"name": "Stop 1", "kind": "delivery", "x": 960, "y": 560},
                  {"name": "CH-6", "kind": "charger", "x": 620, "y": 560, "rate_kw": 120}]
        city = build_city(custom)
        self.assertEqual(city.nodes["Stop 1"]["order_id"], "I")
        self.assertIn("CH-6", city.chargers)
        self.assertEqual(sum("Stop 1" in e for e in city.edges), 3)
        # roads from an added place never cross the river
        for a, b in city.edges:
            if "custom" in city.nodes[a] or "custom" in city.nodes[b]:
                self.assertEqual(bank(city.nodes[a]["x"], city.nodes[a]["y"]),
                                 bank(city.nodes[b]["x"], city.nodes[b]["y"]))
        self.assertLess(city.distance("Depot", "Stop 1"), float("inf"))

        res = self.client.post("/api/optimize", json={
            "stops": ["Stop 1", "Grocery"], "custom_nodes": custom}).json
        self.assertTrue(res["feasible"])
        self.assertIn("Stop 1", res["delivery_order"])
        self.assertIn("Stop 1", res["graph"]["nodes"])

        bnb = branch_and_bound(city, ["Stop 1", "Grocery", "School"], config(soc_pct=30))
        brute = exhaustive_search(city, ["Stop 1", "Grocery", "School"], config(soc_pct=30))
        self.assertAlmostEqual(bnb["cost_km"], brute["cost_km"], places=6)

    def test_bad_places_rejected(self):
        bad = [
            [{"name": "Depot", "kind": "delivery", "x": 10, "y": 10}],
            [{"name": "X", "kind": "house", "x": 10, "y": 10}],
            [{"name": "X", "kind": "delivery", "x": 5000, "y": 10}],
            [{"name": "X", "kind": "delivery", "x": 505, "y": 300}],  # in the river
            [{"name": "<b>", "kind": "delivery", "x": 10, "y": 10}],
            [{"name": "C", "kind": "charger", "x": 10, "y": 10, "rate_kw": 7}],
            [{"name": f"S{i}", "kind": "delivery", "x": 10 * i, "y": 10} for i in range(MAX_CUSTOM_NODES + 1)],
        ]
        for custom in bad:
            r = self.client.post("/api/optimize", json={"custom_nodes": custom})
            self.assertEqual(r.status_code, 400, custom)


class DijkstraTraceTests(unittest.TestCase):
    def replay(self, trace):
        """Rebuild the algorithm state from the recorded steps and check each one."""
        dist, final, heap = {}, set(), []
        for s in trace["steps"]:
            if s["t"] == "start":
                dist[s["node"]] = 0.0
                heap.append((0.0, s["node"]))
            elif s["t"] in ("pop", "stale"):
                smallest = min(heap)
                self.assertAlmostEqual(smallest[0], s["d"], places=3)   # always the minimum
                heap.remove(smallest)
                if s["t"] == "stale":
                    self.assertIn(s["node"], final)
                else:
                    self.assertNotIn(s["node"], final)
                    final.add(s["node"])
            elif s["t"] == "relax":
                self.assertIn(s["from"], final)
                self.assertAlmostEqual(s["new"], dist[s["from"]] + s["w"], places=3)
                if s["better"]:
                    dist[s["to"]] = s["new"]
                    heap.append((s["new"], s["to"]))
        return dist, final

    def test_9_trace_is_dijkstra(self):
        for source in ("Depot", "Hospital", "J8"):
            trace = CITY.dijkstra_trace(source, [])
            dist, final = self.replay(trace)
            self.assertEqual(final, set(CITY.nodes))
            for n in CITY.nodes:
                self.assertAlmostEqual(dist[n], CITY.distance(source, n), places=2)

    def test_9b_trace_stops_when_targets_are_final(self):
        trace = CITY.dijkstra_trace("Hospital", CITY.chargers)
        _, final = self.replay(trace)
        self.assertTrue(set(CITY.chargers) <= final)
        self.assertLess(len(final), len(CITY.nodes))
        for ch, r in trace["results"].items():
            self.assertAlmostEqual(r["km"], CITY.distance("Hospital", ch), places=2)
            self.assertEqual(r["path"], CITY.path("Hospital", ch))
        # the river makes CH-2 much further by road than in a straight line
        r = trace["results"]["CH-2"]
        self.assertGreater(r["km"], 1.5 * r["straight_km"])

    def test_9c_api(self):
        client = app.test_client()
        r = client.post("/api/dijkstra", json={"source": "Office", "targets": ["Hub"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["steps"][-1]["t"], "done")
        self.assertEqual(r.json["results"]["Hub"]["path"][-1], "Hub")
        self.assertEqual(client.post("/api/dijkstra", json={"source": "Nowhere"}).status_code, 400)


class RoadClosureTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_10_closed_bridge_forces_detour(self):
        open_city = self.client.post("/api/optimize", json={"soc_pct": 80}).json
        closed = self.client.post("/api/optimize", json={"soc_pct": 80, "closed_roads": [["J9", "J7"]]}).json
        self.assertEqual(closed["graph"]["closed"], [["J7", "J9"]])
        self.assertGreater(closed["metrics"]["distance_km"], open_city["metrics"]["distance_km"])
        for e in closed["events"]:
            if e["type"] == "travel":
                for a, b in zip(e["path"], e["path"][1:]):
                    self.assertNotEqual({a, b}, {"J7", "J9"})

    def test_10b_cut_off_place_is_reported(self):
        both_bridges = [["J7", "J9"], ["J8", "J10"]]
        res = self.client.post("/api/optimize", json={"closed_roads": both_bridges}).json
        self.assertFalse(res["feasible"])
        self.assertIn("can't be reached", res["reason"])
        trace = self.client.post("/api/dijkstra", json={
            "source": "Depot", "targets": ["Hub"], "closed_roads": both_bridges}).json
        self.assertIsNone(trace["results"]["Hub"])

    def test_10c_unknown_roads_ignored(self):
        res = self.client.post("/api/optimize", json={"closed_roads": [["Depot", "Hub"], "junk"]}).json
        self.assertEqual(res["graph"]["closed"], [])


class ApiValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        self.assertEqual(self.client.get("/api/health").json["status"], "ok")

    def test_index_renders(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_bad_inputs_rejected(self):
        for payload in ({"stops": []},
                        {"stops": ["Nowhere"]},
                        {"capacity_kwh": 2},
                        {"soc_pct": "abc"},
                        {"reserve_pct": 25, "threshold_pct": 10}):
            r = self.client.post("/api/optimize", json=payload)
            self.assertEqual(r.status_code, 400, payload)
            self.assertIn("error", r.json)


if __name__ == "__main__":
    unittest.main()
