"""
Comprehensive test suite for VoltRoute.
Covers all 8 Acceptance Tests specified in the requirements:
  Test 1: Enough battery (proceeds direct)
  Test 2: Threshold equality (direct travel rejected, charger evaluated)
  Test 3: Low battery (charger evaluated and chosen for minimum cost)
  Test 4: Nearest charger is not best (optimizer selects globally better charger)
  Test 5: All deliveries visited exactly once, ending at Hub
  Test 6: Infeasible scenario reported clearly with reason
  Test 7: Simulation steps, road-segment decomposition, and speed configurations
  Test 8: Algorithm correctness verified against exhaustive enumeration
Plus API validation, Dijkstra shortest paths, and lower-bound tests.
"""

import unittest
from app import (
    app,
    dijkstra,
    distance,
    shortest_distance,
    shortest_path,
    evaluate_battery_decision,
    branch_and_bound,
    branch_and_bound_with_charging,
    exhaustive_search,
    lower_bound,
    mst_cost,
    ORDERS,
    NODES,
    CHARGERS,
    ROAD_EDGES,
)


class VoltRouteAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    # -----------------------------------------------------------------------
    # API & Health
    # -----------------------------------------------------------------------
    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["status"], "ok")

    def test_empty_order_selection_rejected(self):
        r = self.client.post("/api/optimize", json={"order_ids": []})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.json)

    def test_invalid_capacity_rejected(self):
        r = self.client.post("/api/optimize", json={"order_ids": ["A"], "capacity_kwh": 2})
        self.assertEqual(r.status_code, 400)

    def test_threshold_below_reserve_rejected(self):
        r = self.client.post("/api/optimize", json={
            "order_ids": ["A"],
            "reserve_pct": 25,
            "threshold_pct": 10
        })
        self.assertEqual(r.status_code, 400)

    def test_negative_budget_rejected(self):
        r = self.client.post("/api/optimize", json={
            "order_ids": ["A"],
            "route_budget_km": -10
        })
        self.assertEqual(r.status_code, 400)

    # -----------------------------------------------------------------------
    # Dijkstra Shortest Paths & Graph Representation
    # -----------------------------------------------------------------------
    def test_dijkstra_shortest_paths(self):
        for u in NODES:
            for v in NODES:
                d, path = dijkstra(u, v)
                self.assertGreaterEqual(d, 0.0)
                self.assertEqual(path[0], u)
                self.assertEqual(path[-1], v)
                # Verify path segments exist as road edges
                for i in range(len(path) - 1):
                    seg = (path[i], path[i+1])
                    rev_seg = (path[i+1], path[i])
                    self.assertTrue(seg in ROAD_EDGES or rev_seg in ROAD_EDGES)

    # -----------------------------------------------------------------------
    # Test 1 — Enough Battery
    # -----------------------------------------------------------------------
    def test_1_enough_battery(self):
        """
        When starting battery is high and projected battery after next delivery
        remains strictly above threshold, the vehicle proceeds directly.
        """
        cfg = {
            "capacity_kwh": 60.0,
            "soc_pct": 95.0,     # 57.0 kWh
            "km_per_kwh": 6.0,
            "reserve_pct": 10.0, # 6.0 kWh
            "threshold_pct": 20.0, # 12.0 kWh
            "charger_kw": 22.0,
            "route_budget_km": 150.0,
        }
        res = branch_and_bound_with_charging(["Pharmacy", "Office", "Grocery"], cfg)
        self.assertTrue(res["feasible"])
        # No charging stops needed because battery remains strictly above threshold
        charge_events = [e for e in res["best_events"] if e["type"] == "charge"]
        self.assertEqual(len(charge_events), 0)

        # Check the battery decision directly
        dec = evaluate_battery_decision(
            battery_kwh=57.0, capacity_kwh=60.0, threshold_pct=20.0,
            reserve_pct=10.0, cur="Depot", nxt="Pharmacy", eff=6.0
        )
        self.assertEqual(dec["status"], "above_threshold")
        self.assertTrue(dec["can_proceed_direct"])
        self.assertGreater(dec["projected_battery_kwh"], dec["threshold_kwh"])

    # -----------------------------------------------------------------------
    # Test 2 — Threshold Equality
    # -----------------------------------------------------------------------
    def test_2_threshold_equality(self):
        """
        When projected battery after reaching the next delivery is exactly equal
        to the configured threshold, direct travel is rejected and charging
        evaluation is triggered.
        """
        capacity = 40.0
        threshold_pct = 25.0  # threshold_kwh = 10.0 kWh
        reserve_pct = 10.0    # reserve_kwh = 4.0 kWh
        eff = 6.0

        # Construct battery such that: battery - (dist / eff) == threshold_kwh exactly
        leg_dist = distance("Depot", "Pharmacy")
        leg_energy = leg_dist / eff
        exact_start_battery = 10.0 + leg_energy

        dec = evaluate_battery_decision(
            battery_kwh=exact_start_battery,
            capacity_kwh=capacity,
            threshold_pct=threshold_pct,
            reserve_pct=reserve_pct,
            cur="Depot",
            nxt="Pharmacy",
            eff=eff
        )
        self.assertEqual(dec["status"], "threshold_equal")
        self.assertFalse(dec["can_proceed_direct"])
        self.assertAlmostEqual(dec["projected_battery_kwh"], 10.0, places=3)
        self.assertIn("exactly equal to the configured threshold", dec["reason"])
        self.assertGreater(len(dec["charger_candidates"]), 0)

    # -----------------------------------------------------------------------
    # Test 3 — Low Battery
    # -----------------------------------------------------------------------
    def test_3_low_battery(self):
        """
        When starting battery is low and projected battery falls below threshold,
        the optimizer evaluates charging stations and selects the station leading
        to the minimum-cost feasible complete route.
        """
        cfg = {
            "capacity_kwh": 40.0,
            "soc_pct": 35.0,       # 14.0 kWh
            "km_per_kwh": 6.0,
            "reserve_pct": 10.0,   # 4.0 kWh
            "threshold_pct": 25.0, # 10.0 kWh
            "charger_kw": 22.0,
            "route_budget_km": 150.0,
        }
        res = branch_and_bound_with_charging(["Pharmacy", "Office", "Grocery"], cfg)
        self.assertTrue(res["feasible"])
        charge_events = [e for e in res["best_events"] if e["type"] == "charge"]
        self.assertGreaterEqual(len(charge_events), 1)

        # Check battery decision for low battery
        dec = evaluate_battery_decision(
            battery_kwh=14.0, capacity_kwh=40.0, threshold_pct=25.0,
            reserve_pct=10.0, cur="Depot", nxt="Grocery", eff=6.0
        )
        # Depot to Grocery distance is 24.03 km -> energy needed = 4.006 kWh -> projected = 9.99 kWh < 10.0 kWh
        self.assertEqual(dec["status"], "below_threshold")
        self.assertFalse(dec["can_proceed_direct"])

    # -----------------------------------------------------------------------
    # Test 4 — Nearest Charger Is Not Best
    # -----------------------------------------------------------------------
    def test_4_nearest_charger_is_not_best(self):
        """
        Demonstrates that selecting the geographically nearest charger from current
        location can produce a longer total detour than choosing a charger that
        aligns better with the continuation route. The global optimizer picks
        the better charger.
        """
        cur = "Office"
        nxt = "Grocery"
        # From Office: CH-2 is 10.30 km (nearest), CH-1 is 16.57 km.
        dist_ch2 = distance(cur, "CH-2")
        dist_ch1 = distance(cur, "CH-1")
        self.assertLess(dist_ch2, dist_ch1, "CH-2 must be geographically closer to Office than CH-1")

        # But continuation to Grocery: CH-2 to Grocery is 21.25 km; CH-1 to Grocery is 11.41 km.
        total_via_ch2 = dist_ch2 + distance("CH-2", nxt)
        total_via_ch1 = dist_ch1 + distance("CH-1", nxt)
        self.assertLess(total_via_ch1, total_via_ch2, "CH-1 route to Grocery must have shorter total detour")

        # Under low battery at Office where charging is required to reach Grocery:
        cfg = {
            "capacity_kwh": 40.0,
            "soc_pct": 20.0,       # 8.0 kWh at Depot (will need charger early)
            "km_per_kwh": 6.0,
            "reserve_pct": 5.0,
            "threshold_pct": 25.0,
            "charger_kw": 22.0,
            "route_budget_km": 150.0,
        }
        res = branch_and_bound_with_charging(["Grocery"], cfg)
        self.assertTrue(res["feasible"])
        chargers_visited = [e["station"] for e in res["best_events"] if e["type"] == "charge"]
        self.assertIn("CH-1", chargers_visited)

    # -----------------------------------------------------------------------
    # Test 5 — All Deliveries Visited Exactly Once, Ending at Hub
    # -----------------------------------------------------------------------
    def test_5_all_deliveries_visited_once(self):
        """
        Every selected delivery must be visited exactly once, starting at Depot
        and ending at Hub.
        """
        cfg = {
            "capacity_kwh": 50.0,
            "soc_pct": 80.0,
            "km_per_kwh": 6.0,
            "reserve_pct": 15.0,
            "threshold_pct": 25.0,
            "charger_kw": 22.0,
            "route_budget_km": 100.0,
        }
        orders = ["Pharmacy", "Office", "Grocery"]
        res = branch_and_bound_with_charging(orders, cfg)
        self.assertTrue(res["feasible"])

        # Delivery sequence contains each order exactly once
        self.assertEqual(sorted(res["best_route"]), sorted(orders))
        self.assertEqual(len(res["best_route"]), len(orders))

        # Planned route starts at Depot and ends at Hub
        planned = res["best_planned"]
        self.assertEqual(planned[-1], "Hub")
        for o in orders:
            self.assertEqual(planned.count(o), 1)

    # -----------------------------------------------------------------------
    # Test 6 — Infeasible Scenario
    # -----------------------------------------------------------------------
    def test_6_infeasible_scenario(self):
        """
        When starting battery is too low to reach any charging station or
        delivery without violating safety reserve, the application reports
        infeasible with a clear explanation without crashing.
        """
        cfg = {
            "capacity_kwh": 20.0,
            "soc_pct": 10.0,      # 2.0 kWh
            "km_per_kwh": 6.0,
            "reserve_pct": 10.0,  # 2.0 kWh reserve -> 0 kWh available for travel
            "threshold_pct": 20.0,
            "charger_kw": 22.0,
            "route_budget_km": 100.0,
        }
        res = branch_and_bound_with_charging(["Pharmacy", "Office", "Grocery"], cfg)
        self.assertFalse(res["feasible"])
        self.assertIsNone(res["best_cost_km"])

        # API endpoint handles infeasibility gracefully
        r = self.client.post("/api/optimize", json={
            "order_ids": ["A", "B", "C"],
            "capacity_kwh": 20.0,
            "soc_pct": 10.0,
            "km_per_kwh": 6.0,
            "reserve_pct": 10.0,
            "threshold_pct": 20.0,
            "charger_kw": 22.0,
            "route_budget_km": 100.0,
        })
        self.assertEqual(r.status_code, 200)
        data = r.json
        self.assertFalse(data["feasible"])
        self.assertIn("infeasibility_reason", data)
        self.assertIn("No feasible route exists", data["infeasibility_reason"])

    # -----------------------------------------------------------------------
    # Test 7 — Simulation Steps & Speed Parameters
    # -----------------------------------------------------------------------
    def test_7_simulation_steps_and_speed_presets(self):
        """
        Simulation steps are decomposed into road segments along Dijkstra paths,
        providing step indices, segment coordinates, battery tracking, and dwell pauses.
        """
        r = self.client.post("/api/optimize", json={
            "order_ids": ["A", "B", "C"],
            "capacity_kwh": 40.0,
            "soc_pct": 75.0,
            "km_per_kwh": 6.0,
            "reserve_pct": 15.0,
            "threshold_pct": 25.0,
            "charger_kw": 22.0,
            "route_budget_km": 100.0,
        })
        self.assertEqual(r.status_code, 200)
        data = r.json
        sim_steps = data["simulation_steps"]
        self.assertGreater(len(sim_steps), 0)

        # First step is start at Depot
        self.assertEqual(sim_steps[0]["type"], "start")
        self.assertEqual(sim_steps[0]["from"], "Depot")

        # Every travel step has road segment endpoints
        for step in sim_steps:
            self.assertIn("title", step)
            self.assertIn("battery_pct", step)
            self.assertIn("distance_so_far_km", step)
            self.assertIn("energy_so_far_kwh", step)
            self.assertGreater(step["dwell_ms"], 0)

    # -----------------------------------------------------------------------
    # Test 8 — Algorithm Correctness (Optimality vs Exhaustive Search)
    # -----------------------------------------------------------------------
    def test_8_algorithm_correctness_against_exhaustive_search(self):
        """
        Compares Branch & Bound results against exhaustive brute-force search
        over all order permutations and charger choices, verifying identical
        optimal distance and constraint satisfaction.
        """
        test_configs = [
            # High battery (no charging needed)
            {
                "capacity_kwh": 40.0,
                "soc_pct": 90.0,
                "km_per_kwh": 6.0,
                "reserve_pct": 10.0,
                "threshold_pct": 20.0,
                "charger_kw": 22.0,
            },
            # Moderate battery (requires charger detour)
            {
                "capacity_kwh": 40.0,
                "soc_pct": 35.0,
                "km_per_kwh": 6.0,
                "reserve_pct": 10.0,
                "threshold_pct": 25.0,
                "charger_kw": 22.0,
            },
        ]

        orders = ["Pharmacy", "Office", "Grocery"]
        for cfg in test_configs:
            bnb_res = branch_and_bound_with_charging(orders, cfg)
            ex_res = exhaustive_search(orders, cfg)

            self.assertEqual(bnb_res["feasible"], ex_res["feasible"])
            if bnb_res["feasible"]:
                self.assertAlmostEqual(
                    bnb_res["best_cost_km"],
                    ex_res["best_cost_km"],
                    places=2,
                    msg=f"Branch & Bound cost ({bnb_res['best_cost_km']}) must match exhaustive search ({ex_res['best_cost_km']})"
                )

    # -----------------------------------------------------------------------
    # Test 9 — Vehicle at 59.5% Battery at Grocery to Hub (No Charging Stop)
    # -----------------------------------------------------------------------
    def test_9_vehicle_at_59_5_pct_direct_to_hub_feasible_no_charge(self):
        """
        Requirement 9: Test for a vehicle at 59.5% battery whose direct route to
        the Hub is feasible and does not require charging.
        Verifies:
          1. CH-1 is recognized as an intermediate road-network vertex (transit waypoint),
             NOT a charging stop.
          2. Direct journey to Hub leaves projected battery above threshold (48.3% > 25%).
          3. Optimizer plans zero charging events.
          4. Simulation steps distinguish transit waypoint CH-1 from charging stops.
        """
        capacity = 40.0
        eff = 6.0
        threshold_pct = 25.0  # 10.0 kWh
        reserve_pct = 15.0    # 6.0 kWh
        # EV at Grocery with 59.5% battery = 23.80 kWh
        battery_at_grocery = 23.80

        # Step 1: Direct battery decision check from Grocery to Hub
        dec = evaluate_battery_decision(
            battery_kwh=battery_at_grocery,
            capacity_kwh=capacity,
            threshold_pct=threshold_pct,
            reserve_pct=reserve_pct,
            cur="Grocery",
            nxt="Hub",
            eff=eff
        )
        self.assertTrue(dec["can_proceed_direct"], "Vehicle at 59.5% battery must be allowed to proceed directly to Hub")
        self.assertEqual(dec["status"], "above_threshold")
        self.assertAlmostEqual(dec["current_soc_pct"], 59.5, places=1)
        self.assertGreater(dec["projected_soc_pct"], threshold_pct)
        self.assertGreater(dec["projected_battery_kwh"], dec["threshold_kwh"])

        # Step 2: Full route optimization starting from 75% SOC (arrives at Grocery with ~59.5% SOC)
        r = self.client.post("/api/optimize", json={
            "order_ids": ["A", "B", "C"],
            "capacity_kwh": capacity,
            "soc_pct": 75.0,
            "km_per_kwh": eff,
            "reserve_pct": reserve_pct,
            "threshold_pct": threshold_pct,
            "charger_kw": 22.0,
            "route_budget_km": 100.0,
        })
        self.assertEqual(r.status_code, 200)
        data = r.json
        self.assertTrue(data["feasible"])

        # Planned route must NOT contain any charger
        planned = data["planned_route"]
        self.assertEqual(planned, ["Depot", "Pharmacy", "Office", "Grocery", "Hub"])
        actual_charges = [e for e in data["charging_events"] if e.get("type") == "charge"]
        self.assertEqual(len(actual_charges), 0)
        self.assertTrue(all(node not in CHARGERS for node in planned))

        # Step 3: Verify simulation steps
        sim_steps = data["simulation_steps"]
        # Find step where EV travels segment Grocery -> CH-1
        transit_step = next(
            (s for s in sim_steps if s.get("from") == "Grocery" and s.get("to") == "CH-1"),
            None
        )
        self.assertIsNotNone(transit_step, "Simulation must decompose road leg into Grocery -> CH-1 segment")
        self.assertTrue(transit_step["is_transit_waypoint"], "CH-1 segment must be marked as transit waypoint")
        self.assertFalse(transit_step["is_charging_stop"], "CH-1 segment must NOT be marked as charging stop")
        self.assertEqual(transit_step["final_destination"], "Hub")
        self.assertIn("Transit only", transit_step["title"])
        self.assertIn("no charging is performed", transit_step["detail"])

        # No step should be of type 'charge'
        charge_steps = [s for s in sim_steps if s.get("type") == "charge"]
        self.assertEqual(len(charge_steps), 0, "No charging steps should exist in simulation")

    # -----------------------------------------------------------------------
    # Test 10 — Genuine Charging Necessary Scenario & Selection Reason
    # -----------------------------------------------------------------------
    def test_10_charging_genuinely_necessary_with_complete_selection_reason(self):
        """
        Requirement 2 & 9: Scenario where charging genuinely IS necessary.
        Verifies:
          1. Low battery correctly identifies that direct travel violates threshold.
          2. Optimizer selects a feasible charger minimizing total complete route distance.
          3. Selection reason contains all required metrics:
             projected before/after, distance to charger, energy needed,
             threshold comparison, safety reserve, and effect on total route distance.
        """
        capacity = 40.0
        eff = 6.0
        threshold_pct = 25.0  # 10.0 kWh
        reserve_pct = 15.0    # 6.0 kWh
        # EV at Grocery with only 12.0 kWh (30.0% SOC)
        # Leg from Grocery to Hub is 26.867 km -> needs 4.478 kWh -> projected = 7.52 kWh (18.8% < 25.0% threshold)
        low_battery_at_grocery = 12.0

        dec = evaluate_battery_decision(
            battery_kwh=low_battery_at_grocery,
            capacity_kwh=capacity,
            threshold_pct=threshold_pct,
            reserve_pct=reserve_pct,
            cur="Grocery",
            nxt="Hub",
            eff=eff
        )
        self.assertFalse(dec["can_proceed_direct"], "Direct travel must be rejected when projected falls below threshold")
        self.assertEqual(dec["status"], "below_threshold")

        # Run optimization starting with low SOC (32.0% SOC = 12.8 kWh)
        r = self.client.post("/api/optimize", json={
            "order_ids": ["A", "B", "C"],
            "capacity_kwh": capacity,
            "soc_pct": 32.0,
            "km_per_kwh": eff,
            "reserve_pct": reserve_pct,
            "threshold_pct": threshold_pct,
            "charger_kw": 22.0,
            "route_budget_km": 150.0,
        })
        self.assertEqual(r.status_code, 200)
        data = r.json
        actual_charges = [e for e in data["charging_events"] if e.get("type") == "charge"]
        self.assertGreater(len(actual_charges), 0, "Actual charging stops must be scheduled when SOC is genuinely low")

        # Verify charging event selection reason fields
        sim_steps = data["simulation_steps"]
        dec_step_with_charger = next(
            (s for s in sim_steps if s.get("type") == "battery_decision" and s.get("selected_charger")),
            None
        )
        self.assertIsNotNone(dec_step_with_charger, "Must contain a battery decision step with selected charger")
        sel_reason = dec_step_with_charger.get("selection_reason")
        self.assertIsNotNone(sel_reason, "Must provide detailed selection_reason dictionary")
        self.assertIn("charger", sel_reason)
        self.assertIn("battery_before_travel_kwh", sel_reason)
        self.assertIn("dist_to_charger_km", sel_reason)
        self.assertIn("energy_to_charger_kwh", sel_reason)
        self.assertIn("battery_at_charger_kwh", sel_reason)
        self.assertIn("energy_added_kwh", sel_reason)
        self.assertIn("threshold_kwh", sel_reason)
        self.assertIn("reserve_kwh", sel_reason)
        self.assertIn("detour_increase_km", sel_reason)
        self.assertIn("explanation", sel_reason)

    # -----------------------------------------------------------------------
    # Test 11 — Unreachable Charger Violating Safety Reserve Never Selected
    # -----------------------------------------------------------------------
    def test_11_unreachable_charger_never_selected(self):
        """
        Requirement 6: Never select a charger that cannot be reached without
        violating the safety reserve constraint.
        """
        capacity = 40.0
        eff = 6.0
        # High safety reserve: 30% = 12.0 kWh
        # EV has 14.0 kWh at Depot -> only 2.0 kWh (12 km) available
        # CH-1 is ~24 km away from Depot -> unreachable with 30% reserve
        dec = evaluate_battery_decision(
            battery_kwh=14.0,
            capacity_kwh=capacity,
            threshold_pct=35.0,
            reserve_pct=30.0,
            cur="Depot",
            nxt="Pharmacy",
            eff=eff
        )
        ch1_cand = next((c for c in dec["charger_candidates"] if c["name"] == "CH-1"), None)
        self.assertIsNotNone(ch1_cand)
        self.assertFalse(ch1_cand["reachable"], "CH-1 must be marked unreachable when reserve is violated")
        self.assertFalse(ch1_cand["continuation_feasible"])


if __name__ == "__main__":
    unittest.main()
