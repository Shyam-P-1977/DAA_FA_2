# VoltRoute — Optimal Delivery and Charging Planning for EV-Based Delivery Partners

A Flask + HTML/CSS/JavaScript interactive educational working model for Design and Analysis of Algorithms (DAA).

VoltRoute is an **interactive two-stage EV delivery route and charging optimization simulator**. Users can configure their own custom road networks, start/end terminals, charging stations, and directional road connections. The application visualizes a baseline unoptimized route (**Stage 1**) alongside a globally optimal Branch & Bound route (**Stage 2**), complete with slow road-by-road animation, real-time battery telemetry, and a side-by-side comparative analysis.

---

## Key Features

1. **User-Configurable Delivery Network (Requirement 2)**:
   - **Delivery Locations**: Select from 1 to 10 delivery locations with custom IDs, names, and descriptions.
   - **Custom Start & End Terminals**: Configurable start depot and ending hub with custom IDs and names.
   - **Charging Stations**: Specify charging stations with individual IDs, names, and charging rates in kW.
   - **Road Connections**: Weighted graph manager supporting bidirectional and one-way directed roads with distance validation.
   - **Quick Actions**: "Load Sample Network", "Clear Network", and "Validate Network" with graph reachability checks.

2. **Two-Stage Simulation Architecture (Requirements 3 & 4)**:
   - **Stage 1 (Unoptimized Baseline)**:
     - Visits delivery locations in user-entered order ($D_1 \to D_2 \to \dots \to D_n \to \text{Hub}$).
     - Traverses shortest road paths using Dijkstra's algorithm.
     - Inserts the nearest reachable charging station when projected battery $\le$ threshold or safety reserve is violated.
   - **Stage 2 (Branch & Bound Global Optimization)**:
     - Exact search over delivery permutations with integrated charging station detours.
     - Prunes non-improving branches using an admissible Prim's Minimum Spanning Tree (MST) lower bound.
     - Globally optimal charging stop selection minimizing complete continuation distance.

3. **Visual Route Comparison (Requirement 6)**:
   - Displays side-by-side measured results:
     - Delivery ordering and complete road path
     - Number of charging stops and charging duration
     - Total road distance (km) and distance saved ($d_{\text{baseline}} - d_{\text{optimized}}$)
     - Percentage improvement: $100 \times (d_{\text{baseline}} - d_{\text{optimized}}) / d_{\text{baseline}}$
     - Total energy consumed (kWh) and energy saved
     - Travel time, charging time, and completion time
     - Branch & Bound search states expanded vs. pruned

4. **Interactive Simulated Road Map (Requirement 7)**:
   - Dynamic SVG rendering adapting to any user-defined network.
   - Visual distinctions:
     - Start terminal (▶), Ending hub (🏁)
     - Delivery stops with sequence number badges (`1`, `2`, `3`)
     - Scheduled charging stops with amber lightning badges (`⚡`)
     - Intermediate road transit waypoints (no stop / no charge)
     - Directional arrow markers for one-way roads
     - Distance badges on all road connections
   - Vehicle animated smoothly along actual road segments with controllable speeds:
     - **Slow**: ~4.0s per road segment
     - **Normal**: ~2.0s per road segment (default)
     - **Fast**: ~0.8s per road segment
   - Playback controls: Play, Pause, Resume, Restart, Previous Step, Next Step.

5. **Battery Decision Rule & Energy Accounting (Requirement 5D)**:
   - Energy consumed on road leg: $E = d / \text{efficiency}$.
   - Projected battery: $B_{\text{proj}} = B - E$.
   - Safety reserve: $B_{\text{proj}} \ge R \times C_{\text{bat}}$.
   - Battery threshold: If $B_{\text{proj}} \le T \times C_{\text{bat}}$, charging evaluation is triggered.
   - Charging duration: $t_{\text{charge}} = (\Delta E / P_{\text{charger}}) \times 60\text{ min}$.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Setup Virtual Environment
Open PowerShell in the `VoltRoute_Flask` directory:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Application
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## User Guide

### A. Configuring a Delivery Network
1. In the sidebar under **Configure Delivery Network**, set your Start Depot ID (e.g. `Depot`) and End Hub ID (e.g. `Hub`).
2. Select the number of delivery locations (1 to 10). Enter custom IDs (e.g. `Pharmacy`, `Office`, `Grocery`), names, and descriptions.
3. Manage charging stations with custom IDs and kW power ratings.
4. Add or remove road connections. Specify source, destination, distance in km, and whether the road is two-way or one-way.
5. Click **Validate Network** to verify that all delivery nodes and terminals are reachable via Dijkstra shortest paths.

### B. Running Stage 1: Unoptimized Baseline
1. Click **Start Stage 1 Simulation**.
2. Click **Play** or step forward with **Next Step**.
3. Watch the EV navigate road segments in the order entered, displaying real-time battery depletion, distance, and charging stops.

### C. Running Stage 2: Optimized Route
1. Click **Start Stage 2 Simulation** (or click **Skip to Stage 2**).
2. The Branch & Bound optimizer evaluates all permutations, prunes branches with the MST lower bound, and calculates the optimal schedule.
3. Click **Play** or step through the optimal route animation.

### D. Viewing Comparative Results
1. Click the **Route Comparison** tab to inspect the side-by-side comparison table showing distance saved, percentage improvement, and energy economy.
2. Click the **Branch & Bound Search Tree** tab to view the expanded, pruned, and optimal search tree nodes.
3. Click the **Battery & Charging Schedule** tab for the event timeline.

---

## Testing

The project contains 23 comprehensive acceptance unit tests covering graph algorithms, energy accounting, B&B search, and API validation:

```powershell
python -m unittest discover -s tests -v
```

### Test Suite Coverage:
- `test_1_enough_battery`: Vehicle proceeds directly when battery remains above threshold.
- `test_2_threshold_equality`: Direct travel rejected when battery equals threshold.
- `test_3_low_battery`: Evaluates chargers when projected battery falls below threshold.
- `test_4_nearest_charger_is_not_best`: Optimizer avoids greedy nearest charger in favor of better global continuation.
- `test_5_all_deliveries_visited_once`: Every delivery visited exactly once, ending at Hub.
- `test_6_infeasible_scenario`: Infeasible battery/network scenarios reported with clear explanations.
- `test_7_simulation_steps_and_speed_presets`: Granular road-segment decomposition and dwell time verification.
- `test_8_algorithm_correctness_against_exhaustive_search`: Verifies Branch & Bound optimality against brute-force permutation search.
- `test_9_vehicle_at_59_5_pct_direct_to_hub_feasible_no_charge`: EV at 59.5% battery visits CH-1 as transit waypoint only; 0 charges planned.
- `test_10_charging_genuinely_necessary_with_complete_selection_reason`: Verifies complete selection reason metrics when charging is necessary.
- `test_11_unreachable_charger_never_selected`: Unreachable chargers violating safety reserve are rejected.
- `test_12_user_defined_network_and_custom_nodes`: Custom start, end, deliveries, chargers, and roads validated and solved.
- `test_13_oneway_and_bidirectional_connections`: Directionality respected on one-way road segments.
- `test_14_disconnected_network_rejected`: Unreachable/disconnected graphs detected and rejected.
- `test_15_stage1_baseline_route_generation_and_simulation`: Baseline route generation and simulation telemetry verified.
- `test_16_route_comparison_calculation`: Route distance saved and percentage improvement formulas verified.
- `test_17_charging_duration_and_amount`: Charging duration formula verified.
- Input validation tests: rejection of empty orders, invalid battery capacity, negative budgets, and threshold < reserve.

---

## API Endpoints

- `GET /api/sample_network`: Returns the default sample network specification and vehicle settings.
- `POST /api/validate_network`: Validates a user-defined network specification and checks graph reachability.
- `POST /api/baseline`: Computes the Stage 1 baseline route, simulation steps, and metrics.
- `POST /api/optimize`: Computes the Stage 2 Branch & Bound optimal route, search tree, simulation steps, and metrics.
- `POST /api/compare`: Computes side-by-side comparative metrics between baseline and optimized results.
- `GET /api/health`: Healthcheck endpoint.
