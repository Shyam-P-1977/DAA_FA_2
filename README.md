# VoltRoute — Optimal Delivery and Charging Planning for EV-Based Delivery Partners

A Flask + HTML/CSS/JavaScript educational virtual working model for Design and Analysis of Algorithms (DAA).

VoltRoute optimizes the multi-stop delivery sequence and charging schedule for an electric delivery partner. It uses **Branch & Bound** with an admissible **MST lower bound** for globally optimal sequencing and charging selection, paired with **Dijkstra's Algorithm** for road network shortest paths.

---

## Key Features

1. **Exact Joint Branch & Bound Optimization**:
   - Searches for the globally minimum-distance feasible route starting at Depot, delivering every selected order exactly once, satisfying battery threshold and reserve constraints, visiting charging stations whenever necessary, and returning to Hub.
   - Charging stops are evaluated **inside the Branch & Bound search state**, not in a disconnected post-processing step.
   - Admissible lower bound using Prim's MST over road network distances guarantees mathematical global optimality.

2. **Dijkstra's Algorithm on Road Network Graph**:
   - Represents the simulated city as an undirected graph $G=(V, E)$ with 8 vertices (Depot, 3 Deliveries, 3 Chargers, Hub) and non-negative weighted road edges matching the visual street map.
   - Computes exact road distances and vertex-by-vertex paths between any pair of stops.

3. **Battery Decision Rule & Threshold Policy**:
   - At each stop, evaluates projected battery $B_{proj} = B - E(cur, nxt)$ before selecting the next destination.
   - **Above Threshold ($B_{proj} > T$ and $B_{proj} \ge R$)**: Proceed directly.
   - **At Threshold ($B_{proj} == T$)**: Direct travel rejected per threshold policy; evaluate reachable charging stations.
   - **Below Threshold ($B_{proj} < T$)**: Direct travel rejected; evaluate reachable charging stations.
   - **Insufficient Battery ($B_{proj} < R$)**: Cannot safely reach destination; evaluate reachable charging stations.
   - Evaluates all reachable charging stations to select the station yielding the lowest-cost feasible continuation. Never chooses a charger purely for geographic proximity if it leads to a worse overall route.

4. **Paced, Controllable Road-Segment Animation**:
   - Animates the EV moving along each individual road segment rather than jumping across the map.
   - Three speed presets:
     - **Slow**: ~4.0 seconds per road segment
     - **Normal**: ~2.0 seconds per road segment (default)
     - **Fast**: ~0.8 seconds per road segment
   - Complete playback controls: **Start Simulation (Play)**, **Pause**, **Resume**, **Restart**, and **Next Step**.
   - Visible progress indicator: `Step X of Y`.
   - Gradual battery percentage animation decreasing during travel and increasing during charging.
   - Brief pauses at deliveries and charging stations for readability.
   - Simulation runs on demand when the user clicks Start Simulation; does not run immediately on page load.

5. **Rich Visualizations & Viva Guide**:
   - Interactive animated SVG road map with delivery stop sequence badges (Stop 1, Stop 2, Stop 3).
   - Dynamic Branch & Bound search tree showing expanded, pruned, and best complete nodes.
   - Contextual Battery Decision card and battery event timeline.
   - College viva reference section with time complexities and algorithm explanations.

---

## Running on Windows

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Setup Virtual Environment
Open PowerShell in the `VoltRoute_Flask` directory:
```powershell
py -m venv .venv
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

## Running Unit Tests

The test suite covers all 8 Acceptance Tests specified in the project requirements:
```powershell
python -m unittest discover -s tests -v
```

### Acceptance Tests Verified:
- **Test 1 — Enough battery**: Next delivery is feasible and battery remains strictly above threshold; proceeds directly.
- **Test 2 — Threshold equality**: Projected battery equals threshold; direct travel rejected, charging route evaluated.
- **Test 3 — Low battery**: Projected battery falls below threshold; optimizer selects minimum-cost charger continuation.
- **Test 4 — Nearest charger is not best**: Nearest charger leads to a longer overall route; optimizer selects the better charger.
- **Test 5 — All deliveries**: Every selected delivery visited exactly once, ending at Hub.
- **Test 6 — Infeasible scenario**: No feasible route exists; reports clear reason without crashing.
- **Test 7 — Slow animation & controls**: Validates speed presets and simulation step decomposition.
- **Test 8 — Algorithm correctness**: Verified against exhaustive brute-force search over all order permutations.
- **API & Validation**: Validates inputs, empty order sets, invalid battery capacities, threshold < reserve, and Dijkstra shortest paths.

---

## Algorithms & Theoretical Summary

### A. Branch and Bound
- **State Representation**: `(current_location, delivered_orders_mask, battery_kWh)`
- **Priority Queue**: Min-heap ordered by admissible lower bound $f = \text{cost\_so\_far} + \text{LB}$.
- **Lower Bound Formula**:
  $$\text{LB} = \min_{r \in \text{rem}} d(u, r) + \text{MST}(\text{rem}) + \min_{r \in \text{rem}} d(r, \text{Hub})$$
- **Admissibility**: Because every valid completion must leave $u$, span all remaining deliveries, and enter Hub, MST is guaranteed to be $\le$ true completion cost. Branches with $\text{LB} \ge \text{best\_cost}$ are safely pruned.

### B. Dijkstra's Algorithm
- Operates on non-negative weighted graph $G = (V, E)$ using a priority queue.
- Computes shortest road distances and exact vertex paths animated segment by segment.
- Time Complexity: $O((V + E) \log V)$.

### C. Battery Constraints
- $B$: Current battery energy (kWh)
- $T$: Configured battery threshold ($C \times T_{\%} / 100$)
- $R$: Configured safety reserve ($C \times R_{\%} / 100$)
- $E(i, j) = \text{distance}(i, j) / \text{efficiency}$
- Direct travel allowed if and only if $B - E(i, j) > T$ and $B - E(i, j) \ge R$.
- At charging stops, EV replenishes to full capacity (100% SOC) at rated charger power.
