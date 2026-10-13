# VoltRoute

**Optimal delivery and charging planning for EV-based delivery partners.**
Design and Analysis of Algorithms project, built with Flask and plain HTML/CSS/JS.

An electric delivery vehicle starts at the Depot, must drop off a set of orders, and
end at the Hub. Its battery is limited, so on some legs it has to detour through a
charging station. VoltRoute finds the **shortest route that delivers every order
without breaking the battery rules**, choosing the delivery order and the charging
stops together. It then animates the trip on a city map.

## Algorithms

| Problem | Algorithm | Complexity |
|---|---|---|
| Step-by-step trace for the lab | The same Dijkstra, recording each pop/relax and stopping once all targets are final | O((V + E) log V) |
| Road distance between any two places | Dijkstra (binary heap), run from every vertex once at startup | O((V + E) log V) per source |
| Best delivery order + charging stops | Best-first Branch & Bound | O(n! · (k+1)ⁿ) worst case, much less in practice |
| Lower bound for pruning | Prim's MST over the remaining stops | O(n²) per node |
| Checking the answer | Brute force over every order and every charger choice | O(n! · (k+1)ⁿ) |

*n* = number of orders, *k* = number of chargers.

**Branch & Bound.** A search state is *(current location, orders delivered, battery left)*.
From each state the EV can go to any undelivered order, either directly or through a
charger. When nothing is left to deliver, the only move is to the Hub. States sit in a
min-heap keyed by

```
bound = distance so far + min(here → remaining) + MST(remaining) + min(remaining → Hub)
```

Every way of finishing the route has to leave the current spot, connect all remaining
stops and then enter the Hub, so the bound never overestimates. Detours through chargers
only make routes longer, so they don't break this. Any state whose bound is no better than
the best complete route found so far gets pruned. The first complete route taken off the
heap is optimal.

**Battery rule.** Before every leg the EV works out how much battery it would have on arrival:

```
projected = battery − distance / efficiency
```

- If `projected` is **strictly above** the threshold, it drives straight there.
- If not (equal to the threshold, below it, or below the safety reserve), it has to go through a
  charger. A charger can be used only if the EV reaches it with at least the reserve and, after
  charging to 100%, can still reach the next stop with at least the reserve.
- Branch & Bound tries every charger that can be used. It doesn't just take the nearest one,
  because the nearest charger can make the whole route longer.

The app runs the brute-force search next to Branch & Bound on every request and shows
whether both found the same distance. Brute force grows as n!, so once it would take more than
120,000 moves it is skipped and the page says so. For example, with all 8 stops Branch & Bound
expands about 20 states, while brute force has to check about 69,000 complete routes.

## The app

- **Scenarios**: five presets (normal day, low battery, closest charger isn't best, north bridge
  closed, not enough charge), or set your own values. The route is replanned as soon as you change an input.
- **City**: two districts split by a river with only two bridges, 8 delivery stops, 5 chargers
  (22, 50 and 120 kW), 15 road junctions and 49 roads. Because of the river and the junctions,
  the distance by road is often much longer than the straight line, so Dijkstra actually matters.
  For example, CH-2 and CH-3 are 6.4 km apart in a straight line but 17.6 km apart by road.
- **Add your own places**: use **+ Delivery stop** or **+ Charger**, then click the map. Each new
  place gets a road to its 3 nearest neighbours on the same side of the river, and the server
  rebuilds the graph and reruns Dijkstra. **Close road** blocks a road (click it again to reopen),
  for example a bridge, and everything is replanned around it. **Remove** deletes a place you added, and **Reset map** removes all of them. You can
  add up to 8 places and select up to 10 stops.
- **Map**: the planned route, numbered delivery stops, and the chargers it will use. The EV
  follows the real road segments. If it drives past a charger without stopping, that is shown too.
- **Dijkstra lab**: switch the map to *Dijkstra lab* and pick a start and target (one place, every
  charger, or every place). It replays the real Python Dijkstra run one step at a time:
  - Places are gray until reached, gold while their distance may still improve, and green once
    their distance is final. The place being expanded has an orange ring.
  - The road being tried is highlighted with its length, and the shortest-path tree grows in gold.
  - The steps follow the stages Start → Choose (pop the minimum) → Compare (relax a road) → Result.
    Each step comes with a plain-English explanation, e.g. "3.4 + 6.0 = 9.4 km, not shorter than
    4.0 km, keep 4.0 km".
  - The priority queue is shown live, including outdated entries that get skipped. A targets table
    compares the straight-line and road distances.
  - From the **Battery check** panel, *Watch Dijkstra search for chargers from …* opens the lab
    at the exact moment the EV needs to charge.
- **Fullscreen** (button or `F`): the map on the left and every panel on the right (metrics,
  battery check, itinerary and algorithm stats). Press `Space` to play/pause, `→` / `←` (or `]` / `[`)
  to step, and `Esc` to exit.
- **Playback**: play/pause, step through one action at a time, restart, change speed (1×/2×/4×),
  or drag the slider to any point.
- **Itinerary**: every stop with the distance and battery level, plus charging time and energy added.
- **Battery check**: for the current leg, why the EV goes direct or charges first. When it
  charges, a table compares all the chargers.

## Running it

Requires Python 3.10+.

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

| Test | Checks |
|---|---|
| 1 Enough battery | Drives directly, no charging |
| 2 Threshold equality | Arriving at exactly the threshold isn't allowed, so it must charge |
| 3 Low battery | A charging stop is added |
| 4 Closest charger isn't best | Hospital → Grocery: CH-3 looks closest, CH-5 is closest by road, but going via CH-3 makes the whole route shortest |
| 5 All deliveries | Every order visited once, starts at Depot, ends at Hub |
| 6 Infeasible | Reports the problem clearly instead of crashing |
| 7 Passing a charger | Driving through a charger on the way somewhere is not a charging stop |
| 8 Optimality | Branch & Bound matches brute force on 3,360 configurations, and on all 8 stops |
| 9 Dijkstra lab trace | Replaying the recorded steps: each pop is the queue minimum, each relaxation is correct, and the final distances match |
| 10 Road closures | Closing a bridge forces a detour; cutting a place off is reported clearly |
| Added places | New places get roads (never across the river), are used by the optimizer, and bad input is rejected |
| Others | Dijkstra paths, graph connected, triangle inequality, bound never overestimates, reserve never broken, brute-force cutoff, input validation |

## Project layout

```
app.py               graph, Dijkstra, Branch & Bound, brute force, Flask routes
templates/index.html the whole frontend (map, playback, panels)
tests/test_app.py    unit and acceptance tests
```

## Assumptions

- The city is a graph of 30 places (15 of them plain junctions) and 49 roads, plus any places you
  add. Distances are taken from map coordinates (1 px = 40 m).
- Energy used is proportional to distance (constant km/kWh). Speed is constant.
- A charging stop always fills the battery to 100%, at that station's own power rating.
- At most one charging stop between two deliveries.
- Each delivery adds 4 minutes of service time to the total time.
