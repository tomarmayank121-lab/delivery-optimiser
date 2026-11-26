import heapq
import json
import math
from itertools import count, permutations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent

graph = {}
coordinates = {
    "Warehouse": (0, 0),
    "A": (2, 0),
    "B": (1, 3),
    "C": (0, 4),
    "D": (4, 0),
    "E": (1, 5),
    "F": (7, 0),
    "G": (8, 2),
    "H": (9, 0),
}

roads = [
    ("Warehouse", "A", 3.0),
    ("Warehouse", "B", 4.2),
    ("Warehouse", "C", 5.0),
    ("A", "D", 3.0),
    ("B", "D", 4.0),
    ("C", "E", 2.0),
    ("D", "F", 3.3),
    ("E", "G", 2.5),
    ("F", "H", 2.4),
    ("G", "H", 3.3),
]


def add_road_to_graph(start, end, distance):
    graph.setdefault(start, [])
    graph.setdefault(end, [])
    graph[start].append((end, distance))
    graph[end].append((start, distance))


for place in coordinates:
    graph[place] = []

for road_start, road_end, road_distance in roads:
    add_road_to_graph(road_start, road_end, road_distance)


def heuristic(node, goal):
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return 0.1 * math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def astar(start, goal):
    if start not in graph or goal not in graph:
        return None, float("inf"), 0

    if start == goal:
        return [start], 0, 0

    pq = []
    tie = count()
    heapq.heappush(pq, (heuristic(start, goal), 0, next(tie), start, [start]))
    best_cost = {start: 0}
    expanded = 0

    while pq:
        _, cost, _, node, path = heapq.heappop(pq)

        if cost > best_cost.get(node, float("inf")):
            continue

        if node == goal:
            return path, cost, expanded

        expanded += 1

        for neighbour, distance in graph[node]:
            new_cost = cost + distance
            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                new_f = new_cost + heuristic(neighbour, goal)
                heapq.heappush(
                    pq,
                    (new_f, new_cost, next(tie), neighbour, path + [neighbour]),
                )

    return None, float("inf"), expanded


def dijkstra(start, goal):
    if start not in graph or goal not in graph:
        return None, float("inf"), 0

    if start == goal:
        return [start], 0, 0

    pq = []
    tie = count()
    heapq.heappush(pq, (0, next(tie), start, [start]))
    best_cost = {start: 0}
    expanded = 0

    while pq:
        cost, _, node, path = heapq.heappop(pq)

        if cost > best_cost.get(node, float("inf")):
            continue

        if node == goal:
            return path, cost, expanded

        expanded += 1

        for neighbour, distance in graph[node]:
            new_cost = cost + distance
            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                heapq.heappush(
                    pq, (new_cost, next(tie), neighbour, path + [neighbour])
                )

    return None, float("inf"), expanded


def network_data():
    return {
        "coordinates": {name: list(point) for name, point in coordinates.items()},
        "roads": [list(road) for road in roads],
        "total_distance": sum(road[2] for road in roads),
    }


def add_network_road(data):
    start = data.get("start")
    end = data.get("end")
    distance = data.get("distance")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("Enter two location names.")

    start = start.strip()
    end = end.strip()
    if not start or not end or start.casefold() == end.casefold():
        raise ValueError("Enter two different location names.")

    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise ValueError("Enter a valid road distance.")
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("Road distance must be a positive number.")

    existing_names = {name.casefold(): name for name in coordinates}
    canonical_start = existing_names.get(start.casefold(), start)
    canonical_end = existing_names.get(end.casefold(), end)

    if any(
        (road_start == canonical_start and road_end == canonical_end)
        or (road_start == canonical_end and road_end == canonical_start)
        for road_start, road_end, _ in roads
    ):
        raise ValueError("A road already connects those locations.")

    submitted_coordinates = data.get("coordinates", {})
    if not isinstance(submitted_coordinates, dict):
        raise ValueError("Location coordinates must be an object.")

    new_coordinates = {}
    for submitted_name, canonical_name in (
        (start, canonical_start),
        (end, canonical_end),
    ):
        if canonical_name in coordinates:
            continue
        point = submitted_coordinates.get(submitted_name)
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in point
            )
        ):
            raise ValueError("Provide valid X and Y coordinates for each new location.")
        new_coordinates[canonical_name] = tuple(point)

    for name, point in new_coordinates.items():
        coordinates[name] = point
        graph[name] = []

    add_road_to_graph(canonical_start, canonical_end, float(distance))
    roads.append((canonical_start, canonical_end, float(distance)))
    return network_data()


def multi_stop_route(start, stops):
    if start not in graph:
        raise ValueError("Starting location does not exist.")
    if not isinstance(stops, list) or not stops:
        raise ValueError("Select at least one delivery stop.")
    if any(not isinstance(stop, str) or stop not in graph for stop in stops):
        raise ValueError("One or more delivery locations do not exist.")
    if len(set(stops)) != len(stops) or start in stops:
        raise ValueError("Choose different delivery locations, not including the start.")
    if len(stops) > 8:
        raise ValueError("Maximum 8 delivery stops allowed.")

    best_distance = float("inf")
    best_order = None
    best_paths = None

    for order in permutations(stops):
        current = start
        total = 0
        paths = []
        possible = True

        for destination in order:
            path, distance, _ = astar(current, destination)
            if path is None:
                possible = False
                break
            total += distance
            paths.append((current, destination, distance, path))
            current = destination

        if possible and total < best_distance:
            best_distance = total
            best_order = order
            best_paths = paths

    if best_order is None:
        return None

    full_path = []
    for _, _, _, path in best_paths:
        if not full_path:
            full_path.extend(path)
        else:
            full_path.extend(path[1:])

    return {
        "order": [start, *best_order],
        "path": full_path,
        "distance": best_distance,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid request body.") from exc
        if length <= 0 or length > 1_000_000:
            raise ValueError("Request body must be between 1 byte and 1 MB.")
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Request body must be valid JSON.") from exc

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/network":
            self.send_json(200, network_data())
            return
        if path not in ("/", "/index.html"):
            self.send_error(404)
            return
        try:
            body = (ROOT / "index.html").read_bytes()
        except OSError:
            self.send_error(500, "Dashboard file could not be read.")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self.read_json()
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object.")

            if path == "/api/route":
                start = data.get("start")
                goal = data.get("goal")
                if not isinstance(start, str) or not isinstance(goal, str):
                    raise ValueError("Select a start and destination.")
                if start not in graph or goal not in graph:
                    raise ValueError("Unknown location.")
                if start == goal:
                    raise ValueError("Choose two different locations.")
                path_result, distance, a_nodes = astar(start, goal)
                if path_result is None:
                    self.send_json(200, {"found": False})
                    return
                _, _, d_nodes = dijkstra(start, goal)
                self.send_json(
                    200,
                    {
                        "found": True,
                        "path": path_result,
                        "distance": distance,
                        "a_nodes": a_nodes,
                        "d_nodes": d_nodes,
                    },
                )
                return

            if path == "/api/multi-stop":
                result = multi_stop_route(data.get("start"), data.get("stops"))
                if result is None:
                    self.send_json(200, {"found": False})
                    return
                self.send_json(200, {"found": True, **result})
                return

            if path == "/api/roads":
                self.send_json(200, add_network_road(data))
                return

            self.send_error(404)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})

    def log_message(self, format_string, *args):
        print("%s - %s" % (self.address_string(), format_string % args))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler)
    print("Routewise dashboard running at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Routewise dashboard.")
    finally:
        server.server_close()
