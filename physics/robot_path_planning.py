"""Grid-based tool-point path planning around static scene obstacles."""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import product

import numpy as np

from physics.robot_scene import Obstacle


def _blocked(point: np.ndarray, obstacles: tuple[Obstacle, ...], clearance: float) -> bool:
    for obstacle in obstacles:
        delta = point - np.asarray(obstacle.position, dtype=float)
        size = np.asarray(obstacle.size, dtype=float)
        if obstacle.shape == "box":
            if np.all(np.abs(delta) <= size + clearance):
                return True
        elif obstacle.shape == "sphere":
            if np.linalg.norm(delta) <= size[0] + clearance:
                return True
        elif obstacle.shape == "cylinder":
            if np.linalg.norm(delta[:2]) <= size[0] + clearance and abs(delta[2]) <= size[1] + clearance:
                return True
        elif obstacle.shape == "capsule":
            axial = max(abs(delta[2]) - size[1], 0.0)
            if np.hypot(np.linalg.norm(delta[:2]), axial) <= size[0] + clearance:
                return True
        else:
            raise ValueError(f"Unsupported obstacle type: {obstacle.shape}")
    return False


def _segment_clear(start: np.ndarray, end: np.ndarray, obstacles: tuple[Obstacle, ...], clearance: float, step: float) -> bool:
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(distance / max(step, 1e-3))))
    return all(
        not _blocked(start + (end - start) * fraction, obstacles, clearance)
        for fraction in np.linspace(0.0, 1.0, count + 1)
    )


def plan_tool_path(
    start_xyz: tuple[float, float, float],
    goal_xyz: tuple[float, float, float],
    obstacles: tuple[Obstacle, ...] = (),
    *,
    resolution_m: float = 0.07,
    clearance_m: float = 0.07,
    max_grid_points: int = 180_000,
) -> np.ndarray:
    """Plan a 3D point path with A* and line-of-sight waypoint smoothing."""
    start = np.asarray(start_xyz, dtype=float)
    goal = np.asarray(goal_xyz, dtype=float)
    if start.shape != (3,) or goal.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError("Path endpoints must each contain three finite coordinates.")
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("Grid resolution must be positive and finite.")
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("clearance must be non-negative and finite.")

    if (
        isinstance(max_grid_points, (bool, np.bool_))
        or not isinstance(max_grid_points, (int, np.integer))
        or max_grid_points <= 0
    ):
        raise ValueError("max_grid_points must be a positive integer.")

    valid_shapes = {"box", "sphere", "cylinder", "capsule"}

    for obstacle in obstacles:
        if obstacle.shape not in valid_shapes:
            raise ValueError(f"Unsupported obstacle type: {obstacle.shape}")

        position = np.asarray(obstacle.position, dtype=float)
        size = np.asarray(obstacle.size, dtype=float)

        if (
            position.shape != (3,)
            or size.shape != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite(size).all()
        ):
            raise ValueError(
                "Obstacle position and size must contain three finite values each."
            )

        if np.any(size <= 0):
            raise ValueError("Obstacle dimensions must be positive.")
    if _blocked(start, obstacles, clearance_m):
        raise ValueError("The current tool point overlaps an obstacle or its clearance margin.")
    if _blocked(goal, obstacles, clearance_m):
        raise ValueError("The target overlaps an obstacle or its clearance margin.")
    safety_step = min(resolution_m / 4, 0.01)
    if _segment_clear(start, goal, obstacles, clearance_m, safety_step):
        return np.vstack([start, goal])

    bounds_min = np.minimum(start, goal) - 0.25
    bounds_max = np.maximum(start, goal) + 0.25
    for obstacle in obstacles:
        center = np.asarray(obstacle.position, dtype=float)
        size = np.asarray(obstacle.size, dtype=float)
        if obstacle.shape == "box":
            extent = size
        elif obstacle.shape == "sphere":
            extent = np.repeat(size[0], 3)
        elif obstacle.shape in {"cylinder", "capsule"}:
            extent = np.array([size[0], size[0], size[1] + size[0]])
        else:
            raise ValueError(f"Unsupported obstacle type: {obstacle.shape}")
        bounds_min = np.minimum(bounds_min, center - extent - clearance_m - resolution_m * 2)
        bounds_max = np.maximum(bounds_max, center + extent + clearance_m + resolution_m * 2)
    bounds_min[2] = max(0.0, bounds_min[2])
    shape = np.ceil((bounds_max - bounds_min) / resolution_m).astype(int) + 1
    point_count = int(np.prod(shape))
    if point_count > max_grid_points:
        raise ValueError(
            f"Path search would require {point_count:,} grid points. Increase the grid spacing or reduce the workspace."
        )

    def world(index: tuple[int, int, int]) -> np.ndarray:
        return bounds_min + np.asarray(index, dtype=float) * resolution_m

    def grid(point: np.ndarray) -> tuple[int, int, int]:
        raw = np.rint((point - bounds_min) / resolution_m).astype(int)
        return tuple(np.clip(raw, 0, shape - 1).tolist())

    start_index, goal_index = grid(start), grid(goal)
    blocked_indices = {
        (x, y, z)
        for x, y, z in product(range(shape[0]), range(shape[1]), range(shape[2]))
        if _blocked(world((x, y, z)), obstacles, clearance_m)
    }
    blocked_indices.discard(start_index)
    blocked_indices.discard(goal_index)
    neighbors = [
        (dx, dy, dz, float(np.linalg.norm((dx, dy, dz))))
        for dx, dy, dz in product((-1, 0, 1), repeat=3)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    frontier = [(float(np.linalg.norm(np.asarray(start_index) - goal_index)), 0.0, start_index)]
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_index: None}
    cost = {start_index: 0.0}
    while frontier:
        _, current_cost, current = heappop(frontier)
        if current == goal_index:
            break
        if current_cost > cost[current] + 1e-9:
            continue
        for dx, dy, dz, edge in neighbors:
            candidate = (current[0] + dx, current[1] + dy, current[2] + dz)
            if any(candidate[axis] < 0 or candidate[axis] >= shape[axis] for axis in range(3)):
                continue
            if candidate in blocked_indices:
                continue
            if not _segment_clear(world(current), world(candidate), obstacles, clearance_m, safety_step):
                continue
            new_cost = current_cost + edge
            if new_cost >= cost.get(candidate, float("inf")):
                continue
            cost[candidate] = new_cost
            came_from[candidate] = current
            heuristic = float(np.linalg.norm(np.asarray(candidate) - goal_index))
            heappush(frontier, (new_cost + heuristic, new_cost, candidate))
    else:
        raise ValueError("No collision-free tool path was found in the current workspace.")

    indices = []
    cursor = goal_index
    while cursor is not None:
        indices.append(cursor)
        cursor = came_from[cursor]
    points = [world(index) for index in reversed(indices)]
    points[0], points[-1] = start, goal
    # Greedily remove grid turns while maintaining the obstacle-clearance buffer.
    smoothed = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        candidate = len(points) - 1
        while candidate > anchor + 1 and not _segment_clear(
            points[anchor], points[candidate], obstacles, clearance_m, safety_step
        ):
            candidate -= 1
        smoothed.append(points[candidate])
        anchor = candidate
    result = np.asarray(smoothed, dtype=float)
    if not all(
        _segment_clear(first, second, obstacles, clearance_m, safety_step)
        for first, second in zip(result[:-1], result[1:])
    ):
        raise ValueError("A collision-free route could not be smoothed safely. Reduce the clearance or grid spacing.")
    return result
