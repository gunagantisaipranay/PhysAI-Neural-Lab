"""Obstacle clearance and route-shape tests for tool-point A* planning."""

import numpy as np
import pytest

from physics.robot_path_planning import _segment_clear, plan_tool_path
from physics.robot_scene import Obstacle


def test_clear_route_is_returned_as_direct_start_and_goal():
    start = (0.0, 0.0, 0.5)
    goal = (0.5, 0.0, 0.5)
    route = plan_tool_path(start, goal)
    assert route.shape == (2, 3)
    np.testing.assert_allclose(route[0], start)
    np.testing.assert_allclose(route[-1], goal)


def test_route_goes_around_box_and_respects_clearance():
    obstacles = (Obstacle("box", (0.5, 0.0, 0.5), (0.12, 0.3, 0.2)),)
    start, goal = np.asarray([0.0, 0.0, 0.5]), np.asarray([1.0, 0.0, 0.5])
    route = plan_tool_path(tuple(start), tuple(goal), obstacles, resolution_m=0.08, clearance_m=0.08)
    assert len(route) > 2
    assert all(_segment_clear(a, b, obstacles, 0.08, 0.02) for a, b in zip(route[:-1], route[1:]))
    assert np.linalg.norm(route[0] - start) == 0
    assert np.linalg.norm(route[-1] - goal) == 0


def test_route_rejects_start_or_goal_inside_an_obstacle_margin():
    obstacle = Obstacle("sphere", (0.0, 0.0, 0.5), (0.15, 0.15, 0.15))
    with pytest.raises(ValueError, match="current tool point"):
        plan_tool_path((0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (obstacle,))


@pytest.mark.parametrize("resolution", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_resolution_is_rejected(resolution):
    with pytest.raises(ValueError, match="resolution"):
        plan_tool_path(
            (0.0, 0.0, 0.5),
            (0.5, 0.0, 0.5),
            resolution_m=resolution,
        )


@pytest.mark.parametrize("clearance", [-0.1, float("nan"), float("inf")])
def test_invalid_clearance_is_rejected(clearance):
    with pytest.raises(ValueError, match="clearance"):
        plan_tool_path(
            (0.0, 0.0, 0.5),
            (0.5, 0.0, 0.5),
            clearance_m=clearance,
        )


@pytest.mark.parametrize("max_grid_points", [0, -1])
def test_invalid_grid_limit_is_rejected(max_grid_points):
    with pytest.raises(ValueError, match="max_grid_points"):
        plan_tool_path(
            (0.0, 0.0, 0.5),
            (0.5, 0.0, 0.5),
            max_grid_points=max_grid_points,
        )


@pytest.mark.parametrize(
    "obstacle",
    [
        Obstacle("sphere", (0.5, 0.0, 0.5), (0.15, 0.15, 0.15)),
        Obstacle("cylinder", (0.5, 0.0, 0.5), (0.15, 0.15, 0.2)),
        Obstacle("capsule", (0.5, 0.0, 0.5), (0.1, 0.1, 0.2)),
    ],
)
def test_route_avoids_non_box_obstacles(obstacle):
    start = (0.0, 0.0, 0.5)
    goal = (1.0, 0.0, 0.5)
    clearance = 0.05

    route = plan_tool_path(
        start,
        goal,
        (obstacle,),
        resolution_m=0.08,
        clearance_m=clearance,
    )

    assert len(route) > 2
    assert np.all(np.isfinite(route))
    np.testing.assert_allclose(route[0], start)
    np.testing.assert_allclose(route[-1], goal)
    assert all(
        _segment_clear(a, b, (obstacle,), clearance, 0.01)
        for a, b in zip(route[:-1], route[1:])
    )



def test_planner_enforces_grid_point_limit():
    wall = Obstacle("box", (0.5, 0.0, 0.5), (0.2, 2.0, 2.0))

    with pytest.raises(ValueError, match="grid points"):
        plan_tool_path(
            (0.0, 0.0, 0.5),
            (1.0, 0.0, 0.5),
            (wall,),
            resolution_m=0.01,
            clearance_m=0.05,
            max_grid_points=100,
        )


@pytest.mark.parametrize(
    "obstacle",
    [
        Obstacle("box", (float("nan"), 0.0, 0.5), (0.1, 0.1, 0.1)),
        Obstacle("box", (0.5, 0.0, 0.5), (float("inf"), 0.1, 0.1)),
        Obstacle("unknown", (0.5, 0.0, 0.5), (0.1, 0.1, 0.1)),
        Obstacle("box", (0.5, 0.0), (0.1, 0.1, 0.1)),
        Obstacle("box", (0.5, 0.0, 0.5), (0.1, 0.1)),
        Obstacle("box", (0.5, 0.0, 0.5), (0.0, 0.1, 0.1)),
        Obstacle("box", (0.5, 0.0, 0.5), (-0.1, 0.1, 0.1)),
    ],
)
def test_invalid_obstacles_are_rejected(obstacle):
    with pytest.raises(ValueError):
        plan_tool_path(
            (0.0, 0.0, 0.5),
            (1.0, 0.0, 0.5),
            (obstacle,),
        )
