"""Animated Plotly representations for simulated physical systems."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _frame_indices(length: int, maximum: int = 140) -> np.ndarray:
    """Keep browser animations responsive for large simulations."""
    return np.unique(np.linspace(0, length - 1, min(length, maximum), dtype=int))


def _spring_trace(anchor: float, mass_position: float) -> go.Scatter:
    end = mass_position - 0.18
    points = 28
    x = np.linspace(anchor, end, points)
    y = np.zeros(points)
    if end > anchor:
        y[1:-1] = np.tile([0.16, -0.16], (points - 2 + 1) // 2)[: points - 2]
    return go.Scatter(x=x, y=y, mode="lines", name="spring", line=dict(color="#22d3ee", width=3), hoverinfo="skip")


def spring_mass_animation(time: np.ndarray, displacement: np.ndarray, title: str = "Virtual mass–spring experiment") -> go.Figure:
    """Return a playable animation of horizontal mass–spring motion."""
    indices = _frame_indices(len(time))
    span = max(float(np.ptp(displacement)), 0.5)
    anchor = float(np.min(displacement) - span * 0.75 - 0.5)
    x_min, x_max = anchor - 0.2, float(np.max(displacement) + span * 0.45 + 0.7)

    def mass_trace(index: int) -> go.Scatter:
        return go.Scatter(
            x=[displacement[index]], y=[0], mode="markers+text", name="mass",
            marker=dict(symbol="square", size=42, color="#e879f9", line=dict(color="#f8fafc", width=1)),
            text=[f"m<br>t = {time[index]:.2f} s"], textposition="top center",
            hovertemplate="displacement = %{x:.5f}<extra>mass</extra>",
        )

    initial = int(indices[0])
    frames = [go.Frame(data=[_spring_trace(anchor, float(displacement[i])), mass_trace(int(i))], name=str(int(i))) for i in indices]
    fig = go.Figure(data=[_spring_trace(anchor, float(displacement[initial])), mass_trace(initial)], frames=frames)
    fig.update_layout(
        title=title,
        height=370,
        template="plotly_dark",
        xaxis=dict(title="displacement axis", range=[x_min, x_max], showgrid=False, zeroline=True),
        yaxis=dict(range=[-0.65, 0.85], visible=False),
        showlegend=False,
        margin=dict(l=12, r=12, t=46, b=24),
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.02, "y": -0.16,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 45, "redraw": True}, "fromcurrent": True}]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
    )
    return fig


def phase_space_animation(time: np.ndarray, x: np.ndarray, v: np.ndarray) -> go.Figure:
    """Return phase space with a playable marker that follows the trajectory."""
    indices = _frame_indices(len(time))
    trajectory = go.Scatter(x=x, y=v, mode="lines", name="trajectory", line=dict(color="rgba(34,211,238,0.45)", width=2))

    def marker(index: int) -> go.Scatter:
        return go.Scatter(
            x=[x[index]], y=[v[index]], mode="markers", name="current state",
            marker=dict(size=12, color="#e879f9"),
            hovertemplate=f"t = {time[index]:.3f} s<br>x = %{{x:.5f}}<br>v = %{{y:.5f}}<extra></extra>",
        )

    first = int(indices[0])
    frames = [go.Frame(data=[marker(int(i))], traces=[1], name=str(int(i))) for i in indices]
    fig = go.Figure(data=[trajectory, marker(first)], frames=frames)
    fig.update_layout(
        title="Animated phase-space trajectory", template="plotly_dark", height=440,
        xaxis_title="x", yaxis_title="v", hovermode="closest",
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.02, "y": -0.16,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 45, "redraw": False}, "fromcurrent": True}]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
    )
    return fig
