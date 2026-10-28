"""Plotly figure builders for the PhysAI Neural Lab dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Shared palette aligned with the dark navy + cyan accent theme.
_BG = "rgba(13, 17, 33, 0)"  # transparent so Streamlit card bg shows through
_GRID = "rgba(120, 140, 180, 0.18)"
_AXIS = "rgba(180, 200, 230, 0.9)"
_C = {
    "cyan": "#22d3ee",
    "blue": "#3b82f6",
    "amber": "#f59e0b",
    "magenta": "#e879f9",
    "green": "#34d399",
    "red": "#f87171",
}


def _base_layout(title: str, x_title: str, y_title: str) -> go.Layout:
    return go.Layout(
        title=dict(text=title, font=dict(size=15, color=_AXIS)),
        xaxis=dict(
            title=x_title,
            gridcolor=_GRID,
            zerolinecolor="rgba(120,140,180,0.35)",
            color=_AXIS,
            showline=True,
            linecolor=_GRID,
        ),
        yaxis=dict(
            title=y_title,
            gridcolor=_GRID,
            zerolinecolor="rgba(120,140,180,0.35)",
            color=_AXIS,
            showline=True,
            linecolor=_GRID,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=_AXIS),
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        font=dict(color=_AXIS),
        margin=dict(l=8, r=8, t=42, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(10,15,30,0.95)", font=dict(color="#e8eef9")),
    )


def timeseries_plot(
    t: np.ndarray,
    x: np.ndarray,
    title: str,
    y_title: str,
    color: str = _C["cyan"],
    name: str = "displacement",
) -> go.Figure:
    """Build a single time-series figure."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=x, mode="lines", name=name,
        line=dict(color=color, width=2.5),
        hovertemplate="t = %{x:.3f} s<br>%{customdata} = %{y:.5f}<extra></extra>",
        customdata=[y_title],
    ))
    fig.update_layout(_base_layout(title, "time t (s)", y_title))
    return fig


def velocity_plot(t: np.ndarray, v: np.ndarray) -> go.Figure:
    """Velocity-versus-time figure."""
    return timeseries_plot(t, v, "Velocity vs time", "velocity v (m/s)", _C["blue"], "velocity")


def phase_space_plot(x: np.ndarray, v: np.ndarray, x2=None, v2=None) -> go.Figure:
    """Phase-space (velocity vs displacement) figure, optionally overlaying a second trajectory."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=v, mode="lines", name="analytical",
        line=dict(color=_C["cyan"], width=2.5),
        hovertemplate="x = %{x:.4f} m<br>v = %{y:.4f} m/s<extra></extra>",
    ))
    if x2 is not None and v2 is not None:
        fig.add_trace(go.Scatter(
            x=x2, y=v2, mode="lines", name="numerical (RK4)",
            line=dict(color=_C["amber"], width=2.5, dash="dash"),
            hovertemplate="x = %{x:.4f} m<br>v = %{y:.4f} m/s<extra></extra>",
        ))
    fig.update_layout(_base_layout("Phase space", "displacement x (m)", "velocity v (m/s)"))
    return fig


def displacement_comparison_plot(
    t: np.ndarray,
    x_reference: np.ndarray,
    x_pred: np.ndarray,
) -> go.Figure:
    """Overlay reference and neural prediction."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=x_reference, mode="lines", name="reference",
        line=dict(color=_C["cyan"], width=2.5),
        hovertemplate="t=%{x:.3f}<br>ref x=%{y:.5f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=x_pred, mode="lines", name="neural prediction",
        line=dict(color=_C["magenta"], width=2.5, dash="dash"),
        hovertemplate="t=%{x:.3f}<br>pred x=%{y:.5f}<extra></extra>",
    ))
    fig.update_layout(_base_layout("Reference vs neural prediction", "time t (s)", "displacement x (m)"))
    return fig


def error_plot(t: np.ndarray, error: np.ndarray) -> go.Figure:
    """Prediction error versus time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=error, mode="lines", name="error",
        line=dict(color=_C["amber"], width=2),
        hovertemplate="t=%{x:.3f}<br>error=%{y:.6f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=_AXIS, width=1, dash="dot"))
    fig.update_layout(_base_layout("Prediction error vs time", "time t (s)", "error x_pred - x_ref (m)"))
    return fig


def loss_plot(loss_history: list[float]) -> go.Figure:
    """Training-loss curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(loss_history) + 1)),
        y=loss_history,
        mode="lines",
        name="loss",
        line=dict(color=_C["green"], width=2.5),
        hovertemplate="epoch %{x}<br>loss=%{y:.6f}<extra></extra>",
    ))
    fig.update_layout(_base_layout("Training loss", "epoch", "MSE loss"))
    return fig


def csv_download(df: pd.DataFrame) -> str:
    """Return CSV text for ``st.download_button``."""
    return df.to_csv(index=False)