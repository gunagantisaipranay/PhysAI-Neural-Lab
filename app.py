"""PhysAI Robotics Lab — entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import pages
from ui_theme import PALETTE, inject_theme

PAGES = {
    "Home": pages.overview,
    "Robot Workspace": pages.robot_studio,
    "Model Training": pages.training,
    "Cart-Pole Benchmark": pages.robotics,
}
PAGE_ROUTES = {
    "Home": "home",
    "Robot Workspace": "robot-workspace",
    "Model Training": "model-training",
    "Cart-Pole Benchmark": "cart-pole-benchmark",
}


def main() -> None:
    st.set_page_config(
        page_title="PhysAI Robotics Lab",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    selected_page = st.navigation(
        [
            st.Page(PAGES["Home"].render, title="Home", icon="🏠", url_path=PAGE_ROUTES["Home"], default=True),
            st.Page(PAGES["Robot Workspace"].render, title="Robot Workspace", icon="🦾", url_path=PAGE_ROUTES["Robot Workspace"]),
            st.Page(PAGES["Model Training"].render, title="Model Training", icon="🧠", url_path=PAGE_ROUTES["Model Training"]),
            st.Page(PAGES["Cart-Pole Benchmark"].render, title="Cart-Pole Benchmark", icon="📈", url_path=PAGE_ROUTES["Cart-Pole Benchmark"]),
        ],
        position="sidebar",
    )
    inject_theme()

    with st.sidebar:
        st.markdown(
            f"<div style='padding:6px 4px 18px 4px'>"
            f"<div style='font-size:18px;font-weight:700;color:{PALETTE['primary']};"
            f"letter-spacing:1px'>PHYS<span style='color:{PALETTE['secondary']}'>AI</span></div>"
            f"<div style='font-size:11px;color:{PALETTE['muted']};"
            f"letter-spacing:2px'>ROBOTICS LAB</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin-top:18px;font-size:11px;color:{PALETTE['muted']};"
            f"border-top:1px solid {PALETTE['border']};padding-top:10px'>"
            f"Laptop robotics simulation<br>MuJoCo · model training · evaluation"
            f"</div>",
            unsafe_allow_html=True,
        )

    selected_page.run()


if __name__ == "__main__":
    main()
