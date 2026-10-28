"""The visible app routes must remain explicit and unique."""

import app


def test_visible_robotics_pages_have_unique_routes():
    assert set(app.PAGE_ROUTES) == set(app.PAGES)
    assert len(set(app.PAGE_ROUTES.values())) == len(app.PAGE_ROUTES)
    assert set(app.PAGE_ROUTES.values()) == {
        "home", "robot-workspace", "model-training", "cart-pole-benchmark"
    }
