# Experiment results

This page documents the status of experiments in **PhysAI Neural Lab**.

## Status

No pre-computed or stored experiment results are shipped with this application.
Every metric displayed in the dashboard is produced at runtime from a real
training run executed in the browser session.

## What is real

- The physics simulation (analytical and RK4 numerical) is computed on demand
  from the closed-form solution and the Runge-Kutta integrator.
- The neural network is a genuine scikit-learn `MLPRegressor` trained live when
  the user presses **Train model**. Its loss curve and predictions are the real
  output of that training run.
- RMSE and MAE on the Model Comparison page are computed from the actual model
  predictions against the actual reference values for the current session.

## What is not yet available

- **Held-out generalisation testing across different physical parameters** is
  not implemented. The model is trained and evaluated within a single parameter
  set; it is not yet tested on unseen $(m, k, x_0, v_0)$ configurations. This is
  labelled as "not yet available" in the UI.
- **Computer-vision motion estimation** (Experimental Analysis page) is planned
  and not implemented. No tracking results are shown.

## Reproducibility note

Training uses a fixed random seed (42) inside scikit-learn, so runs with
identical inputs and parameters are reproducible. Results still depend on the
CPU/interpreter version of the environment.