# Governing equations

## System

Undamped mass-spring oscillator: a mass $m$ attached to an ideal spring of
constant $k$, free to move along one axis with no damping or driving force.

## Equation of motion

$$m\,\ddot{x}(t) + k\,x(t) = 0$$

## Angular frequency and period

$$\omega = \sqrt{\frac{k}{m}}, \qquad T = \frac{2\pi}{\omega} = 2\pi\sqrt{\frac{m}{k}}$$

## Analytical solution

With initial displacement $x_0$ and initial velocity $v_0$:

$$x(t) = x_0 \cos(\omega t) + \frac{v_0}{\omega}\sin(\omega t)$$

Differentiating gives the velocity:

$$\dot{x}(t) = -x_0\,\omega\sin(\omega t) + v_0\cos(\omega t)$$

## Numerical solution (RK4)

The ODE is recast as a first-order system:

$$\dot{x} = v, \qquad \dot{v} = -\frac{k}{m}x$$

and integrated with the classical fourth-order Runge-Kutta scheme. The numerical
solution is used only to cross-check the analytical one; it is never labelled as
the analytical result.

## Evaluation metrics

Given reference values $x_{\text{ref}}$ and predictions $x_{\text{pred}}$ over $N$
samples:

- Root mean squared error: $\displaystyle \text{RMSE} = \sqrt{\frac{1}{N}\sum_i (x_{\text{pred},i} - x_{\text{ref},i})^2}$
- Mean absolute error: $\displaystyle \text{MAE} = \frac{1}{N}\sum_i |x_{\text{pred},i} - x_{\text{ref},i}|$

All metrics are computed from the raw float arrays returned by the model, never
from rounded chart values.