# Optimizers

An optimizer is a routine designed to update some parameters `W` using a
gradient `∇` according to some formula. An optimizer is always needed when solving for the steady state of a Density Matrix.

The most basic type of Optimizer is the Steepest-Gradient-Descent ( also known as Stochastic Gradient Descent - SGD ), which updates the weights ``W_i`` following the gradient ``\nabla = \nabla_W E``, computed as the gradient of the objective function ``E`` against the parameters ``W``. The update formula is:

```math
W_{i+1} = W_i - \epsilon \nabla
```
where the only parameter of the optimizer (hyperparameter, in Machine-Learning jargon) is the step size ``\epsilon``.

There exist some way to improve upon this formula, for example by including momentum and friction, obtaining the Momentum Gradient Descent.

## Types of Optimizers

### Gradient Descent (GD)

```@docs
GD
```

### Gradient Descent with Corrected Momentum (NesterovGD)
```@docs
NesterovGD
```
