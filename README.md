# monte-carlo-option-pricing

This project focuses on the implementation and study of numerical methods for pricing financial derivatives under the Black-Scholes framework.

It explores Monte Carlo simulation, stochastic differential equation discretization, and variance reduction techniques, with applications to European and exotic options.

#Mathematical Framework

The project is based on the Geometric Brownian Motion model:

[
dS_t = rS_t,dt + \sigma S_t,dW_t
]

where:

(S_t): underlying asset price

(r): risk-free interest rate

(\sigma): volatility

(W_t): standard Brownian motion

#Implemented Methods

#1. Black-Scholes Model

Analytical pricing of European call and put options.

Computation of option prices under the Black-Scholes framework.

#2. Monte Carlo Simulation
   Simulation of Brownian motion.

Simulation of the underlying asset price.

Numerical pricing of European options.

#3. Euler-Maruyama Scheme

Numerical discretization of stochastic differential equations.

Simulation of asset price trajectories.

#4. Variance Reduction
   Implementation of antithetic variates.

Comparison with standard Monte Carlo simulation.

5. Exotic Options Pricing

Asian options.

Digital options.

Barrier options: Down-and-Out, Up-and-In and Up-and-Out.

#Objectives

Understand and implement Monte Carlo methods for option pricing.

Study the numerical approximation of stochastic processes.

Investigate variance reduction techniques.

Explore the pricing of exotic financial derivatives.

Compare numerical methods with analytical pricing benchmarks.

#Future Improvements

Refactoring and optimization of the Python implementation.

Numerical validation against Black-Scholes analytical prices.

Convergence analysis with respect to the number of simulations.

Performance comparison between standard Monte Carlo and antithetic variates.

Visualization of simulated trajectories and numerical results.
