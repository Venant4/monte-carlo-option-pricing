
import numpy as np
from scipy.stats import norm


# ============================================================
# 1. INPUT VALIDATION
# ============================================================

def _validate_model(S0, r, vol, T):
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if vol <= 0:
        raise ValueError("vol must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if not np.isfinite(r):
        raise ValueError("r must be finite.")


def _validate_simulation(paths, N):
    if not isinstance(paths, (int, np.integer)) or paths <= 0:
        raise ValueError("paths must be a positive integer.")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be a positive integer.")


# ============================================================
# 2. BLACK-SCHOLES MODEL
# ============================================================

def Black_sholes(S0, K, r, vol, T):

    _validate_model(S0, r, vol, T)

    if K <= 0:
        raise ValueError("K must be positive.")

    d1 = (
        np.log(S0 / K) + (r + 0.5 * vol**2) * T
    ) / (vol * np.sqrt(T))

    d2 = d1 - vol * np.sqrt(T)

    call = (
        S0 * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    put = (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S0 * norm.cdf(-d1)
    )

    return call, put


# ============================================================
# 3. BROWNIAN MOTION
# ============================================================

def Wiener(T, paths, N, rng=None):

    if T <= 0:
        raise ValueError("T must be positive.")

    _validate_simulation(paths, N)

    if rng is None:
        rng = np.random.default_rng()

    dt = T / N

    Z = rng.standard_normal((paths, N))

    W = np.zeros((paths, N + 1))

    W[:, 1:] = np.cumsum(
        np.sqrt(dt) * Z, axis=1
    )

    return W


# ============================================================
# 4. EXACT MONTE CARLO SIMULATION
# ============================================================

def Montecarlo(S0, r, vol, T, paths, N, rng=None):

    _validate_model(S0, r, vol, T)
    _validate_simulation(paths, N)

    W = Wiener(T, paths, N, rng)

    WT = W[:, -1]

    ST = S0 * np.exp(
        (r - 0.5 * vol**2) * T + vol * WT
    )

    return ST


def pricing_Montecarlo(S0, r, vol, T, paths, N, K, rng=None):

    if K <= 0:
        raise ValueError("K must be positive.")

    ST = Montecarlo(S0, r, vol, T, paths, N, rng)

    payoff_call = np.maximum(ST - K, 0)
    payoff_put = np.maximum(K - ST, 0)

    discount = np.exp(-r * T)

    call = discount * np.mean(payoff_call)
    put = discount * np.mean(payoff_put)

    return call, put


# Compatibility with the original function name
pricing_MOntecarlo = pricing_Montecarlo


# ============================================================
# 5. EULER-MARUYAMA SCHEME
# ============================================================

def Euler(S0, r, vol, T, paths, N, rng=None):

    _validate_model(S0, r, vol, T)
    _validate_simulation(paths, N)

    if rng is None:
        rng = np.random.default_rng()

    dt = T / N

    Z = rng.standard_normal((paths, N))

    histo = np.zeros((paths, N + 1))

    histo[:, 0] = S0

    for j in range(N):

        histo[:, j + 1] = (
            histo[:, j]
            + r * dt * histo[:, j]
            + vol * histo[:, j] * np.sqrt(dt) * Z[:, j]
        )

    return histo


def Euler_ant(S0, r, vol, T, paths, N, rng=None):

    _validate_model(S0, r, vol, T)
    _validate_simulation(paths, N)

    if rng is None:
        rng = np.random.default_rng()

    dt = T / N

    Z = rng.standard_normal((paths, N))
    Z_ant = -Z

    histo = np.zeros((paths, N + 1))
    histo2 = np.zeros((paths, N + 1))

    histo[:, 0] = S0
    histo2[:, 0] = S0

    for j in range(N):

        histo[:, j + 1] = (
            histo[:, j]
            + r * dt * histo[:, j]
            + vol * histo[:, j] * np.sqrt(dt) * Z[:, j]
        )

        histo2[:, j + 1] = (
            histo2[:, j]
            + r * dt * histo2[:, j]
            + vol * histo2[:, j] * np.sqrt(dt) * Z_ant[:, j]
        )

    return histo, histo2


# ============================================================
# 6. EUROPEAN OPTIONS WITH EULER
# ============================================================

def Pricing_Euler(S0, r, vol, T, paths, N, K, rng=None):

    if K <= 0:
        raise ValueError("K must be positive.")

    histo = Euler(S0, r, vol, T, paths, N, rng)

    ST = histo[:, -1]

    payoff_call = np.maximum(ST - K, 0)
    payoff_put = np.maximum(K - ST, 0)

    discount = np.exp(-r * T)

    return (
        discount * np.mean(payoff_call),
        discount * np.mean(payoff_put)
    )


def Pricing_Euler_ant(S0, r, vol, T, paths, N, K, rng=None):

    if K <= 0:
        raise ValueError("K must be positive.")

    histo, histo2 = Euler_ant(
        S0, r, vol, T, paths, N, rng
    )

    ST1 = histo[:, -1]
    ST2 = histo2[:, -1]

    payoff = (
        np.maximum(ST1 - K, 0)
        + np.maximum(ST2 - K, 0)
    ) / 2

    return np.exp(-r * T) * np.mean(payoff)


# ============================================================
# 7. ASIAN OPTIONS
# ============================================================

def Asiatique(
    S0, r, vol, T, paths, N, K,
    rng=None, include_initial=True
):

    if K <= 0:
        raise ValueError("K must be positive.")

    histo = Euler(S0, r, vol, T, paths, N, rng)

    observations = histo if include_initial else histo[:, 1:]

    Smoy = np.mean(observations, axis=1)

    payoff_call = np.maximum(Smoy - K, 0)
    payoff_put = np.maximum(K - Smoy, 0)

    discount = np.exp(-r * T)

    return (
        discount * np.mean(payoff_call),
        discount * np.mean(payoff_put)
    )


# ============================================================
# 8. BARRIER OPTIONS
# ============================================================

def DownandOut(S0, r, vol, T, paths, N, K, H, rng=None):

    if K <= 0 or H <= 0:
        raise ValueError("K and H must be positive.")

    histo = Euler(S0, r, vol, T, paths, N, rng)

    ST = histo[:, -1]

    barrier_hit = np.any(histo[:, 1:] <= H, axis=1)

    payoff = np.where(
        barrier_hit, 0, np.maximum(ST - K, 0)
    )

    return np.exp(-r * T) * np.mean(payoff)


def UpandIn(S0, r, vol, T, paths, N, K, H, rng=None):

    if K <= 0 or H <= 0:
        raise ValueError("K and H must be positive.")

    histo = Euler(S0, r, vol, T, paths, N, rng)

    ST = histo[:, -1]

    barrier_hit = np.any(histo[:, 1:] >= H, axis=1)

    payoff = np.where(
        barrier_hit, np.maximum(ST - K, 0), 0
    )

    return np.exp(-r * T) * np.mean(payoff)


def UpandOut(S0, r, vol, T, paths, N, K, H, rng=None):

    if K <= 0 or H <= 0:
        raise ValueError("K and H must be positive.")

    histo = Euler(S0, r, vol, T, paths, N, rng)

    ST = histo[:, -1]

    barrier_hit = np.any(histo[:, 1:] >= H, axis=1)

    payoff = np.where(
        barrier_hit, 0, np.maximum(ST - K, 0)
    )

    return np.exp(-r * T) * np.mean(payoff)


# ============================================================
# 9. DIGITAL OPTION
# ============================================================

def Digitale(S0, r, vol, T, paths, N, K, Q, rng=None):

    if K <= 0:
        raise ValueError("K must be positive.")

    histo = Euler(S0, r, vol, T, paths, N, rng)

    ST = histo[:, -1]

    payoff = np.where(ST > K, Q, 0)

    return np.exp(-r * T) * np.mean(payoff)
