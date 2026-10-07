"""
Geometric Brownian Motion simulation and Monte Carlo option pricing.

Model (risk-neutral):  dS = r S dt + sigma S dW

The script:
  1. Simulates stock price paths with the exact GBM solution
     S_{t+dt} = S_t * exp((r - sigma^2/2) dt + sigma sqrt(dt) Z)
     and, for comparison, the Euler-Maruyama discretization.
  2. Prices a European call and put by Monte Carlo and compares them
     with the closed-form Black-Scholes prices.
  3. Prices an arithmetic-average Asian call, a path-dependent option
     with no simple closed form -- the case where Monte Carlo earns its keep.
  4. Saves a figure with sample paths, the terminal price distribution,
     and Monte Carlo convergence.

Usage:
    python gbm_option_pricing.py
    python gbm_option_pricing.py --S0 150 --K 160 --sigma 0.3 --T 0.5 --paths 200000
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------- simulation

def simulate_gbm_exact(S0, r, sigma, T, n_steps, n_paths, rng):
    """Simulate GBM paths using the exact log-normal solution."""
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


def simulate_gbm_euler(S0, r, sigma, T, n_steps, n_paths, rng):
    """Simulate GBM paths using the Euler-Maruyama scheme."""
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        S = paths[:, t]
        paths[:, t + 1] = S + r * S * dt + sigma * S * np.sqrt(dt) * Z
    return paths


# ------------------------------------------------------------------- pricing

def black_scholes(S0, K, r, sigma, T, kind="call"):
    """Closed-form Black-Scholes price for a European option."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def mc_price(payoffs, r, T):
    """Discounted Monte Carlo price and its standard error."""
    disc = np.exp(-r * T) * payoffs
    return disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--S0", type=float, default=100.0, help="initial stock price")
    p.add_argument("--K", type=float, default=105.0, help="strike price")
    p.add_argument("--r", type=float, default=0.04, help="risk-free rate (annual)")
    p.add_argument("--sigma", type=float, default=0.25, help="volatility (annual)")
    p.add_argument("--T", type=float, default=1.0, help="time to maturity (years)")
    p.add_argument("--steps", type=int, default=252, help="time steps per path")
    p.add_argument("--paths", type=int, default=100_000, help="number of paths")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--out", default="gbm_simulation.png", help="output figure")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)

    paths = simulate_gbm_exact(a.S0, a.r, a.sigma, a.T, a.steps, a.paths, rng)
    ST = paths[:, -1]

    # European options
    call_mc, call_se = mc_price(np.maximum(ST - a.K, 0), a.r, a.T)
    put_mc, put_se = mc_price(np.maximum(a.K - ST, 0), a.r, a.T)
    call_bs = black_scholes(a.S0, a.K, a.r, a.sigma, a.T, "call")
    put_bs = black_scholes(a.S0, a.K, a.r, a.sigma, a.T, "put")

    # Asian (arithmetic average) call -- path dependent
    avg_price = paths[:, 1:].mean(axis=1)
    asian_mc, asian_se = mc_price(np.maximum(avg_price - a.K, 0), a.r, a.T)

    # Euler-Maruyama comparison (fewer paths; it's a Python loop)
    euler = simulate_gbm_euler(a.S0, a.r, a.sigma, a.T, a.steps,
                               min(a.paths, 20_000), rng)
    euler_call, euler_se = mc_price(np.maximum(euler[:, -1] - a.K, 0), a.r, a.T)

    print(f"Parameters: S0={a.S0}  K={a.K}  r={a.r}  sigma={a.sigma}  "
          f"T={a.T}  steps={a.steps}  paths={a.paths:,}\n")
    print(f"{'Option':<28}{'Monte Carlo':>14}{'± 1 s.e.':>10}{'Black-Scholes':>16}")
    print("-" * 68)
    print(f"{'European call (exact GBM)':<28}{call_mc:>14.4f}{call_se:>10.4f}{call_bs:>16.4f}")
    print(f"{'European call (Euler)':<28}{euler_call:>14.4f}{euler_se:>10.4f}{call_bs:>16.4f}")
    print(f"{'European put  (exact GBM)':<28}{put_mc:>14.4f}{put_se:>10.4f}{put_bs:>16.4f}")
    print(f"{'Asian call (arith. avg)':<28}{asian_mc:>14.4f}{asian_se:>10.4f}{'n/a':>16}")
    parity = call_mc - put_mc - (a.S0 - a.K * np.exp(-a.r * a.T))
    print(f"\nPut-call parity residual (MC): {parity:+.4f}  (should be ~0)")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    t = np.linspace(0, a.T, a.steps + 1)

    ax = axes[0]
    ax.plot(t, paths[:50].T, lw=0.7, alpha=0.6)
    ax.plot(t, a.S0 * np.exp(a.r * t), "k--", lw=1.5, label="expected price")
    ax.axhline(a.K, color="crimson", lw=1, ls=":", label=f"strike K={a.K:g}")
    ax.set(title="50 simulated GBM paths", xlabel="time (years)", ylabel="price")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.hist(ST, bins=120, density=True, alpha=0.6, color="steelblue",
            label="simulated $S_T$")
    x = np.linspace(ST.min(), ST.max(), 400)
    mu_ln = np.log(a.S0) + (a.r - 0.5 * a.sigma**2) * a.T
    sd_ln = a.sigma * np.sqrt(a.T)
    pdf = norm.pdf(np.log(x), mu_ln, sd_ln) / x
    ax.plot(x, pdf, "k", lw=1.5, label="log-normal theory")
    ax.axvline(a.K, color="crimson", lw=1, ls=":")
    ax.set(title="Terminal price distribution", xlabel="$S_T$", ylabel="density")
    ax.legend()

    ax = axes[2]
    disc_payoffs = np.exp(-a.r * a.T) * np.maximum(ST - a.K, 0)
    n = np.arange(1, len(disc_payoffs) + 1)
    running = np.cumsum(disc_payoffs) / n
    ax.semilogx(n, running, lw=1, label="MC estimate")
    ax.axhline(call_bs, color="k", ls="--", lw=1.2, label="Black-Scholes")
    ax.set(title="Monte Carlo convergence (European call)",
           xlabel="number of paths", ylabel="price")
    ax.legend()

    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"\nFigure saved to {a.out}")


if __name__ == "__main__":
    main()
