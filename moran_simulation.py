"""
Moran Process Simulation
=========================
Companion code for Chapter 6 of Nowak's "Evolutionary Dynamics".

Simulates a finite population of N individuals where:
  - Type A (mutant) has relative fitness r
  - Type B (resident) has fitness 1
  - At each step: one individual reproduces (fitness-weighted),
    one dies (uniform), offspring replaces dead one.

We verify the fixation probability formula:
    rho = (1 - 1/r) / (1 - 1/r^N)        for r != 1
    rho = 1/N                              for r == 1  (neutral)

Run with:  python moran_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1. SINGLE TRAJECTORY SIMULATION
# ---------------------------------------------------------------

def moran_step(i, N, r):
    """One step of the Moran process. Returns new count of A individuals."""
    # Probability an A is chosen to reproduce (fitness-weighted)
    p_repro_A = (r * i) / (r * i + (N - i))
    # Probability a B is chosen to die (uniform)
    p_die_B = (N - i) / N
    # Probability a B is chosen to reproduce
    p_repro_B = (N - i) / (r * i + (N - i))
    # Probability an A is chosen to die
    p_die_A = i / N

    # Three outcomes
    p_up = p_repro_A * p_die_B      # i -> i+1
    p_down = p_repro_B * p_die_A    # i -> i-1
    # else: stay at i

    u = np.random.random()
    if u < p_up:
        return i + 1
    elif u < p_up + p_down:
        return i - 1
    else:
        return i


def run_trajectory(N, r, i0=1, max_steps=10**7):
    """Run a single Moran trajectory until absorption.
    Returns the full history of counts."""
    history = [i0]
    i = i0
    for _ in range(max_steps):
        if i == 0 or i == N:
            break
        i = moran_step(i, N, r)
        history.append(i)
    return np.array(history)


# ---------------------------------------------------------------
# 2. FIXATION PROBABILITY: ANALYTICAL VS EMPIRICAL
# ---------------------------------------------------------------

def fixation_probability_analytical(N, r):
    """Closed-form fixation probability for one mutant."""
    if abs(r - 1.0) < 1e-9:
        return 1.0 / N
    with np.errstate(over='ignore', divide='ignore'):
        return (1 - 1/r) / (1 - 1/r**N)


def empirical_fixation_probability(N, r, trials=5000, i0=1):
    """Run many trajectories, count fraction that fix (reach N)."""
    fixations = 0
    for _ in range(trials):
        history = run_trajectory(N, r, i0=i0)
        if history[-1] == N:
            fixations += 1
    return fixations / trials


# ---------------------------------------------------------------
# 3. VISUALIZATIONS
# ---------------------------------------------------------------

def plot_trajectories(N=100, r=1.1, n_trajectories=20):
    """Plot multiple sample trajectories to show stochastic variability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fixed = 0
    for _ in range(n_trajectories):
        h = run_trajectory(N, r, i0=1)
        color = 'tab:green' if h[-1] == N else 'tab:red'
        alpha = 0.7 if h[-1] == N else 0.3
        ax.plot(h, color=color, alpha=alpha, linewidth=1)
        if h[-1] == N:
            fixed += 1

    ax.axhline(N, color='black', linestyle='--', alpha=0.5, label=f'Fixation (i=N={N})')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, label='Extinction (i=0)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Number of A individuals')
    ax.set_title(f'Moran trajectories: N={N}, r={r}\n'
                 f'{fixed}/{n_trajectories} fixed  '
                 f'(theoretical fixation prob = {fixation_probability_analytical(N, r):.3f})')
    ax.legend(loc='center right')
    plt.tight_layout()
    plt.savefig('moran_trajectories.png', dpi=120)
    plt.close()
    print("Saved: moran_trajectories.png")


def plot_fixation_vs_r(N=50, r_values=None, trials=2000):
    """Verify the analytical fixation formula across a range of r."""
    if r_values is None:
        r_values = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0])

    analytical = [fixation_probability_analytical(N, r) for r in r_values]
    empirical = []
    for r in r_values:
        emp = empirical_fixation_probability(N, r, trials=trials)
        empirical.append(emp)
        print(f"  r={r:.2f}  analytical={fixation_probability_analytical(N, r):.4f}  "
              f"empirical={emp:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_values, analytical, 'k-', linewidth=2, label='Analytical formula')
    ax.scatter(r_values, empirical, color='tab:red', s=60, zorder=3,
               label=f'Empirical ({trials} trials each)')
    ax.axhline(1/N, color='gray', linestyle=':', label=f'Neutral baseline 1/N = {1/N:.3f}')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Relative fitness r')
    ax.set_ylabel('Fixation probability')
    ax.set_title(f'Fixation probability vs fitness  (N={N})')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('moran_fixation_vs_r.png', dpi=120)
    plt.close()
    print("Saved: moran_fixation_vs_r.png")


def plot_substitution_rate():
    """Demonstrate R = N * mu * rho across regimes.
    Shows the neutral theory result: R = mu (independent of N) when r=1,
    versus adaptive evolution where R grows linearly with N."""
    Ns = np.logspace(2, 5, 20).astype(int)  # 100 to 100,000
    mu = 1e-6  # mutation rate per individual per generation

    R_neutral = np.array([N * mu * fixation_probability_analytical(N, 1.0) for N in Ns])
    R_beneficial_small = np.array([N * mu * fixation_probability_analytical(N, 1.001) for N in Ns])
    R_beneficial_med = np.array([N * mu * fixation_probability_analytical(N, 1.01) for N in Ns])
    R_beneficial_large = np.array([N * mu * fixation_probability_analytical(N, 1.05) for N in Ns])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(Ns, R_neutral, 'o-', label='Neutral (r = 1.000) — Kimura', linewidth=2.5, color='tab:blue')
    ax.loglog(Ns, R_beneficial_small, 's-', label='Beneficial (r = 1.001)', linewidth=2, color='#ffa500')
    ax.loglog(Ns, R_beneficial_med, '^-', label='Beneficial (r = 1.01)', linewidth=2, color='tab:red')
    ax.loglog(Ns, R_beneficial_large, 'D-', label='Beneficial (r = 1.05)', linewidth=2, color='tab:purple')
    ax.axhline(mu, color='gray', linestyle='--', alpha=0.7, label=f'μ = {mu} (neutral asymptote)')
    ax.set_xlabel('Population size N')
    ax.set_ylabel('Substitution rate R = N · μ · ρ')
    ax.set_title('Rate of evolution: neutral vs adaptive\n'
                 'Neutral rate is flat (= μ). Adaptive rate scales with N once Ns > 1.')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('moran_substitution_rate.png', dpi=120)
    plt.close()
    print("Saved: moran_substitution_rate.png")


# ---------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("MORAN PROCESS SIMULATION")
    print("=" * 60)

    print("\n[1] Sample trajectories (N=100, r=1.1)")
    plot_trajectories(N=100, r=1.1, n_trajectories=30)

    print("\n[2] Fixation probability vs r (N=50)")
    print("    Comparing analytical formula to empirical simulation:")
    plot_fixation_vs_r(N=50, trials=2000)

    print("\n[3] Substitution rate: R = N * mu * rho")
    plot_substitution_rate()

    print("\n[4] Quick numerical sanity checks:")
    for N, r in [(100, 1.05), (100, 1.0), (1000, 1.01), (10, 2.0)]:
        rho = fixation_probability_analytical(N, r)
        R = N * 1e-6 * rho
        print(f"    N={N:5d}, r={r:.2f}:  rho={rho:.5f},  R = N*mu*rho = {R:.2e}")

    print("\nDone. Three figures saved to current directory.")
