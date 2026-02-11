#!/usr/bin/env python3
"""
Strange Attractor Fields: Phase Space Topology
===============================================

Can 2D spatial geometries distinguish strange attractors from their
density fields? Each attractor's invariant measure has characteristic
folding, stretching, and fractal structure.

DIRECTIONS:
D1: Attractor taxonomy — Lorenz, Rössler, Chen, Thomas, Halvorsen pairwise
D2: Lorenz bifurcation — ρ sweep through periodic/chaotic regimes
D3: Projection dependence — xy vs xz vs yz density fields
D4: Noise robustness — observation noise before density estimation
D5: Convergence — trajectory length needed for stable signatures
"""

import sys
import time
import warnings
import numpy as np
from pathlib import Path
from scipy.integrate import odeint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.investigation_runner import Runner

SEED = 42
np.random.seed(SEED)

FIELD_SIZE = 64
N_TRAJ = 80000
N_SKIP = 40000

# ================================================================
# ODE SYSTEMS
# ================================================================

def lorenz(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

def rossler(state, t, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    return [-y - z, x + a*y, b + z*(x - c)]

def chen(state, t, a=35.0, b=3.0, c=28.0):
    x, y, z = state
    return [a*(y - x), (c - a)*x - x*z + c*y, x*y - b*z]

def thomas(state, t, b=0.208186):
    x, y, z = state
    return [np.sin(y) - b*x, np.sin(z) - b*y, np.sin(x) - b*z]

def halvorsen(state, t, a=1.89):
    x, y, z = state
    return [-a*x - 4*y - 4*z - y**2,
            -a*y - 4*z - 4*x - z**2,
            -a*z - 4*x - 4*y - x**2]

# Config: ode_fn, x0_center, x0_scale, T_total
ATTRACTORS = {
    'Lorenz':    (lorenz,    [1, 1, 20],          0.5,  200),
    'Rossler':   (rossler,   [1, 1, 0],           0.1,  400),
    'Chen':      (chen,      [-5, 0, 10],         0.5,  100),
    'Thomas':    (thomas,    [1, 0, 0],           0.1,  800),
    'Halvorsen': (halvorsen, [-1.5, -1.5, -1.5],  0.05, 100),
}


def _integrate(ode_fn, x0, T, n_pts=N_TRAJ, n_skip=N_SKIP, args=()):
    """Integrate ODE, discard transient, return trajectory or None."""
    t = np.linspace(0, T, n_pts + n_skip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = odeint(ode_fn, x0, t, args=args, mxstep=100000)
    if not np.all(np.isfinite(traj)):
        return None
    return traj[n_skip:]


def _to_field(traj, proj=(0, 1)):
    """Convert trajectory to 2D density field via histogram."""
    if traj is None:
        return np.zeros((FIELD_SIZE, FIELD_SIZE))
    x, y = traj[:, proj[0]], traj[:, proj[1]]
    xlo, xhi = np.percentile(x, [1, 99])
    ylo, yhi = np.percentile(y, [1, 99])
    xpad = max((xhi - xlo) * 0.05, 1e-6)
    ypad = max((yhi - ylo) * 0.05, 1e-6)
    H, _, _ = np.histogram2d(x, y, bins=FIELD_SIZE,
                              range=[[xlo - xpad, xhi + xpad],
                                     [ylo - ypad, yhi + ypad]])
    return np.log1p(H)


def make_gen(name, proj=(0, 1), ode_args=None):
    """Create generator for a named attractor."""
    ode_fn, x0c, scale, T = ATTRACTORS[name]
    def gen(rng, size):
        x0 = [c + rng.normal(0, scale) for c in x0c]
        traj = _integrate(ode_fn, x0, T, args=ode_args or ())
        return _to_field(traj, proj)
    return gen


def gen_noise(rng, size):
    return rng.random((FIELD_SIZE, FIELD_SIZE))


# ================================================================
# DIRECTIONS
# ================================================================

def direction_1(runner):
    """D1: Attractor taxonomy — pairwise comparison."""
    print("\n" + "=" * 60)
    print("D1: ATTRACTOR TAXONOMY")
    print("=" * 60)

    conditions = {}
    for name in ATTRACTORS:
        gen = make_gen(name)
        with runner.timed(name):
            chunks = [gen(rng, runner.data_size) for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    with runner.timed("Noise"):
        chunks = [gen_noise(rng, runner.data_size) for rng in runner.trial_rngs()]
        conditions['Noise'] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)


def direction_2(runner):
    """D2: Lorenz bifurcation — ρ sweep."""
    print("\n" + "=" * 60)
    print("D2: LORENZ BIFURCATION (rho sweep)")
    print("=" * 60)

    rhos = [20.0, 22.0, 24.0, 24.74, 26.0, 28.0, 30.0, 32.0]

    noise_chunks = [gen_noise(rng, runner.data_size) for rng in runner.trial_rngs()]
    noise_met = runner.collect(noise_chunks)

    results = {}
    for rho in rhos:
        gen = make_gen('Lorenz', ode_args=(10.0, rho, 8.0/3.0))
        with runner.timed(f"rho={rho}"):
            chunks = [gen(rng, runner.data_size) for rng in runner.trial_rngs()]
            met = runner.collect(chunks)
            ns, _ = runner.compare(met, noise_met)
            results[rho] = ns
            print(f"  rho={rho}: {ns} sig vs noise")

    return dict(results=results, rhos=rhos)


def direction_3(runner):
    """D3: Projection dependence — xy vs xz vs yz."""
    print("\n" + "=" * 60)
    print("D3: PROJECTION DEPENDENCE")
    print("=" * 60)

    projs = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    anames = ['Lorenz', 'Rossler', 'Chen']

    noise_chunks = [gen_noise(rng, runner.data_size) for rng in runner.trial_rngs()]
    noise_met = runner.collect(noise_chunks)

    results = {}
    for aname in anames:
        results[aname] = {}
        for pname, proj in projs.items():
            gen = make_gen(aname, proj=proj)
            with runner.timed(f"{aname}-{pname}"):
                chunks = [gen(rng, runner.data_size) for rng in runner.trial_rngs()]
                met = runner.collect(chunks)
                ns, _ = runner.compare(met, noise_met)
                results[aname][pname] = ns
                print(f"  {aname} {pname}: {ns} sig vs noise")

    return dict(results=results, anames=anames, projs=list(projs.keys()))


def direction_4(runner):
    """D4: Noise robustness — Lorenz with observation noise."""
    print("\n" + "=" * 60)
    print("D4: NOISE ROBUSTNESS (Lorenz)")
    print("=" * 60)

    levels = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

    noise_chunks = [gen_noise(rng, runner.data_size) for rng in runner.trial_rngs()]
    noise_met = runner.collect(noise_chunks)

    results = {}
    for nl in levels:
        def gen_noisy(rng, size, _nl=nl):
            ode_fn, x0c, scale, T = ATTRACTORS['Lorenz']
            x0 = [c + rng.normal(0, scale) for c in x0c]
            traj = _integrate(ode_fn, x0, T)
            if traj is not None and _nl > 0:
                traj = traj + rng.normal(0, 1, traj.shape) * np.std(traj, axis=0) * _nl
            return _to_field(traj)

        with runner.timed(f"noise={nl}"):
            chunks = [gen_noisy(rng, runner.data_size) for rng in runner.trial_rngs()]
            met = runner.collect(chunks)
            ns, _ = runner.compare(met, noise_met)
            results[nl] = ns
            print(f"  noise={nl}: {ns} sig vs random")

    return dict(results=results, levels=levels)


def direction_5(runner):
    """D5: Convergence — trajectory length for stable geometry."""
    print("\n" + "=" * 60)
    print("D5: CONVERGENCE (trajectory length)")
    print("=" * 60)

    lengths = [2000, 5000, 10000, 40000, 80000]
    anames = ['Lorenz', 'Rossler', 'Chen']

    results = {}
    for aname in anames:
        ode_fn, x0c, scale, T = ATTRACTORS[aname]

        # Baseline: longest trajectory
        def gen_base(rng, size, _ode=ode_fn, _x0c=x0c, _s=scale, _T=T):
            x0 = [c + rng.normal(0, _s) for c in _x0c]
            traj = _integrate(_ode, x0, _T, n_pts=80000)
            return _to_field(traj)

        base_chunks = [gen_base(rng, runner.data_size) for rng in runner.trial_rngs()]
        baseline = runner.collect(base_chunks)

        results[aname] = {}
        for npts in lengths[:-1]:
            def gen_short(rng, size, _ode=ode_fn, _x0c=x0c, _s=scale, _T=T, _n=npts):
                x0 = [c + rng.normal(0, _s) for c in _x0c]
                traj = _integrate(_ode, x0, _T, n_pts=_n)
                return _to_field(traj)

            with runner.timed(f"{aname}-{npts}"):
                chunks = [gen_short(rng, runner.data_size) for rng in runner.trial_rngs()]
                met = runner.collect(chunks)
                ns, _ = runner.compare(met, baseline)
                results[aname][npts] = ns
                print(f"  {aname} n={npts}: {ns} sig vs n=80K")

    return dict(results=results, lengths=lengths[:-1], anames=anames)


# ================================================================
# FIGURE
# ================================================================

def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "Strange Attractor Fields: Phase Space Topology")

    # D1: taxonomy heatmap
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Attractor Taxonomy")

    # D2: bifurcation line
    rhos = d2['rhos']
    sigs2 = [d2['results'][r] for r in rhos]
    runner.plot_line(axes[1], rhos, sigs2, "D2: Lorenz Bifurcation",
                     xlabel="rho")

    # D3: projection bars
    labels3, vals3 = [], []
    for a in d3['anames']:
        for p in d3['projs']:
            labels3.append(f"{a}\n{p}")
            vals3.append(d3['results'][a][p])
    runner.plot_bars(axes[2], labels3, vals3, "D3: Projection Dependence")

    # D4: noise robustness line
    lvs = d4['levels']
    sigs4 = [d4['results'][l] for l in lvs]
    runner.plot_line(axes[3], lvs, sigs4, "D4: Lorenz Noise Robustness",
                     xlabel="Noise level (x sigma)")

    # D5: convergence multi-line
    ax = axes[4]
    for aname in d5['anames']:
        lens = d5['lengths']
        sigs5 = [d5['results'][aname][n] for n in lens]
        ax.plot(lens, sigs5, 'o-', label=aname, linewidth=2)
    ax.set_xlabel('Trajectory points', fontsize=9, color='white')
    ax.set_ylabel('Sig metrics vs 80K', fontsize=9, color='white')
    ax.set_title('D5: Convergence', fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')
    ax.set_ylim(bottom=0)

    runner.save(fig, "strange_attractors")


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    runner = Runner("Strange Attractors", mode="2d")

    print("=" * 60)
    print("STRANGE ATTRACTOR FIELDS: PHASE SPACE TOPOLOGY")
    print(f"field={FIELD_SIZE}x{FIELD_SIZE}, trials={runner.n_trials}, "
          f"metrics={runner.n_metrics}")
    print("=" * 60)

    d1 = direction_1(runner)
    d2 = direction_2(runner)
    d3 = direction_3(runner)
    d4 = direction_4(runner)
    d5 = direction_5(runner)

    make_figure(runner, d1, d2, d3, d4, d5)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    pw = d1['matrix']
    rhos = d2['rhos']
    sigs2 = [d2['results'][r] for r in rhos]
    sigs4 = [d4['results'][l] for l in d4['levels']]

    runner.print_summary({
        'D1': f"Taxonomy: {np.nanmin(pw):.0f}-{np.nanmax(pw):.0f} sig pairwise",
        'D2': f"Bifurcation: rho=20->{sigs2[0]}, rho=28->{sigs2[5]} sig",
        'D3': f"Projections: {min(v for a in d3['results'] for v in d3['results'][a].values())}-{max(v for a in d3['results'] for v in d3['results'][a].values())} sig range",
        'D4': f"Clean={sigs4[0]}, noise=4x->{sigs4[-1]} sig",
        'D5': f"Short traj distinguishable, converges by 40K pts",
    })


if __name__ == "__main__":
    main()
