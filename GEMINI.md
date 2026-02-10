# Exotic Geometry Framework — Instructions for Gemini

## What This Project Does

This framework applies 24 exotic geometries (E8 lattice, hyperbolic, tropical, Lorentzian, etc.) to arbitrary byte sequences or 2D fields, extracting ~131 metrics (1D) or ~80 metrics (2D). Investigations compare conditions statistically using Cohen's d + Bonferroni-corrected Welch t-tests across 25 independent trials.

## Your Role

You write **investigation scripts** that explore whether exotic geometries can detect structure in a specific domain. Each investigation has exactly **5 directions** (D1–D5), produces a dark-themed figure, and prints a summary.

## How to Write an Investigation

Use the shared runner module at `tools/investigation_runner.py`. It handles all boilerplate:
- Metric discovery, analyzer setup
- `runner.collect(chunks)` — analyze data chunks → metric dict
- `runner.compare(a, b)` — Cohen's d + Bonferroni → (n_sig, findings)
- `runner.compare_pairwise(conditions)` — all-pairs → matrix
- Dark-themed figures: `runner.create_figure()`, `runner.plot_heatmap()`, etc.

### Template

```python
#!/usr/bin/env python3
"""
[Title] Investigation: [Subtitle]
==================================

[1-2 sentence question this investigation answers]

DIRECTIONS:
D1: [taxonomy/detection] — [what conditions, how compared]
D2: [sequential structure] — [real vs shuffled or similar]
D3: [model hierarchy or parameter sweep]
D4: [robustness / variant test]
D5: [delay embedding or cross-scale]
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.investigation_runner import Runner

# ==============================================================
# CONFIG
# ==============================================================
SEED = 42
np.random.seed(SEED)

# ==============================================================
# DATA GENERATORS
# ==============================================================
# Each generator takes (rng, size) and returns np.uint8 array (1D)
# or np.float64 array (2D) of the specified size.

def generate_condition_a(rng, size):
    """Generate data for condition A."""
    # YOUR DOMAIN-SPECIFIC CODE HERE
    return data.astype(np.uint8)

def generate_condition_b(rng, size):
    """Generate data for condition B."""
    return data.astype(np.uint8)

# ==============================================================
# DIRECTIONS
# ==============================================================
def direction_1(runner):
    """D1: [Title]"""
    print("\n" + "=" * 60)
    print("D1: [TITLE]")
    print("=" * 60)

    conditions = {}
    for name, gen_fn in [("cond_a", generate_condition_a),
                         ("cond_b", generate_condition_b)]:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)


def direction_2(runner):
    """D2: Sequential structure — ALL conditions vs shuffled (multi-bar)"""
    print("\n" + "=" * 60)
    print("D2: SEQUENTIAL STRUCTURE")
    print("=" * 60)

    results = {}
    for name, gen_fn in [("cond_a", generate_condition_a),
                         ("cond_b", generate_condition_b),
                         ("cond_c", generate_condition_c)]:
        chunks = [gen_fn(rng, runner.data_size)
                  for rng in runner.trial_rngs()]
        real = runner.collect(chunks)
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, _ = runner.compare(real, shuf)
        results[name] = ns
        print(f"  {name} vs shuffled = {ns:3d} sig")

    return dict(results=results)


def direction_3(runner):
    """D3: Parameter sweep (line plot with 5+ points)"""
    print("\n" + "=" * 60)
    print("D3: PARAMETER SWEEP")
    print("=" * 60)

    params = [0.0, 0.25, 0.5, 0.75, 1.0]
    baseline_chunks = [generate_condition_a(rng, runner.data_size)
                       for rng in runner.trial_rngs()]
    baseline = runner.collect(baseline_chunks)

    results = {}
    for p in params:
        with runner.timed(f"p={p}"):
            chunks = [generate_with_param(rng, runner.data_size, p)
                      for rng in runner.trial_rngs(offset=int(p*100))]
            met = runner.collect(chunks)
            ns, _ = runner.compare(baseline, met)
            results[p] = ns

    return dict(results=results, params=params)

# direction_4, direction_5: ALWAYS produce 3+ data points per panel.

# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "[Investigation Title]")

    # D1: heatmap (pairwise taxonomy)
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Taxonomy")

    # D2: multi-bar (each condition vs shuffled)
    names2 = list(d2['results'].keys())
    vals2 = [d2['results'][n] for n in names2]
    runner.plot_bars(axes[1], names2, vals2, "D2: vs Shuffled")

    # D3: line plot (parameter sweep)
    params = d3['params']
    sigs3 = [d3['results'][p] for p in params]
    runner.plot_line(axes[2], params, sigs3, "D3: Parameter Sweep",
                     xlabel="Parameter")

    # D4, D5: ALWAYS use plot_line or multi-bar plot_bars.
    # NEVER a single bar — that's just a number with extra pixels.
    # ...

    runner.save(fig, "investigation_name")

# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("[Investigation Name]", mode="1d")  # or "2d"

    print("=" * 60)
    print("[INVESTIGATION TITLE]")
    print(f"size={runner.data_size}, trials={runner.n_trials}, "
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

    runner.print_summary({
        'D1': f"Pairwise: ...",
        'D2': f"vs shuffled: ...",
        # ...
    })

if __name__ == "__main__":
    main()
```

## Critical Rules

1. **25 trials, always.** Never N=1. Each condition needs 25 independently generated chunks.
2. **Trials must be INDEPENDENT.** Each call to a generator with a different `rng` MUST produce different data. If your generator is deterministic (e.g., computing a fixed wavefunction), you MUST add randomness — sample from the distribution, add measurement noise, or vary parameters. Zero variance across trials = NaN p-values = 0 sig everywhere.
3. **Use `runner.compare()`** for all statistical tests. Do NOT roll your own t-test.
4. **Data must be `np.uint8`** (1D) or `float64 2D array` (2D). The framework expects this.
4. **Non-overlapping chunks.** If slicing from a fixed sequence, stride = chunk_size.
6. **Generators take `(rng, size)`** and return an array of the right dtype/shape.
7. **5 directions exactly.** Standard pattern: D1=taxonomy, D2=sequential, D3=hierarchy/sweep, D4=robustness, D5=transforms.
8. **Dark theme figures.** The runner handles this — just use `runner.create_figure()`.
9. **Print results as you go.** Each direction prints its findings before the figure.
10. **No external API calls at runtime.** Download/cache data in a setup function if needed.
11. **File placement:** 1D investigations go in `investigations/1d/`, 2D in `investigations/2d/`.
12. **String literals:** Use `"\n"` for newlines in print statements, NOT actual newlines inside the string. Write `print("\n" + "=" * 60)` not `print("[newline]" + "=" * 60)`.
13. **Summary must show actual values.** The `runner.print_summary()` call must include actual computed results (e.g., `f"Taxonomy: {pw.min():.0f}-{pw.max():.0f} sig"`), NOT placeholder text like "Taxonomy completed".

## Runner API Reference

### `Runner(name, mode='1d', data_size=2000, n_trials=25)`
- `mode='1d'`: 24 geometries, ~131 metrics. Data: uint8 arrays.
- `mode='2d'`: 8 spatial geometries, ~80 metrics. Data: 2D float arrays.

### Data Methods
- `runner.trial_rngs(offset=0)` — list of 25 independent `np.random.default_rng` objects
- `runner.collect(chunks)` — analyze chunks → `{metric_name: [values]}`
- `runner.shuffle_chunks(chunks)` — shuffled copies
- `runner.make_chunks(long_array)` — slice a long array into trial-sized chunks

### Statistics
- `runner.compare(metrics_a, metrics_b)` → `(n_sig, [(metric, d, p), ...])`
- `runner.compare_pairwise(dict_of_metrics)` → `(matrix, names, findings_dict)`
- `Runner.cohens_d(a, b)` — static method

### Figures
- `runner.create_figure(n_panels, title)` → `(fig, axes_list)`
- `runner.plot_heatmap(ax, matrix, labels, title)`
- `runner.plot_bars(ax, names, values, title)`
- `runner.plot_line(ax, x, y, title, xlabel=, ylabel=, color=)`
- `runner.save(fig, "name")` — saves to `figures/name.png` at 180 dpi

### Timing
- `with runner.timed("label"):` — prints elapsed time

## Running Investigations

```bash
MPLBACKEND=Agg .venv/bin/python investigations/1d/my_investigation.py
```

## What Makes a Good Investigation

- **Real question:** "Can geometry detect X?" not "Let's compute some numbers."
- **Clean null models:** shuffled (destroys order), dist-matched (same distribution, different sequence), model-based (e.g., Cramér for primes).
- **Surprising findings:** The interesting result is often what the framework *can't* detect.
- **Domain expertise in generators:** The data generators are where your knowledge matters most.

## IMPORTANT: Figure Design — No Single-Bar Panels

**Every figure panel must be visually informative.** A bar chart with 1 bar is just a number — useless as a figure panel.

**BAD** — these produce a single number, wasting a panel:
```python
# Comparing just two things gives one bar. Don't do this.
ns, _ = runner.compare(a_met, b_met)
return dict(ns=ns)
# Then: runner.plot_bars(ax, ["A vs B"], [ns], "Title")  # ONE BAR = USELESS
```

**GOOD** — each direction should produce multiple data points:

1. **Sweep a parameter** (→ line plot with 4-7 points):
   ```python
   for param in [0.1, 0.3, 0.5, 0.7, 0.9]:
       chunks = [gen(rng, size, param=param) for rng in runner.trial_rngs()]
       ns, _ = runner.compare(baseline, runner.collect(chunks))
       results[param] = ns
   # Then: runner.plot_line(ax, params, sigs, "Title")
   ```

2. **Compare all conditions vs shuffled** (→ bar chart with 3-5 bars):
   ```python
   for name, gen_fn in conditions:
       chunks = [gen_fn(rng, size) for rng in runner.trial_rngs()]
       real = runner.collect(chunks)
       shuf = runner.collect(runner.shuffle_chunks(chunks))
       ns, _ = runner.compare(real, shuf)
       results[name] = ns
   # Then: runner.plot_bars(ax, names, values, "Title")
   ```

3. **Pairwise comparison** (→ heatmap):
   ```python
   matrix, names, _ = runner.compare_pairwise(conditions)
   # Then: runner.plot_heatmap(ax, matrix, names, "Title")
   ```

**Design rule: Before writing a direction, ask "will this produce at least 3 data points for the figure?" If not, redesign it as a sweep or multi-condition comparison.**

### Direction Design Patterns

| Pattern | Output | Plot type | Example |
|---------|--------|-----------|---------|
| Taxonomy (3-6 conditions pairwise) | NxN matrix | `plot_heatmap` | "4 ISAs pairwise" |
| Each vs shuffled (3-5 conditions) | N bars | `plot_bars` | "Array/List/Tree vs shuffled" |
| Parameter sweep (5-7 values) | line | `plot_line` | "Load factor 0.1→0.9" |
| Hierarchy (ordered conditions) | line | `plot_line` | "N-gram order 0→8" |
| Scale test (4-5 sizes) | line | `plot_line` | "N=500,1K,2K,4K,8K" |

## Existing Investigations for Reference

Look at these for patterns:
- `investigations/1d/text_geometry.py` — n-gram hierarchy, author fingerprinting
- `investigations/1d/rng.py` — PRNG quality testing
- `investigations/1d/seti.py` — signal detection at low SNR
- `investigations/2d/mandelbrot.py` — escape time fields
- `investigations/2d/elementary_ca.py` — cellular automata classification
