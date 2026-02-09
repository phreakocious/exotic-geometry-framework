#!/usr/bin/env python3
"""
Investigation: Maze Generation Algorithm Fingerprinting via SpatialFieldGeometry

Different maze algorithms produce structurally distinct labyrinths despite
all being "perfect mazes" (exactly one path between any two cells):

- Recursive Backtracker (DFS): long winding corridors, low branching
- Prim's (randomized): short corridors, many branches, bushy
- Kruskal's (randomized): uniform local structure
- Binary Tree: strong diagonal bias (NW to SE)
- Sidewinder: horizontal runs with periodic vertical connections
- Aldous-Broder: uniform random spanning tree (the "gold standard" of uniformity)

Can SpatialFieldGeometry distinguish these from their spatial structure alone?

Methodology: N_TRIALS=25, shuffled baselines, Cohen's d, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
ALPHA = 0.05
MAZE_CELLS = 31  # odd number: cells at odd positions, walls at even
# Grid rendered as (2*MAZE_CELLS+1) x (2*MAZE_CELLS+1) pixels
RENDER_SIZE = 2 * MAZE_CELLS + 1  # 63x63

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn


# =============================================================================
# MAZE GENERATION
# =============================================================================

def _init_grid(rows, cols):
    """Create wall grid: 0=wall, 1=passage. Initialize all walls."""
    H = 2 * rows + 1
    W = 2 * cols + 1
    grid = np.zeros((H, W), dtype=np.float64)
    return grid


def _cell_to_grid(r, c):
    """Convert cell coordinates to grid coordinates."""
    return 2 * r + 1, 2 * c + 1


def _carve(grid, r1, c1, r2, c2):
    """Carve passage between two cells and the wall between them."""
    gr1, gc1 = _cell_to_grid(r1, c1)
    gr2, gc2 = _cell_to_grid(r2, c2)
    grid[gr1, gc1] = 1.0
    grid[gr2, gc2] = 1.0
    grid[(gr1+gr2)//2, (gc1+gc2)//2] = 1.0


def maze_dfs(rows, cols, rng):
    """Recursive backtracker (DFS): long corridors."""
    grid = _init_grid(rows, cols)
    visited = np.zeros((rows, cols), dtype=bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    stack = [(0, 0)]
    visited[0, 0] = True
    grid[_cell_to_grid(0, 0)] = 1.0

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                neighbors.append((nr, nc))

        if neighbors:
            idx = rng.integers(len(neighbors))
            nr, nc = neighbors[idx]
            _carve(grid, r, c, nr, nc)
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


def maze_prim(rows, cols, rng):
    """Randomized Prim's: short corridors, bushy."""
    grid = _init_grid(rows, cols)
    visited = np.zeros((rows, cols), dtype=bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # Start from random cell
    sr, sc = rng.integers(rows), rng.integers(cols)
    visited[sr, sc] = True
    grid[_cell_to_grid(sr, sc)] = 1.0

    frontier = []
    for dr, dc in dirs:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc, sr, sc))

    while frontier:
        idx = rng.integers(len(frontier))
        fr, fc, pr, pc = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()

        if visited[fr, fc]:
            continue

        visited[fr, fc] = True
        _carve(grid, pr, pc, fr, fc)

        for dr, dc in dirs:
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                frontier.append((nr, nc, fr, fc))

    return grid


def maze_kruskal(rows, cols, rng):
    """Randomized Kruskal's: uniform local structure."""
    grid = _init_grid(rows, cols)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # Initialize cells
    for r in range(rows):
        for c in range(cols):
            grid[_cell_to_grid(r, c)] = 1.0

    # Union-Find
    parent = {}
    rank = {}
    for r in range(rows):
        for c in range(cols):
            parent[(r,c)] = (r,c)
            rank[(r,c)] = 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    # List all walls
    edges = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append((r, c, r+1, c))
            if c + 1 < cols:
                edges.append((r, c, r, c+1))

    # Shuffle and process
    order = rng.permutation(len(edges))
    for i in order:
        r1, c1, r2, c2 = edges[i]
        if union((r1,c1), (r2,c2)):
            _carve(grid, r1, c1, r2, c2)

    return grid


def maze_binary_tree(rows, cols, rng):
    """Binary tree: strong NW-to-SE diagonal bias."""
    grid = _init_grid(rows, cols)

    for r in range(rows):
        for c in range(cols):
            grid[_cell_to_grid(r, c)] = 1.0
            choices = []
            if r > 0:
                choices.append((-1, 0))  # North
            if c > 0:
                choices.append((0, -1))  # West

            if choices:
                idx = rng.integers(len(choices))
                dr, dc = choices[idx]
                _carve(grid, r, c, r+dr, c+dc)

    return grid


def maze_sidewinder(rows, cols, rng):
    """Sidewinder: horizontal runs with periodic vertical links."""
    grid = _init_grid(rows, cols)

    for r in range(rows):
        run_start = 0
        for c in range(cols):
            grid[_cell_to_grid(r, c)] = 1.0

            at_east = (c == cols - 1)
            at_north = (r == 0)

            should_close = at_east or (not at_north and rng.random() < 0.5)

            if should_close and not at_north:
                # Pick random cell from run and carve north
                run_c = rng.integers(run_start, c + 1)
                _carve(grid, r, run_c, r-1, run_c)
                run_start = c + 1
            elif not at_east:
                # Continue run east
                _carve(grid, r, c, r, c+1)

    return grid


def maze_aldous_broder(rows, cols, rng):
    """Aldous-Broder: uniform spanning tree. Slow but perfectly uniform."""
    grid = _init_grid(rows, cols)
    visited = np.zeros((rows, cols), dtype=bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    r, c = rng.integers(rows), rng.integers(cols)
    visited[r, c] = True
    grid[_cell_to_grid(r, c)] = 1.0
    n_visited = 1
    total = rows * cols

    while n_visited < total:
        d = dirs[rng.integers(4)]
        nr, nc = r + d[0], c + d[1]
        if 0 <= nr < rows and 0 <= nc < cols:
            if not visited[nr, nc]:
                visited[nr, nc] = True
                _carve(grid, r, c, nr, nc)
                n_visited += 1
            r, c = nr, nc

    return grid


ALGORITHMS = {
    'DFS':           maze_dfs,
    'Prim':          maze_prim,
    'Kruskal':       maze_kruskal,
    'BinaryTree':    maze_binary_tree,
    'Sidewinder':    maze_sidewinder,
    'AldousBroder':  maze_aldous_broder,
}


# =============================================================================
# HELPERS
# =============================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    ps = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def shuffle_field(field, rng):
    flat = field.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(field.shape)


def collect_metrics(analyzer, fields):
    out = {m: [] for m in METRIC_NAMES}
    for f in fields:
        res = analyzer.analyze(f)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


# =============================================================================
# INVESTIGATION
# =============================================================================

def run_investigation():
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    print("=" * 78)
    print("MAZE GENERATION ALGORITHM FINGERPRINTING")
    print(f"Maze: {MAZE_CELLS}x{MAZE_CELLS} cells, rendered {RENDER_SIZE}x{RENDER_SIZE}, "
          f"N={N_TRIALS}")
    print("=" * 78)

    algo_data = {}
    shuffled_algo_data = {}
    example_fields = {}

    for name, gen_func in ALGORITHMS.items():
        print(f"  {name:14s}...", end=" ", flush=True)
        fields, shuf_fields = [], []

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            field = gen_func(MAZE_CELLS, MAZE_CELLS, rng)
            fields.append(field)
            shuf_fields.append(shuffle_field(field, np.random.default_rng(1000 + trial)))

            if trial == 0:
                example_fields[name] = field.copy()

        algo_data[name] = collect_metrics(analyzer, fields)
        shuffled_algo_data[name] = collect_metrics(analyzer, shuf_fields)
        passage_frac = [np.mean(f) for f in fields]
        print(f"passage_frac={np.mean(passage_frac):.3f}")

    names = list(ALGORITHMS.keys())

    # Each vs shuffled
    bonf_s = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  Each algorithm vs SHUFFLED (Bonferroni α={bonf_s:.2e})")
    print(f"{'─' * 78}")

    for name in names:
        sig = 0
        findings = []
        for m in METRIC_NAMES:
            a = np.array(algo_data[name][m])
            b = np.array(shuffled_algo_data[name][m])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if p < bonf_s:
                sig += 1
                findings.append((m, d, p))
        print(f"\n  {name}: {sig} significant metrics")
        findings.sort(key=lambda x: -abs(x[1]))
        for m, d, p in findings[:5]:
            print(f"    {m:30s}  d={d:+8.2f}  p={p:.2e}")

    # Pairwise
    n_pairs = len(names) * (len(names) - 1) // 2
    bonf_p = ALPHA / len(METRIC_NAMES)
    print(f"\n{'─' * 78}")
    print(f"  PAIRWISE comparisons (Bonferroni α={bonf_p:.2e})")
    print(f"{'─' * 78}")

    pair_results = []
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            sig = 0
            best_d, best_m = 0, ""
            for m in METRIC_NAMES:
                a = np.array(algo_data[n1][m])
                b = np.array(algo_data[n2][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf_p:
                    sig += 1
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            pair_results.append((n1, n2, sig, best_m, best_d))
            print(f"  {n1:14s} vs {n2:14s}: {sig:2d} sig  "
                  f"best: {best_m:25s} d={best_d:+8.2f}")

    return algo_data, shuffled_algo_data, example_fields, pair_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(algo_data, example_fields, pair_results):
    print("\nGenerating figure...", flush=True)

    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': '#111111',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    names = list(ALGORITHMS.keys())
    n = len(names)
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0', '#00BCD4']

    fig = plt.figure(figsize=(18, 35), facecolor='black')
    gs = gridspec.GridSpec(4, n, figure=fig, height_ratios=[1.2, 1.0, 1.0, 1.0],
                           hspace=0.45, wspace=0.3)

    # Row 0: Example mazes
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(example_fields[name], cmap='bone', interpolation='nearest')
        ax.set_title(name, fontsize=9, fontweight='bold', color=colors[i])
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1: Key metrics
    compare_metrics = ['SpatialField:coherence_score', 'SpatialField:n_basins',
                       'Surface:gaussian_curvature_mean', 'PersistentHomology2D:persistence_entropy',
                       'Conformal2D:structure_isotropy', 'SpectralPower:spectral_slope']

    for j in range(min(n, len(compare_metrics))):
        metric = compare_metrics[j]
        ax = fig.add_subplot(gs[1, j])
        means = [np.mean(algo_data[nm][metric]) for nm in names]
        stds = [np.std(algo_data[nm][metric]) for nm in names]
        ax.bar(range(n), means, yerr=stds, capsize=3,
               color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=6, rotation=40, ha='right')
        ax.set_title(metric.split(':')[-1].replace('_', ' '), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    # Row 2: Pairwise matrix
    ax_mat = fig.add_subplot(gs[2, :3])
    mat = np.zeros((n, n))
    for n1, n2, sig, _, _ in pair_results:
        i1, i2 = names.index(n1), names.index(n2)
        mat[i1, i2] = sig
        mat[i2, i1] = sig
    im = ax_mat.imshow(mat, cmap='magma', interpolation='nearest', vmin=0)
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(names, fontsize=7, rotation=40, ha='right')
    ax_mat.set_yticklabels(names, fontsize=5)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax_mat.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                           fontsize=7, fontweight='bold',
                           color='white' if mat[i,j] > 8 else '#aaaaaa')
    ax_mat.set_title('Pairwise significant metrics', fontsize=10, fontweight='bold')
    cb = plt.colorbar(im, ax=ax_mat, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    # Summary text
    ax_txt = fig.add_subplot(gs[2, 3:])
    ax_txt.axis('off')
    pair_sorted = sorted(pair_results, key=lambda x: -x[2])
    lines = ['Best discriminator per pair:\n']
    for n1, n2, sig, bm, bd in pair_sorted:
        lines.append(f'{n1:14s} vs {n2:14s}: {sig:2d}  {bm}')
    ax_txt.text(0.05, 0.95, '\n'.join(lines), transform=ax_txt.transAxes,
               fontsize=6.5, fontfamily='monospace', va='top', color='#cccccc')

    # Row 3: Anisotropy profile — Binary Tree should stand out
    ax_aniso = fig.add_subplot(gs[3, :3])
    for i, name in enumerate(names):
        vals = algo_data[name].get('SpatialField:anisotropy_mean', [])
        if vals:
            ax_aniso.hist(vals, bins=15, alpha=0.5, color=colors[i],
                         label=name, density=True)
    ax_aniso.set_xlabel('Anisotropy mean', fontsize=9)
    ax_aniso.set_ylabel('Density', fontsize=9)
    ax_aniso.set_title('Anisotropy distribution by algorithm', fontsize=10,
                       fontweight='bold')
    ax_aniso.legend(fontsize=7, loc='upper right')
    ax_aniso.tick_params(labelsize=8)

    # Coherence profile
    ax_coh = fig.add_subplot(gs[3, 3:])
    for i, name in enumerate(names):
        vals = algo_data[name].get('SpatialField:coherence_score', [])
        if vals:
            ax_coh.hist(vals, bins=15, alpha=0.5, color=colors[i],
                       label=name, density=True)
    ax_coh.set_xlabel('Coherence score', fontsize=9)
    ax_coh.set_ylabel('Density', fontsize=9)
    ax_coh.set_title('Coherence distribution by algorithm', fontsize=10,
                     fontweight='bold')
    ax_coh.legend(fontsize=7, loc='upper right')
    ax_coh.tick_params(labelsize=8)

    fig.suptitle('Maze Generation Algorithm Fingerprinting',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'mazes.png'),
                dpi=180, bbox_inches='tight', facecolor='black')
    print("  Saved mazes.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    algo_data, shuffled_data, example_fields, pair_results = run_investigation()
    make_figure(algo_data, example_fields, pair_results)

    names = list(ALGORITHMS.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    pairs_ok = sum(1 for _, _, s, _, _ in pair_results if s > 0)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Algorithms tested: {len(names)}")
    print(f"  Pairs distinguished: {pairs_ok}/{n_pairs}")
    print(f"  All distinguished: {pairs_ok == n_pairs}")
