#!/usr/bin/env python3
"""
Investigation: Elementary Cellular Automata Spacetime Diagrams
Can spatial geometry distinguish Wolfram's complexity classes?

Elementary CA: 256 rules, 1D cells → 2D spacetime (rows=time, cols=space).
Wolfram's 4 classes:
  I   (homogeneous):  Rule 0, 32, 160   → all cells die or freeze
  II  (periodic):     Rule 4, 50, 90    → Sierpinski / regular patterns
  III (chaotic):      Rule 30, 45, 150  → pseudo-random, sensitive to IC
  IV  (complex):      Rule 110, 124, 54 → localized structures, Turing-complete

Questions:
  D1: Can each class be distinguished from random noise?
  D2: Can we distinguish classes from each other?
  D3: Can we distinguish rules WITHIN the same class?
  D4: How does detection change with spacetime grid size?
  D5: How do geometric signatures evolve as the CA runs longer?

Methodology: N_TRIALS=25, shuffled baselines, Cohen's d > 0.8, Bonferroni correction.
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
WIDTH = 128            # cells
HEIGHT = 128           # time steps (rows)

# --- Discover spatial metric names ---
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF_ALPHA = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn

print(f"Spatial metrics: {N_METRICS}, Bonferroni α={BONF_ALPHA:.2e}")

# --- Rules organized by Wolfram class ---
WOLFRAM_CLASSES = {
    'I':   [0, 32, 160],
    'II':  [4, 50, 90],
    'III': [30, 45, 150],
    'IV':  [110, 124, 54],
}

ALL_RULES = {}
for cls, rules in WOLFRAM_CLASSES.items():
    for r in rules:
        ALL_RULES[f"R{r}"] = (r, cls)


def elementary_ca_step(cells, rule_num):
    """Apply elementary CA rule to 1D cell array."""
    n = len(cells)
    new = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        left = cells[(i - 1) % n]
        center = cells[i]
        right = cells[(i + 1) % n]
        neighborhood = (left << 2) | (center << 1) | right
        new[i] = (rule_num >> neighborhood) & 1
    return new


def run_elementary_ca(rule_num, width, height, rng, single_seed=False):
    """Run elementary CA, return spacetime diagram as 2D float array."""
    if single_seed:
        cells = np.zeros(width, dtype=np.uint8)
        cells[width // 2] = 1
    else:
        cells = rng.integers(0, 2, width, dtype=np.uint8)

    spacetime = np.zeros((height, width), dtype=np.float64)
    spacetime[0] = cells.astype(np.float64)

    for t in range(1, height):
        cells = elementary_ca_step(cells, rule_num)
        spacetime[t] = cells.astype(np.float64)

    return spacetime


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
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


def compare(data_a, data_b, label=""):
    """Compare two metric dictionaries. Return (n_sig, findings)."""
    sig = 0
    findings = []
    for m in METRIC_NAMES:
        a = np.array(data_a[m])
        b = np.array(data_b[m])
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < BONF_ALPHA and abs(d) > 0.8:
            sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# ========================================================================
#  D1: Each class vs random noise
# ========================================================================
def direction_1(analyzer):
    print("\n" + "=" * 78)
    print("D1: WOLFRAM CLASS VS RANDOM NOISE")
    print("=" * 78)

    random_fields = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(9000 + trial)
        random_fields.append(rng.random((HEIGHT, WIDTH)))
    random_data = collect_metrics(analyzer, random_fields)

    class_results = {}
    all_data = {}

    for cls_name in ['I', 'II', 'III', 'IV']:
        rules = WOLFRAM_CLASSES[cls_name]
        print(f"\n  Class {cls_name} (Rules {rules}):")

        for rule in rules:
            fields = []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(42 + trial)
                fields.append(run_elementary_ca(rule, WIDTH, HEIGHT, rng))
            data = collect_metrics(analyzer, fields)
            all_data[f"R{rule}"] = data

            n_sig, findings = compare(data, random_data)
            print(f"    Rule {rule:3d}: {n_sig:2d} significant metrics vs random")
            for m, d, p in findings[:3]:
                print(f"      {m:45s}  d={d:+8.2f}")
            class_results[(cls_name, rule)] = n_sig

    # Summary by class
    print(f"\n  Summary (mean sig metrics by class):")
    for cls in ['I', 'II', 'III', 'IV']:
        vals = [class_results[(cls, r)] for r in WOLFRAM_CLASSES[cls]]
        print(f"    Class {cls}: {np.mean(vals):.0f} avg ({vals})")

    return all_data, random_data


# ========================================================================
#  D2: Pairwise class comparison
# ========================================================================
def direction_2(all_data):
    print("\n" + "=" * 78)
    print("D2: PAIRWISE CLASS COMPARISON")
    print("=" * 78)

    # Use one representative per class
    reps = {'I': 'R0', 'II': 'R90', 'III': 'R30', 'IV': 'R110'}
    classes = ['I', 'II', 'III', 'IV']

    pair_matrix = {}
    print(f"\n  Representatives: {reps}")
    print(f"  {'':12s}", end="")
    for c in classes:
        print(f"{'Class '+c:>12s}", end="")
    print()

    for i, c1 in enumerate(classes):
        print(f"  {'Class '+c1:12s}", end="")
        for j, c2 in enumerate(classes):
            if j <= i:
                print(f"{'—':>12s}", end="")
            else:
                n_sig, findings = compare(all_data[reps[c1]], all_data[reps[c2]])
                pair_matrix[(c1, c2)] = (n_sig, findings)
                print(f"{n_sig:>12d}", end="")
        print()

    # Top discriminators for each pair
    print(f"\n  Top discriminators:")
    for (c1, c2), (n_sig, findings) in sorted(pair_matrix.items()):
        if findings:
            top = findings[0]
            print(f"    Class {c1} vs {c2}: {n_sig:2d} sig  "
                  f"best: {top[0]:40s} d={top[1]:+.2f}")

    return pair_matrix


# ========================================================================
#  D3: Within-class discrimination
# ========================================================================
def direction_3(all_data):
    print("\n" + "=" * 78)
    print("D3: WITHIN-CLASS DISCRIMINATION")
    print("=" * 78)
    print("  Can geometry distinguish rules within the SAME Wolfram class?")

    for cls_name in ['I', 'II', 'III', 'IV']:
        rules = WOLFRAM_CLASSES[cls_name]
        print(f"\n  Class {cls_name}:")
        for i, r1 in enumerate(rules):
            for r2 in rules[i+1:]:
                n_sig, findings = compare(all_data[f"R{r1}"], all_data[f"R{r2}"])
                top_str = ""
                if findings:
                    top_str = f"  best: {findings[0][0]:35s} d={findings[0][1]:+.2f}"
                print(f"    R{r1:3d} vs R{r2:3d}: {n_sig:2d} sig{top_str}")


# ========================================================================
#  D4: Scale dependence
# ========================================================================
def direction_4(analyzer):
    print("\n" + "=" * 78)
    print("D4: SCALE DEPENDENCE")
    print("=" * 78)

    scales = [32, 64, 128, 256]
    test_rules = [30, 90, 110]  # One per interesting class

    for rule in test_rules:
        print(f"\n  Rule {rule}:")
        for size in scales:
            fields, random_fields = [], []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(42 + trial)
                fields.append(run_elementary_ca(rule, size, size, rng))
                random_fields.append(np.random.default_rng(9000 + trial).random((size, size)))

            data = collect_metrics(analyzer, fields)
            rand_data = collect_metrics(analyzer, random_fields)
            n_sig, findings = compare(data, rand_data)
            print(f"    {size:3d}x{size:3d}: {n_sig:2d} sig vs random", end="")
            if findings:
                print(f"  best: {findings[0][0]:35s} d={findings[0][1]:+.2f}", end="")
            print()


# ========================================================================
#  D5: Temporal evolution of spacetime geometry
# ========================================================================
def direction_5(analyzer):
    print("\n" + "=" * 78)
    print("D5: TEMPORAL EVOLUTION OF SPACETIME GEOMETRY")
    print("=" * 78)
    print("  How do geometric signatures change as the CA runs longer?")
    print("  (Fixed width, varying height = more time steps)")

    heights = [32, 64, 128, 256, 512]
    test_rules = [30, 90, 110]
    evo_data = {}

    for rule in test_rules:
        print(f"\n  Rule {rule}:")
        evo_data[rule] = {}
        for h in heights:
            fields = []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(42 + trial)
                fields.append(run_elementary_ca(rule, WIDTH, h, rng))
            data = collect_metrics(analyzer, fields)
            evo_data[rule][h] = data

            # Compare to shortest run
            if h == heights[0]:
                print(f"    h={h:3d}: baseline")
            else:
                n_sig, findings = compare(data, evo_data[rule][heights[0]])
                print(f"    h={h:3d}: {n_sig:2d} sig vs h={heights[0]}", end="")
                if findings:
                    print(f"  best: {findings[0][0]:35s} d={findings[0][1]:+.2f}", end="")
                print()

    return evo_data


# ========================================================================
#  Figure
# ========================================================================
def make_figure(all_data, random_data, pair_matrix, evo_data, analyzer):
    print("\nGenerating figure...", flush=True)

    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 22), facecolor='#181818')
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35,
                           height_ratios=[1.2, 0.8, 0.8, 1.0])

    # Row 0: Example spacetime diagrams (one per class)
    examples = {
        'I: Rule 0': 0, 'II: Rule 90': 90,
        'III: Rule 30': 30, 'IV: Rule 110': 110
    }
    for i, (label, rule) in enumerate(examples.items()):
        ax = _dark_ax(fig.add_subplot(gs[0, i]))
        rng = np.random.default_rng(42)
        field = run_elementary_ca(rule, WIDTH, HEIGHT, rng)
        ax.imshow(field, cmap='hot', interpolation='nearest', aspect='auto')
        ax.set_title(label, fontsize=11, fontweight='bold', color='white')
        ax.set_xlabel('Cell', fontsize=8)
        ax.set_ylabel('Time', fontsize=8)

    # Row 1: D1 — sig metrics vs random per rule
    ax = _dark_ax(fig.add_subplot(gs[1, :]))
    rules_sorted = []
    sigs_sorted = []
    colors_sorted = []
    class_colors = {'I': '#666666', 'II': '#4CAF50', 'III': '#E91E63', 'IV': '#2196F3'}
    for cls in ['I', 'II', 'III', 'IV']:
        for rule in WOLFRAM_CLASSES[cls]:
            name = f"R{rule}"
            if name in all_data:
                n_sig, _ = compare(all_data[name], random_data)
                rules_sorted.append(f"R{rule}\n({cls})")
                sigs_sorted.append(n_sig)
                colors_sorted.append(class_colors[cls])

    bars = ax.bar(range(len(rules_sorted)), sigs_sorted, color=colors_sorted, alpha=0.85)
    ax.set_xticks(range(len(rules_sorted)))
    ax.set_xticklabels(rules_sorted, fontsize=7)
    ax.set_ylabel('Sig metrics vs random', fontsize=9)
    ax.set_title('D1: Detection — significant metrics vs random noise', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='#444444', linewidth=0.5)

    # Row 2: D2 — class pairwise matrix
    ax = _dark_ax(fig.add_subplot(gs[2, :2]))
    classes = ['I', 'II', 'III', 'IV']
    mat = np.zeros((4, 4))
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if (c1, c2) in pair_matrix:
                mat[i, j] = pair_matrix[(c1, c2)][0]
                mat[j, i] = pair_matrix[(c1, c2)][0]
    im = ax.imshow(mat, cmap='magma', interpolation='nearest')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    labels = [f"Class {c}" for c in classes]
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(4):
        for j in range(4):
            if i != j:
                ax.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                       fontsize=12, fontweight='bold',
                       color='white' if mat[i,j] > mat.max()/2 else '#aaa')
    ax.set_title('D2: Class pairwise (sig metrics)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2 right: D3 summary — within-class
    ax = _dark_ax(fig.add_subplot(gs[2, 2:]))
    within_data = []
    within_labels = []
    within_colors_list = []
    for cls_name in ['I', 'II', 'III', 'IV']:
        rules = WOLFRAM_CLASSES[cls_name]
        for i, r1 in enumerate(rules):
            for r2 in rules[i+1:]:
                n1, n2 = f"R{r1}", f"R{r2}"
                if n1 in all_data and n2 in all_data:
                    n_sig, _ = compare(all_data[n1], all_data[n2])
                    within_data.append(n_sig)
                    within_labels.append(f"R{r1}↔R{r2}")
                    within_colors_list.append(class_colors[cls_name])

    ax.barh(range(len(within_data)), within_data, color=within_colors_list, alpha=0.85)
    ax.set_yticks(range(len(within_data)))
    ax.set_yticklabels(within_labels, fontsize=7)
    ax.set_xlabel('Sig metrics', fontsize=9)
    ax.set_title('D3: Within-class discrimination', fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    # Row 3: D4 — scale dependence
    ax = _dark_ax(fig.add_subplot(gs[3, :2]))
    scales = [32, 64, 128, 256]
    rule_colors = {30: '#E91E63', 90: '#4CAF50', 110: '#2196F3'}
    for rule in [30, 90, 110]:
        sigs_at_scale = []
        for size in scales:
            fields, random_fields = [], []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(42 + trial)
                fields.append(run_elementary_ca(rule, size, size, rng))
                random_fields.append(np.random.default_rng(9000 + trial).random((size, size)))
            data = collect_metrics(analyzer, fields)
            rand = collect_metrics(analyzer, random_fields)
            n_sig, _ = compare(data, rand)
            sigs_at_scale.append(n_sig)
        ax.plot(scales, sigs_at_scale, 'o-', color=rule_colors[rule],
                label=f'Rule {rule}', linewidth=2, markersize=6)
    ax.set_xlabel('Grid size', fontsize=9)
    ax.set_ylabel('Sig metrics vs random', fontsize=9)
    ax.set_title('D4: Scale dependence', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # Row 3 right: D5 — temporal evolution
    ax = _dark_ax(fig.add_subplot(gs[3, 2:]))
    heights = sorted(evo_data[30].keys())
    for rule in [30, 90, 110]:
        if rule in evo_data:
            sigs_vs_base = [0]
            for h in heights[1:]:
                n_sig, _ = compare(evo_data[rule][h], evo_data[rule][heights[0]])
                sigs_vs_base.append(n_sig)
            ax.plot(heights, sigs_vs_base, 'o-', color=rule_colors[rule],
                    label=f'Rule {rule}', linewidth=2, markersize=6)
    ax.set_xlabel('Height (time steps)', fontsize=9)
    ax.set_ylabel('Sig metrics vs h=32', fontsize=9)
    ax.set_title('D5: Temporal evolution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    fig.suptitle('Elementary Cellular Automata: Spatial Geometry of Spacetime Diagrams',
                 fontsize=14, fontweight='bold', color='white', y=0.995)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'figures', 'elementary_ca.png'),
                dpi=180, bbox_inches='tight', facecolor='#181818')
    print("  Saved elementary_ca.png")
    plt.close(fig)


# ========================================================================
#  Main
# ========================================================================
if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    all_data, random_data = direction_1(analyzer)
    pair_matrix = direction_2(all_data)
    direction_3(all_data)
    direction_4(analyzer)
    evo_data = direction_5(analyzer)
    make_figure(all_data, random_data, pair_matrix, evo_data, analyzer)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    for cls in ['I', 'II', 'III', 'IV']:
        rules = WOLFRAM_CLASSES[cls]
        sigs = []
        for r in rules:
            n_sig, _ = compare(all_data[f"R{r}"], random_data)
            sigs.append(n_sig)
        print(f"  Class {cls}: {np.mean(sigs):.0f} avg sig vs random ({sigs})")
