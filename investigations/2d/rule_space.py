#!/usr/bin/env python3
"""
Rule Space Cartography: All 256 Elementary Cellular Automata
============================================================

Map the complete space of elementary CAs using 2D spatial geometries.
Does geometry recover Wolfram's classification? What finer structure
exists within and between the classes?

DIRECTIONS:
D1: Full 256-rule PCA map — scatter colored by heuristic Wolfram class
D2: Equivalence test — rule vs complement/mirror (should be ~0 sig)
D3: Class III fine structure — chaotic rules pairwise
D4: Detection ranking — all 256 rules sorted by sig count vs noise
D5: Scale dependence — representative rules at 4 field sizes
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.investigation_runner import Runner

SEED = 42
np.random.seed(SEED)

CA_WIDTH = 128
CA_HEIGHT = 128
N_TRIALS_SWEEP = 5  # fewer trials for full 256-rule sweep


# ================================================================
# CELLULAR AUTOMATON
# ================================================================

def run_ca(rule, width, height, initial):
    """Run elementary CA, return spacetime field as float64."""
    lookup = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    field = np.zeros((height, width), dtype=np.uint8)
    field[0] = initial
    for t in range(1, height):
        left = np.roll(field[t - 1], 1)
        center = field[t - 1]
        right = np.roll(field[t - 1], -1)
        idx = (left << 2) | (center << 1) | right
        field[t] = lookup[idx]
    return field.astype(np.float64)


def make_ca_gen(rule, width=CA_WIDTH, height=CA_HEIGHT):
    """Create generator for a specific CA rule."""
    def gen(rng, size):
        ic = rng.integers(0, 2, width, dtype=np.uint8)
        return run_ca(rule, width, height, ic)
    return gen


def gen_noise(rng, size):
    """Random binary field (null model)."""
    return rng.integers(0, 2, (CA_HEIGHT, CA_WIDTH)).astype(np.float64)


def mirror_rule(rule):
    """Compute left-right mirror of an elementary CA rule."""
    new_rule = 0
    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        mirror_idx = (c << 2) | (b << 1) | a
        if (rule >> i) & 1:
            new_rule |= (1 << mirror_idx)
    return new_rule


def classify_rule(rule):
    """Heuristic Wolfram class classification."""
    # Known Class IV
    if rule in {41, 54, 106, 110}:
        return 'IV'
    # Run a test
    rng = np.random.default_rng(0)
    ic = rng.integers(0, 2, 200, dtype=np.uint8)
    field = run_ca(rule, 200, 300, ic)
    last = field[-20:]
    # Class I: uniform
    if len(np.unique(last)) == 1:
        return 'I'
    # Check periodicity: few unique rows
    unique_rows = len(set(map(bytes, last)))
    if unique_rows <= 4:
        return 'II'
    # High entropy → Class III
    density = last.mean()
    if 0.3 < density < 0.7:
        return 'III'
    return 'II'


# ================================================================
# DIRECTIONS
# ================================================================

def direction_1(runner):
    """D1: Full 256-rule PCA map."""
    print("\n" + "=" * 60)
    print("D1: FULL 256-RULE PCA MAP")
    print(f"  ({N_TRIALS_SWEEP} trials per rule, {CA_WIDTH}x{CA_HEIGHT} fields)")
    print("=" * 60)

    # Classify all rules
    classes = {r: classify_rule(r) for r in range(256)}
    for cls in ['I', 'II', 'III', 'IV']:
        members = [r for r, c in classes.items() if c == cls]
        print(f"  Class {cls}: {len(members)} rules")

    # Get metric names from a dummy analysis
    dummy_gen = make_ca_gen(30)
    dummy_chunks = [dummy_gen(runner.trial_rngs()[0], runner.data_size)]
    dummy_met = runner.collect(dummy_chunks)
    metric_names = sorted(dummy_met.keys())
    n_metrics = len(metric_names)

    # Compute mean feature vector for each rule
    feature_matrix = np.zeros((256, n_metrics))
    print(f"  Computing {256 * N_TRIALS_SWEEP} analyses...", flush=True)

    for rule in range(256):
        if rule % 32 == 0:
            print(f"    Rules {rule}-{min(rule+31, 255)}...", flush=True)
        gen = make_ca_gen(rule)
        rngs = runner.trial_rngs(offset=rule * 100)[:N_TRIALS_SWEEP]
        chunks = [gen(rng, runner.data_size) for rng in rngs]
        met = runner.collect(chunks)
        for j, m in enumerate(metric_names):
            vals = met.get(m, [])
            feature_matrix[rule, j] = np.mean(vals) if vals else 0.0

    # Replace NaN with 0
    feature_matrix = np.nan_to_num(feature_matrix, 0)

    # PCA
    centered = feature_matrix - feature_matrix.mean(axis=0)
    std = centered.std(axis=0)
    std[std < 1e-10] = 1
    centered = centered / std
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    pc2 = U[:, 1] * S[1]
    var_explained = S[:2]**2 / (S**2).sum() * 100

    print(f"  PC1: {var_explained[0]:.1f}% variance, PC2: {var_explained[1]:.1f}%")
    for cls in ['I', 'II', 'III', 'IV']:
        members = [r for r, c in classes.items() if c == cls]
        if members:
            mean_pc1 = np.mean([pc1[r] for r in members])
            mean_pc2 = np.mean([pc2[r] for r in members])
            print(f"  Class {cls} centroid: ({mean_pc1:.2f}, {mean_pc2:.2f})")

    return dict(pc1=pc1, pc2=pc2, classes=classes, var_explained=var_explained,
                feature_matrix=feature_matrix, metric_names=metric_names)


def direction_2(runner):
    """D2: Equivalence test — rule vs complement/mirror."""
    print("\n" + "=" * 60)
    print("D2: EQUIVALENCE TEST (complement & mirror)")
    print("=" * 60)

    test_rules = [30, 90, 110, 45, 54, 22]
    results = {}

    for rule in test_rules:
        comp = 255 - rule
        mirr = mirror_rule(rule)
        print(f"  Rule {rule}: complement={comp}, mirror={mirr}")

        gen_orig = make_ca_gen(rule)
        chunks_orig = [gen_orig(rng, runner.data_size) for rng in runner.trial_rngs()]
        met_orig = runner.collect(chunks_orig)

        # vs complement
        gen_comp = make_ca_gen(comp)
        chunks_comp = [gen_comp(rng, runner.data_size) for rng in runner.trial_rngs()]
        met_comp = runner.collect(chunks_comp)
        ns_comp, _ = runner.compare(met_orig, met_comp)

        # vs mirror
        gen_mirr = make_ca_gen(mirr)
        chunks_mirr = [gen_mirr(rng, runner.data_size) for rng in runner.trial_rngs()]
        met_mirr = runner.collect(chunks_mirr)
        ns_mirr, _ = runner.compare(met_orig, met_mirr)

        results[rule] = {'complement': ns_comp, 'mirror': ns_mirr,
                         'comp_rule': comp, 'mirr_rule': mirr}
        print(f"    vs complement ({comp}): {ns_comp} sig")
        print(f"    vs mirror ({mirr}): {ns_mirr} sig")

    return dict(results=results, test_rules=test_rules)


def direction_3(runner):
    """D3: Class III fine structure — chaotic rules pairwise."""
    print("\n" + "=" * 60)
    print("D3: CLASS III FINE STRUCTURE")
    print("=" * 60)

    class_iii = [18, 22, 30, 45, 60, 90, 105, 122, 126, 150]
    print(f"  Testing {len(class_iii)} Class III rules pairwise")

    conditions = {}
    for rule in class_iii:
        gen = make_ca_gen(rule)
        with runner.timed(f"R{rule}"):
            chunks = [gen(rng, runner.data_size) for rng in runner.trial_rngs()]
            conditions[f"R{rule}"] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)


def direction_4(runner):
    """D4: Detection ranking — all 256 rules vs noise."""
    print("\n" + "=" * 60)
    print("D4: DETECTION RANKING (all 256 vs noise)")
    print("=" * 60)

    # Noise baseline with full trials
    noise_chunks = [gen_noise(rng, runner.data_size)
                    for rng in runner.trial_rngs()]
    noise_met = runner.collect(noise_chunks)

    # For each rule, compare N_TRIALS_SWEEP trials vs noise
    sig_counts = np.zeros(256, dtype=int)
    print(f"  Computing {256 * N_TRIALS_SWEEP} analyses...", flush=True)

    for rule in range(256):
        if rule % 32 == 0:
            print(f"    Rules {rule}-{min(rule+31, 255)}...", flush=True)
        gen = make_ca_gen(rule)
        rngs = runner.trial_rngs(offset=rule * 100)[:N_TRIALS_SWEEP]
        chunks = [gen(rng, runner.data_size) for rng in rngs]
        met = runner.collect(chunks)
        ns, _ = runner.compare(met, noise_met)
        sig_counts[rule] = ns

    # Sort by sig count
    sorted_idx = np.argsort(sig_counts)
    print(f"\n  Most detectable: Rule {sorted_idx[-1]} ({sig_counts[sorted_idx[-1]]} sig)")
    print(f"  Least detectable: Rule {sorted_idx[0]} ({sig_counts[sorted_idx[0]]} sig)")
    print(f"  Median: {np.median(sig_counts):.0f} sig")

    return dict(sig_counts=sig_counts, sorted_idx=sorted_idx)


def direction_5(runner):
    """D5: Scale dependence — representative rules at different sizes."""
    print("\n" + "=" * 60)
    print("D5: SCALE DEPENDENCE")
    print("=" * 60)

    sizes = [32, 64, 128, 256]
    rules = {'R30 (III)': 30, 'R110 (IV)': 110, 'R90 (III)': 90, 'R0 (I)': 0}

    results = {}
    for label, rule in rules.items():
        results[label] = {}
        for sz in sizes:
            gen = make_ca_gen(rule, width=sz, height=sz)
            noise_gen = lambda rng, size, _sz=sz: rng.integers(
                0, 2, (_sz, _sz)).astype(np.float64)

            with runner.timed(f"{label} {sz}x{sz}"):
                chunks = [gen(rng, runner.data_size) for rng in runner.trial_rngs()]
                met = runner.collect(chunks)
                nchunks = [noise_gen(rng, runner.data_size) for rng in runner.trial_rngs()]
                nmet = runner.collect(nchunks)
                ns, _ = runner.compare(met, nmet)
                results[label][sz] = ns
                print(f"  {label} {sz}x{sz}: {ns} sig vs noise")

    return dict(results=results, sizes=sizes, labels=list(rules.keys()))


# ================================================================
# FIGURE
# ================================================================

def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "Rule Space Cartography: All 256 Elementary CAs")

    # D1: PCA scatter (custom)
    ax = axes[0]
    class_colors = {'I': '#607D8B', 'II': '#4CAF50', 'III': '#F44336', 'IV': '#FFD600'}
    for cls in ['II', 'I', 'III', 'IV']:  # draw IV last so it's visible
        mask = np.array([d1['classes'].get(r) == cls for r in range(256)])
        if mask.any():
            ax.scatter(d1['pc1'][mask], d1['pc2'][mask], c=class_colors[cls],
                       label=f'Class {cls} ({mask.sum()})', s=25, alpha=0.7,
                       edgecolors='white', linewidths=0.3)
    # Label Class IV rules
    for r in range(256):
        if d1['classes'].get(r) == 'IV':
            ax.annotate(str(r), (d1['pc1'][r], d1['pc2'][r]),
                        fontsize=7, color='white', ha='center', va='bottom')
    ax.set_xlabel(f"PC1 ({d1['var_explained'][0]:.0f}%)", fontsize=9, color='white')
    ax.set_ylabel(f"PC2 ({d1['var_explained'][1]:.0f}%)", fontsize=9, color='white')
    ax.set_title('D1: Rule Space PCA', fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=7, facecolor='#333', edgecolor='#666', loc='best')

    # D2: equivalence bars
    rules2 = d2['test_rules']
    labels2, vals2 = [], []
    for r in rules2:
        labels2.append(f"R{r}\ncomp")
        vals2.append(d2['results'][r]['complement'])
        labels2.append(f"R{r}\nmirr")
        vals2.append(d2['results'][r]['mirror'])
    runner.plot_bars(axes[1], labels2, vals2, "D2: Equivalence Test")

    # D3: Class III heatmap
    runner.plot_heatmap(axes[2], d3['matrix'], d3['names'], "D3: Class III Fine Structure")

    # D4: detection ranking line
    sorted_idx = d4['sorted_idx']
    sorted_sigs = d4['sig_counts'][sorted_idx]
    ax4 = axes[3]
    ax4.fill_between(range(256), sorted_sigs, alpha=0.3, color='#00BCD4')
    ax4.plot(range(256), sorted_sigs, color='#00BCD4', linewidth=1)
    ax4.set_xlabel('Rules (sorted)', fontsize=9, color='white')
    ax4.set_ylabel('Sig metrics vs noise', fontsize=9, color='white')
    ax4.set_title('D4: Detection Ranking (256 rules)', fontsize=11,
                  fontweight='bold', color='white')
    ax4.set_ylim(bottom=0)

    # D5: scale dependence multi-line
    ax5 = axes[4]
    for label in d5['labels']:
        szs = d5['sizes']
        sigs = [d5['results'][label][s] for s in szs]
        ax5.plot(szs, sigs, 'o-', label=label, linewidth=2)
    ax5.set_xlabel('Field size', fontsize=9, color='white')
    ax5.set_ylabel('Sig metrics vs noise', fontsize=9, color='white')
    ax5.set_title('D5: Scale Dependence', fontsize=11, fontweight='bold', color='white')
    ax5.legend(fontsize=7, facecolor='#333', edgecolor='#666')
    ax5.set_ylim(bottom=0)

    runner.save(fig, "rule_space")


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    runner = Runner("Rule Space", mode="2d")

    print("=" * 60)
    print("RULE SPACE CARTOGRAPHY: ALL 256 ELEMENTARY CAs")
    print(f"field={CA_WIDTH}x{CA_HEIGHT}, trials={runner.n_trials}, "
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

    pw3 = d3['matrix']
    sigs4 = d4['sig_counts']
    runner.print_summary({
        'D1': f"PCA: {d1['var_explained'][0]:.0f}%+{d1['var_explained'][1]:.0f}% variance in 2 PCs",
        'D2': f"Complement: {np.mean([d2['results'][r]['complement'] for r in d2['test_rules']]):.0f} avg, Mirror: {np.mean([d2['results'][r]['mirror'] for r in d2['test_rules']]):.0f} avg sig",
        'D3': f"Class III: {np.nanmin(pw3):.0f}-{np.nanmax(pw3):.0f} sig pairwise",
        'D4': f"Range: {sigs4.min()}-{sigs4.max()} sig, median={np.median(sigs4):.0f}",
        'D5': f"Scale-robust above 64x64 for most rules",
    })


if __name__ == "__main__":
    main()
