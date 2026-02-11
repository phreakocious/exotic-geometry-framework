#!/usr/bin/env python3
"""
Investigation: Bearing Fault Diagnosis via Exotic Geometry
==========================================================

Can exotic geometry metrics distinguish bearing fault types and severities
from raw vibration signals? The CWRU bearing dataset is the gold standard
benchmark for mechanical fault diagnosis.

Data: Case Western Reserve University Bearing Data Center
  - 48 kHz drive-end (DE) accelerometer signals
  - 4 conditions: Normal, Ball fault, Inner race fault, Outer race fault
  - 3 severity levels: 0.007, 0.014, 0.021 inch fault diameter
  - ~480K samples per recording (10 seconds)

DIRECTIONS:
D1: Fault Type Detection — Normal vs Ball vs Inner vs Outer (pooled severity)
D2: Severity Grading — Within each fault type, can we separate 7/14/21 mil?
D3: Early Fault Detection — The hardest case: Normal vs 0.007" (smallest fault)
D4: Sensor Comparison — Drive-end vs fan-end: which carries more signal?
D5: Geometry Heatmap — Which geometries are the best fault detectors?
"""

import sys
import os
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.investigation_runner import Runner

# ==============================================================
# CONFIG
# ==============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'cwru', 'raw')
DATA_SIZE = 2000
N_TRIALS = 25

# File mapping: (filename, DE_key, FE_key, label, fault_type, severity)
FILES = [
    ('Time_Normal_1_098.mat', 'X098_DE_time', 'X098_FE_time', 'Normal', 'normal', 0),
    ('B007_1_123.mat', 'X123_DE_time', 'X123_FE_time', 'Ball-7', 'ball', 7),
    ('B014_1_190.mat', 'X190_DE_time', 'X190_FE_time', 'Ball-14', 'ball', 14),
    ('B021_1_227.mat', 'X227_DE_time', 'X227_FE_time', 'Ball-21', 'ball', 21),
    ('IR007_1_110.mat', 'X110_DE_time', 'X110_FE_time', 'Inner-7', 'inner', 7),
    ('IR014_1_175.mat', 'X175_DE_time', 'X175_FE_time', 'Inner-14', 'inner', 14),
    ('IR021_1_214.mat', 'X214_DE_time', 'X214_FE_time', 'Inner-21', 'inner', 21),
    ('OR007_6_1_136.mat', 'X136_DE_time', 'X136_FE_time', 'Outer-7', 'outer', 7),
    ('OR014_6_1_202.mat', 'X202_DE_time', 'X202_FE_time', 'Outer-14', 'outer', 14),
    ('OR021_6_1_239.mat', 'X239_DE_time', 'X239_FE_time', 'Outer-21', 'outer', 21),
]


# ==============================================================
# DATA LOADING
# ==============================================================

def load_signal(filename, key):
    """Load a single channel from a .mat file."""
    path = os.path.join(DATA_DIR, filename)
    m = loadmat(path)
    return m[key].flatten()


def signal_to_chunks(signal, rng, n_trials=N_TRIALS, chunk_size=DATA_SIZE):
    """Extract n_trials random non-overlapping chunks, quantized to uint8."""
    n_available = len(signal) // chunk_size
    indices = rng.choice(n_available, size=min(n_trials, n_available), replace=False)
    chunks = []
    for idx in indices:
        start = idx * chunk_size
        chunk = signal[start:start + chunk_size]
        # Normalize to [0, 255] per-chunk (preserves local dynamics)
        lo, hi = chunk.min(), chunk.max()
        if hi - lo < 1e-15:
            hi = lo + 1.0
        normalized = ((chunk - lo) / (hi - lo) * 255).astype(np.uint8)
        chunks.append(normalized)
    return chunks


def load_condition(filename, de_key, rng, n_trials=N_TRIALS):
    """Load DE signal and return chunks."""
    signal = load_signal(filename, de_key)
    return signal_to_chunks(signal, rng, n_trials)


def load_condition_fe(filename, fe_key, rng, n_trials=N_TRIALS):
    """Load FE signal and return chunks."""
    signal = load_signal(filename, fe_key)
    return signal_to_chunks(signal, rng, n_trials)


# ==============================================================
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Fault Type Detection — Normal vs Ball vs Inner vs Outer."""
    print("\n" + "=" * 60)
    print("D1: FAULT TYPE DETECTION")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Pool all severities per fault type
    type_groups = {
        'Normal': [f for f in FILES if f[3] == 'Normal'],
        'Ball': [f for f in FILES if f[4] == 'ball'],
        'Inner': [f for f in FILES if f[4] == 'inner'],
        'Outer': [f for f in FILES if f[4] == 'outer'],
    }

    metrics = {}
    for name, group in type_groups.items():
        all_chunks = []
        for fname, de_key, fe_key, label, ftype, sev in group:
            signal = load_signal(fname, de_key)
            # Take ~8 chunks per file to get 25 total per type
            n_per = max(1, N_TRIALS // len(group))
            all_chunks.extend(signal_to_chunks(signal, rng, n_per))
        # Trim to exactly N_TRIALS
        all_chunks = all_chunks[:N_TRIALS]
        metrics[name] = runner.collect(all_chunks)
        print(f"  {name}: {len(all_chunks)} chunks collected")

    matrix, names, _ = runner.compare_pairwise(metrics)
    return dict(matrix=matrix, names=names, metrics=metrics)


def direction_2(runner):
    """D2: Severity Grading — Within each fault type, 7 vs 14 vs 21."""
    print("\n" + "=" * 60)
    print("D2: SEVERITY GRADING")
    print("=" * 60)

    rng = np.random.default_rng(43)
    results = {}

    for fault_type in ['ball', 'inner', 'outer']:
        group = [f for f in FILES if f[4] == fault_type]
        metrics = {}
        for fname, de_key, fe_key, label, ftype, sev in group:
            chunks = load_condition(fname, de_key, rng)
            metrics[f"{sev}mil"] = runner.collect(chunks)

        matrix, names, _ = runner.compare_pairwise(metrics)
        results[fault_type] = dict(matrix=matrix, names=names)

        print(f"\n  {fault_type.upper()} fault severity:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"    {names[i]} vs {names[j]}: {matrix[i,j]} sig")

    return results


def direction_3(runner):
    """D3: Early Fault Detection — Normal vs smallest fault (0.007")."""
    print("\n" + "=" * 60)
    print("D3: EARLY FAULT DETECTION (Normal vs 0.007\")")
    print("=" * 60)

    rng = np.random.default_rng(44)

    normal = FILES[0]
    normal_chunks = load_condition(normal[0], normal[1], rng)
    normal_metrics = runner.collect(normal_chunks)

    results = {}
    for f in FILES:
        if f[5] == 7:  # 0.007" faults only
            chunks = load_condition(f[0], f[1], rng)
            fault_metrics = runner.collect(chunks)
            n_sig, findings = runner.compare(normal_metrics, fault_metrics)
            results[f[3]] = (n_sig, findings[:5])
            print(f"  Normal vs {f[3]}: {n_sig} sig metrics")
            if findings:
                print(f"    Best: {findings[0][0]} (d={findings[0][1]:.2f})")

    return results


def direction_4(runner):
    """D4: Sensor Comparison — DE vs FE for the same fault."""
    print("\n" + "=" * 60)
    print("D4: SENSOR COMPARISON (Drive-End vs Fan-End)")
    print("=" * 60)

    rng = np.random.default_rng(45)

    de_sig = {}
    fe_sig = {}

    for f in FILES:
        de_chunks = load_condition(f[0], f[1], rng)
        fe_chunks = load_condition_fe(f[0], f[2], np.random.default_rng(45))
        de_m = runner.collect(de_chunks)
        fe_m = runner.collect(fe_chunks)

        # Compare Normal-DE vs Fault-DE and Normal-FE vs Fault-FE
        de_sig[f[3]] = de_m
        fe_sig[f[3]] = fe_m

    # Compare each fault vs Normal on both sensors
    results = {}
    for f in FILES[1:]:  # skip Normal
        n_de, _ = runner.compare(de_sig['Normal'], de_sig[f[3]])
        n_fe, _ = runner.compare(fe_sig['Normal'], fe_sig[f[3]])
        results[f[3]] = (n_de, n_fe)
        print(f"  {f[3]:10s}: DE={n_de:3d} sig, FE={n_fe:3d} sig")

    return results


def direction_5(runner, d1_metrics):
    """D5: Geometry Heatmap — Which geometry families detect faults best?"""
    print("\n" + "=" * 60)
    print("D5: GEOMETRY HEATMAP")
    print("=" * 60)

    # For each geometry family, count how many of its metrics are significant
    # across Normal vs each fault type
    normal_m = d1_metrics['Normal']

    families = {}
    for metric_name in runner.metric_names:
        fam = metric_name.split(':')[0]
        families.setdefault(fam, []).append(metric_name)

    fam_names = sorted(families.keys())
    fault_names = ['Ball', 'Inner', 'Outer']

    # Matrix: families x fault_types, cell = fraction of metrics significant
    heatmap = np.zeros((len(fam_names), len(fault_names)))

    for j, fault in enumerate(fault_names):
        fault_m = d1_metrics[fault]
        _, findings = runner.compare(normal_m, fault_m)
        sig_metrics = {f[0] for f in findings}

        for i, fam in enumerate(fam_names):
            n_fam = len(families[fam])
            n_sig = sum(1 for m in families[fam] if m in sig_metrics)
            heatmap[i, j] = n_sig / n_fam if n_fam > 0 else 0

    # Print top families
    mean_rate = np.mean(heatmap, axis=1)
    order = np.argsort(-mean_rate)
    print(f"\n  Top 10 fault-detecting geometries (fraction of metrics sig):")
    for rank, idx in enumerate(order[:10]):
        rates = ' '.join(f'{heatmap[idx, j]:.0%}' for j in range(3))
        print(f"    {rank+1}. {fam_names[idx]:35s} {rates}  (mean {mean_rate[idx]:.0%})")

    return dict(heatmap=heatmap, fam_names=fam_names, fault_names=fault_names,
                order=order)


# ==============================================================
# FIGURE
# ==============================================================

def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5,
        "Bearing Fault Diagnosis via Exotic Geometry (CWRU Dataset)")

    # D1: Fault type confusion matrix
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'],
                        "D1: Fault Type Detection")

    # D2: Severity grading bars
    sev_data = []
    sev_labels = []
    for ft in ['ball', 'inner', 'outer']:
        m = d2[ft]['matrix']
        n = d2[ft]['names']
        for i in range(len(n)):
            for j in range(i + 1, len(n)):
                sev_labels.append(f"{ft[0].upper()}:{n[i]}v{n[j]}")
                sev_data.append(m[i, j])
    runner.plot_bars(axes[1], sev_labels, sev_data,
                     "D2: Severity Grading (within fault type)")

    # D3: Early detection bars
    d3_labels = list(d3.keys())
    d3_vals = [d3[k][0] for k in d3_labels]
    runner.plot_bars(axes[2], d3_labels, d3_vals,
                     "D3: Early Detection (Normal vs 0.007\")")

    # D4: DE vs FE comparison
    d4_labels = list(d4.keys())
    de_vals = [d4[k][0] for k in d4_labels]
    fe_vals = [d4[k][1] for k in d4_labels]
    x = np.arange(len(d4_labels))
    axes[3].bar(x - 0.2, de_vals, 0.35, label='Drive End', color='#e74c3c',
                alpha=0.85)
    axes[3].bar(x + 0.2, fe_vals, 0.35, label='Fan End', color='#3498db',
                alpha=0.85)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(d4_labels, rotation=45, ha='right', fontsize=6)
    axes[3].set_ylabel('Sig. metrics vs Normal', fontsize=8, color='#cccccc')
    axes[3].set_title("D4: Drive End vs Fan End", fontsize=10,
                      fontweight='bold', color='white')
    axes[3].legend(fontsize=7, facecolor='#222222', edgecolor='#444444',
                   labelcolor='#cccccc')

    # D5: Geometry heatmap (top 15)
    top_idx = d5['order'][:15]
    sub_heatmap = d5['heatmap'][top_idx]
    sub_names = [d5['fam_names'][i] for i in top_idx]
    im = axes[4].imshow(sub_heatmap, cmap='YlOrRd', aspect='auto',
                        vmin=0, vmax=1)
    axes[4].set_yticks(range(len(sub_names)))
    axes[4].set_yticklabels([n[:25] for n in sub_names], fontsize=6)
    axes[4].set_xticks(range(3))
    axes[4].set_xticklabels(d5['fault_names'], fontsize=8)
    axes[4].set_title("D5: Best Fault-Detecting Geometries",
                      fontsize=10, fontweight='bold', color='white')
    for i in range(len(sub_names)):
        for j in range(3):
            val = sub_heatmap[i, j]
            axes[4].text(j, i, f'{val:.0%}', ha='center', va='center',
                         fontsize=6, color='white' if val > 0.5 else '#cccccc')

    runner.save(fig, "bearing_fault")


# ==============================================================
# MAIN
# ==============================================================

def main():
    runner = Runner("Bearing Fault", mode="1d")

    d1 = direction_1(runner)
    d2 = direction_2(runner)
    d3 = direction_3(runner)
    d4 = direction_4(runner)
    d5 = direction_5(runner, d1['metrics'])

    make_figure(runner, d1, d2, d3, d4, d5)

    # Summary
    d1m = d1['matrix']
    n = len(d1['names'])
    mean_d1 = np.mean(d1m[np.triu_indices(n, k=1)])

    early_best = max(d3.values(), key=lambda x: x[0])
    early_name = [k for k, v in d3.items() if v is early_best][0]

    runner.print_summary({
        'D1 Mean Pairwise': f'{mean_d1:.0f} sig (fault type detection)',
        'D3 Best Early': f'{early_name} = {early_best[0]} sig (0.007" fault)',
        'D4 Sensor': 'DE > FE' if sum(v[0] for v in d4.values()) >
                     sum(v[1] for v in d4.values()) else 'FE > DE',
    })


if __name__ == "__main__":
    main()
