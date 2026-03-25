#!/usr/bin/env python3
"""
Investigation: ECG Arrhythmia Detection via Geometric Structure
================================================================

Can the exotic geometry framework detect cardiac arrhythmias quickly and
reliably, using geometric structure alone — no ML training, no ECG-specific
feature engineering?

Data:
  - MIT-BIH Arrhythmia Database (Kaggle preprocessed): 87,554 beats
  - 5 classes: Normal (N), Supraventricular (S), Ventricular (V), Fusion (F), Unknown (Q)
  - 187 samples per beat, normalized [0,1]
  - PTB Diagnostic ECG: 4,046 normal, 10,506 abnormal beats

Directions:
  D1: Detection speed — how few beats for reliable Normal vs Ventricular separation?
  D2: All-pairs discrimination — which beat types are geometrically distinguishable?
  D3: Top geometries — which metrics carry the cardiac signal?
  D4: Single-metric detection — can one geometric number classify?
  D5: PTB validation — does it generalize to a different dataset?
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import hashlib
import numpy as np
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
MITBIH_TRAIN = os.path.join(_ROOT, 'data', 'ecg', 'mitbih_train.csv')
MITBIH_TEST = os.path.join(_ROOT, 'data', 'ecg', 'mitbih_test.csv')
PTB_NORMAL = os.path.join(_ROOT, 'data', 'ecg', 'ptbdb_normal.csv')
PTB_ABNORMAL = os.path.join(_ROOT, 'data', 'ecg', 'ptbdb_abnormal.csv')
CACHE_DIR = os.path.join(_ROOT, 'data', 'ecg', '.cache')
FIG_DIR = os.path.join(_ROOT, 'figures')

# ── Config ────────────────────────────────────────────────────
BEAT_LEN = 187
N_SEQ = 40             # sequences per class per condition
ALPHA = 0.05
CLASS_NAMES = {0: 'Normal', 1: 'Supraventr.', 2: 'Ventricular', 3: 'Fusion'}
CLASS_SHORT = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}
SPEED_BEATS = [1, 2, 3, 5, 8, 11]  # beats per sequence for speed curve


# ── Helpers ───────────────────────────────────────────────────
def to_uint8(arr):
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        return np.full(len(arr), 128, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def make_sequences(pool, rng, n_seq, n_beats):
    """Concatenate n_beats random beats from pool, encode to uint8."""
    seqs = []
    for _ in range(n_seq):
        idx = rng.choice(len(pool), size=n_beats, replace=True)
        concat = np.concatenate([pool[i] for i in idx])
        seqs.append(to_uint8(concat))
    return seqs


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                 / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 1e-15 else 0.0


def _fw_hash():
    p = os.path.join(_ROOT, 'exotic_geometry_framework.py')
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def analyze(data, analyzer, metric_names, cache_key, fw_hash):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"{cache_key}_{fw_hash}.npz")
    if os.path.exists(cp):
        return np.load(cp)['profile']
    result = analyzer.analyze(data)
    profile = np.full(len(metric_names), np.nan)
    for r in result.results:
        for mn, mv in r.metrics.items():
            fn = f"{r.geometry_name}:{mn}"
            if fn in metric_names:
                v = float(mv)
                if np.isfinite(v):
                    profile[metric_names.index(fn)] = v
    np.savez_compressed(cp, profile=profile)
    return profile


def compare(profiles_a, profiles_b, metric_names):
    bonf = ALPHA / len(metric_names)
    out = []
    for i, name in enumerate(metric_names):
        av = profiles_a[:, i][~np.isnan(profiles_a[:, i])]
        bv = profiles_b[:, i][~np.isnan(profiles_b[:, i])]
        if len(av) < 3 or len(bv) < 3:
            continue
        _, p = stats.ttest_ind(av, bv, equal_var=False)
        d = cohens_d(av, bv)
        out.append({'metric': name, 'd': d, 'p': p,
                    'sig': p < bonf and abs(d) > 0.8})
    out.sort(key=lambda r: abs(r['d']), reverse=True)
    return out


def nearest_centroid_accuracy(profiles_a, profiles_b, metric_idx):
    """Leave-one-out nearest-centroid accuracy using selected metrics."""
    all_p = np.vstack([profiles_a[:, metric_idx], profiles_b[:, metric_idx]])
    labels = np.array([0] * len(profiles_a) + [1] * len(profiles_b))
    correct = 0
    for i in range(len(all_p)):
        # Leave one out
        mask = np.ones(len(all_p), dtype=bool)
        mask[i] = False
        train_p = all_p[mask]
        train_l = labels[mask]
        centroid_a = np.nanmean(train_p[train_l == 0], axis=0)
        centroid_b = np.nanmean(train_p[train_l == 1], axis=0)
        point = all_p[i]
        # Handle NaNs
        valid = ~np.isnan(point) & ~np.isnan(centroid_a) & ~np.isnan(centroid_b)
        if valid.sum() == 0:
            continue
        da = np.sqrt(np.sum((point[valid] - centroid_a[valid]) ** 2))
        db = np.sqrt(np.sum((point[valid] - centroid_b[valid]) ** 2))
        pred = 0 if da < db else 1
        if pred == labels[i]:
            correct += 1
    return correct / len(all_p)


# ── Figure ────────────────────────────────────────────────────
def make_figure(speed_data, allpairs_matrix, top_metrics, single_metric_data,
                class_labels):
    bg, fg, dim = '#0d1117', '#c9d1d9', '#8b949e'

    fig = plt.figure(figsize=(18, 10), facecolor=bg)
    fig.suptitle('ECG Arrhythmia Detection via Geometric Structure',
                 color=fg, fontsize=16, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             'MIT-BIH Arrhythmia Database · No ML training · No ECG-specific features',
             color=dim, fontsize=10, ha='center')

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                          left=0.06, right=0.97, top=0.88, bottom=0.08)

    def style(ax):
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=9)
        for s in ax.spines.values():
            s.set_color('#30363d')
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)

    # P1: Detection speed curve (N vs V)
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    beats = [s['n_beats'] for s in speed_data]
    n_sig = [s['n_sig'] for s in speed_data]
    top_d = [s['top_d'] for s in speed_data]
    ax.bar(range(len(beats)), n_sig, color='#58a6ff', width=0.6)
    ax.set_xticks(range(len(beats)))
    ax.set_xticklabels([str(b) for b in beats])
    ax.set_xlabel('Heartbeats concatenated')
    ax.set_ylabel('Significant metrics')
    ax.set_title('Detection Speed (Normal vs Ventricular)')
    for i, (ns, td) in enumerate(zip(n_sig, top_d)):
        ax.text(i, ns + 1, f'd={td:.1f}', ha='center', color=dim, fontsize=7)

    # P2: Detection speed — accuracy curve
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    acc_1 = [s['acc_1'] for s in speed_data]
    acc_5 = [s['acc_5'] for s in speed_data]
    ax.plot(beats, [a * 100 for a in acc_1], 'o-', color='#f0883e',
            label='Top-1 metric', linewidth=2, markersize=6)
    ax.plot(beats, [a * 100 for a in acc_5], 's-', color='#58a6ff',
            label='Top-5 metrics', linewidth=2, markersize=6)
    ax.axhline(95, color='#f85149', ls='--', lw=0.8, alpha=0.5)
    ax.text(beats[0], 96, '95%', color='#f85149', fontsize=8)
    ax.set_xlabel('Heartbeats concatenated')
    ax.set_ylabel('LOO Accuracy (%)')
    ax.set_title('Classification Accuracy vs Data')
    ax.set_ylim(40, 102)
    ax.legend(fontsize=8, facecolor=bg, edgecolor='#30363d', labelcolor=fg)

    # P3: All-pairs discrimination matrix
    ax = fig.add_subplot(gs[0, 2])
    style(ax)
    n_cls = len(class_labels)
    im = ax.imshow(allpairs_matrix, cmap='YlOrRd', vmin=0,
                   vmax=max(allpairs_matrix.max(), 1), aspect='equal')
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(class_labels, fontsize=9)
    for i in range(n_cls):
        for j in range(n_cls):
            v = allpairs_matrix[i, j]
            if i != j:
                ax.text(j, i, str(int(v)), ha='center', va='center',
                        color='white' if v > allpairs_matrix.max() * 0.6 else fg,
                        fontsize=11, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center', color=dim, fontsize=11)
    ax.set_title('Significant Metrics (all pairs)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # P4: Top discriminating metrics (N vs V)
    ax = fig.add_subplot(gs[1, 0:2])
    style(ax)
    top = top_metrics[:20]
    if top:
        nm = [r['metric'].split(':')[-1][:25] for r in top]
        ds = [r['d'] for r in top]
        cl = ['#f85149' if r['sig'] else dim for r in top]
        ax.barh(range(len(nm)), ds, color=cl, height=0.7)
        ax.set_yticks(range(len(nm)))
        ax.set_yticklabels(nm, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='#30363d', lw=0.5)
        for v in [0.8, -0.8]:
            ax.axvline(v, color='#f85149', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel("Cohen's d (Normal vs Ventricular)")
    ax.set_title('Top 20 Discriminating Metrics')

    # P5: Single metric distribution
    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    if single_metric_data:
        nm = single_metric_data['name'].split(':')[-1]
        nv = single_metric_data['normal']
        vv = single_metric_data['ventricular']
        parts_n = ax.violinplot([nv], positions=[0], showmedians=True, widths=0.6)
        parts_v = ax.violinplot([vv], positions=[1], showmedians=True, widths=0.6)
        for pc in parts_n['bodies']:
            pc.set_facecolor('#58a6ff')
            pc.set_alpha(0.7)
        for pc in parts_v['bodies']:
            pc.set_facecolor('#f85149')
            pc.set_alpha(0.7)
        for key in ['cmins', 'cmaxes', 'cbars', 'cmedians']:
            if key in parts_n:
                parts_n[key].set_color(fg)
            if key in parts_v:
                parts_v[key].set_color(fg)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Ventricular'], fontsize=10)
        ax.set_title(f'Best Single Metric: {nm}')
        d = single_metric_data['d']
        ax.text(0.5, 0.95, f"d = {d:+.1f}", transform=ax.transAxes,
                ha='center', va='top', color=fg, fontsize=12, fontweight='bold')

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'ecg_arrhythmia.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("ECG Arrhythmia Detection Investigation")
    print("=" * 50)

    # Framework init
    print("Initializing framework...")
    analyzer = GeometryAnalyzer().add_all_geometries()
    dummy = analyzer.analyze(np.zeros(100, dtype=np.uint8))
    metric_names = []
    for r in dummy.results:
        for mn in sorted(r.metrics.keys()):
            metric_names.append(f"{r.geometry_name}:{mn}")
    n_metrics = len(metric_names)
    fw = _fw_hash()
    print(f"  {n_metrics} metrics, framework hash {fw}")

    # Load MIT-BIH
    print("Loading MIT-BIH...")
    raw = np.loadtxt(MITBIH_TRAIN, delimiter=',')
    signals = raw[:, :-1]
    labels = raw[:, -1].astype(int)
    pools = {c: signals[labels == c] for c in CLASS_NAMES}
    for c, name in CLASS_NAMES.items():
        print(f"  {name}: {len(pools[c])} beats")

    rng = np.random.default_rng(42)

    # ── D1: Detection Speed (Normal vs Ventricular) ───────
    print("\n--- D1: Detection Speed (Normal vs Ventricular) ---")
    speed_data = []

    for nb in SPEED_BEATS:
        seq_len = nb * BEAT_LEN
        print(f"\n  {nb} beat(s) = {seq_len} samples ({seq_len/360:.1f}s at 360Hz)")

        norm_seqs = make_sequences(pools[0], rng, N_SEQ, nb)
        vent_seqs = make_sequences(pools[2], rng, N_SEQ, nb)

        t0 = time.time()
        norm_p = np.array([analyze(s, analyzer, metric_names,
                                   f"mitbih_N_b{nb}_s{i}", fw)
                           for i, s in enumerate(norm_seqs)])
        vent_p = np.array([analyze(s, analyzer, metric_names,
                                   f"mitbih_V_b{nb}_s{i}", fw)
                           for i, s in enumerate(vent_seqs)])
        dt = time.time() - t0

        res = compare(norm_p, vent_p, metric_names)
        n_sig = sum(r['sig'] for r in res)
        top_d = abs(res[0]['d']) if res else 0

        # Accuracy with top-1 and top-5 metrics
        top1_idx = [metric_names.index(res[0]['metric'])] if res else [0]
        top5_idx = [metric_names.index(res[i]['metric'])
                    for i in range(min(5, len(res)))]
        acc_1 = nearest_centroid_accuracy(norm_p, vent_p, top1_idx)
        acc_5 = nearest_centroid_accuracy(norm_p, vent_p, top5_idx)

        speed_data.append({
            'n_beats': nb, 'n_sig': n_sig, 'top_d': top_d,
            'acc_1': acc_1, 'acc_5': acc_5, 'time': dt,
        })
        print(f"    {n_sig} significant, top |d|={top_d:.2f}, "
              f"acc1={acc_1:.1%}, acc5={acc_5:.1%}, {dt:.1f}s")

    # ── D2: All-pairs discrimination (11 beats) ──────────
    print("\n--- D2: All-pairs Discrimination (11 beats) ---")
    class_ids = sorted(CLASS_NAMES.keys())
    n_cls = len(class_ids)
    allpairs_matrix = np.zeros((n_cls, n_cls))

    # Generate sequences for all classes (reuse N and V from speed curve at nb=11)
    class_profiles = {}
    for c in class_ids:
        ck_prefix = f"mitbih_{CLASS_SHORT[c]}_b11"
        # Check if already computed (N and V from speed curve)
        seqs = make_sequences(pools[c], rng, N_SEQ, 11)
        profiles = np.array([analyze(s, analyzer, metric_names,
                                     f"{ck_prefix}_s{i}", fw)
                             for i, s in enumerate(seqs)])
        class_profiles[c] = profiles
        print(f"  {CLASS_NAMES[c]}: {len(profiles)} sequences analyzed")

    for i, ci in enumerate(class_ids):
        for j, cj in enumerate(class_ids):
            if i == j:
                continue
            res = compare(class_profiles[ci], class_profiles[cj], metric_names)
            n_sig = sum(r['sig'] for r in res)
            allpairs_matrix[i, j] = n_sig
            if n_sig > 0:
                print(f"  {CLASS_SHORT[ci]} vs {CLASS_SHORT[cj]}: "
                      f"{n_sig} significant, top |d|={abs(res[0]['d']):.2f}")

    # ── D3: Top metrics (N vs V at 11 beats) ─────────────
    print("\n--- D3: Top Metrics (Normal vs Ventricular, 11 beats) ---")
    nv_res = compare(class_profiles[0], class_profiles[2], metric_names)
    n_sig_nv = sum(r['sig'] for r in nv_res)
    print(f"  {n_sig_nv} significant / {n_metrics}")
    for r in nv_res[:15]:
        s = " ***" if r['sig'] else ""
        print(f"    {r['metric']:50s}  d={r['d']:+.3f}  p={r['p']:.2e}{s}")

    # ── D4: Single metric distribution ────────────────────
    top_metric = nv_res[0]['metric'] if nv_res else None
    single_metric_data = None
    if top_metric:
        idx = metric_names.index(top_metric)
        nv_norm = class_profiles[0][:, idx]
        nv_vent = class_profiles[2][:, idx]
        nv_norm = nv_norm[~np.isnan(nv_norm)]
        nv_vent = nv_vent[~np.isnan(nv_vent)]
        single_metric_data = {
            'name': top_metric,
            'normal': nv_norm,
            'ventricular': nv_vent,
            'd': nv_res[0]['d'],
        }
        print(f"\n  Best metric: {top_metric}")
        print(f"    Normal:      mean={nv_norm.mean():.4f} ± {nv_norm.std():.4f}")
        print(f"    Ventricular: mean={nv_vent.mean():.4f} ± {nv_vent.std():.4f}")
        print(f"    Cohen's d:   {nv_res[0]['d']:+.3f}")

    # ── D5: PTB validation ────────────────────────────────
    print("\n--- D5: PTB Validation (Normal vs Abnormal) ---")
    ptb_norm = np.loadtxt(PTB_NORMAL, delimiter=',')[:, :-1]
    ptb_abnorm = np.loadtxt(PTB_ABNORMAL, delimiter=',')[:, :-1]
    print(f"  PTB Normal: {len(ptb_norm)}, Abnormal: {len(ptb_abnorm)}")

    ptb_norm_seqs = make_sequences(ptb_norm, rng, N_SEQ, 11)
    ptb_abnorm_seqs = make_sequences(ptb_abnorm, rng, N_SEQ, 11)

    ptb_norm_p = np.array([analyze(s, analyzer, metric_names,
                                   f"ptb_N_b11_s{i}", fw)
                           for i, s in enumerate(ptb_norm_seqs)])
    ptb_abnorm_p = np.array([analyze(s, analyzer, metric_names,
                                     f"ptb_A_b11_s{i}", fw)
                             for i, s in enumerate(ptb_abnorm_seqs)])

    ptb_res = compare(ptb_norm_p, ptb_abnorm_p, metric_names)
    n_sig_ptb = sum(r['sig'] for r in ptb_res)
    print(f"  {n_sig_ptb} significant / {n_metrics}")
    for r in ptb_res[:10]:
        s = " ***" if r['sig'] else ""
        print(f"    {r['metric']:50s}  d={r['d']:+.3f}  p={r['p']:.2e}{s}")

    # PTB accuracy
    if ptb_res:
        top5_idx = [metric_names.index(ptb_res[i]['metric'])
                    for i in range(min(5, len(ptb_res)))]
        ptb_acc = nearest_centroid_accuracy(ptb_norm_p, ptb_abnorm_p, top5_idx)
        print(f"  Top-5 LOO accuracy: {ptb_acc:.1%}")

    # ── Figure ────────────────────────────────────────────
    print("\n--- Figure ---")
    class_labels = [CLASS_NAMES[c] for c in class_ids]
    make_figure(speed_data, allpairs_matrix, nv_res, single_metric_data,
                class_labels)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"  Normal vs Ventricular: {n_sig_nv} significant / {n_metrics}")
    if speed_data:
        fastest = next((s for s in speed_data if s['acc_5'] >= 0.95), None)
        if fastest:
            print(f"  95% accuracy reached at {fastest['n_beats']} beat(s) "
                  f"({fastest['n_beats'] * BEAT_LEN / 360:.1f}s of ECG)")
        print(f"  Best single-metric accuracy: "
              f"{max(s['acc_1'] for s in speed_data):.1%}")
    print(f"  PTB Normal vs Abnormal: {n_sig_ptb} significant / {n_metrics}")
    print(f"  All-pairs range: {int(allpairs_matrix[allpairs_matrix > 0].min())}"
          f"-{int(allpairs_matrix.max())} significant metrics")


if __name__ == '__main__':
    main()
