#!/usr/bin/env python3
"""
Text Geometry Investigation: Author Fingerprinting & N-gram Hierarchy
=====================================================================

Can exotic geometries fingerprint writing style at the byte level?
At what n-gram order does synthetic text become indistinguishable from real?

DIRECTIONS:
D1: Author taxonomy — 4 Gutenberg authors, pairwise distinguishability
D2: Sequential structure — real vs shuffled for each author
D3: N-gram hierarchy — at what order does synthetic text fool geometry?
D4: Author-dependent detection — who's hardest to mimic?
D5: Delay embedding — does tau amplify text structure?

Data: Project Gutenberg public domain texts (downloaded and cached).
"""

import sys
import time
import warnings
import numpy as np
from pathlib import Path
from collections import Counter
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exotic_geometry_framework import GeometryAnalyzer, delay_embed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================
# CONFIG
# ==============================================================
DATA_SIZE = 2000
N_TRIALS = 25
ALPHA = 0.05
SEED = 42
FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "gutenberg"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

# Gutenberg: (display_name, gutenberg_id)
AUTHORS = {
    'carroll':  ('Lewis Carroll — Alice in Wonderland', 11),
    'austen':   ('Jane Austen — Pride and Prejudice', 1342),
    'melville': ('Herman Melville — Moby Dick', 2701),
    'doyle':    ('Arthur Conan Doyle — Sherlock Holmes', 1661),
}

NGRAM_ORDERS = [0, 1, 2, 3, 4, 6, 8]

# Discover metric names
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.randint(0, 256, DATA_SIZE, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF_ALPHA = ALPHA / N_METRICS
print(f"Geometries: {len(_dummy.results)}, metrics: {N_METRICS}")


# ==============================================================
# TEXT LOADING
# ==============================================================
def download_gutenberg(name, gid):
    """Download and cache a Gutenberg text."""
    cache = DATA_DIR / f"{name}.txt"
    if cache.exists():
        return cache.read_text(encoding='utf-8', errors='ignore')

    urls = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    ]
    for url in urls:
        try:
            print(f"    Downloading {name} ({gid})...", end=" ", flush=True)
            urllib.request.urlretrieve(url, cache)
            print("ok")
            return cache.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            print("failed, trying next...")

    raise RuntimeError(f"Could not download {name} (id={gid})")


def strip_gutenberg(text):
    """Remove Project Gutenberg header/footer."""
    for marker in ["*** START OF THE PROJECT GUTENBERG EBOOK",
                    "*** START OF THIS PROJECT GUTENBERG EBOOK"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]
            text = text[text.find('\n') + 1:]
            break

    for marker in ["*** END OF THE PROJECT GUTENBERG EBOOK",
                    "*** END OF THIS PROJECT GUTENBERG EBOOK"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


def text_to_bytes(text, max_bytes=200000):
    """Normalize whitespace and convert to uint8."""
    text = ' '.join(text.split())
    data = text.encode('utf-8')[:max_bytes]
    return np.frombuffer(data, dtype=np.uint8).copy()


# ==============================================================
# N-GRAM MODEL
# ==============================================================
class ByteNGram:
    """Byte-level Markov chain generator."""

    def __init__(self, data, order):
        self.order = order
        self.byte_dist = np.bincount(data.astype(int), minlength=256).astype(float)
        self.byte_dist /= self.byte_dist.sum() + 1e-10

        self.model = {}
        if order > 0:
            for i in range(len(data) - order):
                ctx = tuple(data[i:i + order].tolist())
                nxt = int(data[i + order])
                if ctx not in self.model:
                    self.model[ctx] = Counter()
                self.model[ctx][nxt] += 1

    def generate(self, length, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        if self.order == 0:
            return rng.choice(256, size=length, p=self.byte_dist).astype(np.uint8)

        contexts = list(self.model.keys())
        ctx = list(contexts[rng.integers(len(contexts))])
        result = list(ctx)

        for _ in range(length - self.order):
            key = tuple(result[-self.order:])
            if key in self.model:
                counter = self.model[key]
                bytes_list = list(counter.keys())
                weights = np.array(list(counter.values()), dtype=float)
                weights /= weights.sum()
                nxt = rng.choice(bytes_list, p=weights)
            else:
                nxt = rng.choice(256, p=self.byte_dist)
            result.append(int(nxt))

        return np.array(result[:length], dtype=np.uint8)


# ==============================================================
# STATISTICS
# ==============================================================
def collect_metrics(analyzer, chunks):
    out = {m: [] for m in METRIC_NAMES}
    for chunk in chunks:
        res = analyzer.analyze(chunk)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def compare(data_a, data_b):
    sig = 0
    findings = []
    for m in METRIC_NAMES:
        a, b = np.array(data_a[m]), np.array(data_b[m])
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        if not np.isfinite(d):
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < BONF_ALPHA and abs(d) > 0.8:
            sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


# ==============================================================
# HELPERS
# ==============================================================
def make_chunks(data, n_chunks=N_TRIALS, chunk_size=DATA_SIZE):
    """Extract evenly-spaced non-overlapping chunks."""
    stride = len(data) // n_chunks
    return [data[i * stride:i * stride + chunk_size] for i in range(n_chunks)]


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# ==============================================================
# D1: AUTHOR TAXONOMY
# ==============================================================
def direction_1(analyzer, author_data, author_metrics):
    print("\n" + "=" * 60)
    print("D1: AUTHOR TAXONOMY — 4 authors pairwise")
    print("=" * 60)

    names = list(author_data.keys())
    n = len(names)
    sig_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            ns, findings = compare(author_metrics[names[i]], author_metrics[names[j]])
            sig_matrix[i, j] = sig_matrix[j, i] = ns
            top = f"  top: {findings[0][0].split(':')[-1]} d={findings[0][1]:+.1f}" if findings else ""
            print(f"  {names[i]:12s} vs {names[j]:12s} = {ns:3d} sig{top}")

    # Each vs uniform random
    print("\n  Each vs uniform random:")
    vs_random = {}
    rand_chunks = [np.random.randint(0, 256, DATA_SIZE, dtype=np.uint8)
                   for _ in range(N_TRIALS)]
    rand_data = collect_metrics(analyzer, rand_chunks)
    for name in names:
        ns, _ = compare(author_metrics[name], rand_data)
        vs_random[name] = ns
        print(f"    {name:12s} vs random = {ns:3d} sig")

    return dict(sig_matrix=sig_matrix, names=names, vs_random=vs_random)


# ==============================================================
# D2: SEQUENTIAL STRUCTURE
# ==============================================================
def direction_2(analyzer, author_data, author_metrics):
    print("\n" + "=" * 60)
    print("D2: SEQUENTIAL STRUCTURE — real vs shuffled")
    print("=" * 60)

    names = list(author_data.keys())
    vs_shuffled = {}

    for name in names:
        chunks = make_chunks(author_data[name])
        shuf_chunks = []
        for chunk in chunks:
            s = chunk.copy()
            np.random.shuffle(s)
            shuf_chunks.append(s)
        shuf_data = collect_metrics(analyzer, shuf_chunks)
        ns, findings = compare(author_metrics[name], shuf_data)
        vs_shuffled[name] = ns
        top = f"  top: {findings[0][0].split(':')[-1]} d={findings[0][1]:+.1f}" if findings else ""
        print(f"  {name:12s} vs shuffled = {ns:3d} sig{top}")

    return dict(vs_shuffled=vs_shuffled, names=names)


# ==============================================================
# D3: N-GRAM HIERARCHY
# ==============================================================
def direction_3(analyzer, author_data, author_metrics):
    print("\n" + "=" * 60)
    print("D3: N-GRAM HIERARCHY — detection vs Markov order")
    print("=" * 60)

    ref = 'carroll'
    ref_data = author_data[ref]
    ref_met = author_metrics[ref]
    print(f"  Reference: {ref} ({len(ref_data):,d} bytes)")

    results = {}
    for order in NGRAM_ORDERS:
        print(f"  Order {order}...", end=" ", flush=True)
        t0 = time.time()
        model = ByteNGram(ref_data, order)
        ngram_chunks = [model.generate(DATA_SIZE, rng=np.random.default_rng(SEED + t))
                        for t in range(N_TRIALS)]
        ngram_met = collect_metrics(analyzer, ngram_chunks)
        ns, findings = compare(ref_met, ngram_met)
        results[order] = (ns, findings)
        print(f"{ns:3d} sig  ({time.time() - t0:.1f}s)")

    return dict(results=results, ref_name=ref)


# ==============================================================
# D4: AUTHOR-DEPENDENT DETECTION
# ==============================================================
def direction_4(analyzer, author_data, author_metrics):
    print("\n" + "=" * 60)
    print("D4: AUTHOR-DEPENDENT DETECTION — mimicry difficulty")
    print("=" * 60)

    test_orders = [2, 4, 8]
    names = list(author_data.keys())
    results = {}

    for name in names:
        results[name] = {}
        for order in test_orders:
            print(f"  {name:12s} order={order}...", end=" ", flush=True)
            t0 = time.time()
            model = ByteNGram(author_data[name], order)
            ngram_chunks = [model.generate(DATA_SIZE, rng=np.random.default_rng(SEED + t))
                            for t in range(N_TRIALS)]
            ngram_met = collect_metrics(analyzer, ngram_chunks)
            ns, _ = compare(author_metrics[name], ngram_met)
            results[name][order] = ns
            print(f"{ns:3d} sig  ({time.time() - t0:.1f}s)")

    return dict(results=results, names=names, orders=test_orders)


# ==============================================================
# D5: DELAY EMBEDDING
# ==============================================================
def direction_5(analyzer, author_data, author_metrics):
    print("\n" + "=" * 60)
    print("D5: DELAY EMBEDDING — optimal tau for text")
    print("=" * 60)

    ref = 'carroll'
    chunks = make_chunks(author_data[ref])

    # Shuffled reference
    shuf_chunks = []
    for chunk in chunks:
        s = chunk.copy()
        np.random.shuffle(s)
        shuf_chunks.append(s)

    taus = [1, 2, 3, 4, 5]
    results = {}

    for tau in taus:
        print(f"  tau={tau}...", end=" ", flush=True)
        t0 = time.time()
        de_real = [delay_embed(c, tau) for c in chunks]
        de_shuf = [delay_embed(s, tau) for s in shuf_chunks]
        de_real_met = collect_metrics(analyzer, de_real)
        de_shuf_met = collect_metrics(analyzer, de_shuf)
        ns, _ = compare(de_real_met, de_shuf_met)
        results[tau] = ns
        print(f"{ns:3d} sig  ({time.time() - t0:.1f}s)")

    # Raw baseline
    raw_met = collect_metrics(analyzer, chunks)
    shuf_met = collect_metrics(analyzer, shuf_chunks)
    ns_raw, _ = compare(raw_met, shuf_met)
    results['raw'] = ns_raw
    print(f"  raw (no DE): {ns_raw:3d} sig")

    return dict(results=results, taus=taus)


# ==============================================================
# FIGURE
# ==============================================================
def make_figure(d1, d2, d3, d4, d5):
    plt.rcParams.update({
        'figure.facecolor': '#181818', 'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444', 'axes.labelcolor': 'white',
        'text.color': 'white', 'xtick.color': '#cccccc', 'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 14), facecolor='#181818')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)
    fig.suptitle("Text Geometry: Author Fingerprinting & N-gram Hierarchy",
                 fontsize=15, fontweight='bold', color='white')

    # --- D1: Author heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)
    names = d1['names']
    mat = d1['sig_matrix']
    im = ax1.imshow(mat, cmap='YlOrRd', vmin=0, vmax=N_METRICS)
    ax1.set_xticks(range(len(names)))
    ax1.set_yticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                c = 'white' if mat[i, j] > N_METRICS * 0.5 else '#cccccc'
                ax1.text(j, i, str(mat[i, j]), ha='center', va='center',
                         fontsize=10, color=c)
    ax1.set_title(f"D1: Author Pairwise (of {N_METRICS})", fontsize=11)

    # --- D2: Sequential structure ---
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)
    snames = list(d2['vs_shuffled'].keys())
    svals = [d2['vs_shuffled'][n] for n in snames]
    colors2 = plt.cm.Set2(np.linspace(0, 1, len(snames)))
    bars = ax2.bar(range(len(snames)), svals, color=colors2, alpha=0.85)
    ax2.set_xticks(range(len(snames)))
    ax2.set_xticklabels(snames, fontsize=8, rotation=20)
    ax2.set_ylabel(f"Sig metrics (of {N_METRICS})")
    ax2.set_title("D2: Real vs Shuffled", fontsize=11)
    for bar, val in zip(bars, svals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 str(val), ha='center', fontsize=9, color='white')

    # --- D3: N-gram hierarchy (KEY CHART) ---
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)
    orders = sorted(d3['results'].keys())
    sig3 = [d3['results'][o][0] for o in orders]
    ax3.plot(orders, sig3, 'o-', color='#e74c3c', markersize=8, linewidth=2.5)
    ax3.fill_between(orders, sig3, alpha=0.15, color='#e74c3c')
    ax3.set_xlabel("N-gram order", fontsize=10)
    ax3.set_ylabel(f"Sig metrics vs real (of {N_METRICS})")
    ax3.set_title(f"D3: N-gram Hierarchy ({d3['ref_name']})", fontsize=11)
    for o, v in zip(orders, sig3):
        ax3.annotate(str(v), (o, v), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, color='white')
    ax3.set_xticks(orders)

    # --- D4: Per-author mimicry ---
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)
    auth_colors = plt.cm.Set1(np.linspace(0, 1, len(d4['names'])))
    for name, color in zip(d4['names'], auth_colors):
        ords = sorted(d4['results'][name].keys())
        vals = [d4['results'][name][o] for o in ords]
        ax4.plot(ords, vals, 'o-', color=color, markersize=7, linewidth=2, label=name)
    ax4.set_xlabel("N-gram order")
    ax4.set_ylabel("Sig metrics vs real")
    ax4.set_title("D4: Mimicry Difficulty by Author", fontsize=11)
    ax4.legend(fontsize=8, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc')
    ax4.set_xticks(d4['orders'])

    # --- D5: Delay embedding ---
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)
    taus = d5['taus']
    de_vals = [d5['results'][t] for t in taus]
    raw_val = d5['results']['raw']
    ax5.plot(taus, de_vals, 'o-', color='#3498db', markersize=7, linewidth=2,
             label='Delay embedded')
    ax5.axhline(y=raw_val, color='#e74c3c', ls='--', lw=1.5,
                label=f'Raw ({raw_val})')
    ax5.set_xlabel("Delay tau")
    ax5.set_ylabel("Sig metrics (real vs shuffled)")
    ax5.set_title("D5: Delay Embedding", fontsize=11)
    ax5.legend(fontsize=8, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc')
    ax5.set_xticks(taus)
    for t, v in zip(taus, de_vals):
        ax5.annotate(str(v), (t, v), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=9, color='white')

    # --- D1b: vs random ---
    ax6 = fig.add_subplot(gs[1, 2])
    _dark_ax(ax6)
    rnames = list(d1['vs_random'].keys())
    rvals = [d1['vs_random'][n] for n in rnames]
    bars = ax6.bar(range(len(rnames)), rvals, color=colors2, alpha=0.85)
    ax6.set_xticks(range(len(rnames)))
    ax6.set_xticklabels(rnames, fontsize=8, rotation=20)
    ax6.set_ylabel(f"Sig metrics (of {N_METRICS})")
    ax6.set_title("D1b: Each Author vs Uniform Random", fontsize=11)
    for bar, val in zip(bars, rvals):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 str(val), ha='center', fontsize=9, color='white')

    out = FIG_DIR / "text_geometry.png"
    fig.savefig(out, dpi=180, facecolor='#181818')
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# ==============================================================
# MAIN
# ==============================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("TEXT GEOMETRY INVESTIGATION")
    print(f"Chunk: {DATA_SIZE} bytes, trials: {N_TRIALS}, metrics: {N_METRICS}")
    print("=" * 60)

    # Load texts
    print("\nLoading Gutenberg texts:")
    author_data = {}
    for key, (desc, gid) in AUTHORS.items():
        # Check local cache first (alice already exists in data/ai_text/)
        local = Path(__file__).resolve().parents[2] / "data" / "ai_text" / f"human_{key}.txt"
        if local.exists():
            text = local.read_text(encoding='utf-8', errors='ignore')
        else:
            text = download_gutenberg(key, gid)
        text = strip_gutenberg(text)
        data = text_to_bytes(text)
        author_data[key] = data
        print(f"  {key:12s}: {len(data):>7,d} bytes  {desc}")

    # Analyze all authors
    print("\nAnalyzing authors:")
    analyzer = GeometryAnalyzer().add_all_geometries()
    author_metrics = {}
    for name, data in author_data.items():
        print(f"  {name:12s}...", end=" ", flush=True)
        t0 = time.time()
        chunks = make_chunks(data)
        author_metrics[name] = collect_metrics(analyzer, chunks)
        print(f"{time.time() - t0:.1f}s")

    # Run directions
    d1 = direction_1(analyzer, author_data, author_metrics)
    d2 = direction_2(analyzer, author_data, author_metrics)
    d3 = direction_3(analyzer, author_data, author_metrics)
    d4 = direction_4(analyzer, author_data, author_metrics)
    d5 = direction_5(analyzer, author_data, author_metrics)

    make_figure(d1, d2, d3, d4, d5)

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    mat = d1['sig_matrix']
    pw = mat[np.triu_indices_from(mat, k=1)]
    print(f"D1: Author pairwise: {pw.min()}-{pw.max()} sig (mean {pw.mean():.0f})")
    for n, v in d1['vs_random'].items():
        print(f"    {n} vs random: {v}")

    for n, v in d2['vs_shuffled'].items():
        print(f"D2: {n} vs shuffled: {v}")

    print(f"D3: N-gram hierarchy ({d3['ref_name']}):")
    for o in sorted(d3['results'].keys()):
        print(f"    order {o}: {d3['results'][o][0]} sig")

    print("D4: Author-dependent (sig at each order):")
    for name in d4['names']:
        vals = [f"o{o}={d4['results'][name][o]}" for o in d4['orders']]
        print(f"    {name}: {', '.join(vals)}")

    print(f"D5: Delay embedding: raw={d5['results']['raw']}", end="")
    for t in d5['taus']:
        print(f", tau={t}:{d5['results'][t]}", end="")
    print()


if __name__ == "__main__":
    main()
