"""
Investigation Runner — shared boilerplate for exotic geometry investigations.

Extracts all repeated patterns (stats, figure theming, metric collection)
so investigation scripts only need to provide domain-specific data generators.

Usage:
    from tools.investigation_runner import Runner

    runner = Runner("DNA Methylation", mode="1d")

    # Generate and analyze data
    chunks = [my_generator(rng, runner.data_size) for rng in runner.trial_rngs()]
    metrics = runner.collect(chunks)

    # Compare two conditions
    n_sig, findings = runner.compare(metrics_a, metrics_b)

    # Figure
    fig, axes = runner.create_figure(5, "DNA Methylation Investigation")
    runner.plot_heatmap(axes[0], matrix, labels, "D1: Taxonomy")
    runner.save(fig, "dna_methylation")
"""

import multiprocessing
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

# Framework imports
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from exotic_geometry_framework import GeometryAnalyzer, delay_embed, bitplane_extract

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ----------------------------------------------------------
# Parallel analysis helpers (module-level for pickling)
# ----------------------------------------------------------
_worker_analyzer = None


def _worker_init(mode, cache_dir):
    """Initialize a per-process GeometryAnalyzer."""
    global _worker_analyzer
    _worker_analyzer = GeometryAnalyzer(cache_dir=cache_dir)
    if mode == '1d':
        _worker_analyzer.add_all_geometries()
    elif mode == '2d':
        _worker_analyzer.add_spatial_geometries()


def _worker_analyze(chunk):
    """Analyze a single chunk using the per-process analyzer."""
    return _worker_analyzer.analyze(chunk)


def _parallel_analyze(chunks, mode, cache_dir, n_workers):
    """Analyze chunks in parallel using a process pool."""
    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(mode, cache_dir),
    ) as pool:
        return pool.map(_worker_analyze, chunks)


class Runner:
    """Reusable investigation runner with all boilerplate baked in."""

    def __init__(self, name, mode='1d', data_size=2000, n_trials=25,
                 alpha=0.05, seed=42, cache=False, n_workers=1):
        self.name = name
        self.mode = mode
        self.data_size = data_size
        self.n_trials = n_trials
        self.alpha = alpha
        self.seed = seed
        self.n_workers = n_workers
        self.fig_dir = _ROOT / "figures"
        self.fig_dir.mkdir(exist_ok=True)

        # Cache setup
        self.cache_dir = None
        if cache:
            self.cache_dir = str(_ROOT / ".cache")

        np.random.seed(seed)

        # Setup analyzer and discover metrics
        self.analyzer = GeometryAnalyzer(cache_dir=self.cache_dir)
        if mode == '1d':
            self.analyzer.add_all_geometries()
            dummy = self.analyzer.analyze(
                np.random.randint(0, 256, data_size, dtype=np.uint8))
        elif mode == '2d':
            self.analyzer.add_spatial_geometries()
            dummy = self.analyzer.analyze(np.random.rand(16, 16))
        else:
            raise ValueError(f"mode must be '1d' or '2d', got '{mode}'")

        self.metric_names = []
        for r in dummy.results:
            for mn in sorted(r.metrics.keys()):
                self.metric_names.append(f"{r.geometry_name}:{mn}")
        self.n_metrics = len(self.metric_names)
        self.bonf_alpha = alpha / self.n_metrics

        print(f"Runner: {name}")
        print(f"  mode={mode}, data_size={data_size}, trials={n_trials}, "
              f"metrics={self.n_metrics}"
              + (f", workers={n_workers}" if n_workers > 1 else "")
              + (", cache=on" if cache else ""))

    # ----------------------------------------------------------
    # RNG helpers
    # ----------------------------------------------------------
    def trial_rngs(self, offset=0):
        """Return a list of n_trials independent RNGs."""
        return [np.random.default_rng(self.seed + offset + i)
                for i in range(self.n_trials)]

    # ----------------------------------------------------------
    # Data collection
    # ----------------------------------------------------------
    def collect(self, chunks):
        """Analyze chunks, return {metric_name: [values]}.

        When n_workers > 1, chunks are processed in parallel using
        multiprocessing.  Each worker creates its own GeometryAnalyzer
        to avoid pickling issues.
        """
        out = {m: [] for m in self.metric_names}

        if self.n_workers > 1 and len(chunks) > 1:
            results = _parallel_analyze(
                chunks, self.mode, self.cache_dir, self.n_workers)
        else:
            results = [self.analyzer.analyze(chunk) for chunk in chunks]

        for res in results:
            for r in res.results:
                for mn, mv in r.metrics.items():
                    key = f"{r.geometry_name}:{mn}"
                    if key in out and np.isfinite(mv):
                        out[key].append(mv)
        return out

    def shuffle_chunks(self, chunks):
        """Return shuffled copies of each chunk."""
        out = []
        for c in chunks:
            s = c.copy()
            if s.ndim == 1:
                np.random.shuffle(s)
            else:
                flat = s.ravel()
                np.random.shuffle(flat)
                s = flat.reshape(s.shape)
            out.append(s)
        return out

    def make_chunks(self, data, n_chunks=None, chunk_size=None):
        """Extract non-overlapping chunks from a long array."""
        n = n_chunks or self.n_trials
        sz = chunk_size or self.data_size
        stride = len(data) // n
        return [data[i * stride:i * stride + sz] for i in range(n)]

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------
    @staticmethod
    def cohens_d(a, b):
        """Pooled-std Cohen's d."""
        na, nb = len(a), len(b)
        sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
        ps = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
        if ps < 1e-15:
            diff = np.mean(a) - np.mean(b)
            return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
        return (np.mean(a) - np.mean(b)) / ps

    def compare(self, data_a, data_b):
        """Compare two metric dicts. Returns (n_sig, [(metric, d, p), ...])."""
        sig = 0
        findings = []
        for m in self.metric_names:
            a = np.array(data_a.get(m, []))
            b = np.array(data_b.get(m, []))
            if len(a) < 3 or len(b) < 3:
                continue
            d = self.cohens_d(a, b)
            if not np.isfinite(d):
                continue
            _, p = sp_stats.ttest_ind(a, b, equal_var=False)
            if p < self.bonf_alpha and abs(d) > 0.8:
                sig += 1
                findings.append((m, d, p))
        findings.sort(key=lambda x: -abs(x[1]))
        return sig, findings

    def compare_pairwise(self, conditions):
        """
        Compare all pairs from a dict of {name: metrics}.
        Returns (matrix, names, findings_dict).
        """
        names = list(conditions.keys())
        n = len(names)
        matrix = np.zeros((n, n), dtype=int)
        all_findings = {}
        for i in range(n):
            for j in range(i + 1, n):
                ns, findings = self.compare(
                    conditions[names[i]], conditions[names[j]])
                matrix[i, j] = matrix[j, i] = ns
                all_findings[(names[i], names[j])] = findings
                top = ""
                if findings:
                    top = (f"  top: {findings[0][0].split(':')[-1]} "
                           f"d={findings[0][1]:+.1f}")
                print(f"  {names[i]:12s} vs {names[j]:12s} = "
                      f"{ns:3d} sig{top}")
        return matrix, names, all_findings

    # ----------------------------------------------------------
    # Timing
    # ----------------------------------------------------------
    def timed(self, label):
        """Context manager for timing blocks."""
        return _Timer(label)

    # ----------------------------------------------------------
    # Figure helpers
    # ----------------------------------------------------------
    @staticmethod
    def _apply_dark_theme():
        plt.rcParams.update({
            'figure.facecolor': '#181818',
            'axes.facecolor': '#181818',
            'axes.edgecolor': '#444444',
            'axes.labelcolor': 'white',
            'text.color': 'white',
            'xtick.color': '#cccccc',
            'ytick.color': '#cccccc',
        })

    @staticmethod
    def dark_ax(ax):
        """Apply dark theme to a single axis."""
        ax.set_facecolor('#181818')
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.tick_params(colors='#cccccc', labelsize=7)
        return ax

    def create_figure(self, n_panels, title=None, rows=2, cols=3,
                      figsize=(20, 14)):
        """Create a dark-themed figure with gridspec panels."""
        self._apply_dark_theme()
        fig = plt.figure(figsize=figsize, facecolor='#181818')
        gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.35,
                               wspace=0.35, left=0.06, right=0.97,
                               top=0.93, bottom=0.06)
        fig.suptitle(title or self.name, fontsize=15, fontweight='bold',
                     color='white')

        axes = []
        for i in range(min(n_panels, rows * cols)):
            r, c = divmod(i, cols)
            ax = fig.add_subplot(gs[r, c])
            self.dark_ax(ax)
            axes.append(ax)
        return fig, axes

    def plot_heatmap(self, ax, matrix, labels, title, vmax=None):
        """Plot a pairwise comparison heatmap."""
        vmax = vmax or self.n_metrics
        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=vmax)
        n = len(labels)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                if i != j:
                    color = ('white' if matrix[i, j] > vmax * 0.5
                             else '#cccccc')
                    ax.text(j, i, str(int(matrix[i, j])), ha='center',
                            va='center', fontsize=10, color=color)
        ax.set_title(f"{title} (of {self.n_metrics})", fontsize=11)

    def plot_bars(self, ax, names, values, title, colors=None):
        """Plot a bar chart with value labels."""
        if colors is None:
            colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=20)
        ax.set_ylabel(f"Sig metrics (of {self.n_metrics})")
        ax.set_title(title, fontsize=11)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    str(val), ha='center', fontsize=9, color='white')

    def plot_line(self, ax, x, y, title, xlabel='', ylabel='',
                  color='#e74c3c', label=None, annotate=True):
        """Plot a line chart with optional annotation."""
        ax.plot(x, y, 'o-', color=color, markersize=7, linewidth=2,
                label=label)
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel or f"Sig metrics (of {self.n_metrics})")
        ax.set_title(title, fontsize=11)
        if annotate:
            for xi, yi in zip(x, y):
                ax.annotate(str(yi), (xi, yi), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=9,
                            color='white')
        if label:
            ax.legend(fontsize=8, facecolor='#222222', edgecolor='#444444',
                      labelcolor='#cccccc')

    def save(self, fig, name=None):
        """Save figure to figures/ directory."""
        name = name or self.name.lower().replace(' ', '_')
        out = self.fig_dir / f"{name}.png"
        fig.savefig(out, dpi=180, facecolor='#181818')
        plt.close(fig)
        print(f"\nFigure saved: {out}")

    # ----------------------------------------------------------
    # Summary printer
    # ----------------------------------------------------------
    def print_summary(self, results):
        """Print a standardized summary from a results dict.

        results: dict with keys like 'D1', 'D2', etc.
            Each value is a string or list of strings to print.
        """
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for key in sorted(results.keys()):
            val = results[key]
            if isinstance(val, list):
                for line in val:
                    print(f"{key}: {line}")
            else:
                print(f"{key}: {val}")


class _Timer:
    """Simple timing context manager."""
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"  {self.label}...", end=" ", flush=True)
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.t0
        print(f"{elapsed:.1f}s")
        self.elapsed = elapsed
