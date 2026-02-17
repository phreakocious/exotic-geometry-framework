#!/usr/bin/env python3
"""
Investigation: ECB Penguin in 2D — Cipher Mode Detection via SpatialFieldGeometry

The classic "ECB penguin": encrypting a structured image with ECB mode preserves
spatial structure because identical plaintext blocks → identical ciphertext blocks.
CBC/CTR modes break this relationship.

Hypothesis: SpatialFieldGeometry detects 2D block structure in ECB ciphertext
that CBC/CTR don't have — even though all ciphertexts have identical byte-level
histograms (uniform).

Methodology: N_TRIALS=25 (different keys), Cohen's d, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from Crypto.Cipher import AES
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
ALPHA = 0.05
IMG_SIZE = 128  # 128×128 = 16384 bytes = 1024 AES blocks

# Discover all metric names from 8 spatial geometries (80 metrics)
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
del _analyzer, _dummy, _r, _mn


# =============================================================================
# ENCRYPTION HELPERS
# =============================================================================

def encrypt_ecb(plaintext: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(plaintext)

def encrypt_cbc(plaintext: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    return cipher.encrypt(plaintext)

def encrypt_ctr(plaintext: bytes, key: bytes, nonce: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    return cipher.encrypt(plaintext)


# =============================================================================
# TEST IMAGE GENERATORS
# =============================================================================

def half_image(H=IMG_SIZE, W=IMG_SIZE):
    """Top half = 0x00, bottom half = 0xFF."""
    img = np.zeros((H, W), dtype=np.uint8)
    img[H//2:, :] = 255
    return img

def bands_image(H=IMG_SIZE, W=IMG_SIZE, band_height=16):
    """Horizontal bands alternating 0x00 and 0xFF."""
    img = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        if (r // band_height) % 2 == 1:
            img[r, :] = 255
    return img

def circle_image(H=IMG_SIZE, W=IMG_SIZE, radius=None):
    """Filled circle on uniform background."""
    if radius is None:
        radius = min(H, W) // 3
    y, x = np.mgrid[:H, :W]
    mask = (x - W//2)**2 + (y - H//2)**2 < radius**2
    img = np.zeros((H, W), dtype=np.uint8)
    img[mask] = 255
    return img

def blocks_image(H=IMG_SIZE, W=IMG_SIZE, block_size=16):
    """Checkerboard of 16×16 blocks (aligned with AES block size in row)."""
    y, x = np.mgrid[:H, :W]
    img = np.zeros((H, W), dtype=np.uint8)
    img[((x // block_size) + (y // block_size)) % 2 == 1] = 255
    return img


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

def encrypt_and_reshape(img, mode, key, iv=None, nonce=None):
    """Encrypt image bytes and reshape back to 2D."""
    plaintext = img.tobytes()
    if mode == 'ECB':
        ct = encrypt_ecb(plaintext, key)
    elif mode == 'CBC':
        ct = encrypt_cbc(plaintext, key, iv)
    elif mode == 'CTR':
        ct = encrypt_ctr(plaintext, key, nonce)
    return np.frombuffer(ct, dtype=np.uint8).reshape(img.shape).astype(np.float64)


# =============================================================================
# INVESTIGATION
# =============================================================================

def run_investigation():
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    test_images = {
        'half':   half_image(),
        'bands':  bands_image(),
        'circle': circle_image(),
        'blocks': blocks_image(),
    }

    print("=" * 78)
    print("ECB PENGUIN IN 2D: Cipher Mode Detection via SpatialFieldGeometry")
    print(f"Image: {IMG_SIZE}x{IMG_SIZE}, N_TRIALS={N_TRIALS} (different AES keys)")
    print("=" * 78)

    all_results = {}  # {(image_name, mode): {metric: [values]}}

    for img_name, img in test_images.items():
        print(f"\n{'─' * 78}")
        print(f"  Image: {img_name}")
        print(f"{'─' * 78}")

        for mode in ['ECB', 'CBC', 'CTR']:
            key_name = f"{img_name}_{mode}"
            metrics = {m: [] for m in METRIC_NAMES}

            for trial in range(N_TRIALS):
                rng = np.random.default_rng(42 + trial)
                key = rng.bytes(16)
                iv = rng.bytes(16)
                nonce = rng.bytes(8)

                ct_2d = encrypt_and_reshape(img, mode, key, iv, nonce)
                res = analyzer.analyze(ct_2d)
                for r in res.results:
                    for mn, mv in r.metrics.items():
                        mkey = f"{r.geometry_name}:{mn}"
                        if mkey in metrics and np.isfinite(mv):
                            metrics[mkey].append(mv)

            all_results[(img_name, mode)] = metrics

        # Random baseline (no encryption, just random bytes)
        rand_metrics = {m: [] for m in METRIC_NAMES}
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(1000 + trial)
            rand_img = rng.integers(0, 256, (IMG_SIZE, IMG_SIZE)).astype(np.float64)
            res = analyzer.analyze(rand_img)
            for r in res.results:
                for mn, mv in r.metrics.items():
                    mkey = f"{r.geometry_name}:{mn}"
                    if mkey in rand_metrics and np.isfinite(mv):
                        rand_metrics[mkey].append(mv)
        all_results[(img_name, 'random')] = rand_metrics

        # Compare ECB, CBC, CTR each vs random
        bonf = ALPHA / len(METRIC_NAMES)
        for mode in ['ECB', 'CBC', 'CTR']:
            sig = 0
            best_d, best_m = 0, ""
            for m in METRIC_NAMES:
                a = np.array(all_results[(img_name, mode)][m])
                b = np.array(all_results[(img_name, 'random')][m])
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf:
                    sig += 1
                if abs(d) > abs(best_d):
                    best_d, best_m = d, m
            marker = "*** DETECTED" if sig > 0 else "    (random-looking)"
            print(f"  {mode:4s} vs random: {sig:2d} sig metrics  "
                  f"best: {best_m:25s} d={best_d:+8.2f}  {marker}")

    # Detailed ECB breakdown for best image
    print(f"\n{'=' * 78}")
    print("DETAILED: ECB 'circle' — all significant metrics vs random")
    print(f"{'=' * 78}")
    bonf = ALPHA / len(METRIC_NAMES)
    for m in METRIC_NAMES:
        a = np.array(all_results[('circle', 'ECB')][m])
        b = np.array(all_results[('circle', 'random')][m])
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        sig = "***" if p < bonf else "   "
        print(f"  {sig} {m:30s}  d={d:+8.2f}  p={p:.2e}  "
              f"ECB={np.mean(a):10.3f}  rand={np.mean(b):10.3f}")

    return all_results, test_images


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(all_results, test_images):
    print("\nGenerating figure...", flush=True)
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    # Pick the circle image for the main visual
    img = test_images['circle']
    rng = np.random.default_rng(42)
    key = rng.bytes(16)
    iv = rng.bytes(16)
    nonce = rng.bytes(8)

    ecb_ct = encrypt_and_reshape(img, 'ECB', key, iv, nonce)
    cbc_ct = encrypt_and_reshape(img, 'CBC', key, iv, nonce)
    ctr_ct = encrypt_and_reshape(img, 'CTR', key, iv, nonce)
    rand_img = np.random.default_rng(999).integers(0, 256, (IMG_SIZE, IMG_SIZE)).astype(float)

    BG = '#181818'
    FG = '#e0e0e0'
    fig = plt.figure(figsize=(16, 22), facecolor=BG)
    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1.3, 0.7, 1.0, 1.0],
                           hspace=0.45, wspace=0.35)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # Row 0: Images — plaintext, ECB, CBC, random
    panels = [
        (img.astype(float), 'Plaintext (circle)', 'gray'),
        (ecb_ct, 'AES-ECB ciphertext', 'viridis'),
        (cbc_ct, 'AES-CBC ciphertext', 'viridis'),
        (rand_img, 'Random (baseline)', 'viridis'),
    ]
    for i, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        _dark_ax(ax)
        ax.imshow(data, cmap=cmap, interpolation='nearest')
        ax.set_title(title, fontsize=10, fontweight='bold', color=FG)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 1:  # ECB
            for spine in ax.spines.values():
                spine.set_edgecolor('#ff3333')
                spine.set_linewidth(2.5)

    # Row 1: Histograms — show they're all uniform
    for i, (data, title, _) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        _dark_ax(ax)
        ax.hist(data.ravel(), bins=50, color='steelblue', alpha=0.7,
                density=True, edgecolor='none')
        ax.set_title('Byte distribution', fontsize=8, color='#999')
        ax.set_xlim(-10, 265)
        if i == 0:
            ax.set_ylabel('Density', fontsize=8, color=FG)

    fig.text(0.5, 0.54, '↑  ECB/CBC/Random all have uniform byte distributions  ↑',
             ha='center', fontsize=9, color='#999', style='italic')

    # Row 2: Metric bars — ECB vs CBC vs CTR vs Random for 'circle'
    compare_metrics = ['Spatial Field:coherence_score', 'Spatial Field:n_basins',
                       'Surface:gaussian_curvature_mean', 'Spectral Power 2D:spectral_slope']
    mode_colors = {'ECB': '#ff3333', 'CBC': '#4CAF50', 'CTR': '#2196F3', 'random': '#999999'}
    mode_list = ['ECB', 'CBC', 'CTR', 'random']
    mode_labels = ['ECB', 'CBC', 'CTR', 'Random']

    for j, metric in enumerate(compare_metrics):
        ax = fig.add_subplot(gs[2, j])
        _dark_ax(ax)
        means = [np.mean(all_results[('circle', m)][metric]) for m in mode_list]
        stds = [np.std(all_results[('circle', m)][metric]) for m in mode_list]
        colors = [mode_colors[m] for m in mode_list]
        ax.bar(range(4), means, yerr=stds, capsize=4,
               color=colors, alpha=0.85, edgecolor='#333')
        ax.set_xticks(range(4))
        ax.set_xticklabels(mode_labels, fontsize=8, color=FG)
        ax.set_title(metric.split(':')[-1].replace('_', ' '), fontsize=10, fontweight='bold', color=FG)

    # Row 3: All four images — show ECB preserves structure across image types
    for j, img_name in enumerate(['half', 'bands', 'circle', 'blocks']):
        ax = fig.add_subplot(gs[3, j])
        _dark_ax(ax)
        img_data = test_images[img_name]
        rng2 = np.random.default_rng(77)
        key2 = rng2.bytes(16)
        ecb2 = encrypt_and_reshape(img_data, 'ECB', key2)
        ax.imshow(ecb2, cmap='viridis', interpolation='nearest')
        ax.set_title(f'ECB({img_name})', fontsize=9, fontweight='bold', color=FG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#ff3333')
            spine.set_linewidth(1.5)

    fig.suptitle('ECB Penguin in 2D: SpatialFieldGeometry Detects Block Structure',
                 fontsize=14, fontweight='bold', y=0.98, color=FG)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'ecb_penguin_2d.png'),
                dpi=180, bbox_inches='tight', facecolor=BG)
    print("  Saved ecb_penguin_2d.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    all_results, test_images = run_investigation()
    make_figure(all_results, test_images)

    # Summary
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    for img_name in test_images:
        ecb_sig = 0
        cbc_sig = 0
        bonf = ALPHA / len(METRIC_NAMES)
        for m in METRIC_NAMES:
            a_ecb = np.array(all_results[(img_name, 'ECB')][m])
            a_cbc = np.array(all_results[(img_name, 'CBC')][m])
            b = np.array(all_results[(img_name, 'random')][m])
            if len(a_ecb) < 3 or len(b) < 3:
                continue
            _, p_ecb = stats.ttest_ind(a_ecb, b, equal_var=False)
            _, p_cbc = stats.ttest_ind(a_cbc, b, equal_var=False)
            if p_ecb < bonf:
                ecb_sig += 1
            if p_cbc < bonf:
                cbc_sig += 1
        print(f"  {img_name:8s}: ECB={ecb_sig:2d} sig, CBC={cbc_sig:2d} sig")
