#!/usr/bin/env python3
"""
Per-Metric Ablation: 2×2 Classification for H3 and H4.

For each metric in a geometry, compute two independent axes:
  1. D1 drop: (|d_exact| - |d_random|) / |d_exact| per source, averaged.
     High D1 drop = structure-dependent (the mathematical object matters).
  2. Source variance: std of metric values across 7 evaluation sources.
     High source variance = discriminative (different data → different readings).

Classification:
  Headline   — high D1 drop AND high source variance (ideal)
  Scientific — high D1 drop, low source variance (proves math, useless for atlas)
  Workhorse  — low D1 drop, high source variance (discriminative but generic)
  Dead       — low D1 drop, low source variance (remove)

Reuses infrastructure from geometry_ablation.py (source generators, Thomson codes,
root variant factory, trial runner).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import argparse
import numpy as np
from copy import deepcopy

# Import shared infrastructure from the main ablation script
from geometry_ablation import (
    N_TRIALS, DATA_SIZE_DEFAULT, GEO_DATA_SIZES,
    EVAL_SOURCES, DIAGNOSTIC_SOURCES, ALL_SOURCES,
    generate_data, run_trials, make_root_variants,
    make_metric_variants_hyperbolic, _hyperbolic_compute_patched,
    make_metric_variants_lorentzian, _lorentzian_compute_patched,
    make_metric_variants_symplectic, _symplectic_compute_patched,
    make_qc_variants, make_hat_variants,
    D1_LEVELS_ROOT, _effective_root_directions,
)

from exotic_geometry_framework import (
    E8Geometry, D4Geometry, H3CoxeterGeometry, H4CoxeterGeometry,
    HyperbolicGeometry, LorentzianGeometry, SymplecticGeometry,
    PenroseGeometry, AmmannBeenkerGeometry, EinsteinHatGeometry,
    DodecagonalGeometry, SeptagonalGeometry,
    GeometryResult,
    _effective_root_directions,
)
import types


def _hyperbolic_compute_patched_decomposed(self, data):
    """Hyperbolic compute_metrics with scaled embedding, returning decomposed metrics."""
    alpha = getattr(self, '_metric_alpha', 1.0)
    if alpha == 1.0:
        return HyperbolicGeometry.compute_metrics(self, data)

    data = self.validate_data(data)
    data_norm = data.astype(float) / 255.0
    n_pairs = len(data_norm) // 2
    zero_metrics = {
        "curvature_structure": 0.0, "temporal_variance": 0.0,
        "spatio_temporal_corr": 0.0, "knn_scale_ratio": 0.0,
        "mean_hyperbolic_radius": 0.0,
    }
    if n_pairs < 30:
        return GeometryResult(geometry_name=self.name, metrics=zero_metrics, raw_data={})

    points = data_norm[:n_pairs * 2].reshape(n_pairs, 2)
    pts_disk = (points - 0.5) * 1.8 * alpha
    norms = np.linalg.norm(pts_disk, axis=1)
    mask = norms >= 0.99
    pts_disk[mask] = pts_disk[mask] / (norms[mask, None] + 1e-10) * 0.98

    def _hyp_dist(z1, z2):
        diff = np.abs(z1 - z2)
        denom = np.abs(1.0 - np.conj(z1) * z2)
        ratio = np.clip(diff / (denom + 1e-15), 0, 0.9999)
        return 2.0 * np.arctanh(ratio)

    z_all = pts_disk[:, 0] + 1j * pts_disk[:, 1]

    radii_all = np.abs(z_all)
    hyp_radii = 2.0 * np.arctanh(np.clip(radii_all, 0, 0.9999))
    mean_hyp_radius = float(np.mean(hyp_radii))

    dists_seq = _hyp_dist(z_all[:-1], z_all[1:])
    temporal_var = float(np.var(dists_seq)) if len(dists_seq) >= 2 else 0.0
    temporal_score = 1.0 + temporal_var

    spatio_temporal_corr = 0.0
    spatio_temporal_score = 1.0
    if len(dists_seq) > 10:
        radii = np.abs(z_all[:-1])
        if np.std(radii) > 1e-9 and np.std(dists_seq) > 1e-9:
            spatio_temporal_corr = float(np.corrcoef(radii, dists_seq)[0, 1])
            spatio_temporal_score = float(np.exp(spatio_temporal_corr))

    z_unique = np.unique(z_all)
    n_u = len(z_unique)
    knn_ratio = 1.0
    if n_u >= 30:
        if n_u > 300:
            u_rng = np.random.default_rng(42)
            z_unique = z_unique[u_rng.choice(n_u, 300, replace=False)]
            n_u = 300
        dist_matrix = _hyp_dist(z_unique[:, np.newaxis], z_unique[np.newaxis, :])
        k = min(5, n_u - 2)
        if k > 0:
            sorted_dists = np.sort(dist_matrix, axis=1)
            local_scale = np.median(sorted_dists[:, k])
            i_idx, j_idx = np.triu_indices(n_u, k=1)
            global_scale = np.median(dist_matrix[i_idx, j_idx])
            if local_scale > 1e-9:
                knn_ratio = float(global_scale / local_scale)

    curvature_structure = float(temporal_score * spatio_temporal_score * knn_ratio)

    return GeometryResult(
        geometry_name=self.name,
        metrics={
            "curvature_structure": curvature_structure,
            "temporal_variance": temporal_var,
            "spatio_temporal_corr": spatio_temporal_corr,
            "knn_scale_ratio": knn_ratio,
            "mean_hyperbolic_radius": mean_hyp_radius,
        },
        raw_data={})


def per_metric_d1_drop(geo_name, geo_instance, rng, n_trials=N_TRIALS,
                       variant_type='root'):
    """Compute per-metric D1 drop and source variance.

    variant_type: 'root' for root-system geometries, 'hyperbolic' for metric-based.

    Returns dict: metric_name -> {
        'd1_drop': mean fractional drop across eval sources,
        'd1_per_source': {source: drop},
        'source_variance': std of exact Cohen's d across eval sources,
        'exact_d_per_source': {source: Cohen's d with exact},
        'random_d_per_source': {source: Cohen's d with random_matched},
    }
    """
    if variant_type == 'hyperbolic':
        variants = make_metric_variants_hyperbolic(geo_instance, rng)
        for k, v in variants.items():
            if hasattr(v, '_metric_alpha'):
                v.compute_metrics = types.MethodType(
                    _hyperbolic_compute_patched_decomposed, v)
    elif variant_type == 'lorentzian':
        variants = make_metric_variants_lorentzian(geo_instance, rng)
        for k, v in variants.items():
            if hasattr(v, '_metric_alpha'):
                v.compute_metrics = types.MethodType(_lorentzian_compute_patched, v)
    elif variant_type == 'symplectic':
        variants = make_metric_variants_symplectic(geo_instance, rng)
        for k, v in variants.items():
            if hasattr(v, '_metric_alpha'):
                v.compute_metrics = types.MethodType(_symplectic_compute_patched, v)
    elif variant_type == 'qc':
        real_ratio = (getattr(geo_instance, 'PHI', None) or
                      getattr(geo_instance, 'SILVER', None) or
                      getattr(geo_instance, 'RATIO', None))
        variants = make_qc_variants(geo_instance, real_ratio, rng)
    elif variant_type == 'hat':
        variants = make_hat_variants(geo_instance, rng)
    else:
        variants = make_root_variants(type(geo_instance), geo_instance, rng)

    exact_geo = variants['exact']
    random_geo = variants['random_matched']

    # Run trials for exact and random_matched on all eval sources
    exact_results = {}
    random_results = {}

    for src in EVAL_SOURCES:
        print(f"  {src}: exact...", end="", flush=True)
        exact_results[src] = run_trials(
            exact_geo, src, n_trials=n_trials, geo_name=geo_name)
        print(f" random...", end="", flush=True)
        random_results[src] = run_trials(
            random_geo, src, n_trials=n_trials, geo_name=geo_name)
        print(" done", flush=True)

    # Extract per-metric results
    # Get metric names from first result
    metric_names = list(exact_results[EVAL_SOURCES[0]]['cohens_d'].keys())

    results = {}
    for metric in metric_names:
        d1_per_source = {}
        exact_d_per_source = {}
        random_d_per_source = {}

        for src in EVAL_SOURCES:
            d_ex = abs(exact_results[src]['cohens_d'].get(metric, 0.0))
            d_rn = abs(random_results[src]['cohens_d'].get(metric, 0.0))
            exact_d_per_source[src] = exact_results[src]['cohens_d'].get(metric, 0.0)
            random_d_per_source[src] = random_results[src]['cohens_d'].get(metric, 0.0)

            if d_ex > 0.3:
                d1_per_source[src] = (d_ex - d_rn) / d_ex
            else:
                d1_per_source[src] = 0.0  # too weak to measure drop

        # D1 drop = mean across sources where exact has signal
        drops = [v for v in d1_per_source.values() if v != 0.0]
        d1_drop = float(np.mean(drops)) if drops else 0.0

        # Source variance = std of exact Cohen's d across sources
        exact_d_values = np.array([exact_d_per_source[s] for s in EVAL_SOURCES])
        source_var = float(np.std(exact_d_values))

        results[metric] = {
            'd1_drop': d1_drop,
            'd1_per_source': d1_per_source,
            'source_variance': source_var,
            'exact_d_per_source': exact_d_per_source,
            'random_d_per_source': random_d_per_source,
        }

    return results


def classify_metric(d1_drop, source_var, d1_threshold=0.30, sv_threshold=0.5):
    """Classify a metric into the 2×2.

    Thresholds:
      d1_threshold: D1 drop above this = structure-dependent
      sv_threshold: source variance above this = discriminative
    """
    high_d1 = d1_drop > d1_threshold
    high_sv = source_var > sv_threshold

    if high_d1 and high_sv:
        return "HEADLINE"
    elif high_d1 and not high_sv:
        return "SCIENTIFIC"
    elif not high_d1 and high_sv:
        return "WORKHORSE"
    else:
        return "DEAD"


def print_results(geo_name, results):
    """Print per-metric 2×2 classification table."""
    print()
    print(f"{'='*80}")
    print(f"  {geo_name}: Per-Metric 2×2 Classification")
    print(f"{'='*80}")
    print()

    # Header
    print(f"{'Metric':<25s} {'D1 drop':>8s} {'Src var':>8s} {'Class':>12s}  "
          f"{'':>8s} ", end="")
    for src in EVAL_SOURCES:
        print(f" {src[:6]:>7s}", end="")
    print()
    print("-" * (60 + 8 * len(EVAL_SOURCES)))

    # Sort: headlines first, then workhorses, scientific, dead
    order = {"HEADLINE": 0, "WORKHORSE": 1, "SCIENTIFIC": 2, "DEAD": 3}
    sorted_metrics = sorted(results.items(),
                            key=lambda x: (order.get(classify_metric(x[1]['d1_drop'],
                                                                      x[1]['source_variance']), 4),
                                           -abs(x[1]['d1_drop'])))

    for metric, data in sorted_metrics:
        cls = classify_metric(data['d1_drop'], data['source_variance'])
        print(f"{metric:<25s} {data['d1_drop']:>7.1%} {data['source_variance']:>8.3f} "
              f"{cls:>12s}  ", end="")

        # Per-source exact Cohen's d
        print(f"{'d_ex':>8s} ", end="")
        for src in EVAL_SOURCES:
            d = data['exact_d_per_source'].get(src, 0.0)
            print(f" {d:>7.2f}", end="")
        print()

        # Per-source D1 drop
        print(f"{'':25s} {'':>8s} {'':>8s} {'':>12s}  {'drop':>8s} ", end="")
        for src in EVAL_SOURCES:
            drop = data['d1_per_source'].get(src, 0.0)
            print(f" {drop:>6.0%} ", end="")
        print()

        # Per-source random Cohen's d
        print(f"{'':25s} {'':>8s} {'':>8s} {'':>12s}  {'d_rn':>8s} ", end="")
        for src in EVAL_SOURCES:
            d = data['random_d_per_source'].get(src, 0.0)
            print(f" {d:>7.2f}", end="")
        print()
        print()

    # Summary
    print("-" * 60)
    classifications = {}
    for metric, data in results.items():
        cls = classify_metric(data['d1_drop'], data['source_variance'])
        classifications.setdefault(cls, []).append(metric)

    for cls in ["HEADLINE", "WORKHORSE", "SCIENTIFIC", "DEAD"]:
        metrics = classifications.get(cls, [])
        if metrics:
            print(f"  {cls:12s}: {', '.join(metrics)}")

    # Aggregate D1 drops (old formula vs new)
    all_d1 = [data['d1_drop'] for data in results.values()]
    headline_d1 = [data['d1_drop'] for m, data in results.items()
                   if classify_metric(data['d1_drop'], data['source_variance']) == "HEADLINE"]

    print()
    print(f"  Aggregate D1 drop (all metrics, old formula): {np.mean(all_d1):.1%}")
    if headline_d1:
        print(f"  Headline-only D1 drop:                        {np.mean(headline_d1):.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Per-metric 2×2 ablation for H3/H4")
    parser.add_argument('--n-trials', type=int, default=N_TRIALS,
                        help=f'Number of trials (default: {N_TRIALS})')
    parser.add_argument('--geometry',
                        choices=['e8', 'd4', 'h3', 'h4', 'hyp', 'lor',
                                 'pen', 'ab', 'hat', 'dodec', 'sept', 'all'],
                        default='all',
                        help='Which geometry to run (default: all)')
    parser.add_argument('--d1-threshold', type=float, default=0.30,
                        help='D1 drop threshold for structure-dependent (default: 0.30)')
    parser.add_argument('--sv-threshold', type=float, default=0.5,
                        help='Source variance threshold for discriminative (default: 0.5)')
    args = parser.parse_args()

    rng = np.random.default_rng(2026)

    geos = []
    if args.geometry in ('e8', 'all'):
        e8 = E8Geometry()
        _ = e8.roots  # force init
        geos.append(('E8 Lattice', e8))

    if args.geometry in ('d4', 'all'):
        d4 = D4Geometry()
        _ = d4.roots
        geos.append(('D4 Triality', d4))

    if args.geometry in ('h3', 'all'):
        h3 = H3CoxeterGeometry()
        _ = h3.roots
        geos.append(('H3 Icosahedral', h3))

    if args.geometry in ('h4', 'all'):
        h4 = H4CoxeterGeometry()
        _ = h4.roots
        geos.append(('H4 600-Cell', h4))

    if args.geometry in ('hyp', 'all'):
        hyp = HyperbolicGeometry()
        geos.append(('Hyperbolic (Poincaré)', hyp))

    if args.geometry in ('lor', 'all'):
        lor = LorentzianGeometry()
        geos.append(('Lorentzian', lor))

    # Symplectic excluded: no swappable object per ablation spec.
    # It's a workhorse-only geometry (no meaningful control variant).

    if args.geometry in ('pen', 'all'):
        pen = PenroseGeometry()
        geos.append(('Penrose', pen))

    if args.geometry in ('ab', 'all'):
        ab = AmmannBeenkerGeometry()
        geos.append(('Ammann-Beenker', ab))

    if args.geometry in ('hat', 'all'):
        hat = EinsteinHatGeometry()
        geos.append(('Einstein Hat', hat))

    if args.geometry in ('dodec', 'all'):
        dodec = DodecagonalGeometry()
        geos.append(('Dodecagonal', dodec))

    if args.geometry in ('sept', 'all'):
        sept = SeptagonalGeometry()
        geos.append(('Septagonal', sept))

    for geo_name, geo_instance in geos:
        print(f"\n{'#'*80}")
        print(f"# {geo_name}")
        # Determine variant type from geometry name
        vtype_map = {
            'Hyperbolic': 'hyperbolic', 'Lorentzian': 'lorentzian',
            'Symplectic': 'symplectic', 'Penrose': 'qc',
            'Ammann-Beenker': 'qc', 'Einstein Hat': 'hat',
            'Dodecagonal': 'qc', 'Septagonal': 'qc',
        }
        vtype = 'root'  # default for root-system geometries
        for prefix, vt in vtype_map.items():
            if prefix in geo_name:
                vtype = vt
                break

        if hasattr(geo_instance, 'roots') and hasattr(geo_instance, '_n_effective_dirs'):
            _ = geo_instance.roots  # ensure init
            print(f"# {len(geo_instance.roots)} roots, dim={geo_instance.dimension}, "
                  f"{geo_instance._n_effective_dirs} effective directions")
        else:
            print(f"# dim={geo_instance.dimension}, variant_type={vtype}")
        print(f"# {args.n_trials} trials per (source × level)")
        print(f"{'#'*80}\n")

        results = per_metric_d1_drop(
            geo_name, geo_instance, rng, n_trials=args.n_trials,
            variant_type=vtype)
        print_results(geo_name, results)


if __name__ == '__main__':
    main()
