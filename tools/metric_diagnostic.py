#!/usr/bin/env python3
"""Metric Performance Diagnostic — reads atlas JSON, prints structured report."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ATLAS = Path(__file__).resolve().parent.parent / "figures" / "structure_atlas_data.json"


def load():
    with open(ATLAS) as f:
        return json.load(f)


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


def run():
    d = load()
    names = d["metric_names"]  # 257
    P = np.array(d["profiles"])  # 180 x 257
    loadings = np.array(d["pca_loadings"])  # 180 x 257
    evr = np.array(d["explained_variance_ratio"])  # 180
    sources = d["sources"]  # 180 dicts with 'domain'
    views = d["views"]

    n_src, n_met = P.shape
    domains = [s["domain"] for s in sources]
    unique_domains = sorted(set(domains))
    domain_idx = {dom: [i for i, d in enumerate(domains) if d == dom]
                  for dom in unique_domains}

    # Geometry grouping from metric name prefix
    geom_of = {}  # metric index -> geometry name
    geom_metrics = {}  # geometry name -> list of metric indices
    for i, name in enumerate(names):
        geom = name.rsplit(":", 1)[0]
        geom_of[i] = geom
        geom_metrics.setdefault(geom, []).append(i)

    # ── 1. Per-Metric Summary Stats ──────────────────────────────────────
    section("1. Per-Metric Summary Stats")
    means = np.nanmean(P, axis=0)
    stds = np.nanstd(P, axis=0)
    cvs = np.where(means != 0, stds / np.abs(means), 0.0)
    mins = np.nanmin(P, axis=0)
    maxs = np.nanmax(P, axis=0)

    # % at floor/ceiling (within 1e-9 of min or max per metric)
    at_floor = np.array([np.mean(np.abs(P[:, j] - mins[j]) < 1e-9)
                         for j in range(n_met)])
    at_ceil = np.array([np.mean(np.abs(P[:, j] - maxs[j]) < 1e-9)
                        for j in range(n_met)])
    # % identical to mode
    mode_frac = np.array([np.max(np.unique(P[:, j], return_counts=True)[1]) / n_src
                          for j in range(n_met)])

    near_const = np.where(cvs < 0.01)[0]
    mode_dominated = np.where(mode_frac > 0.90)[0]

    # Floor/ceiling pile-up: >50% of sources at min or max value
    pileup_floor = np.where(at_floor > 0.50)[0]
    pileup_ceil = np.where(at_ceil > 0.50)[0]
    pileup = np.union1d(pileup_floor, pileup_ceil)

    # Excess kurtosis: heavy-tailed metrics that distort z-score scaling
    kurt = np.array([float(stats.kurtosis(P[:, j], nan_policy='omit'))
                     for j in range(n_met)])
    heavy_tail = np.where(kurt > 10)[0]

    print(f"Metrics: {n_met}   Sources: {n_src}   Domains: {len(unique_domains)}")
    print(f"Near-constant (CV < 0.01): {len(near_const)}")
    for j in near_const:
        print(f"  {names[j]:50s}  CV={cvs[j]:.4f}  mean={means[j]:.4f}")
    print(f"\nMode-dominated (>90% identical): {len(mode_dominated)}")
    for j in mode_dominated:
        print(f"  {names[j]:50s}  mode_frac={mode_frac[j]:.2%}  CV={cvs[j]:.4f}")
    print(f"\nFloor/ceiling pile-up (>50% at min or max): {len(pileup)}")
    for j in pileup:
        end = "floor" if at_floor[j] > 0.50 else "ceil"
        frac = at_floor[j] if end == "floor" else at_ceil[j]
        print(f"  {names[j]:50s}  {end}={frac:.0%}  range=[{mins[j]:.4f}, {maxs[j]:.4f}]")
    print(f"\nHeavy-tailed (excess kurtosis > 10): {len(heavy_tail)}")
    for j in sorted(heavy_tail, key=lambda j: -kurt[j]):
        print(f"  {names[j]:50s}  kurt={kurt[j]:.1f}  range=[{mins[j]:.4f}, {maxs[j]:.4f}]")

    # ── 2. Discrimination Power (ANOVA F-test by domain) ─────────────────
    section("2. Discrimination Power (ANOVA F-test by domain)")
    f_stats = np.zeros(n_met)
    p_vals = np.ones(n_met)
    for j in range(n_met):
        groups = [P[idx, j] for idx in domain_idx.values() if len(idx) > 1]
        # Need at least 2 groups with >1 member and some variance
        if len(groups) >= 2:
            try:
                f_stats[j], p_vals[j] = stats.f_oneway(*groups)
            except Exception:
                pass
    # Handle NaN
    f_stats = np.nan_to_num(f_stats, nan=0.0)

    rank = np.argsort(f_stats)[::-1]
    print("Top-20 discriminating metrics (by ANOVA F-stat across domains):")
    print(f"  {'Rank':>4s}  {'Metric':50s}  {'F-stat':>10s}  {'p-value':>10s}")
    for i, j in enumerate(rank[:20]):
        print(f"  {i+1:4d}  {names[j]:50s}  {f_stats[j]:10.1f}  {p_vals[j]:10.2e}")

    print("\nBottom-20 (least discriminating):")
    for i, j in enumerate(rank[-20:]):
        print(f"  {n_met-19+i:4d}  {names[j]:50s}  {f_stats[j]:10.3f}  {p_vals[j]:10.2e}")

    # ── 3. Global PCA Loadings ───────────────────────────────────────────
    section("3. Global PCA Loadings")

    # Recompute PCA on rank-normalized profiles (matches atlas pipeline).
    # Rank normalization: each column → percentile ranks.  Robust to heavy
    # tails and incommensurate units across metrics.
    from scipy.stats import rankdata as _rankdata
    R = np.zeros_like(P)
    for j in range(n_met):
        col = np.nan_to_num(P[:, j], nan=0.0)
        if np.std(col) < 1e-15:
            R[:, j] = 0.5
        else:
            R[:, j] = _rankdata(col) / n_src
    P_std = R - R.mean(axis=0)
    U, S, Vt = np.linalg.svd(P_std, full_matrices=False)
    total_var = np.sum(S**2)
    evr_recomp = S**2 / total_var

    cum_var = np.cumsum(evr_recomp)
    for thresh in [0.50, 0.80, 0.95]:
        n_needed = int(np.searchsorted(cum_var, thresh)) + 1
        print(f"  PCs for {thresh:.0%} variance: {n_needed}")

    print()
    for pc_idx in range(min(5, len(Vt))):
        comp = Vt[pc_idx]  # length 257
        top5 = np.argsort(np.abs(comp))[::-1][:5]
        print(f"  PC{pc_idx+1} ({evr_recomp[pc_idx]:.1%} variance):")
        for j in top5:
            print(f"    {comp[j]:+.4f}  {names[j]}")

    # ── 4. Per-View Top Metrics ──────────────────────────────────────────
    section("4. Per-View Top Metrics")
    for vname, vdata in sorted(views.items()):
        geoms_in_view = vdata["geometries"]
        # Collect metric indices belonging to this view's geometries
        view_metric_idx = []
        for g in geoms_in_view:
            view_metric_idx.extend(geom_metrics.get(g, []))
        if not view_metric_idx:
            continue
        view_metric_idx = sorted(view_metric_idx)

        # Sub-PCA on this view's metrics (rank-normalized, matching atlas)
        Pv = P[:, view_metric_idx]
        Rv = np.zeros_like(Pv)
        for jj in range(Pv.shape[1]):
            col = np.nan_to_num(Pv[:, jj], nan=0.0)
            if np.std(col) < 1e-15:
                Rv[:, jj] = 0.5
            else:
                Rv[:, jj] = _rankdata(col) / n_src
        Pv_std = Rv - Rv.mean(axis=0)
        _, Sv, Vv = np.linalg.svd(Pv_std, full_matrices=False)
        evr_v = Sv**2 / np.sum(Sv**2)

        print(f"  {vname} ({len(view_metric_idx)} metrics, {len(geoms_in_view)} geometries)")
        for pc_i in range(min(2, len(Vv))):
            comp = Vv[pc_i]
            top3 = np.argsort(np.abs(comp))[::-1][:3]
            print(f"    PC{pc_i+1} ({evr_v[pc_i]:.1%}):", end="")
            for k in top3:
                j = view_metric_idx[k]
                print(f"  {names[j]}({comp[k]:+.3f})", end="")
            print()
        # Cross-ref: how many of this view's metrics are in global top-50?
        global_top50 = set(rank[:50])
        in_top50 = sum(1 for j in view_metric_idx if j in global_top50)
        print(f"    Global top-50 overlap: {in_top50}/{len(view_metric_idx)}")
        print()

    # ── 5. Redundancy (Correlation Clusters) ─────────────────────────────
    section("5. Redundancy (Correlation Clusters)")
    # Spearman rank correlation
    rho, _ = stats.spearmanr(P, axis=0)  # 257 x 257
    if rho.ndim == 0:
        rho = np.array([[1.0]])
    # Distance = 1 - |rho|
    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)  # numerical hygiene
    # Make symmetric
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    # Threshold |r| > 0.9 → distance < 0.1
    labels = fcluster(Z, t=0.1, criterion="distance")
    n_clusters = len(set(labels))
    n_effective = n_clusters  # each cluster ≈ 1 independent metric

    # Find multi-metric clusters
    from collections import Counter
    cluster_sizes = Counter(labels)
    multi = {c: size for c, size in cluster_sizes.items() if size > 1}

    print(f"Total metrics: {n_met}")
    print(f"Correlation clusters (|r| > 0.9): {n_clusters}")
    print(f"Effective independent metrics: {n_effective}")
    print(f"Multi-metric clusters: {len(multi)}")
    print()

    for c, size in sorted(multi.items(), key=lambda x: -x[1])[:15]:
        members = [i for i, l in enumerate(labels) if l == c]
        # Pick representative (highest F-stat)
        rep = max(members, key=lambda j: f_stats[j])
        print(f"  Cluster (n={size}), rep: {names[rep]}")
        for j in members:
            tag = " *" if j == rep else ""
            print(f"    {names[j]:50s}  F={f_stats[j]:8.1f}  CV={cvs[j]:.3f}{tag}")
        print()

    # ── 6. Per-Geometry Scorecard ────────────────────────────────────────
    section("6. Per-Geometry Scorecard")
    global_top50 = set(rank[:50])
    # Top-3 loading metrics for first 5 PCs
    pca_top3 = set()
    for pc_idx in range(min(5, len(Vt))):
        top3 = np.argsort(np.abs(Vt[pc_idx]))[::-1][:3]
        pca_top3.update(top3)

    f_quartile_25 = np.percentile(f_stats, 25)
    print(f"  {'Geometry':40s} {'#Met':>4s} {'Med F':>8s} {'Top50':>5s} "
          f"{'Dead%':>5s} {'PCA?':>4s}")
    print(f"  {'-'*40} {'----':>4s} {'--------':>8s} {'-----':>5s} "
          f"{'-----':>5s} {'----':>4s}")

    geom_scores = []
    for geom in sorted(geom_metrics.keys()):
        idxs = geom_metrics[geom]
        n = len(idxs)
        med_f = float(np.median([f_stats[j] for j in idxs]))
        in_top50 = sum(1 for j in idxs if j in global_top50)
        dead = sum(1 for j in idxs if cvs[j] < 0.05) / n if n else 0
        has_pca = any(j in pca_top3 for j in idxs)
        geom_scores.append((geom, n, med_f, in_top50, dead, has_pca))

    # Sort by median F descending
    geom_scores.sort(key=lambda x: -x[2])
    for geom, n, med_f, in_top50, dead, has_pca in geom_scores:
        print(f"  {geom:40s} {n:4d} {med_f:8.1f} {in_top50:5d} "
              f"{dead:5.0%} {'yes' if has_pca else '':>4s}")

    # ── 7. Dead Metric Candidates ────────────────────────────────────────
    section("7. Dead Metric Candidates")
    # Two-gate: low F-stat AND redundant with a better metric.
    # (CV gate dropped — all 257 metrics have CV > 0.05 in this dataset.)
    low_f = set(np.where(f_stats < f_quartile_25)[0])

    # For each metric, check if it's |r|>0.90 with a better-F-stat metric
    redundant = set()
    proxy_of = {}  # j -> best correlated better metric
    for j in range(n_met):
        cluster_members = [i for i, l in enumerate(labels) if l == labels[j]]
        better = [i for i in cluster_members if i != j and f_stats[i] > f_stats[j]]
        if better:
            for b in sorted(better, key=lambda i: -f_stats[i]):
                if abs(rho[j, b]) > 0.90:
                    redundant.add(j)
                    proxy_of[j] = b
                    break

    dead = low_f & redundant
    print(f"Low F-stat (bottom quartile, F < {f_quartile_25:.1f}): {len(low_f)}")
    print(f"Redundant (|r| > 0.90 with better metric): {len(redundant)}")
    print(f"Dead metrics (low F AND redundant): {len(dead)}")
    print()

    if dead:
        print(f"  {'Metric':50s} {'F-stat':>8s} {'|r|':>5s} {'Better Proxy'}")
        print(f"  {'-'*50} {'--------':>8s} {'-----':>5s} {'-'*40}")
        for j in sorted(dead, key=lambda j: f_stats[j]):
            b = proxy_of[j]
            print(f"  {names[j]:50s} {f_stats[j]:8.2f} {abs(rho[j,b]):5.3f} "
                  f"{names[b]}")
    else:
        print("  (none — all low-F metrics are uncorrelated with better ones)")

    # Summary
    section("Summary")
    print(f"  Total metrics:                 {n_met}")
    print(f"  Effective independent metrics:  {n_effective}")
    print(f"  Near-constant (CV < 0.01):     {len(near_const)}")
    print(f"  Dead metric candidates:        {len(dead)}")
    print(f"  Top discriminator:             {names[rank[0]]} (F={f_stats[rank[0]]:.0f})")
    print(f"  PCs for 80% variance:          {int(np.searchsorted(cum_var, 0.80)) + 1}")
    print(f"  Correlation clusters:          {n_clusters}")
    print()


if __name__ == "__main__":
    run()
