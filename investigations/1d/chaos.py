#!/usr/bin/env python3
"""
Investigation: Can exotic geometries fingerprint different chaotic maps?

We know chaos differs from random (E8 and Sol detect this). But can we
tell WHICH chaotic system generated the data? This would be useful for:
- Identifying dynamical systems from time series
- Classifying attractors
- Detecting specific chaos generators in applications

Maps tested: logistic, Henon, tent, Lorenz, Rossler, standard, baker's, Arnold cat
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer

N_TRIALS = 25
DATA_SIZE = 2000

def generate_chaotic_data(map_name, trial_seed, size=DATA_SIZE):
    """Generate data from various chaotic maps."""
    warmup = 500  # discard transient

    if map_name == 'logistic':
        r = 3.99
        x = 0.1 + 0.001 * (trial_seed % 100)
        vals = []
        for _ in range(size + warmup):
            x = r * x * (1 - x)
            vals.append(x)
        arr = np.array(vals[warmup:warmup+size])
        return (arr * 255).astype(np.uint8)

    elif map_name == 'logistic_edge':
        # Logistic at edge of chaos (r=3.57, period-doubling cascade)
        r = 3.57
        x = 0.1 + 0.001 * trial_seed
        vals = []
        for _ in range(size + warmup):
            x = r * x * (1 - x)
            vals.append(x)
        arr = np.array(vals[warmup:warmup+size])
        return (arr * 255).astype(np.uint8)

    elif map_name == 'henon':
        x, y = 0.1 + 0.001 * trial_seed, 0.1
        vals = []
        for _ in range(size + warmup):
            x_new = 1 - 1.4 * x * x + y
            y_new = 0.3 * x
            x, y = x_new, y_new
            vals.append(x)
        arr = np.array(vals[warmup:warmup+size])
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        return (arr * 255).astype(np.uint8)

    elif map_name == 'tent':
        mu = 1.99
        x = 0.1 + 0.001 * trial_seed
        vals = []
        for _ in range(size + warmup):
            x = mu * min(x, 1 - x)
            vals.append(x)
        arr = np.array(vals[warmup:warmup+size])
        return (arr * 255).astype(np.uint8)

    elif map_name == 'lorenz_x':
        # Lorenz system, x component
        dt = 0.01
        x, y, z = 1.0 + 0.01 * trial_seed, 1.0, 1.0
        sigma, rho, beta = 10.0, 28.0, 8/3
        vals = []
        for _ in range((size + warmup) * 10):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            vals.append(x)
        vals = vals[warmup*10::10]  # subsample
        arr = np.array(vals[:size])
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        return (arr * 255).astype(np.uint8)

    elif map_name == 'lorenz_z':
        # Lorenz system, z component (different character)
        dt = 0.01
        x, y, z = 1.0 + 0.01 * trial_seed, 1.0, 1.0
        sigma, rho, beta = 10.0, 28.0, 8/3
        vals = []
        for _ in range((size + warmup) * 10):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            vals.append(z)
        vals = vals[warmup*10::10]
        arr = np.array(vals[:size])
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        return (arr * 255).astype(np.uint8)

    elif map_name == 'rossler':
        dt = 0.02
        x, y, z = 1.0 + 0.01 * trial_seed, 1.0, 1.0
        a, b, c = 0.2, 0.2, 5.7
        vals = []
        for _ in range((size + warmup) * 10):
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            x += dx * dt
            y += dy * dt
            z += dz * dt
            vals.append(x)
        vals = vals[warmup*10::10]
        arr = np.array(vals[:size])
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        return (arr * 255).astype(np.uint8)

    elif map_name == 'standard_map':
        # Chirikov standard map (K=6, chaotic)
        K = 6.0
        p, theta = 0.5 + 0.01 * trial_seed, 0.1
        vals = []
        for _ in range(size + warmup):
            p = (p + K / (2 * np.pi) * np.sin(2 * np.pi * theta)) % 1
            theta = (theta + p) % 1
            vals.append(theta)
        arr = np.array(vals[warmup:warmup+size])
        return (arr * 255).astype(np.uint8)

    elif map_name == 'baker':
        # Baker's map
        x, y = 0.1 + 0.001 * trial_seed, 0.3
        vals = []
        for _ in range(size + warmup):
            if x < 0.5:
                x, y = 2 * x, y / 2
            else:
                x, y = 2 * x - 1, (y + 1) / 2
            vals.append(x)
        arr = np.array(vals[warmup:warmup+size])
        return (arr * 255).astype(np.uint8)

    elif map_name == 'arnold_cat':
        # Arnold's cat map on unit square
        x, y = 0.1 + 0.001 * trial_seed, 0.3
        vals = []
        for _ in range(size + warmup):
            x_new = (2 * x + y) % 1
            y_new = (x + y) % 1
            x, y = x_new, y_new
            vals.append(x)
        arr = np.array(vals[warmup:warmup+size])
        return (arr * 255).astype(np.uint8)

    elif map_name == 'random':
        rng = np.random.RandomState(trial_seed)
        return rng.randint(0, 256, size, dtype=np.uint8)

    else:
        raise ValueError(f"Unknown map: {map_name}")


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    print("=" * 70)
    print("INVESTIGATION: Chaotic Map Fingerprinting via Exotic Geometries")
    print("=" * 70)

    maps = ['random', 'logistic', 'logistic_edge', 'henon', 'tent', 'lorenz_x',
            'lorenz_z', 'rossler', 'standard_map', 'baker', 'arnold_cat']

    key_metrics = [
        'E8 Lattice:unique_roots',
        'E8 Lattice:alignment_mean',
        'Torus T^2:coverage',
        'Torus T^2:chi2_uniformity',
        'Sol:anisotropy',
        'Sol:stretch_ratio',
        'Heisenberg (Nil):twist_rate',
        'Heisenberg (Nil) (centered):twist_rate',
        'Tropical:linearity',
        'Persistent Homology:n_significant_features',
        'Penrose (Quasicrystal):fivefold_balance',
        'Fisher Information:trace_fisher',
        'Cantor (base 3):cantor_dimension',
        'Lorentzian:causal_density',
        'Hyperbolic (Poincaré):mean_radius',
    ]

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Collect metrics
    all_metrics = {m: defaultdict(list) for m in maps}

    for map_name in maps:
        print(f"  Processing {map_name}...", end=" ", flush=True)
        for trial in range(N_TRIALS):
            try:
                data = generate_chaotic_data(map_name, trial)
                results = analyzer.analyze(data)
                for r in results.results:
                    for mn, mv in r.metrics.items():
                        all_metrics[map_name][f"{r.geometry_name}:{mn}"].append(mv)
            except Exception as e:
                if trial == 0:
                    print(f"Error: {e}", end=" ")
        print("done")

    # 1. Fingerprint table
    print(f"\n{'='*70}")
    print("1. GEOMETRIC FINGERPRINT TABLE")
    print(f"{'='*70}\n")

    header = f"{'Map':<16}" + "".join(f"{km.split(':')[1][:10]:>12}" for km in key_metrics[:8])
    print(header)
    print("-" * len(header))

    for m in maps:
        row = f"{m:<16}"
        for km in key_metrics[:8]:
            vals = all_metrics[m].get(km, [0])
            row += f"{np.mean(vals):>12.3f}"
        print(row)

    # 2. Can we distinguish each chaotic map from random?
    print(f"\n{'='*70}")
    print("2. EACH MAP vs RANDOM")
    print(f"{'='*70}\n")

    n_tests = len(key_metrics)
    alpha = 0.05 / max(n_tests, 1)

    map_detections = defaultdict(list)
    for m in maps:
        if m == 'random':
            continue
        for km in key_metrics:
            if km not in all_metrics['random'] or km not in all_metrics[m]:
                continue
            if len(all_metrics['random'][km]) < 2 or len(all_metrics[m][km]) < 2:
                continue
            d = cohens_d(all_metrics[m][km], all_metrics['random'][km])
            _, p = stats.ttest_ind(all_metrics[m][km], all_metrics['random'][km], equal_var=False)
            if abs(d) > 0.8 and p < alpha:
                map_detections[m].append((km, d, p))

    for m in maps:
        if m == 'random':
            continue
        dets = map_detections.get(m, [])
        print(f"\n{m}: {len(dets)} significant detections")
        for km, d, p in sorted(dets, key=lambda x: -abs(x[1]))[:5]:
            print(f"  {km:<45} d={d:+.2f} p={p:.2e}")

    # 3. Pairwise: can we distinguish chaos from chaos?
    print(f"\n{'='*70}")
    print("3. PAIRWISE CHAOTIC MAP DISCRIMINATION")
    print(f"{'='*70}\n")

    chaos_maps = [m for m in maps if m not in ('random',)]
    n_pairs = len(list(__import__('itertools').combinations(chaos_maps, 2)))
    n_pair_tests = len(key_metrics)
    alpha_pair = 0.05 / max(n_pair_tests, 1)

    pair_results = []
    for i, m1 in enumerate(chaos_maps):
        for m2 in chaos_maps[i+1:]:
            best_d = 0
            best_km = None
            best_p = 1
            for km in key_metrics:
                if km not in all_metrics[m1] or km not in all_metrics[m2]:
                    continue
                if len(all_metrics[m1][km]) < 2 or len(all_metrics[m2][km]) < 2:
                    continue
                d = cohens_d(all_metrics[m1][km], all_metrics[m2][km])
                _, p = stats.ttest_ind(all_metrics[m1][km], all_metrics[m2][km], equal_var=False)
                if abs(d) > abs(best_d):
                    best_d = d
                    best_km = km
                    best_p = p
            pair_results.append((m1, m2, best_km, best_d, best_p))

    # Sort by distinguishability
    pair_results.sort(key=lambda x: -abs(x[3]))

    print(f"{'Map1':<16} {'Map2':<16} {'Best Metric':<35} {'d':>8} {'p':>12} {'Dist?':>6}")
    print("-" * 95)
    for m1, m2, km, d, p in pair_results:
        sig = "YES" if p < alpha_pair and abs(d) > 0.8 else "no"
        km_short = km.split(':')[1] if km else '?'
        print(f"{m1:<16} {m2:<16} {km_short:<35} {d:>+8.2f} {p:>12.2e} {sig:>6}")

    # 4. Which geometry is best for chaos fingerprinting?
    print(f"\n{'='*70}")
    print("4. BEST GEOMETRIES FOR CHAOS FINGERPRINTING")
    print(f"{'='*70}\n")

    # For each geometry, count how many pairwise distinctions it enables
    geom_power = defaultdict(int)
    for m1, m2, best_km, d, p in pair_results:
        if p < alpha_pair and abs(d) > 0.8 and best_km:
            geom = best_km.split(':')[0]
            geom_power[geom] += 1

    print(f"{'Geometry':<30} {'Pairwise distinctions':>25}")
    print("-" * 60)
    for g, count in sorted(geom_power.items(), key=lambda x: -x[1]):
        n_possible = n_pairs
        print(f"{g:<30} {count:>10}/{n_possible}")

    # 5. Unique geometric signatures
    print(f"\n{'='*70}")
    print("5. UNIQUE SIGNATURES (metrics that uniquely identify a map)")
    print(f"{'='*70}\n")

    for m in chaos_maps:
        unique_features = []
        for km in key_metrics:
            if km not in all_metrics[m]:
                continue
            m_vals = all_metrics[m][km]
            # Is this metric significantly different from ALL other maps?
            all_different = True
            min_d = float('inf')
            for m2 in chaos_maps:
                if m2 == m:
                    continue
                if km not in all_metrics[m2]:
                    all_different = False
                    break
                d = abs(cohens_d(m_vals, all_metrics[m2][km]))
                _, p = stats.ttest_ind(m_vals, all_metrics[m2][km], equal_var=False)
                if d < 0.8 or p >= alpha_pair:
                    all_different = False
                    break
                min_d = min(min_d, d)
            if all_different and min_d < float('inf'):
                unique_features.append((km, min_d))

        if unique_features:
            print(f"{m}: {len(unique_features)} unique signature metric(s)")
            for km, min_d in sorted(unique_features, key=lambda x: -x[1]):
                print(f"  {km}: min |d| = {min_d:.2f}")
        else:
            print(f"{m}: no unique signature (overlaps with other maps)")

    # Shuffle validation
    print(f"\n{'='*70}")
    print("6. SHUFFLE VALIDATION")
    print(f"{'='*70}\n")

    for m in ['logistic', 'henon', 'lorenz_x', 'tent']:
        data = generate_chaotic_data(m, 999)
        results_real = analyzer.analyze(data)
        shuffled = data.copy()
        np.random.shuffle(shuffled)
        results_shuf = analyzer.analyze(shuffled)

        print(f"{m}:")
        for r, rs in zip(results_real.results, results_shuf.results):
            for mn in r.metrics:
                fk = f"{r.geometry_name}:{mn}"
                if fk in key_metrics[:6]:
                    real_v = r.metrics[mn]
                    shuf_v = rs.metrics[mn]
                    ratio = real_v / (shuf_v + 1e-10)
                    if abs(ratio - 1.0) > 0.15:
                        print(f"  {fk[:44]}: real={real_v:.3f} shuf={shuf_v:.3f} ratio={ratio:.2f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    n_detected = sum(1 for m in chaos_maps if map_detections.get(m))
    n_pairwise = sum(1 for _, _, _, d, p in pair_results if abs(d) > 0.8 and p < alpha_pair)
    print(f"\nMaps detected as non-random: {n_detected}/{len(chaos_maps)}")
    print(f"Distinguishable pairs: {n_pairwise}/{n_pairs}")

    print("\n[Investigation complete]")


if __name__ == '__main__':
    main()
