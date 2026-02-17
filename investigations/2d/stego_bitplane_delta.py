#!/usr/bin/env python3
"""
Investigation: 2D Bit-Plane Delta Steganalysis.
Analyzes each bit-plane as a 64x64 grid and measures the 'tension shock'
between planes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import importlib.util

# Load generators from stego_deep
stego_mod = importlib.util.spec_from_file_location("stego", "investigations/1d/stego_deep.py")
stego = importlib.util.module_from_spec(stego_mod)
stego_mod.loader.exec_module(stego)

N_TRIALS = 30
DATA_SIZE = 4096

def compute_plane_metrics(data, analyzer):
    """Compute spatial metrics for each of the 8 bit-planes."""
    bits = np.unpackbits(data).reshape(8, 64, 64).astype(np.float64)
    all_plane_metrics = []
    for p in range(8):
        res = analyzer.analyze(bits[p])
        combined = {}
        for r in res.results:
            for mn, mv in r.metrics.items():
                combined[f"{r.geometry_name}:{mn}"] = mv
        all_plane_metrics.append(combined)
    return all_plane_metrics

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def main():
    print("=" * 70)
    print("INVESTIGATION: 2D Bit-Plane Delta Steganalysis")
    print("=" * 70)
    
    analyzer = GeometryAnalyzer().add_spatial_geometries()
    techs = ['LSB Replace', 'LSB Match', 'PVD', 'Spread Spectrum', 'Matrix Embed']
    carrier_type = 'natural_texture'
    
    # 1. Profile Clean Baseline
    print(f"Profiling Clean Baseline ({carrier_type})...")
    # Store: bit_index -> metric_name -> [values]
    clean_profile = {p: defaultdict(list) for p in range(8)}
    
    for trial in range(N_TRIALS):
        data = stego.generate_carrier(carrier_type, trial, size=DATA_SIZE)
        p_metrics = compute_plane_metrics(data, analyzer)
        for p in range(8):
            for mn, mv in p_metrics[p].items():
                clean_profile[p][mn].append(mv)
                
    # Show clean tension profile (mean)
    print("")
    print("Clean Tension Profile (Bit 7 -> Bit 0):")
    t_means = [np.mean(clean_profile[p]['Spatial Field:tension_mean']) for p in range(8)]
    print("  " + " -> ".join([f"{v:.3f}" for v in t_means]))
    
    # 2. Test Stego Shocks
    for tech_name in techs:
        print("")
        print(f"Testing {tech_name}...", end=" ", flush=True)
        stego_profile = {p: defaultdict(list) for p in range(8)}
        embed_fn = stego.TECHNIQUES[tech_name]
        
        for trial in range(N_TRIALS):
            carrier = stego.generate_carrier(carrier_type, trial, size=DATA_SIZE)
            stego_data = embed_fn(carrier, rate=1.0, seed=trial + 1000)
            p_metrics = compute_plane_metrics(stego_data, analyzer)
            for p in range(8):
                for mn, mv in p_metrics[p].items():
                    stego_profile[p][mn].append(mv)
        
        # Compare Bit 0 metrics
        detections = []
        n_metrics = len(clean_profile[0])
        alpha = 0.05 / n_metrics
        for mn in clean_profile[0]:
            d = cohens_d(stego_profile[0][mn], clean_profile[0][mn])
            _, p = stats.ttest_ind(stego_profile[0][mn], clean_profile[0][mn], equal_var=False)
            if p < alpha and abs(d) > 0.8:
                detections.append((mn, d, p))
                
        if detections:
            print(f"DETECTED! ({len(detections)} metrics at Bit 0)")
            for mn, d, p in sorted(detections, key=lambda x: -abs(x[1]))[:3]:
                print(f"    {mn:<25} d={d:+.2f} p={p:.2e}")
        else:
            print("Invisible at Bit 0.")

    print("")
    print("[Investigation complete]")

if __name__ == "__main__":
    main()
