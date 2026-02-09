#!/usr/bin/env python3
"""
Training Engine for the Exotic Geometry Framework.
Computes and saves high-dimensional signatures for mathematical systems.
"""

import os
import json
import numpy as np
from exotic_geometry_framework import GeometryAnalyzer, delay_embed

DEFAULT_SCALES = [1, 2, 5, 10]
DEFAULT_TRIALS = 20
DEFAULT_SIZE = 2000

def train_system_signature(name, generator_fn, analyzer=None, scales=DEFAULT_SCALES, 
                           n_trials=DEFAULT_TRIALS, size=DEFAULT_SIZE):
    """
    Train a signature for a system.
    
    generator_fn: (trial_seed, size) -> np.ndarray (uint8)
    """
    print(f"Training signature for '{name}'...")
    if analyzer is None:
        analyzer = GeometryAnalyzer().add_all_geometries()

    # Collect all signatures across trials
    all_trial_sigs = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...")
        data = generator_fn(trial, size)
        
        trial_sig_parts = []
        for tau in scales:
            if tau == 1:
                scaled_data = data
            else:
                scaled_data = delay_embed(data, tau=tau)
                # Ensure consistent length for small data
                if len(scaled_data) > size:
                    scaled_data = scaled_data[:size]
                elif len(scaled_data) < size // 2:
                    # Skip if too short
                    continue
            
            results = analyzer.analyze(scaled_data)
            # Flatten all metrics into a consistent list
            for r in results.results:
                for mname in sorted(r.metrics.keys()):
                    trial_sig_parts.append(r.metrics[mname])
        
        all_trial_sigs.append(trial_sig_parts)
    print(f"  Done. Computing statistics...")

    all_trial_sigs = np.array(all_trial_sigs)
    means = np.mean(all_trial_sigs, axis=0)
    stds = np.std(all_trial_sigs, axis=0)
    
    # Get metric names for documentation/debugging in the JSON
    metric_names = []
    # Use trial 0 results to get names
    test_data = generator_fn(0, size)
    for tau in scales:
        if tau > 1:
            scaled_data = delay_embed(test_data, tau=tau)
        else:
            scaled_data = test_data
            
        results = analyzer.analyze(scaled_data)
        for r in results.results:
            for mname in sorted(r.metrics.keys()):
                metric_names.append(f"{r.geometry_name}:{mname}:tau{tau}")

    signature_data = {
        "name": name,
        "scales": scales,
        "n_trials": n_trials,
        "size": size,
        "metrics": metric_names,
        "means": means.tolist(),
        "stds": stds.tolist()
    }
    
    os.makedirs("signatures", exist_ok=True)
    filename = f"signatures/{name.lower().replace(' ', '_')}.json"
    with open(filename, 'w') as f:
        json.dump(signature_data, f, indent=2)
    print(f"  Saved to {filename}")
    return filename

# --- Example generators for initial library ---

def gen_random(seed, size):
    return np.random.RandomState(seed).randint(0, 256, size, dtype=np.uint8)

def gen_henon(seed, size):
    # Keep initial condition small to stay in basin of attraction
    x, y = 0.1 + 0.0001 * (seed % 1000), 0.1
    vals = []
    for _ in range(size + 500):
        x_new = 1 - 1.4 * x * x + y
        y_new = 0.3 * x
        x, y = x_new, y_new
        # Clamp to prevent overflow if it diverges
        if abs(x) > 1000: x = 1000.0 * (1 if x > 0 else -1)
        vals.append(x)
    arr = np.array(vals[500:])
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
    return (arr * 255).astype(np.uint8)

def gen_logistic(seed, size):
    r = 3.99
    x = 0.1 + 0.001 * seed
    vals = []
    for _ in range(size + 500):
        x = r * x * (1 - x)
        vals.append(x)
    arr = np.array(vals[500:])
    return (arr * 255).astype(np.uint8)

if __name__ == "__main__":
    # Train some defaults
    analyzer = GeometryAnalyzer().add_all_geometries()
    train_system_signature("Random", gen_random, analyzer)
    train_system_signature("Henon Chaos", gen_henon, analyzer)
    train_system_signature("Logistic Chaos", gen_logistic, analyzer)
