import pytest
import numpy as np
from exotic_geometry_framework import (
    CantorGeometry,
    TropicalGeometry,
    SymplecticGeometry,
    PersistentHomologyGeometry,
    UltrametricGeometry
)

def generate_cantor_set_approximation(size, depth=4):
    """
    Generate points approximating a middle-third Cantor set.
    We'll construct numbers that only use digits 0 and 2 in base 3,
    mapped to 0-255.
    """
    # Cantor set construction in [0, 1]
    # Filter uniform random numbers to keep only those with ternary digits 0 or 2
    # This is hard to do perfectly with fixed size in one go, so we generate more and filter.
    
    # Alternative: Deterministic construction
    # Just generate random integers using only bits that map to "0" or "2" behavior?
    # Better: Use the geometry's own logic reversed, or just known properties.
    # The CantorGeometry maps bytes to [0,1] via ternary expansion.
    # If we supply bytes that, when expanded, only have 0s and 2s, we simulate the set.
    
    # 0 = 00000000 (base 3: 0)
    # 255 = 11111111 (base 3: large)
    # Actually, simpler: Generate numbers in [0,1] that are Cantor, then map to bytes.
    # But the geometry takes bytes.
    
    # Let's try a statistical approach:
    # A Cantor set has dimension ~0.63.
    # Uniform noise has dimension ~1.0.
    
    # To make "Cantor-like" bytes:
    # Avoid values that land in the middle third [1/3, 2/3].
    # Byte range [0, 255]. 1/3 is ~85, 2/3 is ~170.
    # Remove [85, 170].
    # Recursively remove middle thirds of remaining chunks.
    
    valid_ranges = [(0, 255)]
    for _ in range(depth):
        new_ranges = []
        for start, end in valid_ranges:
            width = end - start
            if width < 3:
                new_ranges.append((start, end))
                continue
            
            third = width // 3
            # Keep first third
            new_ranges.append((start, start + third))
            # Skip middle third
            # Keep last third
            new_ranges.append((end - third, end))
        valid_ranges = new_ranges
        
    # Generate data from valid ranges
    data = []
    rng = np.random.default_rng(42)
    while len(data) < size:
        r = valid_ranges[rng.integers(len(valid_ranges))]
        if r[1] > r[0]:
            val = rng.integers(r[0], r[1] + 1)
            data.append(val)
    
    return np.array(data[:size], dtype=np.uint8)

def test_cantor_dimension():
    """
    CantorGeometry should estimate dimension < 1.0 for Cantor-like data,
    and ~1.0 for uniform random data.
    """
    geom = CantorGeometry(base=3)
    size = 2000
    
    # 1. Cantor-like data
    cantor_data = generate_cantor_set_approximation(size, depth=3)
    res_cantor = geom.compute_metrics(cantor_data)
    dim_cantor = res_cantor.metrics['estimated_dimension']
    
    # 2. Uniform noise
    noise = np.random.randint(0, 256, size, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)
    dim_noise = res_noise.metrics['estimated_dimension']
    
    print(f"Cantor Dimension - Fractal: {dim_cantor:.3f}, Noise: {dim_noise:.3f}")
    
    # Known Cantor dimension is ln(2)/ln(3) ~= 0.63
    # Our approximation is crude (byte discretization), but should be distinct.
    assert dim_cantor < dim_noise - 0.1, "Fractal data should have lower dimension"
    assert dim_cantor < 0.9, "Cantor set dimension should be clearly fractional"

def test_tropical_linearity():
    """
    TropicalGeometry should detect piecewise linear structures.
    Triangle wave -> High linearity, few slope changes.
    Random walk -> Low linearity, many changes.
    """
    geom = TropicalGeometry()
    size = 1000
    
    # 1. Triangle Wave (Perfectly Piecewise Linear)
    # 0 -> 255 -> 0 over 100 steps
    x = np.linspace(0, 10, size)
    triangle = (np.abs((x % 2) - 1) * 255).astype(np.uint8)
    
    res_tri = geom.compute_metrics(triangle)
    
    # 2. White Noise (No linear structure)
    noise = np.random.randint(0, 256, size, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)
    
    lin_tri = res_tri.metrics['linearity']
    lin_noise = res_noise.metrics['linearity']
    
    print(f"Tropical Linearity - Triangle: {lin_tri:.3f}, Noise: {lin_noise:.3f}")
    
    assert lin_tri > lin_noise, "Piecewise linear signal should have higher linearity score"
    assert lin_tri > 0.8, "Triangle wave linearity should be high"

def test_symplectic_recurrence():
    """
    SymplecticGeometry checks phase space recurrence.
    Harmonic Oscillator (Sine) -> Stable closed orbit -> Low min_return_distance.
    Random Walk -> Wandering open trajectory -> High min_return_distance.
    """
    geom = SymplecticGeometry()
    size = 1000
    
    # 1. Harmonic Oscillator (Sine wave)
    # Perfectly periodic
    t = np.linspace(0, 20*np.pi, size)
    sine_wave = (np.sin(t) * 100 + 128).astype(np.uint8)
    
    res_sine = geom.compute_metrics(sine_wave)
    
    # 2. Random Walk (Brownian)
    # Non-recurrent (on short timescales)
    rng = np.random.default_rng(42)
    walk = np.cumsum(rng.standard_normal(size))
    walk = (walk - walk.min()) / (walk.max() - walk.min()) * 255
    walk = walk.astype(np.uint8)
    
    res_walk = geom.compute_metrics(walk)
    
    ret_sine = res_sine.metrics['min_return_distance']
    ret_walk = res_walk.metrics['min_return_distance']
    
    print(f"Symplectic Return Dist - Sine: {ret_sine:.3f}, Walk: {ret_walk:.3f}")
    
    # Sine wave loops back on itself many times (low return distance).
    # Random walk wanders off (high return distance).
    assert ret_sine < ret_walk, "Periodic orbit should have better recurrence (lower return distance) than random walk"

def test_persistent_homology_loops():
    """
    PersistentHomology should detect H1 loops.
    Noisy Circle -> Significant H1 feature (long lifetime).
    Noise -> Short H1 lifetimes (topological noise).
    """
    # Use fewer points for TDA speed
    geom = PersistentHomologyGeometry(n_points=40, max_dim=1)
    
    # 1. Noisy Circle (should have 1 major loop)
    # The geometry uses lag-1 embedding: (x[i], x[i+1]).
    # To get a circle x^2 + y^2 = r^2, we need x[i] and x[i+1] to be roughly orthogonal (pi/2 phase shift).
    # So we need period ~ 4 samples.
    # 40 points, period 4 -> 10 cycles.
    t = np.linspace(0, 10*2*np.pi, 41)[:-1] 
    circle_signal = (np.sin(t) * 100 + 128).astype(np.uint8)
    
    res_circle = geom.compute_metrics(circle_signal)
    
    # 2. Noise
    noise = np.random.randint(0, 256, 40, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)
    
    h1_life_circle = res_circle.metrics['max_h1_lifetime']
    h1_life_noise = res_noise.metrics['max_h1_lifetime']
    
    print(f"Max H1 Lifetime - Circle: {h1_life_circle:.3f}, Noise: {h1_life_noise:.3f}")
    
    # Circle should have a persistent loop.
    # Noise loops die quickly (diagonal proximity in persistence diagram).
    assert h1_life_circle > h1_life_noise, "Circle should have more persistent H1 topology than noise"

def test_ultrametric_divisibility():
    """
    UltrametricGeometry (2-adic) should detect even/odd/divisibility structure.
    Data with multiples of 4 -> 'Close' in 2-adic metric.
    Random data -> spread out.
    """
    geom = UltrametricGeometry(p=2)
    size = 500
    
    # 1. Divisible by 16 (High 2-adic closeness)
    # Differences will be divisible by 16, so distances p^-v will be small (2^-4).
    div_data = np.arange(0, size) * 16
    div_data = (div_data % 256).astype(np.uint8)
    
    res_div = geom.compute_metrics(div_data)
    
    # 2. Odd numbers (Low 2-adic closeness)
    # Differences of odds are even (div by 2), but not necessarily by 4, 8, etc.
    # Actually, consecutive odd numbers: 1, 3, 5... diff is 2.
    # Random odd numbers: diff can be anything.
    # Let's use Random data.
    noise = np.random.randint(0, 256, size, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)
    
    dist_div = res_div.metrics['mean_distance']
    dist_noise = res_noise.metrics['mean_distance']
    
    print(f"Mean 2-adic Distance - Multiples of 16: {dist_div:.3f}, Noise: {dist_noise:.3f}")
    
    # Multiples of 16 are "close" in 2-adic (distance <= 1/16 = 0.0625).
    # Random data has avg distance ~ 1/p = 0.5 or higher.
    assert dist_div < dist_noise, "Highly divisible data should be close in p-adic metric"
