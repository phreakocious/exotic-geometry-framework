import pytest
import numpy as np
from exotic_geometry_framework import (
    LorentzianGeometry,
    ProjectiveGeometry,
    SpiralGeometry,
    SolGeometry
)

def test_lorentzian_causality():
    """
    LorentzianGeometry checks spacetime intervals (dt^2 - dx^2).
    - Causal (Timelike): |dx/dt| < 1.
    - Acausal (Spacelike): |dx/dt| > 1.
    
    The geometry maps t=index/N, x=value [0,1].
    So max dx is 1.0. dt is 1/N.
    Velocity = dx/dt = N * delta_x.
    
    If N is large, almost any change in x is superluminal (Spacelike).
    To get Timelike, we need very small dx, or small N.
    
    Let's adjust input_scale or data to simulate this.
    Or relying on the geometry's internal normalization.
    
    Embed: t = i/len, x = val_norm.
    dt = 1/len.
    Condition for Timelike: -(1/len)^2 + (dx)^2 < 0  => dx^2 < (1/len)^2 => |dx| < 1/len.
    
    So x must change very slowly.
    """
    size = 100
    geom = LorentzianGeometry()
    
    # 1. Causal Signal (Slow drift)
    # Change < 1/100 per step.
    # 1/100 of 255 is 2.5.
    # So if steps are 0, 1, or 2, it should be mostly Timelike.
    causal = np.cumsum(np.random.randint(0, 2, size, dtype=np.uint8))
    
    # 2. Acausal Signal (Fast jumps)
    # Jumps of 10+
    acausal = np.random.randint(0, 256, size, dtype=np.uint8)
    
    res_causal = geom.compute_metrics(causal)
    res_acausal = geom.compute_metrics(acausal)
    
    time_c = res_causal.metrics['timelike_fraction']
    time_a = res_acausal.metrics['timelike_fraction']
    
    print(f"Lorentzian Timelike Frac - Causal: {time_c:.3f}, Acausal: {time_a:.3f}")
    
    assert time_c > time_a, "Slow drifting signal should be more timelike than random noise"
    assert time_c > 0.5, "Slow signal should be predominantly timelike"

def test_projective_collinearity():
    """
    ProjectiveGeometry detects collinear points in P^2.
    Triples (x,y,z) are collinear if det([p1, p2, p3]) = 0.
    
    We construct data such that consecutive triples lie on a plane (great circle in S^2).
    """
    geom = ProjectiveGeometry()
    
    # 1. Collinear Data (Points on equator z=0)
    # (x, y, 0) -> these are all on the line z=0 in P^2.
    # Any 3 points of form (x,y,0) have det=0.
    # We construct bytes: x, y, 0, x, y, 0...
    size = 300
    collinear = []
    for _ in range(100):
        collinear.extend([
            np.random.randint(1, 256), 
            np.random.randint(1, 256), 
            0 # z=0
        ])
    collinear = np.array(collinear, dtype=np.uint8)
    
    res_col = geom.compute_metrics(collinear)
    
    # 2. Random Data
    random = np.random.randint(0, 256, size, dtype=np.uint8)
    res_rand = geom.compute_metrics(random)
    
    score_col = res_col.metrics['collinearity']
    score_rand = res_rand.metrics['collinearity']
    
    print(f"Projective Collinearity - Planar: {score_col:.3f}, Random: {score_rand:.3f}")
    
    # The 'z=0' points are all on the "line at infinity" relative to the affine patch z=1,
    # or just a standard line in P^2. They are perfectly collinear.
    assert score_col > score_rand, "Points on a plane (z=0) should be detected as collinear"
    assert score_col > 0.9, "Perfectly planar points should have near 1.0 collinearity score"

def test_spiral_growth():
    """
    SpiralGeometry detects logarithmic growth.
    Input: Constant value -> constant angle step -> constant growth factor.
    Result should be a perfect spiral.
    Metric: 'angular_uniformity' should be high.
    Metric: 'tightness' might be stable.
    """
    geom = SpiralGeometry(spiral_type="logarithmic")
    size = 1000
    
    # 1. Constant Input (Smooth Spiral)
    # Theta increases by constant amount. r increases by constant factor.
    smooth = np.full(size, 128, dtype=np.uint8)
    res_smooth = geom.compute_metrics(smooth)
    
    # 2. Random Input (Jagged Spiral)
    # Theta step varies. r growth varies.
    random = np.random.randint(0, 256, size, dtype=np.uint8)
    res_rand = geom.compute_metrics(random)
    
    uni_s = res_smooth.metrics['angular_uniformity']
    uni_r = res_rand.metrics['angular_uniformity']
    
    print(f"Spiral Angular Uniformity - Smooth: {uni_s:.3f}, Random: {uni_r:.3f}")
    
    assert uni_s > uni_r, "Constant signal should produce a more uniform spiral"
    assert uni_s > 0.95, "Constant signal should be near perfect uniformity"

def test_sol_anisotropy():
    """
    SolGeometry metric ds^2 = e^(2z)dx^2 + e^(-2z)dy^2 + dz^2.
    If z is large positive, x-steps are amplified, y-steps are suppressed.
    
    We construct a signal where z (3rd byte) is consistently High (255).
    This should lead to high Anisotropy (std(x) >> std(y)).
    
    Contrast with z consistently Low (0) -> z mapped to -5.
    Then y-steps amplified, x-steps suppressed.
    """
    geom = SolGeometry()
    
    # 1. High Z (x expansion)
    # triples: (random, random, 255)
    high_z = []
    for _ in range(100):
        high_z.extend([
            np.random.randint(100, 156), # x step
            np.random.randint(100, 156), # y step
            255 # z step (drifts positive)
        ])
    high_z = np.array(high_z, dtype=np.uint8)
    res_high = geom.compute_metrics(high_z)
    
    # 2. Low Z (y expansion)
    # triples: (random, random, 0)
    low_z = []
    for _ in range(100):
        low_z.extend([
            np.random.randint(100, 156),
            np.random.randint(100, 156),
            0 # z step (drifts negative)
        ])
    low_z = np.array(low_z, dtype=np.uint8)
    res_low = geom.compute_metrics(low_z)
    
    stretch_high = res_high.metrics['stretch_ratio']  # x_range / y_range
    stretch_low = res_low.metrics['stretch_ratio']

    print(f"Sol Stretch Ratio - High Z: {stretch_high:.3f}, Low Z: {stretch_low:.3f}")

    # High Z -> X expands -> stretch_ratio > 1
    # Low Z -> Y expands -> stretch_ratio < 1
    assert stretch_high > 1.0, "High Z should favor X expansion"
    assert stretch_low < 1.0, "Low Z should favor Y expansion"
    assert stretch_high > stretch_low
