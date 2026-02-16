import pytest
import numpy as np
from exotic_geometry_framework import (
    WassersteinGeometry,
    FisherGeometry,
    SphericalGeometry,
    FractalMandelbrotGeometry
)

def test_wasserstein_distribution_shift():
    """
    WassersteinGeometry measures Earth Mover's Distance (EMD).
    - Distance from Uniform:
        - Uniform Data -> Low Distance.
        - Peaked/Biased Data -> High Distance.
    """
    geom = WassersteinGeometry(n_bins=10)
    size = 1000
    
    # 1. Uniform Data
    uniform = np.random.randint(0, 256, size, dtype=np.uint8)
    res_uni = geom.compute_metrics(uniform)
    
    # 2. Peaked Data (All values near 0)
    # EMD to uniform (which is flat 0.1 per bin) will be high.
    # Our data is in bin 0 (prob 1.0).
    peaked = np.zeros(size, dtype=np.uint8)
    res_peak = geom.compute_metrics(peaked)
    
    dist_uni = res_uni.metrics['dist_from_uniform']
    dist_peak = res_peak.metrics['dist_from_uniform']
    
    print(f"Wasserstein Dist from Uniform - Uniform: {dist_uni:.3f}, Peaked: {dist_peak:.3f}")
    
    assert dist_peak > dist_uni, "Peaked distribution should be further from uniform than random noise"

def test_fisher_information_content():
    """
    FisherGeometry measures information/curvature of the probability manifold.
    - Low Entropy (Peaked) -> High curvature / information content (Trace Fisher).
    - High Entropy (Uniform) -> Low curvature / information content.
    
    Fisher Info Matrix (diagonal approx) F_ii = 1/p_i.
    If p_i is small, F_ii is huge.
    If distribution is uniform (p_i = 1/N), F_ii = N.
    If distribution is peaked (p_0 ~ 1, others ~ 0), F for others is HUGE.
    
    Actually, let's check the implementation:
    p = (hist + 1) / (N + bins)  (Laplace smoothing)
    F = diag(1/p)
    Trace = sum(1/p).
    
    Uniform: p ~ 1/16. Trace ~ 16 * 16 = 256.
    Peaked: p_0 ~ 1. p_others ~ small (1/N).
    Trace ~ 1/1 + 15 * (1/(1/N)) = 1 + 15N. -> HUGE.
    
    So Trace Fisher should be much higher for Peaked data.
    """
    geom = FisherGeometry(n_bins=16)
    size = 1000
    
    # 1. Uniform Data
    uniform = np.random.randint(0, 256, size, dtype=np.uint8)
    res_uni = geom.compute_metrics(uniform)
    
    # 2. Peaked Data
    peaked = np.zeros(size, dtype=np.uint8)
    res_peak = geom.compute_metrics(peaked)
    
    trace_uni = res_uni.metrics['trace_fisher']
    trace_peak = res_peak.metrics['trace_fisher']
    
    print(f"Fisher Trace - Uniform: {trace_uni:.1f}, Peaked: {trace_peak:.1f}")
    
    assert trace_peak > trace_uni * 2, "Peaked distribution should have much higher Fisher Information Trace"

def test_spherical_clustering():
    """
    SphericalGeometry (S^2) detects directional clustering.
    - Uniform Random -> 'concentration' (mean vector length) near 0.
    - Clustered (e.g., North Pole) -> 'concentration' near 1.
    """
    geom = SphericalGeometry()
    size = 1000
    
    # 1. Uniform Random on S^2 (approx)
    # Random bytes map to theta/phi.
    # Note: Uniform in theta/phi is NOT uniform on sphere (crowding at poles),
    # but it's much more spread out than a single point.
    uniform = np.random.randint(0, 256, size, dtype=np.uint8)
    res_uni = geom.compute_metrics(uniform)
    
    # 2. Clustered (North Pole)
    # Theta = 0 -> z=1.
    # Byte 0 -> Theta=0.
    clustered = np.zeros(size, dtype=np.uint8)
    res_clust = geom.compute_metrics(clustered)
    
    conc_uni = res_uni.metrics['concentration']
    conc_clust = res_clust.metrics['concentration']
    
    print(f"Spherical Concentration - Uniform: {conc_uni:.3f}, Clustered: {conc_clust:.3f}")
    
    assert conc_clust > 0.9, "Clustered data should have high concentration"
    assert conc_uni < 0.2, "Random data should have low concentration (mean vector length)"

def test_mandelbrot_stability():
    """
    FractalMandelbrotGeometry iterates z = z^2 + c.
    Mapping: value v in [0, 255] -> x in [0, 1] -> c_real in [-1.5, 1.5].
    
    Stable Orbit (Inside Set):
    We want c ~ 0.
    3x - 1.5 = 0 => x = 0.5.
    v = 128.
    
    Escape Orbit (Outside Set):
    v = 0 -> x = 0 -> c = -1.5 - 1.5i.
    Magnitude |c| ~ 2.12 > 2. Escapes immediately.
    """
    geom = FractalMandelbrotGeometry(input_scale=255.0)
    
    # 1. Stable Data (Center of range -> Center of Mandelbrot set)
    data_stable = np.full(100, 128, dtype=np.uint8)
    res_stable = geom.compute_metrics(data_stable)
    
    # 2. Escape Data (Corner of range -> Outside Mandelbrot set)
    data_escape = np.zeros(100, dtype=np.uint8)
    res_escape = geom.compute_metrics(data_escape)
    
    print(f"Mandelbrot Metrics Stable: {res_stable.metrics}")
    print(f"Mandelbrot Metrics Escape: {res_escape.metrics}")
    
    # Interior fraction should be 1.0 for stable, 0.0 for escape
    frac_stable = res_stable.metrics['interior_fraction']
    frac_escape = res_escape.metrics['interior_fraction']
    
    assert frac_stable > 0.9, "Data mapped to c=0 should stay in the interior"
    assert frac_escape < 0.1, "Data mapped to c=-1.5-1.5i should escape"
