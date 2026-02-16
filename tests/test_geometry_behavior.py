import pytest
import numpy as np
from exotic_geometry_framework import (
    TorusGeometry, 
    HeisenbergGeometry, 
    E8Geometry, 
    HyperbolicGeometry
)

def generate_periodic_signal(size, period=16):
    """Generate a repeating byte sequence."""
    base = np.linspace(0, 255, period, dtype=np.uint8)
    return np.resize(base, size)

def generate_correlated_walk(size, step_std=5):
    """Generate a random walk (Brownian motion)."""
    steps = np.random.normal(0, step_std, size)
    walk = np.cumsum(steps)
    # Normalize to 0-255
    walk = (walk - walk.min()) / (walk.max() - walk.min() + 1e-10) * 255
    return walk.astype(np.uint8)

def generate_white_noise(size):
    """Generate uniform random bytes."""
    return np.random.randint(0, 256, size, dtype=np.uint8)

def test_torus_detects_periodicity():
    """
    Torus Geometry should detect periodicity.
    Periodic data -> Traces a closed loop -> Low coverage, Low entropy.
    Random data -> Fills the space -> High coverage, High entropy.
    """
    size = 2000
    geom = TorusGeometry(bins=16)
    
    # 1. Periodic Signal
    periodic = generate_periodic_signal(size, period=32)
    res_periodic = geom.compute_metrics(periodic)
    
    # 2. Random Signal
    random = generate_white_noise(size)
    res_random = geom.compute_metrics(random)
    
    # Check Coverage: Periodic should cover significantly LESS of the torus
    cov_p = res_periodic.metrics['coverage']
    cov_r = res_random.metrics['coverage']
    
    print(f"Torus Coverage - Periodic: {cov_p:.3f}, Random: {cov_r:.3f}")
    assert cov_p < cov_r * 0.5, "Periodic signal should have much lower torus coverage than noise"
    
    # Check Entropy: Periodic should have lower entropy
    ent_p = res_periodic.metrics['normalized_entropy']
    ent_r = res_random.metrics['normalized_entropy']
    
    print(f"Torus Entropy - Periodic: {ent_p:.3f}, Random: {ent_r:.3f}")
    assert ent_p < ent_r, "Periodic signal should have lower entropy on torus"

def test_heisenberg_detects_correlation():
    """
    Heisenberg Geometry should detect correlations (twist).
    Correlated data (Brownian) -> Systematic accumulation of area (z).
    Uncorrelated data (White Noise) -> Random cancellation (lower relative z).
    
    We use center_data=True to focus on correlation, not mean bias.
    """
    size = 2000
    geom = HeisenbergGeometry(center_data=True)
    
    # 1. Correlated Walk
    correlated = generate_correlated_walk(size)
    res_corr = geom.compute_metrics(correlated)
    
    # 2. White Noise
    random = generate_white_noise(size)
    res_rand = geom.compute_metrics(random)
    
    # Metric: twist_rate (rate of z accumulation) or final_z
    twist_c = abs(res_corr.metrics['twist_rate'])
    twist_r = abs(res_rand.metrics['twist_rate'])
    
    # Correlated walks tend to sweep out large coherent areas in the Heisenberg group
    print(f"Heisenberg Twist Rate - Correlated: {twist_c:.3f}, Random: {twist_r:.3f}")
    
    # Note: Brownian motion typically scales differently, but generally produces 
    # larger area accumulation than white noise which self-cancels more frequently.
    assert twist_c > twist_r, "Correlated data should accumulate more Heisenberg twist/area"

def test_e8_detects_lattice_structure():
    """
    E8 Geometry should detect data aligned with the E8 lattice.
    Lattice Data -> High alignment, low diversity (reusing specific roots).
    Random Data -> Lower alignment, high diversity.
    """
    geom = E8Geometry(window_size=8, normalize=True)
    
    # 1. Synthesize Lattice Data (perfect E8 roots)
    # Get the roots directly from the geometry
    roots = geom.roots  # (240, 8)
    
    # Create a signal that is just repeated roots
    # We need to map these float roots back to uint8-ish behavior or just raw floats
    # The framework handles float input if we passed it, but let's simulate 
    # 'perfect' alignment by constructing data that windows exactly to roots.
    
    # Easiest way: just chain roots together.
    # We use a subset of roots to ensure 'low diversity' metric works too.
    chosen_roots = roots[:10] 
    lattice_data_list = []
    for _ in range(100):
        root = chosen_roots[np.random.choice(len(chosen_roots))]
        lattice_data_list.append(root)
    
    lattice_data = np.array(lattice_data_list).flatten()
    
    # Add offset/scale to make it look like "signal" (E8 embed normalizes anyway)
    lattice_data = (lattice_data * 10) + 128 
    
    res_lattice = geom.compute_metrics(lattice_data)
    
    # 2. Random Data
    random_data = np.random.uniform(0, 255, len(lattice_data))
    res_random = geom.compute_metrics(random_data)
    
    # Metric: alignment_mean (how close are windows to real roots?)
    align_l = res_lattice.metrics['alignment_mean']
    align_r = res_random.metrics['alignment_mean']
    
    print(f"E8 Alignment - Lattice: {align_l:.3f}, Random: {align_r:.3f}")
    assert align_l > align_r, "Lattice-derived data should have better root alignment"
    
    # Metric: diversity_ratio (fraction of 240 roots used)
    # Our lattice data only used 10 roots. Random data should use many more.
    div_l = res_lattice.metrics['diversity_ratio']
    div_r = res_random.metrics['diversity_ratio']
    
    print(f"E8 Diversity - Lattice: {div_l:.3f}, Random: {div_r:.3f}")
    assert div_l < div_r, "Restricted lattice data should have lower root diversity"

def test_hyperbolic_detects_hierarchy():
    """
    Hyperbolic Geometry should detect hierarchy.
    Tree-like/Exponential data -> Clusters near boundary.
    Uniform data -> Fills space (less boundary clustering).
    """
    geom = HyperbolicGeometry()
    size = 2000
    
    # 1. Hierarchical Proxy (Exponential distribution)
    # Exponential data looks like a "branching" process where few values are large
    # When embedded in Poincare disk, large values map to the edge.
    hierarchical = np.random.exponential(scale=50, size=size)
    hierarchical = np.clip(hierarchical, 0, 255).astype(np.uint8)
    
    res_hier = geom.compute_metrics(hierarchical)
    
    # 2. Uniform Random
    uniform = np.random.uniform(0, 255, size).astype(np.uint8)
    res_unif = geom.compute_metrics(uniform)
    
    # Metric: boundary_proximity
    bound_h = res_hier.metrics['boundary_proximity']
    bound_u = res_unif.metrics['boundary_proximity']
    
    print(f"Hyperbolic Boundary Proximity - Hierarchical: {bound_h:.3f}, Uniform: {bound_u:.3f}")
    
    # The exponential distribution has a "heavy tail" which in our 
    # [0,1] normalization (if auto-scaled) might actually look like sparse spikes.
    # But usually, hierarchy = edge clustering in hyperbolic space.
    
    # Let's ensure our assumption holds. 
    # If not, we learn something about the metric!
    # Exponential data has many small values and few large ones.
    # Small values -> Center? Large values -> Edge?
    # Embed maps (x,y) pairs.
    # If x,y are small, z is small -> Center.
    # If x,y are large, z is large -> Edge.
    # Exponential has MANY small values (Center) and FEW large (Edge).
    # Uniform has MANY large values (relative to exponential).
    
    # WAIT. The test premise might be flipped depending on embedding details.
    # Embed: z = complex(x-0.5, y-0.5) * 1.8
    # Uniform: fills the square [-0.5, 0.5].
    # Exponential (normalized): clusters at 0 (which is -0.5 in embedding coords -> Edge!).
    
    # Let's trace:
    # Data 0..255. Exponential: most values ~0.
    # Normalized: most values ~0.0.
    # Embedded x = 0.0 - 0.5 = -0.5.
    # Distance from center = 0.5 * 1.8 = 0.9.
    # So Exponential data -> Clusters at the EDGE (radius ~0.9).
    # Uniform data -> Clusters everywhere, mean radius smaller.
    
    assert bound_h > bound_u, "Hierarchical (exponential) data should cluster at the hyperbolic boundary"
