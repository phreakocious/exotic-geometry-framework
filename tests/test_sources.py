import pytest
import numpy as np
from tools.sources import get_sources, gen_forest_fire, gen_sandpile, _to_uint8

def test_source_contract():
    """Ensure all registered sources return correct shape and type."""
    rng = np.random.default_rng(42)
    size = 1000
    
    # Test a subset of sources to save time, or all if feasible.
    # We'll test all since they are fast enough usually.
    sources = get_sources()
    print(f"Testing {len(sources)} sources...")
    
    for s in sources:
        try:
            data = s.gen_fn(rng, size)
            
            # Check type
            assert isinstance(data, np.ndarray), f"{s.name}: Result should be ndarray"
            assert data.dtype == np.uint8, f"{s.name}: Result should be uint8"
            
            # Check shape
            assert data.shape == (size,), f"{s.name}: Shape mismatch. Expected ({size},), got {data.shape}"
            
            # Check range (soft check, just ensure it's not totally broken)
            # Some sparse sources might be all 0s if size is small, but generally shouldn't be
            # We won't assert >0 variance globally because some constant sources might exist (though unlikely in this repo)
            
        except Exception as e:
            pytest.fail(f"{s.name} raised exception: {e}")

def test_forest_fire_behavior():
    """
    Forest Fire should exhibit 'sawtooth' behavior:
    - Long periods of slow growth (accumulation)
    - Sudden sharp drops (fires)
    """
    rng = np.random.default_rng(123)
    size = 5000
    # Note: gen_forest_fire returns normalized uint8 data.
    # We need to interpret the relative changes.
    data = gen_forest_fire(rng, size)
    
    # Convert back to float for analysis to avoid overflow/wrap issues with diff
    data_f = data.astype(float)
    diffs = np.diff(data_f)
    
    # 1. Growth phase dominance:
    # Most steps should be small positive increments (growth)
    # The normalization might mess with the exact '1' increment, but the trend should be positive.
    # Actually, _to_uint8 normalizes the *entire* sequence. 
    # In the raw model: growth is +1 (mostly), fire is -N.
    # So we expect many small positive diffs and fewer large negative diffs.
    
    positive_diffs = diffs[diffs > 0]
    negative_diffs = diffs[diffs < 0]
    
    # There should be significantly more growth steps than fire steps
    # (unless f is very high relative to p, which it isn't in default params)
    assert len(positive_diffs) > len(negative_diffs), "Forest Fire should grow more often than it burns"
    
    # 2. Asymmetry of magnitude
    # The average drop size should be significantly larger than the average growth step
    if len(negative_diffs) > 0:
        avg_growth = np.mean(positive_diffs)
        avg_drop = np.mean(np.abs(negative_diffs))
        assert avg_drop > avg_growth * 2, f"Fires should be sudden crashes. Drop {avg_drop:.2f} vs Growth {avg_growth:.2f}"

def test_sandpile_behavior():
    """
    Sandpile should be 'bursty':
    - Many zeros or small values
    - Heavy tail of large values
    - Not uniform noise
    """
    rng = np.random.default_rng(456)
    size = 5000
    data = gen_sandpile(rng, size)
    
    # 1. Sparsity / Low activity baseline
    # In SOC sandpiles, depending on the drive rate, we might see activity.
    # The current implementation drives it constantly (drops a grain every step).
    # But avalanches are intermittent.
    # However, _to_uint8 normalizes the output.
    
    # Check for heavy tail: kurtosis should be high?
    # Or simply compare mean to median. 
    # For a heavy tailed distribution, mean is often >> median.
    
    mean_val = np.mean(data)
    median_val = np.median(data)
    
    # It's not strictly guaranteed to be mean > median if it's purely power law with a cutoff,
    # but usually avalanche sizes are skewed.
    # Let's check skewness instead?
    
    from scipy.stats import skew, kurtosis
    k = kurtosis(data)
    s = skew(data)
    
    # SOC systems typically have high kurtosis (fat tails) compared to Gaussian (k=0)
    # or Uniform (k=-1.2).
    # We expect k > 0.
    assert k > 0, f"Sandpile should be leptokurtic (fat-tailed). Got kurtosis {k:.2f}"
    
    # 2. Ensure it's not just static
    assert np.std(data) > 0, "Sandpile data should have variance"
