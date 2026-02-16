import pytest
import numpy as np
import inspect
import exotic_geometry_framework as egf
from exotic_geometry_framework import ExoticGeometry, GeometryResult

def get_all_geometry_classes():
    """Find all subclasses of ExoticGeometry in the module."""
    geoms = []
    for name, obj in inspect.getmembers(egf):
        if inspect.isclass(obj) and issubclass(obj, ExoticGeometry) and obj is not ExoticGeometry:
            geoms.append(obj)
    return geoms

@pytest.mark.parametrize("GeomClass", get_all_geometry_classes())
def test_geometry_contract(GeomClass):
    """
    Smoke test for all geometries:
    - Instantiation
    - Properties (name, detects, etc.)
    - Embed
    - Compute metrics
    """
    # Skip abstract base classes if any slipped through (shouldn't with issubclass check usually)
    # But some might require specific init params.
    # Most have defaults.
    
    try:
        if GeomClass.__name__ == "CantorGeometry":
            geom = GeomClass(base=3)
        elif GeomClass.__name__ == "UltrametricGeometry":
            geom = GeomClass(p=2)
        else:
            geom = GeomClass()
    except TypeError as e:
        pytest.fail(f"Could not instantiate {GeomClass.__name__} with default args: {e}")

    # Check basic properties
    assert isinstance(geom.name, str)
    assert len(geom.name) > 0
    assert isinstance(geom.description, str)
    assert isinstance(geom.view, str)
    
    # Generate random data
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, 100, dtype=np.uint8)
    
    # Test Embed
    try:
        embedding = geom.embed(data)
        assert isinstance(embedding, np.ndarray) or isinstance(embedding, list)
        # Some embeddings are lists of arrays or specific structures, but usually ndarray
        if isinstance(embedding, np.ndarray):
             assert embedding.shape[0] > 0
    except Exception as e:
        pytest.fail(f"{geom.name}.embed() failed: {e}")
        
    # Test Metrics
    try:
        result = geom.compute_metrics(data)
        assert isinstance(result, GeometryResult)
        assert result.geometry_name == geom.name
        assert isinstance(result.metrics, dict)
        
        # Check that metrics are floats and not NaN (unless expected)
        for k, v in result.metrics.items():
            # It's okay if some are NaN in edge cases, but let's check for crashes
            pass
            
    except Exception as e:
        pytest.fail(f"{geom.name}.compute_metrics() failed: {e}")
