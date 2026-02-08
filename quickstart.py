#!/usr/bin/env python3
"""
Exotic Geometry Quickstart - Run this to verify the framework and see examples.

Usage:
    pip install -r requirements.txt
    python quickstart.py
"""

import numpy as np

print("=" * 70)
print("EXOTIC GEOMETRY FRAMEWORK - QUICKSTART")
print("=" * 70)

from exotic_geometry_framework import (
    GeometryAnalyzer, E8Geometry, TorusGeometry,
    HeisenbergGeometry, PenroseGeometry, SolGeometry
)
print("\n[OK] Framework imported successfully")

# Check all geometries load
analyzer = GeometryAnalyzer().add_all_geometries()
print(f"[OK] Loaded {len(analyzer.geometries)} geometries")

# Generate test data
np.random.seed(42)
print("\n" + "-" * 70)
print("GENERATING TEST DATA")
print("-" * 70)

# Random baseline
random_data = np.random.randint(0, 256, 1000, dtype=np.uint8)
print(f"  Random:     {len(random_data)} bytes")

# Chaotic (logistic map)
def logistic(r, n):
    x = 0.1
    result = []
    for _ in range(n + 100):
        x = r * x * (1 - x)
        result.append(x)
    return np.array([int(v * 255) for v in result[100:]], dtype=np.uint8)

chaos_data = logistic(3.9, 1000)
print(f"  Chaos:      {len(chaos_data)} bytes (logistic r=3.9)")

# Periodic
periodic_data = np.array([int(127 + 100*np.sin(2*np.pi*i/50)) for i in range(1000)], dtype=np.uint8)
print(f"  Periodic:   {len(periodic_data)} bytes (sine wave)")

# Structured (fibonacci)
fib = [1, 1]
for _ in range(998):
    fib.append((fib[-1] + fib[-2]) % 256)
fib_data = np.array(fib, dtype=np.uint8)
print(f"  Fibonacci:  {len(fib_data)} bytes")

# Key geometries comparison
print("\n" + "-" * 70)
print("KEY GEOMETRY COMPARISON")
print("-" * 70)

e8 = E8Geometry()
heisenberg_centered = HeisenbergGeometry(center_data=True)
sol = SolGeometry()
penrose = PenroseGeometry()

data_sources = [
    ("Random", random_data),
    ("Chaos (logistic)", chaos_data),
    ("Periodic (sine)", periodic_data),
    ("Fibonacci", fib_data),
]

print("\n{:<20} | {:>10} | {:>12} | {:>12} | {:>12}".format(
    "Data", "E8 roots", "Heis. twist", "Sol aniso", "Penrose 5f"))
print("-" * 75)

for name, data in data_sources:
    e8_res = e8.compute_metrics(data)
    heis_res = heisenberg_centered.compute_metrics(data)
    sol_res = sol.compute_metrics(data)
    pen_res = penrose.compute_metrics(data)

    print("{:<20} | {:>10.0f} | {:>12.1f} | {:>12.2f} | {:>12.4f}".format(
        name,
        e8_res.metrics["unique_roots"],
        heis_res.metrics["twist_rate"],
        sol_res.metrics["anisotropy"],
        pen_res.metrics["fivefold_balance"]
    ))

# Shuffle test demonstration
print("\n" + "-" * 70)
print("SHUFFLE TEST DEMONSTRATION (Critical for validation!)")
print("-" * 70)

print("\nE8 roots - Real vs Shuffled:")
for name, data in data_sources:
    real = e8.compute_metrics(data).metrics["unique_roots"]

    shuffled = data.copy()
    np.random.shuffle(shuffled)
    shuf = e8.compute_metrics(shuffled).metrics["unique_roots"]

    ratio = real / (shuf + 1e-10)
    sig = "***" if ratio < 0.7 or ratio > 1.3 else ""
    print(f"  {name:<20}: real={real:3.0f}, shuffled={shuf:3.0f}, ratio={ratio:.2f} {sig}")

print("\n*** = Significant difference (structure preserved through ordering)")

# What to explore next
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
Run an investigation:
    python investigations/1d/chaos.py      # Chaotic map fingerprinting
    python investigations/2d/ising.py      # Ising phase transition

Use the framework in your own code:
    from exotic_geometry_framework import GeometryAnalyzer
    analyzer = GeometryAnalyzer().add_all_geometries()
    results = analyzer.analyze(your_data_as_uint8)
""")

print("[OK] Quickstart complete!")
