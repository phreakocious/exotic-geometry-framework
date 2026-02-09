#!/usr/bin/env python3
"""
CLI tool to classify arbitrary binary data against the signature library.

Usage:
    python quick_classify.py data.bin
    python quick_classify.py data.bin --top 10 --verbose
    python quick_classify.py data.bin --signatures /path/to/sigs
"""

import argparse
import os
import sys
import numpy as np
from exotic_geometry_framework import GeometryAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Classify binary data using exotic geometry signatures."
    )
    parser.add_argument("filename", help="Binary file to classify")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top matches to show (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show best/worst matching metrics for top match")
    parser.add_argument("--signatures", default="signatures",
                        help="Signature directory (default: signatures)")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.filename, "rb") as f:
        data = f.read(4000)

    if len(data) < 500:
        print("Warning: Data is very short. Classification may be unreliable.",
              file=sys.stderr)

    byte_array = np.frombuffer(data, dtype=np.uint8)

    print(f"Classifying {args.filename} ({len(byte_array)} bytes)...")
    analyzer = GeometryAnalyzer().add_all_geometries()
    results = analyzer.classify(byte_array, signature_dir=args.signatures)

    if not results:
        print("No signatures found. Run train_signature.py / harvest_*.py first.")
        sys.exit(1)

    # Table header
    print()
    print(f"  {'#':<4} {'System':<25} {'Median Z':>9} {'Match %':>8} {'Confidence':>11}")
    print(f"  {'─'*4} {'─'*25} {'─'*9} {'─'*8} {'─'*11}")

    for i, res in enumerate(results[:args.top]):
        rank = i + 1
        conf_str = f"{res['confidence']:.1%}" if i == 0 else ""
        print(f"  {rank:<4} {res['system']:<25} {res['median_z']:>9.3f} "
              f"{res['match_fraction']:>7.0%} {conf_str:>11}")

    # High-entropy / ambiguous warning
    top = results[0]
    if top["median_z"] > 2.0 and top["match_fraction"] < 0.5:
        print(f"\n  Note: Weak best match (z={top['median_z']:.1f}, {top['match_fraction']:.0%}).")
        print(f"  No signature in the library is a strong match.")
    elif top["confidence"] < 0.30 and len(results) >= 3:
        # Check if top matches are clustered (high-entropy pack)
        z3 = results[2]["median_z"]
        spread = z3 - top["median_z"]
        if spread < 0.5:
            print(f"\n  Note: High-entropy data — top 3 matches clustered within "
                  f"z={top['median_z']:.2f}-{z3:.2f}.")
            print(f"  Likely compressed or encrypted. Classification is ambiguous.")
        else:
            gap = results[1]["median_z"] - top["median_z"]
            print(f"\n  Note: Top two matches are close (gap={gap:.2f}).")
            print(f"  Ambiguous between {top['system']} and {results[1]['system']}.")

    if args.verbose and results:
        details = top.get("metric_details", [])
        if details:
            n_show = 5
            print(f"\n  Top match: {top['system']}")
            print(f"\n  5 best-matching metrics (lowest |z|):")
            for name, z in details[:n_show]:
                print(f"    {z:>7.2f}  {name}")
            print(f"\n  5 worst-matching metrics (highest |z|):")
            for name, z in details[-n_show:]:
                print(f"    {z:>7.2f}  {name}")
    print()


if __name__ == "__main__":
    main()
