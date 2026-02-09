#!/usr/bin/env python3
"""
Generalization evaluation: held-out test with seeds far from training range.

Training used seeds 0-19. We test with seeds 500-509 (10 trials per system).
Reports per-system top-1 and top-3 accuracy, plus identifies which systems
have unstable signatures.
"""

import numpy as np
from exotic_geometry_framework import GeometryAnalyzer
from classifier_heatmap import GENERATORS
import json
import os

N_TEST = 10
TEST_SEEDS = range(500, 500 + N_TEST)


def main():
    print("Initializing analyzer...")
    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get signature names in library order
    sig_names = []
    for fn in sorted(os.listdir("signatures")):
        if fn.endswith(".json"):
            with open(os.path.join("signatures", fn)) as f:
                sig_names.append(json.load(f)["name"])

    gen_names = [s for s in sig_names if s in GENERATORS]
    print(f"Testing {len(gen_names)} systems x {N_TEST} held-out seeds "
          f"(seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1})\n")

    # Per-system results
    system_results = {}

    for gi, gen_name in enumerate(gen_names):
        top1 = 0
        top3 = 0
        self_zs = []
        best_others = []

        for seed in TEST_SEEDS:
            data = GENERATORS[gen_name](seed, 2000)
            rankings = analyzer.classify(data)

            rank_map = {r["system"]: (i + 1, r["median_z"])
                        for i, r in enumerate(rankings)}
            self_rank, self_z = rank_map.get(gen_name, (999, 99.0))
            best_name = rankings[0]["system"]
            best_z = rankings[0]["median_z"]

            self_zs.append(self_z)
            if self_rank == 1:
                top1 += 1
            if self_rank <= 3:
                top3 += 1
            if best_name != gen_name:
                best_others.append(best_name)

        mean_z = np.mean(self_zs)
        std_z = np.std(self_zs)
        system_results[gen_name] = {
            "top1": top1, "top3": top3,
            "mean_z": mean_z, "std_z": std_z,
            "confusors": best_others,
        }

        status = "GOOD" if top1 >= 7 else ("WEAK" if top1 >= 4 else "POOR")
        confusor_str = ""
        if best_others:
            from collections import Counter
            c = Counter(best_others).most_common(2)
            confusor_str = "  <- " + ", ".join(f"{name}({n})" for name, n in c)

        print(f"  [{gi+1:2d}/48] {gen_name:<30} top1={top1:>2}/{N_TEST}  "
              f"top3={top3:>2}/{N_TEST}  self_z={mean_z:.2f}+/-{std_z:.2f}  "
              f"{status}{confusor_str}")

    # Summary
    all_top1 = sum(r["top1"] for r in system_results.values())
    all_top3 = sum(r["top3"] for r in system_results.values())
    total = len(gen_names) * N_TEST

    good = sum(1 for r in system_results.values() if r["top1"] >= 7)
    weak = sum(1 for r in system_results.values() if 4 <= r["top1"] < 7)
    poor = sum(1 for r in system_results.values() if r["top1"] < 4)

    print(f"\n{'='*64}")
    print(f"GENERALIZATION SUMMARY ({len(gen_names)} systems x {N_TEST} trials)")
    print(f"{'='*64}")
    print(f"  Overall top-1: {all_top1}/{total} ({100*all_top1/total:.0f}%)")
    print(f"  Overall top-3: {all_top3}/{total} ({100*all_top3/total:.0f}%)")
    print(f"  Systems: {good} good (>=70%), {weak} weak (40-70%), {poor} poor (<40%)")

    if poor:
        print(f"\n  Poor systems (need more training diversity):")
        for name, r in system_results.items():
            if r["top1"] < 4:
                from collections import Counter
                c = Counter(r["confusors"]).most_common(3)
                conf = ", ".join(f"{n}({ct})" for n, ct in c)
                print(f"    {name}: top1={r['top1']}/{N_TEST}, "
                      f"self_z={r['mean_z']:.2f}+/-{r['std_z']:.2f}, "
                      f"confused with: {conf}")


if __name__ == "__main__":
    main()
