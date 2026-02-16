#!/usr/bin/env python3
"""
CLI wrapper for the Structure Atlas.

Usage:
    python tools/atlas.py list                  # show all registered sources
    python tools/atlas.py test "Logistic Chaos" # profile one source, print top metrics
    python tools/atlas.py run                   # full atlas rebuild (cached sources load instantly)
    python tools/atlas.py run --fresh           # clear cache, full rebuild
    python tools/atlas.py run --tier standard   # quick rebuild with fewer geometries
"""

import argparse
import os
import sys
import shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def cmd_list(args):
    """Show all registered sources with their flags."""
    from tools.sources import get_sources, DOMAIN_COLORS

    # Import structure_atlas to trigger file-based registrations
    sys.path.insert(0, os.path.join(ROOT, 'investigations', '1d'))
    import structure_atlas  # noqa: F401

    sources = get_sources()
    print(f"\n{'Name':<30s} {'Domain':<15s} {'Atlas':<7s} {'Sig':<5s}  Description")
    print("-" * 90)
    for s in sources:
        atlas_flag = "yes" if s.atlas else "-"
        sig_flag = "yes" if s.signature else "-"
        desc = s.description[:35] if s.description else ""
        print(f"  {s.name:<28s} {s.domain:<15s} {atlas_flag:<7s} {sig_flag:<5s}  {desc}")

    n_atlas = sum(1 for s in sources if s.atlas)
    n_sig = sum(1 for s in sources if s.signature)
    print(f"\nTotal: {len(sources)} sources  |  atlas={n_atlas}  |  signature={n_sig}")
    domains = sorted(set(s.domain for s in sources))
    print(f"Domains: {', '.join(domains)}")


def cmd_test(args):
    """Profile a single source and print top metrics."""
    from tools.sources import get_sources
    from tools.investigation_runner import Runner

    # Import structure_atlas to trigger file-based registrations
    sys.path.insert(0, os.path.join(ROOT, 'investigations', '1d'))
    import structure_atlas  # noqa: F401

    sources = {s.name: s for s in get_sources()}
    if args.name not in sources:
        print(f"Unknown source: {args.name!r}")
        print(f"Available: {', '.join(sorted(sources.keys()))}")
        return 1

    src = sources[args.name]
    print(f"\nProfiling: {src.name}  [{src.domain}]")
    print(f"  {src.description}")

    runner = Runner(f"test:{src.name}", mode="1d", data_size=16384,
                    n_trials=10, cache=True, n_workers=1)

    rngs = runner.trial_rngs()
    chunks = [src.gen_fn(rng, runner.data_size) for rng in rngs]
    metrics = runner.collect(chunks)

    # Compute mean values and sort by magnitude
    mean_vals = {}
    for m in runner.metric_names:
        vals = metrics.get(m, [])
        if len(vals) > 0:
            mean_vals[m] = np.mean(vals)
        else:
            mean_vals[m] = 0.0

    ranked = sorted(mean_vals.items(), key=lambda x: -abs(x[1]))
    n_show = min(20, len(ranked))
    print(f"\n  Top {n_show} metrics by magnitude:")
    for m, v in ranked[:n_show]:
        print(f"    {m:<50s}  {v:+.4f}")

    n_nonzero = sum(1 for v in mean_vals.values() if abs(v) > 1e-15)
    print(f"\n  Active metrics: {n_nonzero}/{len(runner.metric_names)}")
    return 0


def cmd_run(args):
    """Full atlas rebuild, delegating to structure_atlas.main()."""
    if args.fresh:
        cache_dir = os.path.join(ROOT, 'figures', '.atlas_cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")

    sys.path.insert(0, os.path.join(ROOT, 'investigations', '1d'))
    from structure_atlas import main
    main(tier=args.tier)


def main():
    parser = argparse.ArgumentParser(
        description="Structure Atlas CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('list', help='Show all registered sources')

    p_test = sub.add_parser('test', help='Profile one source')
    p_test.add_argument('name', help='Source name (e.g. "Logistic Chaos")')

    p_run = sub.add_parser('run', help='Full atlas rebuild')
    p_run.add_argument('--fresh', action='store_true',
                       help='Clear cache before rebuild')
    p_run.add_argument('--tier', default='complete',
                       choices=['quick', 'standard', 'full', 'complete'],
                       help='Geometry tier (default: complete)')

    args = parser.parse_args()

    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'test':
        sys.exit(cmd_test(args) or 0)
    elif args.command == 'run':
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
