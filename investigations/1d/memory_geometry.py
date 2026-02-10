#!/usr/bin/env python3
"""
Memory Geometry Investigation: Data Structure Topologies
========================================================

Can exotic geometries detect the underlying topology of data structures 
in memory? This investigation analyzes the byte-level signatures of 
common organizational patterns used in computer science.

DIRECTIONS:
D1: Topology Taxonomy — Array, Linked List, Hash Table, and Binary Tree.
D2: Pointer Chasing — Sequential vs Random memory walk traces.
D3: Fragmentation — Contiguous vs Scattered memory allocation.
D4: Data Density — Sparse vs Dense organizational signatures.
D5: Traversal Order — DFS vs BFS vs Random walk traces.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.investigation_runner import Runner

# ==============================================================
# CONFIG
# ==============================================================
SEED = 42
np.random.seed(SEED)

# ==============================================================
# DATA GENERATORS
# ==============================================================

def gen_array(rng, size):
    """Simulate a contiguous array of structs [ID(4), Val(4), Padding(8)]."""
    data = np.zeros(size, dtype=np.uint8)
    for i in range(0, size - 16, 16):
        # ID as 4-byte int
        data[i:i+4] = np.frombuffer(np.int32(i//16).tobytes(), dtype=np.uint8)
        # Random payload
        data[i+4:i+8] = rng.integers(0, 256, 4, dtype=np.uint8)
    return data

def gen_linked_list(rng, size):
    """Simulate a linked list with scattered nodes [Val(4), NextPtr(4), Padding(8)]."""
    data = np.zeros(size, dtype=np.uint8)
    nodes = list(range(0, size - 16, 16))
    rng.shuffle(nodes)
    for i, addr in enumerate(nodes):
        data[addr:addr+4] = rng.integers(0, 256, 4, dtype=np.uint8)
        if i < len(nodes) - 1:
            nxt = nodes[i+1]
            data[addr+4:addr+8] = np.frombuffer(np.int32(nxt).tobytes(), dtype=np.uint8)
    return data

def gen_hash_table(rng, size, load_factor=0.7):
    """Simulate a hash table with collisions [Key(4), Val(4), Padding(8)]."""
    data = np.zeros(size, dtype=np.uint8)
    n_buckets = size // 16
    for _ in range(int(n_buckets * load_factor)):
        bucket = rng.integers(0, n_buckets)
        addr = bucket * 16
        while data[addr] != 0: # Linear probing
            addr = (addr + 16) % (n_buckets * 16)
        data[addr:addr+4] = rng.integers(1, 256, 4, dtype=np.uint8)
        data[addr+4:addr+8] = rng.integers(0, 256, 4, dtype=np.uint8)
    return data

def gen_binary_tree(rng, size):
    """Simulate a binary tree [Val(4), Left(4), Right(4), Padding(4)]."""
    data = np.zeros(size, dtype=np.uint8)
    nodes = list(range(0, size - 16, 16))
    for i, addr in enumerate(nodes):
        data[addr:addr+4] = rng.integers(0, 256, 4, dtype=np.uint8)
        left, right = 2 * i + 1, 2 * i + 2
        if left < len(nodes):
            data[addr+4:addr+8] = np.frombuffer(np.int32(nodes[left]).tobytes(), dtype=np.uint8)
        if right < len(nodes):
            data[addr+8:addr+12] = np.frombuffer(np.int32(nodes[right]).tobytes(), dtype=np.uint8)
    return data

def gen_trace(rng, size, random=False):
    """Simulate memory access traces (sequence of addresses)."""
    addresses = np.zeros(size, dtype=np.uint8)
    current = 0
    for i in range(size // 4):
        if random:
            current = rng.integers(0, 1024 * 1024) # 1MB range
        else:
            current += rng.integers(4, 64) # Sequential-ish
        addresses[i*4:(i+1)*4] = np.frombuffer(np.int32(current).tobytes(), dtype=np.uint8)
    return addresses

# ==============================================================
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Topology Taxonomy"""
    print("\n" + "=" * 60)
    print("D1: TOPOLOGY TAXONOMY")
    print("=" * 60)

    conditions = {}
    for name, gen_fn in [("Array", gen_array),
                         ("LinkedList", gen_linked_list),
                         ("HashTable", gen_hash_table),
                         ("BinaryTree", gen_binary_tree)]:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names, conditions=conditions)

def direction_2(runner):
    """D2: Each topology vs shuffled — which has most sequential structure?"""
    print("\n" + "=" * 60)
    print("D2: SEQUENTIAL STRUCTURE — each vs shuffled")
    print("=" * 60)

    results = {}
    for name, gen_fn in [("Array", gen_array),
                         ("LinkedList", gen_linked_list),
                         ("HashTable", gen_hash_table),
                         ("BinaryTree", gen_binary_tree)]:
        chunks = [gen_fn(rng, runner.data_size)
                  for rng in runner.trial_rngs()]
        real = runner.collect(chunks)
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, _ = runner.compare(real, shuf)
        results[name] = ns
        print(f"  {name:12s} vs shuffled = {ns:3d} sig")

    return dict(results=results)

def direction_3(runner, d1_conditions):
    """D3: Fragmentation (Array vs Linked List)"""
    print("\n" + "=" * 60)
    print("D3: FRAGMENTATION (CONTIGUOUS vs SCATTERED)")
    print("=" * 60)
    
    ns, _ = runner.compare(d1_conditions["Array"], d1_conditions["LinkedList"])
    print(f"  Array vs LinkedList = {ns:3d} sig")
    return dict(ns=ns)

def direction_4(runner):
    """D4: Load factor sweep — hash table detection vs fill ratio."""
    print("\n" + "=" * 60)
    print("D4: LOAD FACTOR SWEEP — hash table fill ratio")
    print("=" * 60)

    load_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Compare each to empty (load=0.1 as baseline)
    base_chunks = [gen_hash_table(rng, runner.data_size, load_factor=0.1)
                   for rng in runner.trial_rngs()]
    base_met = runner.collect(base_chunks)

    results = {}
    for lf in load_factors:
        with runner.timed(f"lf={lf:.1f}"):
            chunks = [gen_hash_table(rng, runner.data_size, load_factor=lf)
                      for rng in runner.trial_rngs(offset=int(lf * 100))]
            met = runner.collect(chunks)
            ns, _ = runner.compare(base_met, met)
            results[lf] = ns

    return dict(results=results, load_factors=load_factors)

def direction_5(runner):
    """D5: Access traces — sequential vs random at varying stride."""
    print("\n" + "=" * 60)
    print("D5: ACCESS TRACES — stride sweep")
    print("=" * 60)

    # Random access baseline
    rnd_chunks = [gen_trace(rng, runner.data_size, random=True)
                  for rng in runner.trial_rngs()]
    rnd_met = runner.collect(rnd_chunks)

    strides = [4, 16, 64, 256, 1024]
    results = {}
    for stride in strides:
        with runner.timed(f"stride={stride}"):
            def _gen(rng, size, s=stride):
                addrs = np.zeros(size, dtype=np.uint8)
                cur = 0
                for i in range(size // 4):
                    cur += rng.integers(s // 2, s * 2)
                    addrs[i*4:(i+1)*4] = np.frombuffer(
                        np.int32(cur % (1024*1024)).tobytes(), dtype=np.uint8)
                return addrs
            chunks = [_gen(rng, runner.data_size)
                      for rng in runner.trial_rngs(offset=stride)]
            met = runner.collect(chunks)
            ns, _ = runner.compare(rnd_met, met)
            results[stride] = ns

    return dict(results=results, strides=strides)

# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "Memory Geometry: Data Structure Topologies")

    # D1: Pairwise heatmap
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Topology Taxonomy")

    # D2: Sequential structure bars
    names2 = list(d2['results'].keys())
    vals2 = [d2['results'][n] for n in names2]
    runner.plot_bars(axes[1], names2, vals2, "D2: vs Shuffled")

    # D3: Single comparison — use text annotation on a bar
    runner.plot_bars(axes[2], ["Array vs LL"], [d3['ns']], "D3: Fragmentation")

    # D4: Load factor sweep
    lfs = d4['load_factors']
    sigs4 = [d4['results'][lf] for lf in lfs]
    runner.plot_line(axes[3], lfs, sigs4, "D4: Hash Load Factor",
                     xlabel="Load factor", color='#e74c3c')

    # D5: Stride sweep
    strides = d5['strides']
    sigs5 = [d5['results'][s] for s in strides]
    runner.plot_line(axes[4], strides, sigs5, "D5: Access Stride",
                     xlabel="Stride (bytes)", color='#3498db')
    axes[4].set_xscale('log', base=2)

    runner.save(fig, "memory_geometry")

# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("Memory Geometry", mode="1d")

    print("=" * 60)
    print("MEMORY GEOMETRY: DATA STRUCTURE TOPOLOGIES")
    print(f"size={runner.data_size}, trials={runner.n_trials}, "
          f"metrics={runner.n_metrics}")
    print("=" * 60)

    d1 = direction_1(runner)
    d2 = direction_2(runner)
    d3 = direction_3(runner, d1['conditions'])
    d4 = direction_4(runner)
    d5 = direction_5(runner)

    make_figure(runner, d1, d2, d3, d4, d5)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    runner.print_summary({
        'D1': f"Topology taxonomy: {np.mean(d1['matrix'][np.triu_indices_from(d1['matrix'], k=1)]):.1f} avg sig",
        'D2': [f"{n} vs shuffled = {v}" for n, v in d2['results'].items()],
        'D3': f"Array vs LinkedList = {d3['ns']} sig",
        'D4': f"Load factor sweep: " + ", ".join(
            f"{lf}→{d4['results'][lf]}" for lf in d4['load_factors']),
        'D5': f"Stride sweep: " + ", ".join(
            f"{s}→{d5['results'][s]}" for s in d5['strides']),
    })

if __name__ == "__main__":
    main()
