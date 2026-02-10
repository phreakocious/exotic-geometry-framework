#!/usr/bin/env python3
"""
Sorting Algorithms Investigation: Geometry of Memory Access Patterns
====================================================================

Can exotic geometries distinguish the memory access traces of different
sorting algorithms (Quick, Merge, Heap, Radix, Insertion)?

DIRECTIONS:
D1: Taxonomy — Pairwise comparison of memory traces (Quick, Merge, Heap, Radix, Insertion).
D2: Sequential structure — Real traces vs shuffled traces for each algorithm.
D3: Input Sortedness — Parameter sweep of QuickSort on 0% to 100% sorted inputs.
D4: Robustness — QuickSort pivot strategies (First, Last, Median-3) vs Random Pivot.
D5: Scale Test — Distinguishability of QuickSort vs HeapSort across array sizes.
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
# TRACKING & ALGORITHMS
# ==============================================================
class TrackedList:
    """Wraps a list to record access indices."""
    def __init__(self, data):
        self.data = list(data)
        self.accesses = []
        self.n = len(data)

    def __getitem__(self, key):
        # Record read
        if isinstance(key, int):
            self.accesses.append(key % 256)
        return self.data[key]

    def __setitem__(self, key, value):
        # Record write
        if isinstance(key, int):
            self.accesses.append(key % 256)
        self.data[key] = value

    def __len__(self):
        return len(self.data)

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def quicksort(arr, low, high, pivot_strategy='last', rng=None):
    if low < high:
        pi = partition(arr, low, high, pivot_strategy, rng)
        quicksort(arr, low, pi - 1, pivot_strategy, rng)
        quicksort(arr, pi + 1, high, pivot_strategy, rng)

def partition(arr, low, high, pivot_strategy, rng):
    pivot_idx = high
    if pivot_strategy == 'random' and rng:
        pivot_idx = rng.integers(low, high + 1)
    elif pivot_strategy == 'first':
        pivot_idx = low
    elif pivot_strategy == 'median3':
        mid = (low + high) // 2
        # Simple median of low, mid, high
        candidates = [(arr[low], low), (arr[mid], mid), (arr[high], high)]
        candidates.sort(key=lambda x: x[0])
        pivot_idx = candidates[1][1]
    
    # Swap pivot to high
    if pivot_idx != high:
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def heapsort(arr):
    n = len(arr)
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def mergesort(arr, l, r):
    if l < r:
        m = l + (r - l) // 2
        mergesort(arr, l, m)
        mergesort(arr, m + 1, r)
        merge(arr, l, m, r)

def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m
    # Create temp arrays (these reads/writes are not on 'arr' directly, 
    # but we will access 'arr' to copy to them)
    L = [arr[l + i] for i in range(n1)]
    R = [arr[m + 1 + j] for j in range(n2)]
    
    i = 0; j = 0; k = l
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def radix_sort(arr):
    # Simplified LSD radix sort for integers
    # We only care about access patterns, assume max val is handled
    max_val = max(arr.data) # cheat slightly to get max, but it triggers reads
    exp = 1
    while max_val // exp > 0:
        counting_sort(arr, exp)
        exp *= 10

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
        
    for i in range(1, 10):
        count[i] += count[i - 1]
        
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
        
    for i in range(n):
        arr[i] = output[i]


# ==============================================================
# GENERATORS
# ==============================================================
def run_algorithm(algo_name, arr, rng, **kwargs):
    if algo_name == 'quick':
        strategy = kwargs.get('pivot', 'random')
        quicksort(arr, 0, len(arr)-1, strategy, rng)
    elif algo_name == 'merge':
        mergesort(arr, 0, len(arr)-1)
    elif algo_name == 'heap':
        heapsort(arr)
    elif algo_name == 'radix':
        radix_sort(arr)
    elif algo_name == 'insertion':
        insertion_sort(arr)

def generate_trace(rng, size, algo='quick', n_elements=256, sortedness=0.0, **kwargs):
    """
    Generates a memory access trace.
    Repeats the sort on new random data until 'size' bytes are collected.
    """
    trace = []
    
    while len(trace) < size:
        # Create data
        if sortedness == 0.0:
            data = rng.permutation(n_elements)
        else:
            # Mix of sorted and random
            n_sorted = int(n_elements * sortedness)
            part1 = np.arange(n_sorted)
            part2 = rng.permutation(np.arange(n_sorted, n_elements))
            data = np.concatenate([part1, part2])
            # Add slight noise to sorted part to prevent perfect-case optimizations if any
            # (though our impls are naive)
        
        tracked = TrackedList(data)
        run_algorithm(algo, tracked, rng, **kwargs)
        trace.extend(tracked.accesses)
        
    return np.array(trace[:size], dtype=np.uint8)

# Specific generators for directions
def gen_quick(rng, size): return generate_trace(rng, size, 'quick')
def gen_merge(rng, size): return generate_trace(rng, size, 'merge')
def gen_heap(rng, size): return generate_trace(rng, size, 'heap')
def gen_radix(rng, size): return generate_trace(rng, size, 'radix')
def gen_insertion(rng, size): return generate_trace(rng, size, 'insertion')

# ==============================================================
# DIRECTIONS
# ==============================================================
def direction_1(runner):
    """D1: Taxonomy — Pairwise comparison of memory traces."""
    print("\n" + "=" * 60)
    print("D1: TAXONOMY (Pairwise Distinguishability)")
    print("=" * 60)

    conditions = {}
    algos = [('Quick', gen_quick), 
             ('Merge', gen_merge), 
             ('Heap', gen_heap), 
             ('Radix', gen_radix),
             ('Insert', gen_insertion)]
             
    for name, gen_fn in algos:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size) 
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)

def direction_2(runner):
    """D2: Sequential structure — Real vs Shuffled traces."""
    print("\n" + "=" * 60)
    print("D2: SEQUENTIAL STRUCTURE (vs Shuffled)")
    print("=" * 60)

    algos = [('Quick', gen_quick), 
             ('Merge', gen_merge), 
             ('Heap', gen_heap), 
             ('Radix', gen_radix)]
             
    results = {}
    for name, gen_fn in algos:
        chunks = [gen_fn(rng, runner.data_size) for rng in runner.trial_rngs()]
        real = runner.collect(chunks)
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, _ = runner.compare(real, shuf)
        results[name] = ns
        print(f"  {name} vs shuffled = {ns:3d} sig")

    return dict(results=results)

def direction_3(runner):
    """D3: Input Sortedness — QuickSort trace changes."""
    print("\n" + "=" * 60)
    print("D3: INPUT SORTEDNESS (QuickSort)")
    print("=" * 60)

    params = [0.0, 0.25, 0.5, 0.75, 0.95] # 1.0 might be too fast/empty trace for some
    
    # Baseline is random input (0.0)
    baseline_chunks = [generate_trace(rng, runner.data_size, 'quick', sortedness=0.0)
                       for rng in runner.trial_rngs()]
    baseline = runner.collect(baseline_chunks)

    results = {}
    for p in params:
        with runner.timed(f"sorted={p}"):
            chunks = [generate_trace(rng, runner.data_size, 'quick', sortedness=p)
                      for rng in runner.trial_rngs(offset=int(p*100))]
            met = runner.collect(chunks)
            ns, _ = runner.compare(baseline, met)
            results[p] = ns
            print(f"  Sortedness {p:.2f} vs Random = {ns:3d} sig")

    return dict(results=results, params=params)

def direction_4(runner):
    """D4: Robustness — QuickSort Pivot Strategies."""
    print("\n" + "=" * 60)
    print("D4: ROBUSTNESS (Pivot Strategies)")
    print("=" * 60)

    # Baseline: Random Pivot
    baseline_chunks = [generate_trace(rng, runner.data_size, 'quick', pivot='random')
                       for rng in runner.trial_rngs()]
    baseline = runner.collect(baseline_chunks)
    
    strategies = ['first', 'last', 'median3']
    results = {}
    
    for strat in strategies:
        chunks = [generate_trace(rng, runner.data_size, 'quick', pivot=strat)
                  for rng in runner.trial_rngs(offset=len(strat))]
        met = runner.collect(chunks)
        ns, _ = runner.compare(baseline, met)
        results[strat] = ns
        print(f"  {strat} vs random_pivot = {ns:3d} sig")

    return dict(results=results)

def direction_5(runner):
    """D5: Scale Test — QuickSort vs HeapSort across array sizes."""
    print("\n" + "=" * 60)
    print("D5: SCALE TEST (Array Size)")
    print("=" * 60)

    sizes = [64, 128, 256, 512, 1024]
    results = {}

    for s in sizes:
        with runner.timed(f"N={s}"):
            # Compare Quick vs Heap at this scale
            chunks_q = [generate_trace(rng, runner.data_size, 'quick', n_elements=s)
                        for rng in runner.trial_rngs()]
            chunks_h = [generate_trace(rng, runner.data_size, 'heap', n_elements=s)
                        for rng in runner.trial_rngs(offset=s)]
            
            met_q = runner.collect(chunks_q)
            met_h = runner.collect(chunks_h)
            
            ns, _ = runner.compare(met_q, met_h)
            results[s] = ns
            print(f"  N={s}: Quick vs Heap = {ns:3d} sig")

    return dict(results=results, sizes=sizes)

# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "Sorting Memory Geometry")

    # D1: Taxonomy (Heatmap)
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Taxonomy (Pairwise)")

    # D2: Structure (Bars)
    names2 = list(d2['results'].keys())
    vals2 = [d2['results'][n] for n in names2]
    runner.plot_bars(axes[1], names2, vals2, "D2: Structure (vs Shuffled)")

    # D3: Sortedness (Line)
    params3 = d3['params']
    vals3 = [d3['results'][p] for p in params3]
    runner.plot_line(axes[2], params3, vals3, "D3: Input Sortedness Impact",
                     xlabel="Fraction Sorted", ylabel="Sig. vs Random Input")

    # D4: Pivots (Bars)
    names4 = list(d4['results'].keys())
    vals4 = [d4['results'][n] for n in names4]
    runner.plot_bars(axes[3], names4, vals4, "D4: Pivot vs Random Pivot")
    
    # D5: Scale (Line)
    sizes5 = d5['sizes']
    vals5 = [d5['results'][s] for s in sizes5]
    runner.plot_line(axes[4], sizes5, vals5, "D5: Quick vs Heap by Size",
                     xlabel="Array Size (N)", ylabel="Sig. Difference")

    runner.save(fig, "sorting_algorithms")

# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("Sorting Algorithms", mode="1d")

    print("=" * 60)
    print("SORTING ALGORITHMS INVESTIGATION")
    print(f"size={runner.data_size}, trials={runner.n_trials}")
    print("=" * 60)

    d1 = direction_1(runner)
    d2 = direction_2(runner)
    d3 = direction_3(runner)
    d4 = direction_4(runner)
    d5 = direction_5(runner)

    make_figure(runner, d1, d2, d3, d4, d5)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    pw = d1['matrix'][np.triu_indices_from(d1['matrix'], k=1)]
    runner.print_summary({
        'D1': f"Taxonomy: {pw.min()}-{pw.max()} sig (mean {pw.mean():.0f})",
        'D2': [f"{n} vs shuffled = {v}" for n, v in d2['results'].items()],
        'D3': "Sortedness: " + ", ".join(
            f"{p}→{d3['results'][p]}" for p in d3['params']),
        'D4': [f"{s} vs random = {v}" for s, v in d4['results'].items()],
        'D5': "Scale: " + ", ".join(
            f"N={s}→{d5['results'][s]}" for s in d5['sizes']),
    })

if __name__ == "__main__":
    main()
