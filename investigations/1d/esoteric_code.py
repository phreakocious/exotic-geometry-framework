#!/usr/bin/env python3
"""
Esoteric Code Investigation: Geometry of the Bizarre
====================================================

Can exotic geometries detect structure in programming languages designed 
to be impossible, invisible, or hellish? This investigation explores 
the structural signatures of Esoteric Programming Languages (Esolangs).

DIRECTIONS:
D1: Taxonomy — Brainfuck vs Whitespace vs Malbolge vs Zalgo.
D2: The Hell Test — Malbolge (complex state) vs Random Noise.
D3: Invisible Structure — Structured Whitespace vs "White Noise".
D4: Fractal Depth — Flat vs Deeply Nested Brainfuck loops.
D5: The Zalgo Singularity — Evolution from clean text to glitch horror.
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

def gen_brainfuck(rng, size, depth_prob=0.3):
    """
    Generate Brainfuck-like code (><+-.,[]).
    Maintains bracket balance to simulate valid loop structures.
    """
    chars = [b'>', b'<', b'+', b'-', b'.', b',']
    data = []
    stack = 0
    
    while len(data) < size:
        r = rng.random()
        
        # Closing bracket (if we have open ones)
        if stack > 0 and r < 0.15:
            data.append(b']')
            stack -= 1
        # Opening bracket (if we have space)
        elif stack < 20 and r < (0.15 + depth_prob):
            data.append(b'[')
            stack += 1
        # Normal command
        else:
            data.append(rng.choice(chars))
            
    # Pad or truncate
    return np.frombuffer(b''.join(data[:size]), dtype=np.uint8)

def gen_brainfuck_flat(rng, size):
    return gen_brainfuck(rng, size, depth_prob=0.0) # No loops

def gen_brainfuck_deep(rng, size):
    return gen_brainfuck(rng, size, depth_prob=0.5) # Many loops

def gen_whitespace(rng, size, structured=True):
    """
    Generate Whitespace code (Space, Tab, LF).
    structured=True follows valid IMP (Instruction Modification Parameter) patterns.
    """
    S, T, L = b' ', b'\t', b'\n'
    
    if not structured:
        # "White Noise" - random selection
        chars = [S, T, L]
        data = rng.choice(chars, size=size)
        return np.frombuffer(b''.join(data), dtype=np.uint8)
        
    # Structured generation
    data = []
    
    # Possible IMPs (Stack, Arith, Heap, Flow, IO)
    imps = [
        [S],         # Stack Manipulation
        [T, S],      # Arithmetic
        [T, T],      # Heap Access
        [L],         # Flow Control
        [T, L]       # I/O
    ]
    
    while len(data) < size:
        # Select IMP pattern manually to avoid numpy inhomogeneous array error
        imp_idx = rng.integers(0, len(imps))
        imp = imps[imp_idx]
        
        # Command within IMP
        cmd = []
        if imp == [S]: # Stack
            stack_cmds = [[S], [L, S], [L, L]]
            cmd = stack_cmds[rng.integers(0, len(stack_cmds))] # Push, Dup, Discard
            if cmd == [S]: # Push requires number
                # Number: Sign + Binary + Terminal
                num = [rng.choice([S, T])] + \
                      [rng.choice([S, T]) for _ in range(rng.integers(1, 8))] + \
                      [L]
                cmd += num
                
        elif imp == [T, S]: # Arith
            # All same length (2), so choice is safe here? No, better be safe.
            arith_cmds = [[S, S], [S, T], [T, S], [T, T]]
            cmd = arith_cmds[rng.integers(0, len(arith_cmds))]
            
        elif imp == [L]: # Flow
            flow_cmds = [[S, S], [S, T], [L, L]]
            cmd = flow_cmds[rng.integers(0, len(flow_cmds))] # Label, Call, End
            if cmd != [L, L]: # Label/Call needs label arg
                lbl = [rng.choice([S, T]) for _ in range(rng.integers(1, 8))] + [L]
                cmd += lbl
        
        # Add to stream
        block = imp + cmd
        # Flatten
        for item in block:
            if isinstance(item, list):
                for sub in item: data.append(sub)
            else:
                data.append(item)
                
    return np.frombuffer(b''.join(data[:size]), dtype=np.uint8)

def gen_malbolge(rng, size):
    """
    Generate Malbolge-like code.
    Malbolge has a specific valid character set and cycles.
    """
    # The valid Malbolge alphabet (subset)
    valid_chars = b"+b(29e*j1VMEKLyC})8&m#~W>qxdRp0w<oINShkzaF45U|s?36rvgtuBq0"
    # It's essentially a substitution cipher on execution, but the static
    # code looks like a high-entropy selection from this set.
    
    # We'll simulate it by picking randomly from the valid set.
    # While true Malbolge has constraints, visually/statistically it's dense.
    indices = rng.integers(0, len(valid_chars), size)
    data = np.array([valid_chars[i] for i in indices], dtype=np.uint8)
    return data

def gen_zalgo(rng, size, intensity=1):
    """
    Generate Zalgo text (Glitch).
    Base ASCII characters + [intensity] combining diacritics per char.
    """
    # Base: Printable ASCII
    base_chars = [i for i in range(32, 127)]
    
    # Combining diacritics range: U+0300 to U+036F
    # In UTF-8, these are 2 bytes: 0xCC 0x80 to 0xCD 0xAF
    
    out = bytearray()
    while len(out) < size:
        # 1. Base char
        out.append(rng.choice(base_chars))
        
        # 2. Add n diacritics
        n_glitch = rng.poisson(intensity)
        for _ in range(n_glitch):
            # Pick a random diacritic code point
            cp = rng.integers(0x0300, 0x0370)
            # Encode to utf-8 bytes
            try:
                b = chr(cp).encode('utf-8')
                out.extend(b)
            except:
                pass
                
    return np.frombuffer(out[:size], dtype=np.uint8)

# ==============================================================
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Taxonomy of the Bizarre"""
    print("\n" + "=" * 60)
    print("D1: TAXONOMY OF THE BIZARRE")
    print("=" * 60)

    conditions = {}
    for name, gen_fn in [("Brainfuck", gen_brainfuck),
                         ("Whitespace", lambda r, s: gen_whitespace(r, s, structured=True)),
                         ("Malbolge", gen_malbolge),
                         ("Zalgo", lambda r, s: gen_zalgo(r, s, intensity=5))]:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names, conditions=conditions)

def direction_2(runner):
    """D2: The Hell Test (Malbolge vs Random)"""
    print("\n" + "=" * 60)
    print("D2: THE HELL TEST (MALBOLGE vs RANDOM)")
    print("=" * 60)

    with runner.timed("malbolge"):
        mal_chunks = [gen_malbolge(rng, runner.data_size) for rng in runner.trial_rngs()]
        mal_met = runner.collect(mal_chunks)
    
    with runner.timed("random_ascii"):
        # Random printable ASCII for fair comparison
        rnd_chunks = [rng.integers(33, 127, runner.data_size, dtype=np.uint8) 
                      for rng in runner.trial_rngs()]
        rnd_met = runner.collect(rnd_chunks)

    ns, findings = runner.compare(mal_met, rnd_met)
    print(f"  Malbolge vs Random ASCII = {ns:3d} sig")
    if findings:
        for m, d, p in findings[:3]:
            print(f"    {m:45s} d={d:+8.2f}")

    return dict(ns=ns)

def direction_3(runner):
    """D3: Invisible Structure (Whitespace vs White Noise)"""
    print("\n" + "=" * 60)
    print("D3: INVISIBLE STRUCTURE")
    print("=" * 60)
    
    with runner.timed("structured"):
        struct_chunks = [gen_whitespace(rng, runner.data_size, structured=True) 
                         for rng in runner.trial_rngs()]
        struct_met = runner.collect(struct_chunks)

    with runner.timed("white_noise"):
        noise_chunks = [gen_whitespace(rng, runner.data_size, structured=False) 
                        for rng in runner.trial_rngs()]
        noise_met = runner.collect(noise_chunks)

    ns, findings = runner.compare(struct_met, noise_met)
    print(f"  Whitespace Code vs White Noise = {ns:3d} sig")
    return dict(ns=ns)

def direction_4(runner):
    """D4: Fractal Depth (Brainfuck Nesting)"""
    print("\n" + "=" * 60)
    print("D4: FRACTAL DEPTH (BRAINFUCK)")
    print("=" * 60)

    with runner.timed("flat"):
        flat_chunks = [gen_brainfuck_flat(rng, runner.data_size) for rng in runner.trial_rngs()]
        flat_met = runner.collect(flat_chunks)
    
    with runner.timed("deep"):
        deep_chunks = [gen_brainfuck_deep(rng, runner.data_size) for rng in runner.trial_rngs()]
        deep_met = runner.collect(deep_chunks)

    ns, _ = runner.compare(flat_met, deep_met)
    print(f"  Flat vs Deep Nesting = {ns:3d} sig")
    return dict(ns=ns)

def direction_5(runner):
    """D5: The Zalgo Singularity"""
    print("\n" + "=" * 60)
    print("D5: THE ZALGO SINGULARITY")
    print("=" * 60)
    
    results = []
    intensities = [0, 1, 5, 20]
    
    # Baseline: Clean text (intensity 0)
    base_chunks = [gen_zalgo(rng, runner.data_size, intensity=0) for rng in runner.trial_rngs()]
    base_met = runner.collect(base_chunks)
    
    for i in intensities[1:]:
        print(f"  Testing Intensity {i} vs 0...", end=" ", flush=True)
        t0 = time.time()
        test_chunks = [gen_zalgo(rng, runner.data_size, intensity=i) for rng in runner.trial_rngs()]
        test_met = runner.collect(test_chunks)
        ns, _ = runner.compare(base_met, test_met)
        results.append(ns)
        print(f"{ns:3d} sig ({time.time()-t0:.1f}s)")
        
    return dict(ns_list=results, intensities=intensities[1:])

# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "The Geometry of Esoteric Code")

    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Esolang Taxonomy")
    runner.plot_bars(axes[1], ["Mal vs Rnd"], [d2['ns']], "D2: The Hell Test")
    runner.plot_bars(axes[2], ["Code vs Noise"], [d3['ns']], "D3: Whitespace Structure")
    runner.plot_bars(axes[3], ["Flat vs Deep"], [d4['ns']], "D4: Brainfuck Depth")
    runner.plot_line(axes[4], d5['intensities'], d5['ns_list'], "D5: Zalgo Corruption",
                     xlabel="Glitch Intensity", ylabel="Sig metrics vs Clean")

    runner.save(fig, "esoteric_code")

# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("Esoteric Code", mode="1d")

    print("=" * 60)
    print("ESOTERIC CODE: GEOMETRY OF THE BIZARRE")
    print(f"size={runner.data_size}, trials={runner.n_trials}, "
          f"metrics={runner.n_metrics}")
    print("=" * 60)

    d1 = direction_1(runner)
    d2 = direction_2(runner)
    d3 = direction_3(runner)
    d4 = direction_4(runner)
    d5 = direction_5(runner)

    make_figure(runner, d1, d2, d3, d4, d5)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    runner.print_summary({
        'D1': f"Taxonomy: {np.mean(d1['matrix'][np.triu_indices_from(d1['matrix'], k=1)]):.1f} avg sig",
        'D2': f"Malbolge vs Random: {d2['ns']} sig metrics (Is Hell distinguishable?)",
        'D3': f"Whitespace: {d3['ns']} sig metrics (Structure in the void)",
        'D4': f"Brainfuck Depth: {d4['ns']} sig metrics (Fractal dimension)",
        'D5': f"Zalgo Singularity: {d5['ns_list'][-1]} sig at max corruption"
    })

if __name__ == "__main__":
    main()
