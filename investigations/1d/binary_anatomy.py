#!/usr/bin/env python3
"""
Binary Anatomy Investigation: Structural Geometry of Executables
================================================================

Can exotic geometries detect the underlying architecture and structural
patterns of machine code? This investigation explores the geometric
signatures of different instruction sets and compilation artifacts.

DIRECTIONS:
D1: Architecture Taxonomy — x86-64, ARM64, WASM, and Java Bytecode.
D2: Sequential structure — real (synthetic) machine code vs shuffled.
D3: Entropy Sweep — raw vs compressed (UPX-like) vs encrypted (AES).
D4: Obfuscation — simulated control flow flattening (central dispatcher).
D5: Payload Detection — structured binary vs embedded high-entropy shellcode.
"""

import sys
import time
import numpy as np
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

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

def gen_x86_64(rng, size):
    """Synthetic x86-64: Variable length, REX prefixes, INT3 padding."""
    data = np.zeros(size, dtype=np.uint8)
    i = 0
    while i < size:
        r = rng.random()
        if r < 0.15: # REX.W + MOV/ADD
            instr = [0x48, rng.choice([0x89, 0x8B, 0x01, 0x03]), rng.integers(0, 256)]
        elif r < 0.30: # PUSH/POP reg
            instr = [rng.integers(0x50, 0x60)]
        elif r < 0.40: # CALL/JMP rel32
            instr = [rng.choice([0xE8, 0xE9]), 0, 0, 0, 0] # simplified rel32
        elif r < 0.50: # LEA
            instr = [0x48, 0x8D, rng.integers(0, 256), rng.integers(0, 256)]
        elif r < 0.60: # Padding / INT3
            instr = [0xCC] * rng.integers(1, 4)
        elif r < 0.70: # TEST/CMP
            instr = [0x48, 0x85, rng.integers(0, 256)]
        else: # Miscellaneous small instructions
            instr = [rng.integers(0, 256) for _ in range(rng.integers(1, 3))]
        
        for b in instr:
            if i < size:
                data[i] = b
                i += 1
    return data

def gen_arm64(rng, size):
    """Synthetic ARM64: 4-byte aligned, specific opcode distributions."""
    data = np.zeros(size, dtype=np.uint8)
    for i in range(0, size - 3, 4):
        r = rng.random()
        if r < 0.30: # ADD/SUB (shifted register) - 0x91...
            instr = [rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 256), 0x91]
        elif r < 0.60: # LDR/STR - 0xF9...
            instr = [rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 256), 0xF9]
        elif r < 0.75: # Branch - 0x14... or 0x94...
            instr = [rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 256), rng.choice([0x14, 0x94])]
        elif r < 0.85: # RET - 0xD65F03C0
            instr = [0xC0, 0x03, 0x5F, 0xD6]
        else: # System/Other
            instr = [rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 256), 0xD5]
        
        # ARM64 is little-endian usually, but the patterns stay in 4-byte blocks
        data[i:i+4] = instr
    return data

def gen_wasm(rng, size):
    """Synthetic WASM: LEB128 constants, stack opcodes, 'end' markers."""
    data = np.zeros(size, dtype=np.uint8)
    i = 0
    while i < size:
        r = rng.random()
        if r < 0.40: # local.get / local.set
            instr = [rng.choice([0x20, 0x21]), rng.integers(0, 128)] # Simple LEB128
        elif r < 0.60: # i32.const
            instr = [0x41, rng.integers(0, 256)] # Simplified const
        elif r < 0.80: # i32.add/sub/etc
            instr = [rng.integers(0x6A, 0x7F)]
        elif r < 0.90: # Block end
            instr = [0x0B]
        else: # Br / Call
            instr = [rng.choice([0x0C, 0x10]), rng.integers(0, 128)]
            
        for b in instr:
            if i < size:
                data[i] = b
                i += 1
    return data

def gen_java(rng, size):
    """Synthetic Java Bytecode: Method invocations, aload, return."""
    data = np.zeros(size, dtype=np.uint8)
    i = 0
    while i < size:
        r = rng.random()
        if r < 0.30: # aload_n / iload_n
            instr = [rng.integers(0x1A, 0x2F)]
        elif r < 0.50: # getfield / putfield
            instr = [rng.choice([0xB4, 0xB5]), 0, rng.integers(0, 256)] # const pool idx
        elif r < 0.70: # invokevirtual / invokespecial
            instr = [rng.choice([0xB6, 0xB7]), 0, rng.integers(0, 256)]
        elif r < 0.85: # return / ireturn
            instr = [rng.integers(0xAC, 0xB1)]
        else: # new / checkcast
            instr = [rng.choice([0xBB, 0xC0]), 0, rng.integers(0, 256)]
            
        for b in instr:
            if i < size:
                data[i] = b
                i += 1
    return data

def gen_compressed(rng, size):
    """Simulate compressed data (high entropy but not perfectly random)."""
    # We'll use a simple RLE + Random to simulate 'packed' but structured data
    data = rng.integers(0, 256, size, dtype=np.uint8)
    # Introduce some repeated bytes to simulate compression artifacts
    for _ in range(size // 100):
        pos = rng.integers(0, size - 10)
        data[pos:pos+10] = data[pos]
    return data

def gen_encrypted(rng, size):
    """Simulate encrypted data using AES-CTR."""
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CTR, nonce=get_random_bytes(8))
    pt = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
    ct = cipher.encrypt(pt)
    return np.frombuffer(ct, dtype=np.uint8)

def gen_obfuscated(rng, size):
    """Simulate control-flow flattening (central dispatcher)."""
    # Lots of jumps to the same 'dispatcher' address
    data = gen_x86_64(rng, size)
    dispatcher_addr = [0xAA, 0xBB, 0xCC, 0xDD]
    for i in range(0, size - 10, 20):
        # Insert a JMP <dispatcher>
        data[i:i+5] = [0xE9] + dispatcher_addr
    return data

def gen_benign_with_payload(rng, size):
    """Benign x86-64 binary with an embedded high-entropy payload."""
    data = gen_x86_64(rng, size)
    payload_size = size // 4
    payload = gen_encrypted(rng, payload_size)
    start = size // 2
    data[start:start+payload_size] = payload
    return data

# ==============================================================
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Architecture Taxonomy"""
    print("\n" + "=" * 60)
    print("D1: ARCHITECTURE TAXONOMY")
    print("=" * 60)

    conditions = {}
    for name, gen_fn in [("x86-64", gen_x86_64),
                         ("ARM64", gen_arm64),
                         ("WASM", gen_wasm),
                         ("Java", gen_java)]:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names, conditions=conditions)

def direction_2(runner, d1_conditions):
    """D2: Sequential structure — real vs shuffled"""
    print("\n" + "=" * 60)
    print("D2: SEQUENTIAL STRUCTURE")
    print("=" * 60)

    results = {}
    for name in ["x86-64", "ARM64"]:
        # We need the original chunks to shuffle them
        chunks = [gen_x86_64(rng, runner.data_size) if name == "x86-64" else gen_arm64(rng, runner.data_size)
                  for rng in runner.trial_rngs()]
        real = d1_conditions[name]
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, findings = runner.compare(real, shuf)
        results[name] = ns
        print(f"  {name} vs shuffled = {ns:3d} sig")

    return dict(results=results)

def direction_3(runner, d1_conditions):
    """D3: Entropy Sweep — raw vs packed vs encrypted"""
    print("\n" + "=" * 60)
    print("D3: ENTROPY SWEEP")
    print("=" * 60)

    conditions = {"raw_x86": d1_conditions["x86-64"]}
    for name, gen_fn in [("packed", gen_compressed),
                         ("encrypted", gen_encrypted)]:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)

def direction_4(runner, d1_conditions):
    """D4: Obfuscation — Control Flow Flattening"""
    print("\n" + "=" * 60)
    print("D4: OBFUSCATION (CF-FLATTENING)")
    print("=" * 60)

    with runner.timed("obfuscated"):
        chunks = [gen_obfuscated(rng, runner.data_size)
                  for rng in runner.trial_rngs()]
        obf_metrics = runner.collect(chunks)

    ns, findings = runner.compare(d1_conditions["x86-64"], obf_metrics)
    print(f"  x86-64 vs Obfuscated = {ns:3d} sig")
    if findings:
        for m, d, p in findings[:3]:
            print(f"    {m:45s} d={d:+8.2f}")
    
    return dict(ns=ns)

def direction_5(runner, d1_conditions):
    """D5: Payload Detection"""
    print("\n" + "=" * 60)
    print("D5: PAYLOAD DETECTION")
    print("=" * 60)

    with runner.timed("with_payload"):
        chunks = [gen_benign_with_payload(rng, runner.data_size)
                  for rng in runner.trial_rngs()]
        payload_metrics = runner.collect(chunks)

    ns, findings = runner.compare(d1_conditions["x86-64"], payload_metrics)
    print(f"  x86-64 vs With Payload = {ns:3d} sig")
    
    return dict(ns=ns)

# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "Binary Anatomy: Structural Geometry of Machine Code")

    # D1: Taxonomy Heatmap
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Architecture Taxonomy")

    # D2: Bars for vs Shuffled
    names = list(d2['results'].keys())
    vals = [d2['results'][n] for n in names]
    runner.plot_bars(axes[1], names, vals, "D2: Sequential Structure (vs Shuffled)")

    # D3: Entropy Sweep Heatmap
    runner.plot_heatmap(axes[2], d3['matrix'], d3['names'], "D3: Entropy Sweep")

    # D4 & D5: Bars
    axes[3].bar(["Obfuscated", "With Payload"], [d4['ns'], d5['ns']], color=['#9C27B0', '#F44336'], alpha=0.85)
    runner.dark_ax(axes[3])
    axes[3].set_ylabel(f"Sig metrics (vs x86-64)")
    axes[3].set_title("D4 & D5: Structural Deviations")
    for i, v in enumerate([d4['ns'], d5['ns']]):
        axes[3].text(i, v + 0.5, str(v), ha='center', color='white')

    # D1b: Key metric across architectures
    metric = "Tropical:linearity"
    if metric in d1['conditions']['x86-64']:
        arch_names = d1['names']
        arch_vals = [np.mean(d1['conditions'][n][metric]) for n in arch_names]
        runner.plot_bars(axes[4], arch_names, arch_vals, f"Metric: {metric.split(':')[-1]}")
        axes[4].set_ylabel("Value")

    runner.save(fig, "binary_anatomy")

# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("Binary Anatomy", mode="1d")

    print("=" * 60)
    print("BINARY ANATOMY: STRUCTURAL GEOMETRY OF EXECUTABLES")
    print(f"size={runner.data_size}, trials={runner.n_trials}, "
          f"metrics={runner.n_metrics}")
    print("=" * 60)

    d1 = direction_1(runner)
    d2 = direction_2(runner, d1['conditions'])
    d3 = direction_3(runner, d1['conditions'])
    d4 = direction_4(runner, d1['conditions'])
    d5 = direction_5(runner, d1['conditions'])

    make_figure(runner, d1, d2, d3, d4, d5)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    runner.print_summary({
        'D1': f"Arch taxonomy: {np.mean(d1['matrix'][np.triu_indices_from(d1['matrix'], k=1)]):.1f} avg sig",
        'D2': f"vs Shuffled: x86={d2['results']['x86-64']} sig, ARM={d2['results']['ARM64']} sig",
        'D3': f"Entropy sweep: packed vs encrypted = {d3['matrix'][1,2]} sig",
        'D4': f"Obfuscation: {d4['ns']} sig metrics detected deviation",
        'D5': f"Payload: {d5['ns']} sig metrics detected deviation"
    })

if __name__ == "__main__":
    main()
