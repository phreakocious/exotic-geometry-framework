#!/usr/bin/env python3
"""
Music Theory Investigation: MIDI-like Byte Streams
==================================================

Can exotic geometries distinguish between musical structures like scales,
arpeggios, chord progressions, and drum patterns in raw byte streams?

DIRECTIONS:
D1: Taxonomy — Scales vs Arpeggios vs Drums vs Random (Heatmap)
D2: Sequential Structure — Real vs Shuffled for each type (Bar chart)
D3: Tonality Sweep — Diatonic (Order) to Chromatic (Chaos) (Line plot)
D4: Polyphony Density — 1 vs 2 vs 3 vs 4 interleaved voices (Line plot)
D5: Rhythmic Periodicity — Drum loop lengths (4, 8, 16, 32) (Line plot)
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
SEED = 314159
np.random.seed(SEED)

# Music Theory Constants
SCALE_MAJOR = [0, 2, 4, 5, 7, 9, 11]
SCALE_MINOR = [0, 2, 3, 5, 7, 8, 10]
CHORD_MAJ = [0, 4, 7]
CHORD_MIN = [0, 3, 7]
CHORD_DOM7 = [0, 4, 7, 10]
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HAT = 42
DRUM_OHAT = 46

# ==============================================================
# DATA GENERATORS
# ==============================================================

def generate_random_chromatic(rng, size):
    """Unstructured random notes (0-127)."""
    return rng.integers(0, 128, size=size, dtype=np.uint8)

def generate_scale_run(rng, size, scale_type='major'):
    """Continuous run up/down a scale."""
    root = rng.integers(0, 12)
    intervals = SCALE_MAJOR if scale_type == 'major' else SCALE_MINOR
    
    # Construct full keyboard map for this scale
    notes = []
    octave = 0
    while True:
        for interval in intervals:
            note = octave * 12 + root + interval
            if note > 127:
                break
            notes.append(note)
        if notes and notes[-1] > 115: # Stop near top
            break
        octave += 1
    
    notes = np.array(notes, dtype=np.uint8)
    if len(notes) == 0: return generate_random_chromatic(rng, size)

    # Generate random walk on scale indices
    result = np.zeros(size, dtype=np.uint8)
    current_idx = rng.integers(len(notes) // 3, 2 * len(notes) // 3)
    
    for i in range(size):
        result[i] = notes[current_idx]
        step = rng.choice([-1, 1])
        current_idx += step
        # Bounce off edges
        if current_idx < 0:
            current_idx = 1
        elif current_idx >= len(notes):
            current_idx = len(notes) - 2
            
    return result

def generate_arpeggio(rng, size, chord_type='maj'):
    """Arpeggiated chords (Root-3-5-Root-3-5...)."""
    root = rng.integers(40, 70) # Middle range
    intervals = CHORD_MAJ
    if chord_type == 'min': intervals = CHORD_MIN
    if chord_type == 'dom7': intervals = CHORD_DOM7
    
    pattern = [(root + x) % 128 for x in intervals]
    
    # Repeat pattern to fill size
    # Add slight variation: occasionally jump octave
    result = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        base_note = pattern[i % len(pattern)]
        octave_shift = rng.choice([0, 0, 0, 12, -12]) # Occasional octave jump
        note = np.clip(base_note + octave_shift, 0, 127)
        result[i] = note
        
    return result

def generate_drums(rng, size, loop_len=16):
    """Drum pattern repeating every loop_len events."""
    # Create a basic beat
    pattern = np.zeros(loop_len, dtype=np.uint8)
    
    # Basic Rock Beat framework
    for i in range(loop_len):
        r = rng.random()
        if i % 8 == 0: # Kick on 1
            pattern[i] = DRUM_KICK
        elif i % 4 == 0: # Snare on 2/4
            pattern[i] = DRUM_SNARE
        elif i % 2 == 0: # Hat 8ths
            pattern[i] = DRUM_HAT
        else: # Offbeats / Fills
            if r < 0.2: pattern[i] = DRUM_KICK
            elif r < 0.4: pattern[i] = DRUM_SNARE
            elif r < 0.8: pattern[i] = DRUM_HAT
            else: pattern[i] = DRUM_OHAT
            
    # Tile the pattern
    full = np.tile(pattern, size // loop_len + 1)[:size]
    
    # Add slight "humanize" error? 
    # Actually, for byte streams, exact repetition is the signal.
    # But we need trial independence.
    # The pattern generation above uses `rng`, so each trial has a unique beat.
    
    return full

def generate_tonality_mix(rng, size, randomness=0.0):
    """Mix of Scale (ordered) and Random (chromatic)."""
    scale_data = generate_scale_run(rng, size)
    random_data = generate_random_chromatic(rng, size)
    mask = rng.random(size) < randomness
    return np.where(mask, random_data, scale_data)

def generate_polyphony(rng, size, voices=1):
    """Interleave 'voices' independent scale runs."""
    # Generate 'voices' separate buffers
    sub_size = (size // voices) + 10
    streams = [generate_scale_run(rng, sub_size) for _ in range(voices)]
    
    result = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        voice_idx = i % voices
        stream_idx = i // voices
        result[i] = streams[voice_idx][stream_idx]
        
    return result

# ==============================================================
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Taxonomy — Can we distinguish basic musical textures?"""
    print("\n" + "=" * 60)
    print("D1: TAXONOMY")
    print("=" * 60)

    conditions = {
        "Scale": generate_scale_run,
        "Arpeggio": generate_arpeggio,
        "Drums": generate_drums,
        "Random": generate_random_chromatic
    }

    results = {}
    for name, gen_fn in conditions.items():
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            results[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(results)
    return dict(matrix=matrix, names=names)

def direction_2(runner):
    """D2: Sequential Structure — Real vs Shuffled"""
    print("\n" + "=" * 60)
    print("D2: SEQUENTIAL STRUCTURE")
    print("=" * 60)

    results = {}
    # Compare 3 distinct types against their shuffled versions
    for name, gen_fn in [("Scale", generate_scale_run),
                         ("Arpeggio", generate_arpeggio),
                         ("Drums", generate_drums)]:
        chunks = [gen_fn(rng, runner.data_size) 
                  for rng in runner.trial_rngs()]
        real = runner.collect(chunks)
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, _ = runner.compare(real, shuf)
        results[name] = ns
        print(f"  {name} vs Shuffled: {ns} sig")

    return dict(results=results)

def direction_3(runner):
    """D3: Tonality Sweep — From Scale (0.0) to Random (1.0)"""
    print("\n" + "=" * 60)
    print("D3: TONALITY SWEEP")
    print("=" * 60)

    # 0.0 = Pure Scale, 1.0 = Pure Random
    params = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Baseline is pure scale (0.0)
    # Actually, let's compare each step to Pure Random (1.0) to see detection fade?
    # Or compare each step to Baseline (0.0) to see detection rise?
    # Usually "distance from baseline" is good.
    
    baseline_chunks = [generate_tonality_mix(rng, runner.data_size, 0.0)
                       for rng in runner.trial_rngs()]
    baseline = runner.collect(baseline_chunks)

    results = {}
    for p in params:
        chunks = [generate_tonality_mix(rng, runner.data_size, p)
                  for rng in runner.trial_rngs(offset=int(p*100))]
        met = runner.collect(chunks)
        ns, _ = runner.compare(baseline, met)
        results[p] = ns
        print(f"  p={p:.2f} vs Baseline: {ns} sig")

    return dict(results=results, params=params)

def direction_4(runner):
    """D4: Polyphony Density — Interleaved Voices (1, 2, 3, 4, 8)"""
    print("\n" + "=" * 60)
    print("D4: POLYPHONY DENSITY")
    print("=" * 60)

    voices_list = [1, 2, 3, 4, 8]
    
    # Baseline is 1 voice
    baseline_chunks = [generate_polyphony(rng, runner.data_size, 1)
                       for rng in runner.trial_rngs()]
    baseline = runner.collect(baseline_chunks)

    results = {}
    for v in voices_list:
        chunks = [generate_polyphony(rng, runner.data_size, v)
                  for rng in runner.trial_rngs(offset=v*10)]
        met = runner.collect(chunks)
        ns, _ = runner.compare(baseline, met)
        results[v] = ns
        print(f"  Voices={v} vs 1-Voice: {ns} sig")

    return dict(results=results, params=voices_list)

def direction_5(runner):
    """D5: Rhythmic Periodicity — Drum Loop Lengths"""
    print("\n" + "=" * 60)
    print("D5: RHYTHMIC PERIODICITY")
    print("=" * 60)

    # Compare different loop lengths to Random (infinite loop length)
    # This checks if we can detect the repetition period.
    lengths = [4, 8, 16, 32, 64]
    
    # Baseline: Random Chromatic (no loop / infinite loop)
    # Or Baseline: Length 4?
    # Let's use Random as baseline to see "Detection of Loop"
    
    baseline_chunks = [generate_random_chromatic(rng, runner.data_size)
                       for rng in runner.trial_rngs()]
    baseline = runner.collect(baseline_chunks)

    results = {}
    for l in lengths:
        chunks = [generate_drums(rng, runner.data_size, loop_len=l)
                  for rng in runner.trial_rngs(offset=l*5)]
        met = runner.collect(chunks)
        ns, _ = runner.compare(baseline, met)
        results[l] = ns
        print(f"  Loop={l} vs Random: {ns} sig")

    return dict(results=results, params=lengths)

# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "Music Theory: MIDI Byte Streams")

    # D1: Heatmap
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Taxonomy")

    # D2: Bars (Structure)
    names2 = list(d2['results'].keys())
    vals2 = [d2['results'][n] for n in names2]
    runner.plot_bars(axes[1], names2, vals2, "D2: Structure vs Shuffled")

    # D3: Line (Tonality)
    runner.plot_line(axes[2], d3['params'], [d3['results'][p] for p in d3['params']],
                     "D3: Tonality Sweep (vs Scale)", xlabel="Randomness")

    # D4: Line (Polyphony)
    runner.plot_line(axes[3], d4['params'], [d4['results'][p] for p in d4['params']],
                     "D4: Polyphony (vs Monophonic)", xlabel="Voices")

    # D5: Line (Loop Length)
    runner.plot_line(axes[4], d5['params'], [d5['results'][p] for p in d5['params']],
                     "D5: Drum Loop vs Random", xlabel="Loop Length")

    runner.save(fig, "music_theory")

# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("MusicTheory", mode="1d")

    print("=" * 60)
    print("MUSIC THEORY INVESTIGATION")
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
        'D1': f"Taxonomy: {pw.min():.0f}-{pw.max():.0f} sig (mean {pw.mean():.0f})",
        'D2': [f"{n} vs shuffled = {v}" for n, v in d2['results'].items()],
        'D3': "Tonality: " + ", ".join(
            f"{p}→{d3['results'][p]}" for p in d3['params']),
        'D4': "Polyphony: " + ", ".join(
            f"v={p}→{d4['results'][p]}" for p in d4['params']),
        'D5': "Loop length: " + ", ".join(
            f"{p}→{d5['results'][p]}" for p in d5['params']),
    })

if __name__ == "__main__":
    main()
