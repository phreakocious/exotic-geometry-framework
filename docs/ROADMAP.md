# Roadmap

Next investigations and improvements, roughly priority-ordered.

## New Investigations

### 1. Prime Numbers (`investigations/1d/primes.py`, `primes_deep.py`) — COMPLETE
Can exotic geometries detect structure in prime number sequences?
- 7 encodings: prime gaps, primes mod 256, last digits, binary expansion, gap pairs, mod 30#, gap differences
- Primes vs Cramér random model (fake primes with probability 1/ln(n))
- Deep follow-up: 52 pure-primality metrics, model hierarchy (Cramér→sieved→dist-matched: 54→30→14), 31 always-sig across scales

### 2. Number Theory Deep Dive (`investigations/1d/number_theory.py`) — COMPLETE
8 arithmetic functions: d(n), φ(n), φ/n, Ω(n), μ(n), λ(n), Mertens M(n), ζ zero spacings.
- All 8 detected (71-105 sig), all 28 pairwise distinguished, all ordering-dependent
- Mertens vs random walk: 37 sig metrics — structure beyond what RH predicts
- d(n) vs distribution-matched: 85 sig — massive sequential correlation in divisor function
- Zeta spacings vs Wigner surmise: 90 sig — structure beyond GUE

### 3. Continued Fractions (`investigations/1d/continued_fractions.py`) — COMPLETE
CF coefficients of 5 constants (√2, √3, e, π, ln 2) vs Gauss-Kuzmin surrogates.
- π vs GK = 0 sig — CF coefficients are indistinguishable from iid Gauss-Kuzmin
- ln 2 vs GK = 11 sig (weak sequential signal); e vs GK = 65; √2/√3 trivially different
- π ordering = 0, ln 2 ordering = 0 — no detectable sequential dependence
- π vs ln 2 = only 4 sig pairwise (both look GK-like); algebraic vs transcendental = 96-100
- Khinchin check: π geo_mean = 2.6624 ≈ K = 2.6854

### 4. Deep DNA (`investigations/1d/dna_deep.py`)
Follow the collatz_deep pattern: 5 directions probing the 291 findings.
- Which findings are ordering vs distribution?
- Which geometries are organism-specialists?
- Codon structure (3-periodicity) via delay embedding
- GC-content vs sequence complexity separation

### 5. Deep Chaos (`investigations/1d/chaos_deep.py`) — COMPLETE
Address the baker/logistic_edge degeneracy and probe deeper.
- Lyapunov exponent correlation: 14 metrics with |r| > 0.8
- Route to chaos (logistic map r parameter sweep)
- Poincaré section analysis: 107 metrics distinguish r=3.6 from r=4.0
- Float precision recovers structure lost to quantization

### 6. PRNG Deep (`investigations/1d/prng_deep.py`) — COMPLETE
Can preprocessing catch MT19937 or XorShift? 5 directions:
- D1 Multiscale delay embedding: XorShift128 detected at τ=3 (2 sig), MINSTD at τ=1 (4 sig)
- D2 Bitplane decomposition: MT19937 and XorShift128 remain invisible
- D3 2D spatial analysis: all three generators indistinguishable from urandom as images
- D4 Spectral preprocessing: confirms time-domain quality, no hidden spectral bias
- D5 Shuffle test: MINSTD shows 2 sig (Spiral geometry detects determinism), MT19937/XorShift128 = 0

## Framework Improvements

### 7. Audit Remaining d Value Claims
- DNA: d = -18.3 (E8, microsatellites) — unverified
- NN weights: d = 7.12 (Wasserstein, backdoor) — unverified
- Chaos: d = 266 (Rossler), d = 60 (Lorenz) — VERIFIED ✓
- Stego: d = 1.06 (Fisher, LSB) — VERIFIED ✓, d = 1166 was FALSE ✗

## Completed
- `investigations/1d/continued_fractions.py` — 5 constants, GK surrogates, π passes iid test ✓
- `investigations/1d/chaos_deep.py` — Lyapunov correlation, bifurcation sweep ✓
- `investigations/1d/prng_deep.py` — 5 directions: delay, bitplane, spatial, spectral, shuffle ✓
- `investigations/1d/number_theory.py` — 8 arithmetic functions, 5 directions ✓
- `investigations/1d/primes_deep.py` — 5 directions probing what Cramér misses ✓
- `investigations/1d/stego_deep.py` — 6 techniques, 5 directions ✓
- Doc cleanup: false d=1166 claim corrected across 6 files ✓
- `investigations/1d/bearing_fault.py` — CWRU bearing fault diagnosis, fractal geometry specialization ✓
- `investigations/1d/structure_atlas.py` — 56-source atlas, 13 domains, surrogate decomposition, multi-scale ✓
- `investigations/1d/math_constants.py` — Rewritten: base-256 digits, CF taxonomy, Gauss-Kuzmin null, representation fingerprints ✓
- `investigations/1d/binary_anatomy.py` — ISA taxonomy (x86/ARM/WASM/JVM) ✓
- `investigations/1d/quantum_geometry.py` — Quantum wavefunctions ✓
- `investigations/1d/memory_geometry.py` — Data structure topology ✓
- `investigations/1d/esoteric_code.py` — Esolang fingerprinting ✓
- `investigations/1d/sorting_algorithms.py` — Sort memory access traces ✓
- `investigations/1d/music_theory.py` — Musical structure ✓
- `investigations/1d/network_protocols.py` — Protocol fingerprinting ✓
- Mandelbrot + Julia fractal geometries added to framework (10 new metrics) ✓
- Runner module (`tools/investigation_runner.py`) — shared boilerplate ✓
- Gemini CLI integration (`tools/spawn_gemini.sh`) ✓
- Meta-investigation: 7D signature space, 98.7% metric redundancy ✓

## Next Steps
- Phase 2 datasets: GTZAN music genres, ESC-50 environmental sounds, TESS emotional speech
- Lean mode: pruned ~40-metric fast path using meta-investigation redundancy analysis
- Re-harvest classifier signatures with working Mandelbrot/Julia metrics
- Structure Atlas at even larger scales (24K, 48K bytes)
- BBP partial sums investigation (Pi convergence dynamics)
