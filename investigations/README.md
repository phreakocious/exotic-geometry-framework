# Investigations

27 self-contained investigation scripts demonstrating the framework across diverse domains.

## Running

Each script is standalone. From the repo root:

```bash
pip install -r requirements.txt
python investigations/1d/chaos.py
python investigations/2d/ising.py
```

Three scripts require `pycryptodome`:
```bash
pip install -r requirements-crypto.txt
python investigations/1d/ciphers.py
python investigations/1d/reduced_aes.py
python investigations/2d/ecb_penguin.py
```

Figures are saved to `figures/` (gitignored, generated on demand).

## 1D Investigations (15)

| Script | Domain | Key Result | Dependencies |
|--------|--------|------------|-------------|
| `1d/hashes.py` | Hash functions (MD5, SHA-1, SHA-256, SHA-3, BLAKE2) | 0 significant differences from random. Validates methodology. | hashlib |
| `1d/prng.py` | PRNGs (RANDU, LCG, MT19937, XorShift) | RANDU d=-19.89, glibc LCG d=-15.00. MT19937 passes. | numpy |
| `1d/ciphers.py` | Cipher modes (ECB, CTR, CBC) | ECB d=19-146, stream/block CTR invisible. | pycryptodome |
| `1d/reduced_aes.py` | Reduced-round AES (1-10 rounds) | Sharp cliff at R=4: 40 metrics → 0 metrics. | pycryptodome |
| `1d/stego.py` | LSB steganography via bitplane | 1 sig metric (Fisher d=1.06) via LSB correlation. | numpy |
| `1d/stego_deep.py` | Advanced stego (6 techniques) | PVD: 42 sig raw bytes. Matrix embed: invisible. Bitplane: 0 sig. | numpy, scipy |
| `1d/chaos.py` | 10 chaotic maps (logistic, Henon, Lorenz, ...) | 10/10 detected, 45/45 pairwise distinguished. | numpy |
| `1d/dna.py` | DNA sequences (8 organism types) | 291 significant findings. All types distinguishable. | numpy |
| `1d/nn_weights.py` | Neural network weights | Backdoor detected (d=7.12). Dense vs conv d=11.9. | numpy |
| `1d/compression_algos.py` | Compressed data (zlib, bz2, lzma) | bz2 vs zlib d=7.75. Algorithms distinguishable. | zlib, bz2, lzma |
| `1d/collatz.py` | Collatz (3n+1) sequences | 7/7 encodings detected. 21/21 pairwise. 3n+1 vs 5n+1: 71 sig. | numpy |
| `1d/collatz_deep.py` | Deep Collatz I: phase transitions | Sharp k=1→2 boundary (86→35 sig). Tropical slopes match theory. | numpy, scipy |
| `1d/collatz_deep2.py` | Deep Collatz II: convergence anatomy | 45 convergence-specific metrics. Composition order matters. 37 drift detectors. | numpy, scipy |
| `1d/primes.py` | Prime number sequences (7 encodings) | 7/7 detected, 21/21 pairwise. Primes vs Cramér model: 55 sig. All ordering-dependent. | numpy, scipy |
| `1d/primes_deep.py` | Deep prime gaps: what Cramér misses | 52 pure-primality metrics. Sieved Cramér closes gap 54→30. 31 always-sig across scales. | numpy, scipy |
| `1d/number_theory.py` | Arithmetic functions (μ, λ, d, φ, Ω, Mertens, ζ zeros) | 8/8 detected. Mertens vs random walk: 37 sig (beyond RH). All ordering-dependent. | numpy, scipy, mpmath |

## 2D Investigations (12)

All 2D scripts use `SpatialFieldGeometry` for native spatial analysis.

| Script | Domain | Key Result | Dependencies |
|--------|--------|------------|-------------|
| `2d/ising.py` | Ising model phase transition | All temps vs random. T_c detected via multiscale coherence. | numpy |
| `2d/reaction_diffusion.py` | Gray-Scott morphologies | 15/15 pairs (spots/stripes/worms/coral/mazes/chaos). | numpy, scipy |
| `2d/percolation.py` | Site percolation | 28/28 probability pairs. n_basins strongest. | numpy, scipy |
| `2d/cellular_automata.py` | 2D cellular automata rules | 14/15 pairs. GoL ≈ HighLife (expected). | numpy |
| `2d/ecb_penguin.py` | ECB block structure in 2D | ECB 12-15 sig metrics. CBC/CTR: 0 sig. | pycryptodome |
| `2d/mazes.py` | Maze generation algorithms | 15/15 algorithm pairs distinguished. | numpy |
| `2d/wave_equation.py` | Wave equation source configs | 15/15 configs. coherence_score d=-496. | numpy |
| `2d/voronoi.py` | Point process signatures | 10/10 pairs (Poisson/clustered/regular/...). | numpy, scipy |
| `2d/growth_models.py` | DLA, Eden, random growth | 3/3 pairs. Eden anisotropy d=-178. | numpy |
| `2d/sandpile.py` | Abelian sandpile (SOC) | 5/6 pairs. 10k ≈ 50k (both at SOC). | numpy |
| `2d/lenia.py` | Lenia continuous cellular automata | 12/15 configs (3 died to density=0). | numpy, scipy |
| `2d/stego_bitmatrix.py` | 2D stego via byte representations | Co-occurrence: SS 14/15 (d=−5.6). Diff grid: LSBMR 14 sig. Matrix embed: still invisible. | numpy, scipy |

## Investigation Template

Each script follows the same statistical protocol (see [docs/METHODOLOGY.md](../docs/METHODOLOGY.md)):

1. Generate N trials of each condition
2. Compute geometric metrics per trial
3. Welch's t-test with Bonferroni correction
4. Report Cohen's d effect sizes
5. Shuffle validation for significant findings
