# Complete Findings

All validated discoveries and negative results from 29 investigations across 8 rounds.

## Validated Positive Findings (45)

### Methodology Validation
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 1 | All hash functions (MD5, SHA-1, SHA-256, SHA-3, BLAKE2, SHA-512) are geometrically indistinguishable from random | d ≈ 0, 0 significant metrics | `1d/hashes.py` |
| 2 | 21 geometries collapse to ~6 independent dimensions | r > 0.85 clusters | Redundancy analysis |

### Cryptography
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 3 | ECB mode massively detectable with structured plaintext | d = 19-146 | `1d/ciphers.py` |
| 4 | All ECB block ciphers (AES, DES3, Blowfish) produce identical geometric signatures | E8 = 2 roots for all | `1d/ciphers.py` |
| 5 | Stream ciphers (AES-CTR, ChaCha20, RC4) look random | d ≈ 0 | `1d/ciphers.py` |
| 6 | AES becomes indistinguishable from random at exactly 4 rounds | 40 metrics at R=1, 0 at R=4 | `1d/reduced_aes.py` |
| 7 | LSB steganography cracked via bitplane decomposition | d = 1166 | `1d/stego.py` |
| 8 | Fisher detects stego at 10% embedding rate | d = 3.33 | `1d/stego.py` |
| 9 | Bitplane analysis has perfect specificity (no crosstalk between planes) | 0 false positives | `1d/stego.py` |

### PRNG Testing
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 10 | E8 catches RANDU from raw bytes | d = -19.89 | `1d/prng.py` |
| 11 | E8 catches glibc LCG from raw bytes | d = -15.00 | `1d/prng.py` |
| 12 | Tropical geometry is the second-best PRNG detector | d = -8 | `1d/prng.py` |
| 13 | RANDU detectable at ALL bit planes | d > 10 for most planes | `1d/stego.py` |

### Chaotic Systems
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 14 | 10/10 chaotic maps detected as non-random | varies | `1d/chaos.py` |
| 15 | 45/45 pairwise chaotic map distinctions | varies | `1d/chaos.py` |
| 16 | Each chaotic map has a unique geometric fingerprint | Fisher: 17/45 pairs | `1d/chaos.py` |
| 17 | Lorenz uniquely destroys Penrose 5-fold symmetry | d = 60 | `1d/chaos.py` |
| 18 | Rossler shows extreme Tropical linearity | d = 266 | `1d/chaos.py` |

### Biology
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 19 | All DNA sequence types have distinct geometric signatures | 291 significant findings | `1d/dna.py` |
| 20 | Microsatellites are the most detectable DNA feature | d = -18.3 (E8) | `1d/dna.py` |
| 21 | Viral DNA uniquely caught by Sol geometry | d = 6.2 | `1d/dna.py` |
| 22 | Protein-coding DNA caught by S2xR sphere concentration | d = 7.2 | `1d/dna.py` |
| 23 | E. coli vs shuffled E. coli: geometry sees DNA ordering | 7 significant metrics | `1d/dna.py` |

### Neural Networks
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 24 | All NN weight types distinguishable from random | 57-71 significant metrics | `1d/nn_weights.py` |
| 25 | Backdoored weights detected by 15 metrics | d = 7.12 (Wasserstein) | `1d/nn_weights.py` |
| 26 | Dense vs conv weights distinguished | d = 11.9 (Cantor) | `1d/nn_weights.py` |

### Compression
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 27 | bz2 output distinguishable from zlib/lzma | d = 7.75 | `1d/compression_algos.py` |
| 28 | Compressed structured data ≠ random | d = 128 (coverage) | `1d/compression_algos.py` |

### Preprocessing
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 29 | Delay embedding gives 52x improvement for lag detection | d: 0.17 → 9.1 | Delay embedding study |
| 30 | FFT and raw analysis are complementary (17+20 exclusive pairs) | varies | Spectral study |
| 31 | kurt_mean (4th order) is independent from all 23 geometries | max r = 0.22 | Higher-order study |
| 32 | Permutation entropy: 33x better for recurrence detection | d = 250 vs 7.5 | Higher-order study |

### 2D Spatial Field
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 33 | Ising model: all temperatures vs random distinguished | varies | `2d/ising.py` |
| 34 | Ising: multiscale_coherence_4 peaks near T_c | scale-free structure | `2d/ising.py` |
| 35 | Reaction-diffusion: 15/15 morphology pairs | d = 97 (tension_std) | `2d/reaction_diffusion.py` |
| 36 | Percolation: 28/28 probability pairs | d = 261 (n_basins) | `2d/percolation.py` |
| 37 | Cellular automata: 14/15 rule pairs | d = 78.5 (anisotropy) | `2d/cellular_automata.py` |
| 38 | ECB penguin detected in 2D, CBC/CTR invisible | d = -283 | `2d/ecb_penguin.py` |
| 39 | Maze algorithms: 15/15 pairs | d = 108.9 | `2d/mazes.py` |
| 40 | Wave equation: 15/15 source configs | d = -496 | `2d/wave_equation.py` |
| 41 | Voronoi tessellations: 10/10 point process pairs | d = -154 | `2d/voronoi.py` |
| 42 | Growth models: 3/3 (DLA/Eden/random) | d = -178 | `2d/growth_models.py` |
| 43 | Sandpile SOC: 5/6 pairs, convergence detected | d = 261 | `2d/sandpile.py` |
| 44 | Lenia continuous CA: 12/15 configs | d = 249 | `2d/lenia.py` |
| 45 | Near-identical rules detected: GoL ≈ HighLife, Kruskal ≈ AldousBroder | d ≈ 0 | Various 2D |

## Negative Results (17)

These are equally important — they define the boundaries of what geometric analysis can and cannot do.

| # | Finding | Implication |
|---|---------|-------------|
| 1 | **AES-CTR indistinguishable from random** across ALL geometries, preprocessings, higher-order stats, and bit planes | Fundamental limit; AES works |
| 2 | MT19937 passes all geometric tests | Mersenne Twister is geometrically random at byte level |
| 3 | XorShift32 passes all geometric tests | Simple but sufficient for byte-level randomness |
| 4 | MINSTD passes all geometric tests | Park-Miller is geometrically random |
| 5 | RC4 stream looks random (even 24-bit key) | Stream cipher output is geometrically indistinguishable |
| 6 | Standard map ≈ random | Uniformly mixing on torus |
| 7 | Arnold cat map ≈ random | Uniformly mixing on torus |
| 8 | LSB steganography invisible at byte level | Must use bitplane decomposition |
| 9 | GBM indistinguishable from IID normal | Random walk is geometrically random |
| 10 | Ornstein-Uhlenbeck ≈ IID normal | Mean-reversion too subtle for byte encoding |
| 11 | Random vs encrypted is the hardest classification boundary | Both designed to look random |
| 12 | Multi-scale analysis doesn't improve general classification | 69.5% < 79.5% baseline |
| 13 | Cepstrum mostly useless after uint8 quantization | Information crushed |
| 14 | GARCH hard to detect after byte quantization | Variance-of-variance destroyed |
| 15 | GoL ≈ HighLife | Near-identical rules (differ only in B6) |
| 16 | Sandpile 10k ≈ 50k iterations | Both at SOC steady state |
| 17 | Kruskal ≈ Aldous-Broder maze generation | Both produce uniform spanning trees |

## Key Takeaway

The framework detects genuine structure with massive effect sizes (d = 7-1166) while producing zero false positives on validated random sources. The AES-CTR negative result confirms that the methodology is honest — geometries report "no structure" when encryption is working correctly.
