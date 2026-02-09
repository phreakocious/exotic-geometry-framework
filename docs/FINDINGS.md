# Complete Findings

All validated discoveries and negative results from 37 investigations.

## Validated Positive Findings (114)

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
| 7 | LSB stego weakly detectable via LSB-correlation extraction | d = 1.06 (Fisher trace), 1 sig metric at 100% rate | `1d/stego.py` |
| 8 | PVD steganography massively detectable at raw byte level | 42 sig metrics (natural_texture), detectable at 10% rate | `1d/stego_deep.py` |
| 9 | Spread spectrum stego detectable at raw byte level | 25 sig metrics (natural_texture), detectable at 25% rate | `1d/stego_deep.py` |
| 9a | Matrix embedding (Hamming syndrome coding) invisible to all geometries | 0 sig at all rates, both carriers | `1d/stego_deep.py` |
| 9b | Bitplane extraction does NOT improve stego detection (8x sample reduction) | 0 sig across all techniques with bitplane | `1d/stego_deep.py` |
| 9c | Delay embedding amplifies detectable stego signal | PVD: 16 sig via DE2 at 50% rate | `1d/stego_deep.py` |
| 9d | 2D co-occurrence matrix detects PVD and SS via spatial metrics alone | PVD: 13/15, SS: 14/15; tension_mean d=−5.6 | `2d/stego_bitmatrix.py` |
| 9e | 2D diff grid amplifies LSBMR from near-invisible to solidly detected | 14 sig (up from 2–6 in 1D); Wasserstein d=−1.36 | `2d/stego_bitmatrix.py` |
| 9f | 2D representation choice is critical — co-occurrence Σ=27, diff grid Σ=24, bitplane tiled Σ=0 | binary grids kill SpatialField metric variance | `2d/stego_bitmatrix.py` |
| 9g | 8 spatial geometries (80 metrics) nearly 4x PVD/SS detection on co-occurrence | PVD: 13→49, SS: 14→52; every geometry contributes unique detections | Framework 2D battery |
| 9h | HodgeLaplacian saturates co-occurrence detection (8/9 metrics); SpectralPower adds 7/8 | Laplacian energy, Poisson recovery, spectral slope are new detection axes | Framework 2D battery |
| 9i | 10 diverse 2D field types: all 45 pairwise distinguished by 8 spatial geometries | min 45/80 sig (Stripes vs Spirals), max 78/80 (Checkerboard vs Voronoi) | 2D field type validation |

### PRNG Testing
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 10 | E8 catches RANDU from raw bytes | d = -19.89 | `1d/prng.py` |
| 11 | E8 catches glibc LCG from raw bytes | d = -15.00 | `1d/prng.py` |
| 12 | Tropical geometry is the second-best PRNG detector | d = -8 | `1d/prng.py` |
| 13 | RANDU detectable at ALL bit planes | d > 10 for most planes | `1d/prng.py` |

### RNG Quality Testing (10 generators, systematic)
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 13a | Self-check (urandom vs urandom) returns 0 sig — validates methodology | 0/131 | `1d/rng.py` |
| 13b | CRYPTO (os.urandom, SHA-256 CTR) and GOOD (PCG64, SFC64) = 0 sig | 0/131 | `1d/rng.py` |
| 13c | RANDU massively detected: 44 sig metrics, Penrose index_diversity d=-37 | 44/131 | `1d/rng.py` |
| 13d | Middle-Square massively detected: 78 sig metrics | 78/131 | `1d/rng.py` |
| 13e | Geometric metrics outperform standard: RANDU 44 vs 4, Middle-Square 78 vs 7 | 11x, 11x | `1d/rng.py` |
| 13f | Delay embedding reveals XorShift128 (0→2 sig via Thurston height_drift) | DE2 amplification | `1d/rng.py` |
| 13g | RANDU detectable at just 500 bytes (41 sig) | N=500 | `1d/rng.py` |
| 13h | Middle-Square detectable at 500 bytes (72 sig) | N=500 | `1d/rng.py` |
| 13i | E8, Higher-Order Stats, Torus T², Ammann-Beenker are top PRNG weakness detectors | heatmap | `1d/rng.py` |
| 13j | Penrose index_diversity is single most discriminating metric | d=-37 (RANDU) | `1d/rng.py` |

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

### Collatz Sequences
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 29 | All 7 Collatz encodings detected as non-random | 73-103 significant metrics | `1d/collatz.py` |
| 30 | 21/21 pairwise Collatz encoding distinctions | d = 7.4 (hailstone_small vs hailstone_large) | `1d/collatz.py` |
| 31 | Hailstone sequences carry sequential structure (destroyed by shuffling) | coverage ratio 0.31 | `1d/collatz.py` |
| 32 | Parity sequence is purely marginal (survives shuffling) | all ratios ≈ 1.0 | `1d/collatz.py` |
| 33 | 3n+1 vs 5n+1 variant geometrically distinguishable | 71 sig, d = 12.8 | `1d/collatz.py` |
| 34 | Starting number magnitude affects geometric signature | 90-98 sig metrics across ranges | `1d/collatz.py` |
| 35a | Sharp phase transition at k=1→2 in (2k+1)n+1 family | 86 sig (k=1) → 35 sig (k=2) | `1d/collatz_deep.py` |
| 35b | Tropical slopes match theory exactly: log₂(3) for 3n+1 | μ=1.5855 vs theory 1.5850 | `1d/collatz_deep.py` |
| 35c | Optimal delay embedding at τ=2, monotonic decrease beyond | 78 sig at τ=2, 42 at τ=21 | `1d/collatz_deep.py` |
| 35d | U-shaped bitplane signal: both LSB and MSB carry structure | LSB=77, MSB=68 sig metrics | `1d/collatz_deep.py` |
| 35e | Ordering matters most at mod 4 (not mod 2 or higher) | 63 metrics (mod 4) vs 40 (mod 2) | `1d/collatz_deep.py` |
| 36a | Syracuse encoding of v₂ sequence is the richest Collatz representation | 105 sig metrics | `1d/collatz_deep2.py` |
| 36b | Mean v₂(3n+1) = 2.001, exactly matching Geometric(1/2) theory | empirical vs theoretical | `1d/collatz_deep2.py` |
| 36c | 45 metrics are convergence-specific (sig for 3n+1, vanish for 5n+1) | 45 vs 3 divergence-specific | `1d/collatz_deep2.py` |
| 36d | Fisher, Heisenberg, Sol, Spherical, Wasserstein are 100% convergence-aware | all sig metrics vanish at k=2 | `1d/collatz_deep2.py` |
| 36e | Composition order matters: DE2→BP0 beats BP0→DE2 (86 vs 75 sig) | exceeds both individual baselines | `1d/collatz_deep2.py` |
| 36f | 37 metrics track drift rate across (2k+1)n+1 with \|r\| > 0.8 | kurt_mean r=+0.986 | `1d/collatz_deep2.py` |

### Prime Numbers
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 37a | All 7 prime encodings detected as non-random | 49-100 significant metrics | `1d/primes.py` |
| 37b | 21/21 pairwise encoding distinctions | min 71 sig (last_digit vs binary), max 109 sig | `1d/primes.py` |
| 37c | Prime gaps distinguishable from Cramér random model | 55 sig metrics | `1d/primes.py` |
| 37d | Prime gaps distinguishable from semiprime gaps | 80 sig metrics | `1d/primes.py` |
| 37e | All 7 encodings have ordering-dependent structure | orig vs shuf: 12-83 sig | `1d/primes.py` |
| 37f | Gap geometry evolves with prime size | all 6 range pairs: 68-82 sig | `1d/primes.py` |
| 37g | gap_pairs is richest encoding (100 sig vs random) | Torus chi2 d=441, Lorentzian d=307 | `1d/primes.py` |
| 38a | 52 metrics detect pure primality (sig vs both Cramér AND random) | 2-adic mean_distance \|d\|=128 | `1d/primes_deep.py` |
| 38b | Heisenberg and Algebraic geometries are 100% Cramér-sensitive | all sig metrics also distinguish from Cramér | `1d/primes_deep.py` |
| 38c | Model hierarchy closes the gap: Cramér=54 → Even-gap=37 → Sieved=30 → Dist-matched=14 | progressive improvement | `1d/primes_deep.py` |
| 38d | 14 metrics are pure sequential correlation in prime gaps | Lorentzian causal_order d=10.5 | `1d/primes_deep.py` |
| 38e | Delay embedding does NOT amplify real-vs-Cramér signal | 47-52 sig across τ=1-10 vs 54 raw | `1d/primes_deep.py` |
| 38f | Cramér gap shrinks with scale: 75 (1K) → 54 (1M) | 31 always-sig, 61 scale-dependent | `1d/primes_deep.py` |
| 38g | 14 dist-matched survivors confirmed: pure sequential correlation in prime gaps | Lorentzian d=10.5, HOS:perm_entropy d=4.2 | `1d/primes_deep2.py` |
| 38h | Prime gaps are time-irreversible (4 sig forward vs reversed) | Spiral:growth_rate d=+4.01 is strongest asymmetry detector | `1d/primes_deep2.py` |
| 38i | 57% linear, 43% nonlinear sequential structure in prime gaps | IAAFT: 6 sig (nonlinear), block-shuffle: 3 sig | `1d/primes_deep2.py` |
| 38j | mod6=1 vs mod6=5 gaps geometrically distinct (Lemke Oliver–Soundararajan) | 46 sig; Projective ℙ² d=−5.90 top discriminator | `1d/primes_deep2.py` |
| 38k | Both residue classes retain independent sequential structure | mod6=1: 9 sig, mod6=5: 7 sig vs dist-match | `1d/primes_deep2.py` |
| 38l | 7/14 survivors persist at ALL scales (1K through 1M) — fundamental | Lorentzian (3), Aperiodic (3), HOS:perm_entropy (1) | `1d/primes_deep2.py` |

### Number Theory
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 39a | All 8 arithmetic function encodings detected as non-random | 50-105 sig metrics | `1d/number_theory.py` |
| 39b | Even degenerate μ(n) (3 values) and λ(n) (2 values) strongly detected | μ=78, λ=71 sig metrics | `1d/number_theory.py` |
| 39c | All 28 pairwise encoding distinctions | min 75 (μ vs λ), max 118 (ζ vs d) | `1d/number_theory.py` |
| 39d | Mertens function has 37 metrics beyond random walk model | E8 diversity d=4.62, Clifford regularity d=-3.79 | `1d/number_theory.py` |
| 39e | d(n) has 85 metrics beyond distribution-matched model | Cantor max_gap d=64.7; massive sequential correlation | `1d/number_theory.py` |
| 39f | Ω(n) has 88 metrics beyond Erdős–Kac normal approximation | Heisenberg xy_spread d=-54.8 | `1d/number_theory.py` |
| 39g | Zeta zero spacings have 90 metrics beyond Wigner surmise (GUE) | Fisher jeffreys d=-24.3, Torus entropy d=22.5 | `1d/number_theory.py` |
| 39h | ALL 8 encodings are ordering-dependent | d(n) strongest (57 orig vs shuf), λ weakest (8) | `1d/number_theory.py` |
| 39i | d(n) and Ω(n) detection stable across 4 decades of scale | 99-107 sig across 1K-1M | `1d/number_theory.py` |

### Continued Fractions
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 40a | All 5 constants' CF coefficients detected as non-random | √3=100, √2=96, π=89, e=87, ln2=86 sig | `1d/continued_fractions.py` |
| 40b | **π CF coefficients indistinguishable from iid Gauss-Kuzmin** — geometry confirms "almost all" applies to π | 0 sig vs GK surrogates | `1d/continued_fractions.py` |
| 40c | ln 2 shows weak sequential signal beyond GK distribution | 11 sig vs GK | `1d/continued_fractions.py` |
| 40d | e strongly distinguishable from GK (deterministic [1,2k,1] pattern) | 65 sig vs GK | `1d/continued_fractions.py` |
| 40e | π and ln 2 have zero ordering dependence (orig vs shuffled) | π=0, ln2=0 sig | `1d/continued_fractions.py` |
| 40f | √3 ordering = 48 sig (period-2 destroyed by shuffling); √2 = 0 (constant, perfect control) | validates methodology | `1d/continued_fractions.py` |
| 40g | π vs ln 2 = only 4 sig pairwise (both look GK-like) | weakest pair among 10 | `1d/continued_fractions.py` |
| 40h | Algebraic vs transcendental CFs: 96-100 sig pairwise | √2 vs π=98, √3 vs π=100 | `1d/continued_fractions.py` |
| 40i | Khinchin's constant: π geo_mean=2.6624 ≈ K=2.6854, ln 2 geo_mean=2.6266 | both approach K | `1d/continued_fractions.py` |
| 40j | √3 delay embedding peaks at τ=2, matching its period-2 CF structure | τ=2 → 103 sig (vs 101 at τ=1) | `1d/continued_fractions.py` |

### Unsolved Problems: Goldbach's Comet
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 41a | Goldbach g(2n) massively non-random | 86 sig metrics vs random | `1d/unsolved.py` |
| 41b | Hardy-Littlewood prediction captures most but not all structure | 17 sig metrics beyond HL (E8, HOS lead) | `1d/unsolved.py` |
| 41c | g(2n) has strong sequential correlations (destroyed by shuffling) | 51 sig vs shuffled | `1d/unsolved.py` |
| 41d | g(2n) sequential structure beyond marginal distribution | 56 sig vs distribution-matched | `1d/unsolved.py` |
| 41e | HL predicts g(2n) with r=0.992 but systematic underestimate (+19 residual) | mean residual = +19.07 | `1d/unsolved.py` |
| 41f | Goldbach structure robust across scales (n=100 to n=200K) | 17-32 sig vs HL across ranges | `1d/unsolved.py` |

### Preprocessing
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 40 | Delay embedding gives 52x improvement for lag detection | d: 0.17 → 9.1 | Delay embedding study |
| 41 | FFT and raw analysis are complementary (17+20 exclusive pairs) | varies | Spectral study |
| 42 | kurt_mean (4th order) is independent from all 23 geometries | max r = 0.22 | Higher-order study |
| 43 | Permutation entropy: 33x better for recurrence detection | d = 250 vs 7.5 | Higher-order study |

### 2D Spatial Field (8 geometries, 80 metrics)
| # | Finding | Effect Size | Investigation |
|---|---------|-------------|---------------|
| 44 | Ising model: all temperatures vs random distinguished | varies | `2d/ising.py` |
| 45 | Ising: multiscale_coherence_4 peaks near T_c | scale-free structure | `2d/ising.py` |
| 46 | Reaction-diffusion: 15/15 morphology pairs | d = 97 (tension_std) | `2d/reaction_diffusion.py` |
| 47 | Percolation: 28/28 probability pairs | d = 261 (n_basins) | `2d/percolation.py` |
| 48 | Cellular automata: 14/15 rule pairs | d = 78.5 (anisotropy) | `2d/cellular_automata.py` |
| 49 | ECB penguin detected in 2D, CBC/CTR invisible | d = -283 | `2d/ecb_penguin.py` |
| 50 | Maze algorithms: 15/15 pairs | d = 108.9 | `2d/mazes.py` |
| 51 | Wave equation: 15/15 source configs | d = -496 | `2d/wave_equation.py` |
| 52 | Voronoi tessellations: 10/10 point process pairs | d = -154 | `2d/voronoi.py` |
| 53 | Growth models: 3/3 (DLA/Eden/random) | d = -178 | `2d/growth_models.py` |
| 54 | Sandpile SOC: 5/6 pairs, convergence detected | d = 261 | `2d/sandpile.py` |
| 55 | Lenia continuous CA: 15/15 configs | d = 249 | `2d/lenia.py` |
| 56 | Near-identical rules detected: GoL ≈ HighLife, Kruskal ≈ AldousBroder | d ≈ 0 | Various 2D |

## Negative Results (21)

These are equally important — they define the boundaries of what geometric analysis can and cannot do.

| # | Finding | Implication |
|---|---------|-------------|
| 1 | **AES-CTR indistinguishable from random** across ALL geometries, preprocessings, higher-order stats, and bit planes | Fundamental limit; AES works |
| 2 | MT19937 essentially passes (1 borderline detection: S² × ℝ sphere_concentration d=-1.27) | Byte-level structure is marginal |
| 3 | XorShift128 undetected raw (0 sig), but DE2 reveals 2 metrics | Thurston height_drift exposed by delay embedding |
| 4 | MINSTD nearly passes (1 sig: S² × ℝ; standard chi2+entropy catch 2) | Distributional bias caught by standard tests first |
| 4a | SFC64 and PCG64 completely indistinguishable from os.urandom | Modern PRNGs are geometrically random |
| 5 | RC4 stream looks random (even 24-bit key) | Stream cipher output is geometrically indistinguishable |
| 6 | Standard map ≈ random | Uniformly mixing on torus |
| 7 | Arnold cat map ≈ random | Uniformly mixing on torus |
| 8 | LSB steganography nearly invisible (d=1.06, 1 metric at 100% rate only) | Byte-level changes too small for most geometries |
| 8a | Matrix embedding completely invisible to all geometries (confirmed 1D and 2D) | d=0.00 across 131 metrics, 4 representations, all rates |
| 8b | Bitplane extraction does not improve stego detection | 8x sample reduction destroys statistical power |
| 9 | GBM indistinguishable from IID normal | Random walk is geometrically random |
| 10 | Ornstein-Uhlenbeck ≈ IID normal | Mean-reversion too subtle for byte encoding |
| 11 | Random vs encrypted is the hardest classification boundary | Both designed to look random |
| 12 | Multi-scale analysis doesn't improve general classification | 69.5% < 79.5% baseline |
| 13 | Cepstrum mostly useless after uint8 quantization | Information crushed |
| 14 | GARCH hard to detect after byte quantization | Variance-of-variance destroyed |
| 15 | GoL ≈ HighLife | Near-identical rules (differ only in B6) |
| 16 | Sandpile 10k ≈ 50k iterations | Both at SOC steady state |
| 17 | Kruskal ≈ Aldous-Broder maze generation | Both produce uniform spanning trees |
| 18 | **π CF coefficients indistinguishable from iid Gauss-Kuzmin** (0 sig, 0 ordering) | Geometry confirms "almost all" applies to π |
| 19 | **π digits indistinguishable from random in base-256 AND base-10** (0 sig vs random, 0 vs shuffled, 0 across positions 0K-48K, 0 under delay embedding τ=1-5) | Strong geometric evidence for normality of π |
| 19a | **e and √2 digits also indistinguishable from random** (same battery: all 0 sig) | Evidence extends to e and √2 |

## Key Takeaway

The framework detects genuine structure with large effect sizes (d = 7-266) while producing zero false positives on validated random sources. The AES-CTR negative result confirms that the methodology is honest — geometries report "no structure" when encryption is working correctly.

The 2D spatial geometry battery (8 geometries, 80 metrics) demonstrates that genuinely different mathematical lenses — differential geometry (Surface), algebraic topology (PersistentHomology2D), complex analysis (Conformal2D), integral geometry (MinkowskiFunctional), scaling analysis (MultiscaleFractal), Hodge theory (HodgeLaplacian), and spectral analysis (SpectralPower) — each contribute unique discriminative power. On stego co-occurrence matrices, PVD detection jumps from 13/15 (SpatialField alone) to 49/80 (all 8 geometries). On 10 diverse field types, all 45 pairs are distinguished.

The deep Collatz investigations (collatz_deep, collatz_deep2) demonstrate a particularly striking application: five specific geometry families (Fisher, Heisenberg, Sol, Spherical, Wasserstein) detect convergence-specific structure in 3n+1 that categorically vanishes in divergent variants. The convergence mechanism has a geometric character — information-geometric, nilpotent, solvable — that is absent, not merely attenuated, in divergent maps.

The deep prime gap investigations (`primes_deep.py`, `primes_deep2.py`) trace the sequential structure of prime gaps to its roots. Of 131 metrics, 14 survive distribution-matching — detecting ordering that no marginal model explains. These 14 decompose into 57% linear (autocorrelation) and 43% nonlinear (IAAFT-surviving) components. Prime gaps are temporally irreversible (4 sig metrics forward vs reversed), consistent with the sieve's inherent directionality. Residue classes mod 6 are massively geometrically distinct (46 sig), with both classes retaining independent sequential structure — the Lemke Oliver–Soundararajan bias made geometric. Seven of the 14 survivors persist from primes near 1K through 1M, establishing them as fundamental features of prime distribution rather than small-prime artifacts.

The number theory investigations reveal that classical limit theorems leave substantial geometric structure unexplained. The Mertens function M(n) — whose random-walk behavior is equivalent to RH — has 37 metrics beyond what a random walk with matching step probabilities can produce. Zeta zero spacings differ from the GUE/Wigner prediction in 90 metrics. Even the divisor function d(n) has 85 metrics of sequential correlation destroyed by shuffling. These results suggest that exotic geometries detect multiplicative number-theoretic structure that standard probabilistic models do not capture.

The continued fractions investigation provides a striking validation of Khinchin's theorem: π's CF coefficients are geometrically indistinguishable from iid Gauss-Kuzmin samples (0 sig, 0 ordering dependence), confirming that "almost all" applies to π as far as 131 exotic geometric metrics can detect. ln 2 shows a faint crack (11 sig vs GK), while algebraic constants are trivially distinguishable. The √2 control (constant sequence, 0 ordering sig) and random self-check (0 sig) validate the methodology.

The RNG quality testing investigation (`rng.py`) provides a clean validation story: 10 generators spanning a quality gradient from cryptographic to historically broken. CRYPTO/GOOD generators return 0 significant metrics (self-check urandom vs urandom also 0), while RANDU (44 sig) and Middle-Square (78 sig) are massively detected. Geometric metrics outperform standard statistical tests by 11x on the worst generators. Delay embedding newly reveals XorShift128 (undetected raw), and RANDU is detectable from just 500 bytes. Different weaknesses have distinct geometric fingerprints — Penrose quasicrystal metrics detect lattice structure (d=-37), while Higher-Order Statistics catches nonlinear correlations.

The unsolved problems investigation (`unsolved.py`) addresses two famous open questions. For normality of π: digits of π, e, and √2 are completely indistinguishable from random bytes across 131 geometric metrics, in both base-256 and base-10, at four different digit positions (0K-48K), and under delay embedding at τ=1-5. Combined with the CF result (π passes iid Gauss-Kuzmin), this is comprehensive geometric evidence for normality. For Goldbach's comet: g(2n) is massively structured (86 sig vs random), with strong sequential correlations (51 sig vs shuffled). The Hardy-Littlewood prediction captures most structure (r=0.992) but 17 metrics detect patterns beyond HL — primarily via E8 Lattice and Higher-Order Statistics. This gap persists across scales from n=100 to n=200K.
