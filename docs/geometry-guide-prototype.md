# Geometry Guide — Prototype

Content for all 8 encoding-invariant geometries (Tier 1) and 8 high-discrimination geometries (Tier 2). Each entry explains what the geometry detects through contrast and atlas examples, not mathematical definitions.

---

## Boltzmann

**What it measures:** Whether nearby values cooperate or compete.

Treats each window of your data as a miniature spin glass — a system where local constraints may be impossible to satisfy simultaneously. Fits the maximum-entropy pairwise interaction model and extracts three properties of the resulting energy landscape.

### Metrics

**coupling_strength** — How strongly do positions in a window constrain each other? Monotonic signals have extreme coupling (each value predicts the next). Noise has near-zero coupling. Periodic orbits of the logistic map dominate the atlas: the period-3 window (r=3.83) scores 15.3, while white noise scores 0.01.

**frustration** — What fraction of local constraints are mutually contradictory? Three positions A, B, C are frustrated when A wants B up, B wants C up, but A wants C down — they can't all be satisfied. The Fibonacci word scores 1.0 (maximally frustrated): the golden ratio creates irreconcilable correlations at every scale. Pink noise scores 0.0 (all correlations agree). White noise scores ~0.5 (random constraints, half frustrated by chance). Quantum walk probability amplitudes also hit 1.0 — a different mechanism (interference) producing the same geometric signature.

**spectral_gap_J** — How degenerate is the coupling spectrum? Near 0 means one dominant interaction mode (hierarchical structure). Near 1 means all modes are equally important. L-System (Dragon Curve) and Rule 110 score ~1.0; logistic period-4 scores 0.14.

### When it lights up

Boltzmann frustration is the framework's most specific quasicrystal detector. In the atlas, Fibonacci word is the only non-quantum source to achieve frustration = 1.0. The metric is encoding-invariant: it depends only on which positions are above or below the median, not on the byte values themselves.

---

## Hölder Regularity

**What it measures:** How rough or smooth the signal is at each point — and how much that roughness varies.

If you zoom into a smooth signal, it looks linear. If you zoom into a rough signal, it stays jagged no matter how far you zoom. The Holder exponent quantifies this: high = smooth, low = rough. This geometry computes it at every point and asks: is the roughness uniform (monofractal) or does it vary wildly from place to place (multifractal)?

### Metrics

**hurst_exponent** — The global roughness summary. Sine waves score 0.92 (very smooth, persistent). White noise scores ~0 (uncorrelated). Logistic maps in the period-doubling cascade score -2.0 (actively anti-persistent — each value fights the previous one). In the seismic P-wave investigation, earthquake arrivals scored dramatically lower than ambient noise (d = 9.06): P-waves are impulsive, ambient microseisms are smooth.

**holder_mean** — Average local regularity. Wigner semicircle (0.99) and triangle wave (0.98) are the smoothest signals in the atlas. Logistic period-2 (-3.2) is the roughest — it alternates between two values with no interpolation.

**holder_std** — How much does roughness vary? English literature scores highest (1.87): some passages are smooth (common words), others are jagged (rare words, punctuation). Fibonacci word scores 0.0 (perfectly uniform roughness at every point — it's monofractal).

**multifractal_width** — The range of the singularity spectrum. Wide = the signal has both very smooth and very rough regions. Financial returns, turbulence, and English text have wide spectra. Pure tones and constants have zero width.

### When it lights up

Holder Regularity was the #2 discriminator in the seismic P-wave investigation (Cohen's d = 9.06), detecting earthquake arrivals through the collapse of local smoothness. It separates the Distributional view's C1 (smooth oscillators) from C4 (anti-persistent chaos) along the persistence axis — the biggest gap in ordinal space.

---

## Recurrence Quantification

**What it measures:** Does the signal ever return to places it's been before — and when it does, does it follow the same path?

Embeds the signal in 3D via delay coordinates, then asks: for every pair of points in the trajectory, are they close? The answers form a binary recurrence matrix whose texture reveals the dynamics.

### Metrics

**determinism** — Of the recurrent points, what fraction form diagonal lines in the recurrence matrix? Diagonal lines mean: when the signal returns to a previous state, it follows the same trajectory it followed last time. All logistic periodic orbits score 1.0 (perfectly deterministic — they retrace their paths exactly). White noise scores 0.71 (some diagonal structure by chance). Constants score 0.0 (everything is recurrent, but there's no trajectory to follow).

**laminarity** — What fraction form vertical lines? Vertical lines mean: the signal gets stuck near one state for extended periods. Thue-Morse, Rule 30, and Symbolic Henon all score 1.0. Logistic period-4 scores 0.0 — it recurs but never lingers.

**trapping_time** — When the signal gets stuck, how long does it stay? Thue-Morse and Rule 30 score 249.5 (maximal trapping — the symbolic dynamics creates long laminar stretches). Exponential chirp scores 0 (never stays anywhere).

**entropy_diagonal** — How varied are the diagonal line lengths? Thue-Morse and Rule 30 maximize this at 6.2 bits: they revisit their trajectory at many different timescales. Simple periodic orbits have low entropy (one dominant recurrence period).

### When it lights up

Recurrence Quantification drives the gap between the ordinal view's C3 (symbolic dynamics: high recurrence, high laminarity) and C2 (smooth correlated noise: low recurrence). In the solar eclipse VLF investigation, laminarity and determinism were significant only during the eclipse-active interval — the D-layer collapse changed the signal's recurrence structure in real time.

---

## Ordinal Partition

**What it measures:** The grammar of ups and downs.

Ignores the actual values entirely and looks only at their rank order. In a window of 5 consecutive values, there are 120 possible orderings (permutations). Which orderings appear? Which transitions between orderings are allowed? This is the most purely encoding-invariant geometry in the framework.

### Metrics

**transition_entropy** — How predictable is the next rank pattern given the current one? Wind speed, BTC range, and Poisson spacings max out at 1.0 (any pattern can follow any other). Logistic period-4 scores 0.0 (each pattern has exactly one successor). Chaos lives in between: logistic r=3.9 scores 0.82.

**forbidden_transitions** — What fraction of theoretically possible pattern transitions never occur? Square wave and Morse code score 0.88 (only 12% of transitions are allowed — the signal's grammar is highly constrained). White noise and prime gaps score 0.0 (all transitions observed). Deterministic chaos forbids specific transitions that noise allows — this is a zero-training chaos detector.

**time_irreversibility** — Does the signal look different played backwards? Phyllotaxis scores 1.0 (strongly irreversible — the golden-angle rotation has a definite direction). Stern-Brocot walk scores 0.008 (nearly reversible). Financial returns, being famously irreversible (crashes are fast, recoveries are slow), score 0.3-0.5.

**statistical_complexity** — The sweet spot between order and disorder. Peaks at the edge of chaos. L-System Dragon Curve (0.104) and logistic r=3.9 (0.102) are the most complex signals in the atlas. Both constants and perfect noise score 0.0 — neither is complex, for opposite reasons.

**memory_order** — Does the signal have hidden higher-order dependencies? Stern-Brocot walk scores highest (1.08): its next value depends on the last two values, not just the last one. This detects Markov order that transition_entropy misses.

### When it lights up

Ordinal Partition's forbidden_transitions is the primary separator between deterministic and stochastic sources in the atlas. Combined with transition_entropy, it places every signal on a 2D map from "fully constrained" (periodic) through "partially constrained" (chaos) to "unconstrained" (noise) — without any training data or parameter tuning.

---

## Heisenberg (Nil) (centered)

**What it measures:** How much consecutive values twist around each other.

Lifts pairs of successive values into 3D Heisenberg group coordinates, where the z-axis accumulates the signed area swept by the (x, y) path. Correlated data twists the path into a helix; uncorrelated data stays flat. The centered version subtracts the mean first, so it measures correlation, not bias.

### Metrics

**final_z** — Total accumulated twist after traversing the entire signal. Logistic period-2 scores 33.5 million — each alternation adds massive twist in the same direction. L-System Dragon Curve scores 1.0 (the path barely twists because the symbolic dynamics has no consistent correlation direction). The 7-order-of-magnitude range makes this the framework's most dynamic metric.

**xy_spread** — Standard deviation of the path in the x-y plane. Measures how widely the signal explores its amplitude space. Logistic period-2 again dominates (4,730): the two alternating values create maximum spread.

### When it lights up

Heisenberg twist is mathematically identical to the lag-1 autocorrelation — but computed through group multiplication rather than arithmetic. Its value is structural: it connects correlation detection to the geometry of 3-manifolds (Nil geometry is one of Thurston's eight). In the ordinal view, the Heisenberg axis (PC3) separates sources by their correlation polarity: strongly anti-correlated chaos at one extreme, positively correlated drift at the other.

---

## Spectral Analysis

**What it measures:** Where the energy lives in frequency space.

Computes the power spectral density via FFT and characterizes its shape. Not just "what's the dominant frequency" but "is the spectrum flat (noise), peaked (periodic), or power-law (fractal)?"

### Metrics

**spectral_slope** — The exponent in P(f) ~ f^beta. Brown noise = -2, pink noise = -1, white noise = 0. Positive slopes (blue spectra) are rare in nature — Thue-Morse (+1.75) and Fibonacci word (+1.63) are the bluest signals in the atlas, their substitution structure concentrating energy at high frequencies. Kilauea tremor (-3.33) and Lotka-Volterra (-3.14) are the reddest: slow dynamics dominate.

**spectral_r2** — How well does a single power law fit the spectrum? Anti-persistent fBm scores 0.99 (textbook power law). Logistic period-2 scores 0.0 (all energy at one frequency, not a power law at all). This distinguishes genuine 1/f processes from signals that happen to have similar average slope.

**spectral_entropy** — Shannon entropy of the normalized spectrum. RANDU and dice rolls score 0.95 (nearly flat — energy spread uniformly). Logistic period-2 scores 0.0 (all energy at one frequency). Higher entropy = more frequencies contributing = broader bandwidth signal.

**spectral_flatness** — Wiener entropy: ratio of geometric to arithmetic mean of the spectrum. 1.0 = perfectly flat (white noise). 0.0 = all power at one frequency (pure tone). Rule 30 scores 0.57 — its pseudorandom output is flatter than most chaos but not as flat as true randomness.

**peak_frequency** — Where is the maximum power? Stern-Brocot walk peaks at the Nyquist frequency (0.5). Most natural signals peak near DC (0.0). Values near 0.5 indicate the signal's dominant variation is at the shortest timescale — rapid alternation or anti-persistence.

### When it lights up

Spectral slope is the strongest separator between the ordinal view's C1 (red-spectrum oscillators, slope -1.9) and C4 (blue-spectrum chaos, slope +0.3). In the seismic P-wave investigation, spectral metrics were among the top discriminators: earthquake P-waves flatten the ambient spectrum by injecting broadband impulsive energy.

---

## D4 Triality

**What it measures:** Whether 4-byte structural patterns respect the Spin(8) triality symmetry.

Takes each group of 4 consecutive values and projects them onto the 24 roots of D4 — the vectors ±eᵢ ± eⱼ in 4D space. D4 is the only Lie algebra with triality: an order-3 automorphism that cyclically permutes three fundamentally different representations of Spin(8). The geometry asks: does the data's root usage look the same after applying this triality rotation? If so, the data has a structural symmetry that only D4 can detect.

### Metrics

**triplet_temporal** — Temporal coherence of root assignment across overlapping 4-byte windows. Measures whether the root trajectory has predictable structure rather than random jumps. Higher for signals with smooth dynamics.

**neighborhood_asymmetry** — Asymmetry between the 16 edge-neighbors per root in the D4 graph. Measures whether transitions favor certain neighbor directions over others. High for signals with directional bias in 4D root space.

**structural_coherence** — Combined structural regularity of root usage patterns. Captures how "crystalline" the data's 4-byte structure is in D4 coordinates.

**spectral_transition** — Projects root-to-root transition dynamics onto D4's edge-adjacency eigenspaces {-8, 0, 16}, returns weighted (E₁₆ - E₋₈) / E_total. Thomson control points have no edges, so this eigenspace decomposition doesn't exist — giving strong D1 drop. *Evolved via ShinkaEvolve atlas v1.*

**diversity_ratio** — Fraction of the 24 roots actually used. High diversity means the data explores the full D4 geometry; low diversity means it's confined to a subspace.

**normalized_entropy** — Shannon entropy of root usage, normalized by maximum. Uniform root distributions score high; periodic orbits score 0.

### When it lights up

D4 triality is the only Lie algebra with an order-3 automorphism cyclically permuting three representations of Spin(8). The spectral_transition metric exploits D4's unique edge-adjacency spectrum: the eigenvalues {-8, 0, 16} are algebraic properties of the 24-root graph that no generic point arrangement shares. D4's 4-byte window size complements G2 (pairs) and E8 (octets), catching intermediate-scale correlations with crystallographic structure.

---

## Inflation (Substitution)

**What it measures:** Whether the data was generated by a substitution rule — a process that replaces symbols with fixed patterns at every scale.

A substitution rule maps each symbol to a word: Fibonacci does a→ab, b→a; Thue-Morse does a→ab, b→ba. Iterating the rule produces an infinite aperiodic sequence whose statistical properties repeat at geometrically spaced scales (powers of the inflation factor λ). This geometry binarizes the data and tests five signatures that substitution sequences must satisfy and random or periodic sequences cannot.

### Metrics

**complexity_linearity** — How linear is the growth of distinct subwords? For a substitution sequence, the number of distinct n-grams p(n) grows as c·n + d (exactly linear). For random sequences, p(n) grows exponentially (2ⁿ). For periodic sequences, p(n) flatlines. Fibonacci word and Thue-Morse both score near 1.0 (textbook linear). White noise scores 0.76 (exponential growth, poor linear fit). Logistic period-5 scores 0.46 — its near-periodicity confuses the linearity test.

**entropy_rate** — Topological entropy: does the subword count grow? Zero for substitution and periodic, positive for random and chaos. White noise and PRNG outputs score ~1.0 (exponential growth = maximal entropy). Fibonacci word scores 0.19 (barely above zero — its linear complexity growth produces a small but nonzero growth ratio). Periodic orbits score exactly 0.0.

**discrepancy** — How evenly are the symbols distributed? Measures max|D(n)|/√N, where D(n) is the deviation of cumulative symbol counts from expected. Substitution sequences have bounded discrepancy (D*/√N → 0). Random sequences have D*/√N ≈ 0.5 (random walk). Thue-Morse (0.006) and Fibonacci word (0.006) have the lowest nonzero discrepancy in the atlas — their symbols are distributed with almost crystalline uniformity. Devil's staircase (28.1) has the highest: its long constant plateaus create massive cumulative imbalance.

**return_concentration** — How regular are the gaps between repeated subwords? Substitution sequences have few distinct return times because the hierarchical tiling constrains where each pattern can appear. Logistic period-2 and edge-of-chaos score 1.0 (perfectly regular returns). Fibonacci word scores 0.82. fBm scores 0.08 (widely scattered, irregular returns).

**acf_geometric** — Are autocorrelation peaks at geometric (λ^k) rather than arithmetic (nT) spacings? This is the substitution fingerprint: Fibonacci has ACF peaks at Fibonacci numbers (φ^k spacings), Thue-Morse at powers of 2. Quantum walk scores highest (0.91) — its interference pattern creates geometric-ratio ACF peaks through a completely different mechanism. Fibonacci word (0.88), phyllotaxis (0.89), and circle map quasiperiodic (0.89) cluster together. White noise and logistic chaos score 0.0 (no ACF peaks at all).

### When it lights up

Inflation is the framework's substitution-rule detector. The combination of linear complexity + near-zero discrepancy + geometric ACF peaks is essentially unique to substitution sequences. In the ordinal view, it separates quasicrystalline sources (Fibonacci, Thue-Morse, L-System) from both periodic and random, which the other ordinal geometries struggle to do — Ordinal Partition sees Fibonacci as "moderately constrained" without distinguishing it from other aperiodic sequences. Inflation's acf_geometric metric also catches quantum walk and ocean/wave signals, suggesting that geometric self-similarity in autocorrelation structure is a broader phenomenon than pure substitution dynamics.

---

## Tier 2 — High-Discrimination Geometries

---

## Multifractal Spectrum

**What it measures:** Whether the signal's roughness is uniform or varies from place to place.

A monofractal signal (like fractional Brownian motion) has the same scaling behavior everywhere — zoom in anywhere and you see the same texture. A multifractal signal (like turbulence or financial returns) has smooth regions and rough regions interspersed. This geometry computes the singularity spectrum f(α) via structure functions and measures how wide it is.

### Metrics

**spectrum_width** — The range of local scaling exponents (α_max − α_min). Zero means monofractal: every point has the same roughness. Seismic noise (ANMO) dominates the atlas at 0.78 — earthquake background has wildly varying local regularity from microseisms to cultural noise. GOES X-ray flux (0.49) is next: solar flares inject bursts of irregular scaling into a smooth background. Sawtooth wave (0.45) is surprisingly wide because the discontinuous resets create a sharp transition between smooth ramps and singular jumps. Constants score 0.0 (no scaling structure at all).

**hurst_estimate** — The self-affinity exponent H = τ(2)/2. Sine wave scores 0.90 (strongly persistent, smooth). Chua's circuit (0.86) is also highly persistent — its double-scroll attractor stays in each lobe for extended stretches. Prime gaps and divisor count score 0.0: number-theoretic sequences have no self-affine scaling.

### When it lights up

Multifractal width is the primary separator between "simple" dynamics (periodic, noise, low-dimensional chaos) and "complex" dynamics (turbulence, geophysical processes, financial markets). In the seismic P-wave investigation, multifractal metrics contributed to the top discriminators — earthquake arrivals temporarily broaden the local spectrum width by injecting a new scaling regime into the ambient noise.

---

## Predictability

**What it measures:** How much does knowing the past help predict the future?

Computes conditional entropy H(X_t | X_{t-1}, ..., X_{t-k}) at increasing depths k = 1, 2, 4, 8. If the conditional entropy drops as you add more history, the signal has memory — past values constrain the future. Also includes sample entropy (SampEn), a phase-space regularity measure.

### Metrics

**excess_predictability** — Total information gain from knowing the past 8 values: H(X) - H(X | past_8). De Bruijn sequence scores 3.0 (maximally predictable — it's constructed so every 8-bit pattern appears exactly once, making the next bit deterministic given the last 7). Hilbert walk (2.90) and sawtooth (2.89) are nearly as predictable. White noise scores 0.0 (the past tells you nothing).

**sample_entropy** — Regularity of the phase-space trajectory. Low = self-similar, predictable. High = complex, unpredictable. Pi digits (2.22) and BSL residues (2.21) are the most irregular signals in the atlas — close to the theoretical maximum for byte-valued data. Devil's staircase scores 0.019 (nearly zero — its long constant plateaus create trivially self-similar trajectories). Constants score exactly 0.0.

**entropy_decay_rate** — How fast does conditional entropy decrease with depth? Baker map has the steepest positive slope (0.26): it reveals more structure at each depth. De Bruijn has the steepest negative slope (-0.31): it becomes maximally predictable at depth 7 and the entropy collapses. Near zero means either unpredictable at all depths (noise) or already fully predicted at depth 1 (simple periodic).

**cond_entropy_k1 / cond_entropy_k8** — Conditional entropy at depths 1 and 8. White noise scores ~3.0 at both (8 bits of uncertainty, knowing the past doesn't help). Logistic period-3 scores 0.0 at both (past fully determines future). The gap between k1 and k8 reveals hidden long-range dependencies — PRNG outputs score identically at k1 and k8 (memoryless), while the baker map drops from 2.8 to 2.3 (its 2D structure creates long-range predictability invisible at lag 1).

### When it lights up

Predictability's sample_entropy and entropy_decay_rate were key discriminators in the negative re-evaluation study (2026-03-05). Standard map (44 significant metrics), Arnold cat (19), and GARCH (29) were reclassified from negative to positive detections largely on Predictability metrics. Sample entropy distinguishes deterministic chaos (moderate, ~1.5) from true noise (high, ~2.2) and periodicity (low, <0.5).

---

## Information Theory

**What it measures:** How compressible is the signal, and where does the redundancy come from?

Shannon entropy at multiple block sizes, compression ratio via Lempel-Ziv (zlib), and mutual information at lags 1 and 8. Together these measure the signal's intrinsic complexity from three complementary angles: distributional (are byte patterns uniform?), algorithmic (can a compressor find structure?), and temporal (does the past predict the future?).

### Metrics

**compression_ratio** — Lempel-Ziv compression ratio: compressed_size / original_size. 1.0 means incompressible (Wichmann-Hill, MINSTD, XorShift32 — good PRNGs are incompressible). 0.0 means trivially compressible (constants). Logistic period-2 scores 0.002 (alternating between two values compresses almost completely). This is a direct proxy for Kolmogorov complexity.

**mutual_info_1** — Mutual information between consecutive bytes: how much does byte t tell you about byte t+1? Logistic periodic orbits score 1.0 (each byte perfectly predicts the next). L-System Dragon Curve scores 0.0 (its symbolic dynamics are unpredictable one step ahead despite being deterministic). This separates "locally predictable" from "locally random" deterministic systems.

**mutual_info_8** — Mutual information at lag 8. Rule 30 scores 0.0 (no 8-step memory), while logistic period-2 still scores 1.0 (period divides 8). The comparison between lag-1 and lag-8 mutual information reveals the memory timescale: signals where MI_8 ≈ MI_1 have long memory; signals where MI_8 ≪ MI_1 have short memory.

**excess_entropy** — The total shared information between past and future. Hilbert walk (0.95) and sawtooth (0.95) maximize this: their deterministic structure creates maximum past-future coupling. Random steps (0.94) are high too — the random-walk integration creates long-range correlations even from IID increments.

**block_entropy_2 / block_entropy_4** — Shannon entropy of byte pairs and 4-grams, normalized. PRNGs and white noise score ~1.0 (all patterns equally likely). Constants and periodic orbits score 0.0. The drop from block_entropy_2 to block_entropy_4 measures how much additional structure emerges at longer pattern lengths.

### When it lights up

Information Theory metrics are the framework's workhorse for separating noise from structure. Compression ratio alone separates PRNGs (incompressible) from all other sources. Mutual information at multiple lags provides the temporal skeleton that static entropy measures miss. In the atlas, Information Theory drives the distributional view's separation between C1 (compressible, high MI: oscillators and periodic chaos) and C5 (incompressible, zero MI: noise and PRNGs).

---

## Cayley

**What it measures:** The large-scale shape of the signal's state space.

Delay-embeds the signal into 5D, builds a k-nearest-neighbor graph on the resulting point cloud, and measures three properties of the graph that are invariants of the "Cayley graph" of the underlying dynamics: how curved it is (hyperbolicity), how fast it grows (dimension), and how connected it is (spectral gap).

### Metrics

**growth_exponent** — How does the number of points within graph distance r scale? β ≈ 1 means the dynamics live on a curve (1D). β ≈ 2 means area-filling (2D manifold). β > 2 means volume-filling or branching. BTC returns score 2.07 (the highest in the atlas — financial returns fill a 2D manifold in delay space). Lotka-Volterra scores 0.56 (its limit cycles live on a 1D curve). DNA scores moderately (1.5-1.7), consistent with a branching structure in sequence space.

**delta_hyp_norm** — Gromov δ-hyperbolicity, normalized by diameter. δ ≈ 0 means the graph is tree-like (hyperbolic, negative curvature). δ/diam ≈ 0.25 means flat (Euclidean). Logistic chaos (0.26) and DNA (0.26) are the most "flat" — their delay embeddings fill space uniformly, without tree-like branching. Periodic orbits score 0.0 (trivially tree-like — just a cycle).

**spectral_gap** — Fiedler eigenvalue of the graph Laplacian: how well-connected is the state space? Large gap means rapid mixing (the dynamics explore the full space quickly). BTC returns (0.035) and neural net dense (0.029) have the largest gaps — both are highly mixing processes. Symbolic Henon scores 0.0 (its state space has disconnected components in the graph).

### When it lights up

Cayley's spectral_gap was a key discriminator in the negative re-evaluation study: it helped reclassify Arnold cat map and GARCH as positive detections. Growth exponent provides an intrinsic dimension estimate that complements the extrinsic dimension from fractal geometries. In the atlas, Cayley separates the distributional view's C3 (low growth, tree-like: symbolic dynamics and periodic orbits) from C2 (high growth, flat: continuous chaos and noise).

---

## Tropical

**What it measures:** The piecewise-linear skeleton of the signal.

Tropical geometry — named for the Brazilian mathematician Imre Simon — replaces addition with max and multiplication with addition, turning smooth curves into piecewise-linear ones. This geometry detects how many distinct linear regimes the signal passes through, how diverse their slopes are, and how much area the signal sweeps above its running minimum envelope.

### Metrics

**slope_changes** — How many times does the signal change its linear regime? Stern-Brocot walk, logistic edge-of-chaos, and Thue-Morse all hit 16,382 (maximum for the sample size — essentially every point is a slope change). Rössler attractor and constants score 0 (smooth or flat). This metric separates "jagged" dynamics (symbolic, chaotic, quasicrystalline) from "smooth" dynamics (oscillators, attractors, drifts).

**unique_slopes** — How many distinct slope values appear? Arnold cat map and beta noise reach 21 (maximum diversity — the signal visits every possible local gradient). fBm scores 1.0 (its smooth increments have one dominant slope). This measures the "vocabulary" of the signal's piecewise-linear representation.

**envelope_area** — Area between the signal and its running minimum. Forest fire (15,966) dominates: its bursty avalanche dynamics create tall, wide excursions above the baseline. Stern-Brocot walk (11,609) is next — its fractal staircase accumulates massive area. Rainfall (32) is among the lowest nonzero scores: precipitation is close to its own minimum most of the time (many dry hours).

### When it lights up

Tropical geometry provides a "complexity fingerprint" that's independent of the signal's amplitude distribution. Two signals with identical histograms can have completely different tropical profiles if one is smooth and the other is jagged. In the atlas, slope_changes is tightly correlated with ordinal transition_entropy (both measure local variability), but envelope_area captures a global property (excursion magnitude) that no ordinal metric touches. Tropical is most useful in the symmetry view, where it separates bursty processes (forest fire, Stern-Brocot) from smooth oscillators (Lorenz, Rössler) along the envelope axis.

---

## Zariski

**What it measures:** Whether the signal lives on an algebraic variety — a surface defined by polynomial equations.

Delay-embeds the signal and tests whether the resulting point cloud lies near the zero set of a low-degree polynomial. In algebraic geometry, the Zariski topology's closed sets are exactly these zero sets, making the topology non-Hausdorff (most pairs of points can't be separated by open sets). This geometry also measures the signal's pattern lattice: does it obey Boolean logic (classical), or does it deviate toward a Heyting algebra (intuitionistic)?

### Metrics

**heyting_gap** — How far is the signal's pattern lattice from Boolean? 1.0 means maximally non-Boolean: the law of excluded middle fails for many pattern pairs. Forest fire scores 1.0 — its intermittent dynamics create pattern relationships that violate classical logic (a pattern can be "neither always present nor always absent" in a meaningful sense). Logistic period-3 (0.90) and period-5 (0.78) are high too: their windows of periodicity within chaos create ambiguous pattern membership. Logistic full chaos and Rössler score 0.0 — fully chaotic systems are "Boolean" because every pattern either appears or doesn't, with no intermediate states.

**nonsep_fraction** — What fraction of point pairs are non-separable in the Zariski topology? Collatz gap lengths score 0.9995 (almost all pairs are non-separable — the data lies almost exactly on an algebraic variety). Rainfall scores 0.96 (its exponential-like distribution creates a low-dimensional algebraic structure). Wichmann-Hill (0.007) is nearly zero — good PRNGs fill delay space uniformly, avoiding any algebraic surface.

**algebraic_residual** — How well does a low-degree polynomial fit the delay-embedded cloud? Dice rolls (0.064) have the highest residual — the data is maximally far from any algebraic variety. Constants score 0.0 (trivially on a variety: the point {(c,c,c,...)}). This is the "anti-algebraic" metric: high values mean the signal has no polynomial recurrence relation.

### When it lights up

Zariski's heyting_gap (CV=1.76) is the framework's most variable metric across sources. It detects a specific structural property — intermittency — that other geometries approximate but don't directly measure. The forest fire / logistic period-3 cluster at the top of the heyting_gap ranking corresponds to "intermittent chaos": systems that switch between qualitatively different dynamical regimes. In investigations, heyting_gap discriminated between normal and abnormal ECG with moderate effect size, because cardiac arrhythmias are fundamentally intermittent.

---

## Julia Set

**What it measures:** How the signal's values behave as starting conditions for a complex dynamical system.

Maps pairs of consecutive data values to complex starting positions z₀ in the plane, then iterates z → z² + c (with a fixed parameter c). Some starting conditions escape to infinity; others are trapped in bounded orbits. The fractal boundary between escaping and trapped regions is the Julia set. The data's distribution of escape times and orbit stabilities reveals its "dynamical texture" as seen through this particular lens.

### Metrics

**escape_entropy** — Shannon entropy of the escape-time distribution. High entropy means the data samples many different escape times — the starting conditions are spread across different dynamical basins. Coupled map lattice (4.45) scores highest: its spatiotemporal chaos creates a rich diversity of starting conditions. Gaussian noise (4.39) is close behind. Logistic period-2 and Collatz parity score 0.0 — their starting conditions are degenerate (only 1-2 distinct z₀ values), so there's only one escape time.

**stability** — Mean absolute value of the final iterate for non-escaping orbits. Devil's staircase (938) has the highest stability: its starting conditions land in far-flung bounded orbits that stay large without escaping. Berry random wave (911) and seismograph (911) are similar — their broad amplitude distributions create starting conditions near the boundary of the Julia set, where orbits are large but trapped. Constants score 0.0 (degenerate single-point orbit).

### When it lights up

Julia Set geometry acts as a "nonlinear filter" — it transforms the data through a fixed dynamical system and asks what the output looks like. This is conceptually different from geometries that analyze the data's own dynamics. It's most useful for separating signals with similar linear statistics but different amplitude distributions: two signals with the same mean and variance can have very different Julia escape profiles if one has heavy tails (creating starting conditions near the Julia set boundary) and the other is Gaussian (starting conditions far from the boundary).

---

## Lorentzian

**What it measures:** How much of the signal's structure is causal — does the future follow from the past in a timelike way, or do successive values jump acausally?

Embeds consecutive (time, value) pairs as events in 1+1 Minkowski spacetime with metric ds² = −dt² + dx². For each pair of events, the Minkowski interval classifies the separation: timelike (|Δx| < |Δt|, causally connected), spacelike (|Δx| > |Δt|, causally disconnected), or lightlike (|Δx| = |Δt|, on the light cone).

### Metrics

**causal_order_preserved** — Fraction of consecutive event pairs that are timelike-separated (causally ordered). Smooth, slow-varying signals score near 1.0 (each value follows causally from the previous one). Logistic edge-of-chaos scores 0.0 (every successive value is an acausal jump). This is the framework's most direct "smoothness vs. jumpiness" metric.

**spacelike_fraction** — Fraction of sampled pairs (at log-spaced separations) that are acausally separated. High for chaotic maps that make large jumps between values; zero for continuous-time attractors (Lorenz, Rössler) whose smooth trajectories keep consecutive samples causally close.

**crossing_density** — How often consecutive steps switch between timelike (subluminal) and spacelike (superluminal). Measures burstiness relative to the lightcone boundary c=1. Rule 30 scores ~0.5 (maximally bursty, random-looking transitions), Collatz Parity ~0.32 (structured runs of same causal character), smooth signals score 0.0 (always subluminal). *Evolved via ShinkaEvolve atlas v1.*

**causal_persistence** — Lag-1 autocorrelation of the timelike/spacelike binary sequence, scaled to [0,1]. High values indicate long runs of the same causal character; 0.5 indicates random alternation. Collatz Parity scores ~0.62 (correlated causal runs), Rule 30 ~0.49 (uncorrelated). Degenerate when causal_order is near 0 or 1. *Evolved via ShinkaEvolve atlas v1.*

### When it lights up

Lorentzian geometry captures the "speed" of the signal relative to its sampling rate. A high-frequency oscillation sampled slowly appears spacelike (large jumps); the same oscillation sampled fast appears timelike (smooth evolution). The evolved metrics (crossing_density and causal_persistence) add temporal dynamics to the static causal fraction: two signals with the same causal_order can differ in whether their lightcone crossings are bursty (low persistence, high crossing) or structured (high persistence, moderate crossing). In the atlas, these four metrics separate smooth oscillators from chaotic/noise sources along a "causality axis" orthogonal to the entropy axis.

---

## Tier 3 — Remaining Geometries

## Torus T^2

**What it measures:** How thoroughly the signal fills its available space.

Maps each pair of consecutive values to a point on a flat torus — a square where the top edge is glued to the bottom and the left edge to the right, so there are no boundaries. Bins the torus into a 16x16 grid and asks: how many cells are occupied, how uniform is the distribution, and how close are neighboring points?

### Metrics

**coverage** — What fraction of the 256 torus cells contain at least one point? Arnold cat map, MINSTD, and XorShift32 all hit 1.0 (perfect coverage — every cell occupied). Constant signals score 0.0 (one cell forever). Logistic period-2 scores 0.004 (two values produce one point, landing in a single cell). Coverage is the simplest possible test of distributional richness: does the data ever go everywhere?

**entropy** — Shannon entropy of the torus cell histogram. Wichmann-Hill, XorShift32, and AES encrypted cluster at 7.98 (near the maximum of 8.0 bits — uniform occupancy across all 256 cells). Constants and logistic period-2 score 0.0. High coverage with low entropy means the data visits many cells but spends most of its time in a few — the signature of a non-uniform but space-filling process.

**toroidal_mean_nn_distance** — Average distance from each point to its nearest neighbor, measured with wrap-around (torus metric, not Euclidean). Pi digits (0.023), Wichmann-Hill (0.022), and white noise (0.022) cluster together — well-spread points on the torus. Logistic period-4 scores 0.0 (all points land in the same tiny region). This catches clumping that coverage misses: a signal might occupy many cells but pile points into adjacent ones.

### When it lights up

Torus coverage is the framework's simplest binary test — it immediately separates data that explores its value space from data that doesn't. In the atlas, coverage and entropy together form the "distributional baseline" that all other distributional geometries refine. PRNGs cluster at the (1.0, 7.98) corner; periodic orbits cluster at the origin. The interesting signals are the ones in between: chaotic maps with 0.8-0.95 coverage, natural signals with high coverage but moderate entropy.

---

## Spherical S²

**What it measures:** Whether the signal's values cluster around a preferred direction.

Maps each pair of consecutive values to a point on the 2-sphere via spherical coordinates: the first value becomes the polar angle (north-south), the second becomes the azimuthal angle (around the equator). The resulting point cloud on the sphere reveals directional bias that a histogram would miss — two signals with identical value distributions can have completely different spherical profiles if their consecutive-pair correlations differ.

### Metrics

**angular_spread** — Standard deviation of the angles between each point and the cloud's mean direction. Thue-Morse, L-System Dragon, and Rule 30 all score 1.57 (maximum spread — points uniformly scattered, no preferred direction). Constants score 0.0 (all points at one pole). This is the spherical analogue of standard deviation.

**concentration** — Resultant length of the mean direction vector: how tightly do the points cluster? 1.0 means all points at the same location (Logistic period-2, Collatz gap lengths, Rainfall). Near 0.0 means perfectly diffuse (Thue-Morse at 0.0001). A signal can have high entropy in the Torus geometry but high concentration on the sphere if its consecutive pairs always point in the same angular direction.

**hemisphere_balance** — How evenly are points split between the northern and southern hemispheres? 1.0 means perfect 50/50 (Thue-Morse, L-System Dragon, Clipped Sine). 0.0 means all points on one side (constants, logistic period-2). This detects asymmetry in the polar angle distribution — signals that spend more time in one "half" of their range.

**mean_z** — Average z-coordinate on the sphere (cosine of polar angle). Positive means points cluster toward the north pole (low first-of-pair values, since small values map to theta near 0 where cos(theta) = 1). Collatz gap lengths (1.0) and Rainfall (1.0) are maximally northern: their values are dominated by small numbers, which map to the north pole. Forest fire (-0.96) is maximally southern: its large avalanche values map to the south. This metric encodes distributional skewness through geometry.

### When it lights up

Spherical geometry separates sources by their directional structure in a way that scalar statistics cannot. Two signals with the same mean and variance can have opposite mean_z values if one is right-skewed and the other is left-skewed. In the atlas, the concentration axis separates noise-like sources (diffuse, concentration near 0) from spike-dominated sources (concentrated, Collatz gaps and rainfall near 1.0). The hemisphere_balance axis is orthogonal to this, catching symmetry vs. asymmetry regardless of concentration.

---

## Cantor Set

**What it measures:** How gappy the signal's ternary address space is.

Each byte is re-interpreted as a base-3 address into the unit interval, the way the classical Cantor set construction works: each bit selects "left third" or "right third," skipping the middle. The resulting coordinates cluster around Cantor-set-like positions if the data has ternary self-similarity, or spread uniformly if it doesn't. The geometry then measures the gap structure of these coordinates.

### Metrics

**coverage** — Fraction of distinct embedded coordinates relative to total data length. Sprott-B, Projectile, and Damped Pendulum score 0.016 (many repeated coordinates — smooth dynamics map many different byte values to the same Cantor address). Fibonacci word scores 0.0001 (its binary structure creates extreme degeneracy in ternary representation). High coverage means the data explores many distinct positions in the Cantor embedding; low coverage means it collapses to a dust-like subset.

**max_gap** — The largest gap between adjacent sorted coordinates. L-System Dragon, Morse code, and Rule 110 all hit 1.0 (a gap spanning the full interval — the data avoids an entire region of the ternary address space). Collatz gap lengths scores 0.005 (tiny gaps, nearly uniform coverage). A large max_gap means the data has a forbidden zone in its ternary structure, like the middle third removed in the classical Cantor construction.

**mean_gap** — Average spacing between consecutive sorted coordinates. Accel walk, Kepler exoplanet, and Zipf distribution cluster at 6.1e-5 (tightly packed — many distinct coordinates with small gaps). Collatz gap lengths scores 7.8e-7 (extremely dense). Mean gap complements max_gap: a signal can have large max_gap but small mean_gap if it has one big hole and is densely packed everywhere else.

### When it lights up

Cantor Set geometry detects structure in the low bits of byte values — the ternary address depends on the full bit pattern, not just the magnitude. Signals with repetitive low-bit patterns (periodic orbits, symbolic dynamics) produce degenerate Cantor embeddings with extreme gaps. In the atlas, the combination of low coverage and high max_gap identifies signals whose byte values avoid specific ternary regions, which is a different kind of regularity than the distributional uniformity measured by Torus or Wasserstein.

---

## 2-adic

**What it measures:** How much structure the signal has in its divisibility pattern.

Samples random pairs of byte values and measures the distance between them using the 2-adic metric: two numbers are "close" if their difference is divisible by a high power of 2. The number 48 and the number 80 differ by 32 = 2^5, so they are 2-adically very close (distance 2^-5) despite being far apart on the number line. This geometry detects hierarchical modular structure — patterns in which bits the signal's values share.

### Metrics

**distance_entropy** — Shannon entropy of the distinct 2-adic distances. High entropy means the pairwise distances span many different powers of 2 — the signal explores multiple levels of the divisibility hierarchy. Devil's staircase scores highest (3.02): its plateaus create consecutive values whose differences hit every power of 2 as the staircase descends through its fractal levels. Classical MIDI (2.49) and ECG supraventricular (2.27) are also high — both have quantized amplitude levels that create diverse divisibility patterns. Rainfall scores lowest among nontrivial signals (0.66): its near-zero values produce differences that are almost always odd (2-adic distance 1.0), collapsing the entropy.

**mean_distance** — Average 2-adic distance across sampled pairs. VLF Radio Eclipse, Van der Pol, and Triangle Wave all score 0.668 (the maximum). Rainfall scores 0.09 — its small integer values have differences divisible by high powers of 2 more often than random data would. The expectation for uniform random bytes is near 2/3 (matching the ~0.668 maxima above); values significantly below indicate non-trivial divisibility structure.

### When it lights up

The 2-adic geometry detects structure invisible to every other distributional metric: the pattern of trailing bits. Two signals with identical histograms and identical Torus coverage can have very different 2-adic profiles if one tends to produce differences divisible by 4 and the other doesn't. Devil's staircase topping distance_entropy is diagnostic: its Cantor-function structure creates a precise hierarchy of jump sizes that maps perfectly onto the 2-adic distance hierarchy. This makes 2-adic geometry the framework's most direct detector of fractal staircase dynamics.

---

## Wasserstein

**What it measures:** How the distribution of values differs from uniform, and how stable that distribution is across the signal.

Bins the data into a 32-bin histogram and treats it as a probability distribution. Then asks three questions: how far is this distribution from uniform (optimal transport cost)? How concentrated is it (peak height)? And does the first half of the signal look like the second half (self-similarity)?

### Metrics

**concentration** — The peak bin height times the number of bins. 1.0 means uniform (De Bruijn scores exactly 1.0 — its construction guarantees every byte pattern appears equally). Above 1.0 means the distribution has a spike. Collatz gap lengths (31.8), Rainfall (31.5), and Forest fire (29.4) are the most concentrated signals in the atlas — their heavy-tailed distributions pile most of their mass into the lowest bin.

**dist_from_uniform** — Earth mover's distance from the uniform distribution: the minimum amount of "dirt" you'd need to move to make the histogram flat. Collatz gap lengths (0.48) and Rainfall (0.48) are farthest from uniform. Neural net pruned weights (0.46) are close behind — pruning creates a spike at zero. De Bruijn scores near 0 (already uniform).

**entropy** — Shannon entropy of the 32-bin histogram. De Bruijn, circle map quasiperiodic, and phyllotaxis all score 5.0 (near the maximum of 5 bits — flat distribution). Collatz gap lengths scores 0.05 (almost all mass in one bin). This is the classical measure of distributional spread, here computed on the Wasserstein embedding.

**self_similarity** — One minus the earth mover's distance between the first-half and second-half histograms. 1.0 means the distribution is perfectly stable over time (logistic period-4, logistic period-2, De Bruijn). Hilbert walk scores 0.60 (its deterministic sweep creates different distributions in the first and second halves). This catches nonstationarity that entropy and concentration miss: a signal can have high entropy overall but low self_similarity if its distribution drifts.

### When it lights up

Wasserstein self_similarity is the distributional lens's nonstationarity detector. Signals that change character midstream — sensor drift, regime switches, concatenated recordings — score low on self_similarity while potentially scoring high on all other distributional metrics. In the atlas, Wasserstein's concentration axis separates the heavy-tailed cluster (Collatz, rainfall, forest fire) from the uniform-distribution cluster (PRNGs, De Bruijn), while self_similarity provides an orthogonal axis that catches temporal instability invisible to any single-histogram metric.

---

## Fisher Information

**What it measures:** How sharply the signal's distribution changes when you perturb the histogram.

Treats the 16-bin histogram as a point on a statistical manifold — a curved space where each point represents a probability distribution. The Fisher information matrix measures the curvature at that point: high curvature means a small change in the data would radically shift the distribution (informationally sensitive). Low curvature means the distribution is robust to perturbation.

### Metrics

**effective_dimension** — How many independent directions matter in the Fisher matrix? Computed as the participation ratio of eigenvalues. De Bruijn, phyllotaxis, and circle map quasiperiodic all score 16.0 (the maximum — all 16 bins are equally important, so the statistical manifold is fully 16-dimensional). Seismic b-value scores 2.27 (its distribution has only 2-3 effective degrees of freedom, despite occupying many bins). This measures the intrinsic dimensionality of the signal's distributional footprint.

**log_det_fisher** — Logarithm of the determinant of the Fisher matrix. This is the log-volume element of the statistical manifold at the data's location. Collatz parity (137.4), Symbolic Henon (137.4), and Fibonacci word (137.3) score highest — their sparse, peaked distributions create enormous Fisher curvature (tiny probabilities in many bins produce large 1/p terms). De Bruijn scores 44.4 (the minimum for a 16-bin uniform — all probabilities equal 1/16). The 93-unit range across the atlas spans 40 orders of magnitude in actual determinant value.

**trace_fisher** — Sum of diagonal Fisher matrix entries (sum of 1/p_i for each bin). Collatz parity (229,605), Symbolic Henon (229,604), and Fibonacci word (229,604) score highest for the same reason as log_det: near-empty bins dominate the trace. De Bruijn scores 256 (16 bins, each with probability 1/16, so 16 * 16 = 256). Trace and log_det are correlated but not identical: log_det captures the product of per-bin information (sensitive to the emptiest bin), while trace captures the sum (sensitive to the total information budget).

### When it lights up

Fisher Information geometry detects a specific distributional property: how many bins are nearly empty. Signals that use only a few of 16 bins — binary sequences, periodic orbits, sparse event streams — create Fisher matrices with explosive curvature because the nearly-empty bins contribute 1/p terms near infinity (Laplace smoothing prevents actual infinity but preserves the relative ranking). In the atlas, effective_dimension separates "genuinely multi-valued" signals (dimension near 16) from "effectively binary" signals (dimension 2-4), providing a distributional complexity measure independent of entropy.

---

## Zipf–Mandelbrot (8-bit)

**What it measures:** How the frequencies of byte values decay from most-common to least-common.

Counts the frequency of each of the 256 possible byte values, sorts them from most to least common, and fits the Zipf-Mandelbrot law: does frequency drop off as a power of rank? Natural language follows Zipf's law closely (the 2nd most common word appears half as often as the 1st). Random data has flat frequency — no decay. This geometry characterizes the "vocabulary" structure of the byte stream.

### Metrics

**zipf_alpha** — The Zipf exponent: how steeply does frequency decay with rank? Alpha = 0 means flat (all bytes equally common). Alpha = 1 is Zipf's law (natural language). Poker hands (3.87) and Collatz gap lengths (3.62) score highest — extremely steep decay, a few values dominate completely. Sensor event streams (3.04) are similarly top-heavy. Collatz parity scores 0.0 (two values only, not enough for a power-law fit).

**zipf_r_squared** — How well does the Zipf-Mandelbrot model actually fit? De Bruijn (1.0) scores perfect: its uniform distribution is a trivial special case (alpha = 0, perfect fit). Divisor count (0.99) and logistic near-full chaos (0.99) also fit well. Collatz parity scores 0.0 (too few unique values). A high alpha with low r_squared means the signal is concentrated but not in a power-law way — useful for distinguishing genuine Zipf behavior from arbitrary concentration.

**mandelbrot_q** — The Mandelbrot offset parameter: how much do the low ranks deviate from pure Zipf? Large q means the most common values are less dominant than Zipf would predict — the top of the frequency curve is flattened. Solar wind IMF, Solar wind speed, and Sunspot all score 10.0 (maximum q — their distributions have a plateau at the top before the power-law tail kicks in). Logistic chaos and constants score 0.0.

**gini_coefficient** — Income-inequality measure applied to byte frequencies. 0.0 means perfect equality (all bytes equally common). 1.0 means maximal inequality (one byte gets all the count). Rainfall (0.97), Forest fire (0.95), and Neural net pruned (0.94) are the most unequal — a handful of values dominate. Constants score 0.0 (only one value — no inequality when there is only one entity).

**hapax_ratio** — Fraction of distinct byte values that appear exactly once (hapax legomena). Rainfall (0.31), Accel sit (0.30), and EEG tumor (0.26) score highest — many byte values appear only once, indicating a sparse tail. Logistic chaos, Henon map, and Tent map score 0.0 (chaotic maps visit enough values often enough that none are unique). High hapax ratio signals have "rare words" — a linguistic fingerprint of sparse, heavy-tailed data.

### When it lights up

Zipf-Mandelbrot (8-bit) is the framework's vocabulary profiler at single-byte resolution. The combination of alpha (decay steepness), r_squared (fit quality), and gini (concentration) gives a three-dimensional characterization of the frequency curve that entropy alone collapses to a single number. In the atlas, rainfall and forest fire cluster together on the high-gini, high-alpha, high-hapax corner — both are "natural language-like" in having a few dominant values and a long sparse tail. PRNGs and De Bruijn occupy the opposite corner: flat frequencies, low gini, zero hapax.

---

## Zipf–Mandelbrot (16-bit)

**What it measures:** The same vocabulary structure as the 8-bit version, but at the resolution of byte pairs.

Packs each pair of consecutive bytes into a 16-bit symbol (65,536 possible values) using a sliding window, then performs the same Zipf-Mandelbrot analysis. The larger alphabet dramatically increases sensitivity to sequential structure: two signals with identical 8-bit histograms can have completely different 16-bit Zipf profiles if their byte-to-byte transitions differ.

### Metrics

**zipf_alpha** — Zipf exponent at 16-bit resolution. Poker hands (2.74) and Sensor event streams (2.56) still top the ranking, but with lower exponents than the 8-bit version — the larger vocabulary dilutes the concentration. Collatz gap lengths drops from 3.62 (8-bit) to 2.39 (16-bit), indicating its dominance is mainly a single-byte phenomenon. Signals where alpha is similar at both resolutions have concentration that extends to pair structure.

**zipf_r_squared** — Fit quality at 16-bit resolution. Collatz gap lengths (0.99), Poker hands (0.98), and Continued fractions (0.98) score highest — their frequency decay is genuinely power-law even at pair resolution. This is more discriminating than the 8-bit version: many signals that fit Zipf at byte level fall apart at the pair level because their byte transitions are not power-law distributed.

**mandelbrot_q** — The same plateau parameter. Solar wind IMF and Solar wind speed still score 10.0 — their distributional plateau persists at pair resolution. Devil's staircase drops from nonzero (8-bit) to 0.0 (16-bit): its plateau structure is a single-byte phenomenon that doesn't extend to pairs. Tidal gauge (10.0) joins the top at 16-bit — its slow tidal oscillations create repeated byte pairs that flatten the top of the frequency curve.

**gini_coefficient** — Frequency inequality at pair resolution. Rainfall (0.97), Forest fire (0.95), and Neural net pruned (0.91) remain the most unequal. The pruned network drops from 0.94 (8-bit) to 0.91 (16-bit) — its zero-dominated weight distribution is slightly less extreme when viewed as pairs. Logistic period-3 scores 0.0 (its 3 repeated values produce 3 repeated pairs, all equally common).

**hapax_ratio** — Fraction of 16-bit symbols appearing exactly once. Champernowne (0.95) tops the ranking — its digit-concatenation construction creates an enormous number of unique byte pairs. Middle-Square (0.93) and Intermittent silence (0.90) follow. The hapax explosion at 16-bit resolution is expected: with 65,536 possible symbols and finite data, many pairs will be seen only once. The signals at the bottom (Sawtooth wave at 0.0) are those whose pair vocabulary is so constrained that every pair repeats.

### When it lights up

The 16-bit version's primary value over the 8-bit is detecting transition structure. A signal with a flat byte histogram (high 8-bit entropy, low 8-bit alpha) can still have extreme 16-bit concentration if certain byte transitions dominate. Comparing alpha and hapax between the two resolutions reveals how much of the signal's vocabulary structure is local (single-byte) versus sequential (pair). In the atlas, Champernowne's jump from low hapax at 8-bit (it uses all digits) to 0.95 hapax at 16-bit (its digit pairs are almost all unique) is the clearest example: its structure is entirely in the sequence of bytes, not in their individual distribution.

## Hyperbolic (Poincaré)

**What it measures:** How much your signal's structure looks like a tree -- branching, hierarchical, pushing toward the boundary of a curved disk.

Maps consecutive byte pairs into the Poincaré disk, a model of hyperbolic space where distances grow exponentially near the edge. Flat, uniform data spreads evenly across the disk. Hierarchical or heavy-tailed data gets shoved toward the boundary, because the disk has exponentially more room out there -- the same reason tree-like data embeds naturally in hyperbolic space. The geometry computes where the mass centroid sits, how far points are from the origin, and how far apart they are from each other under the curved metric.

### Metrics

**centroid_offset** — How far the hyperbolic centroid drifts from the disk's origin. Collatz Gap Lengths (4.71) and Rainfall (4.69) push the centroid far off-center, reflecting asymmetric, heavy-tailed distributions. Prime Gaps (4.52) does the same. Constants score 0.0 (collapsed to a single point). L-System Dragon scores 0.0002 despite being complex -- its byte distribution is too symmetric to shift the centroid.

**mean_hyperbolic_radius** — Average distance of embedded points from the disk's origin. L-System Dragon (4.72), Pulse-Width Mod (4.72), and Morse Code (4.72) all push points to the boundary, where hyperbolic distances explode. Bearing Inner (0.22) stays close to the center. This metric separates signals with extreme byte-value concentrations from those with more centered distributions.

**mean_pairwise_distance** — Average hyperbolic distance between sampled point pairs. L-System Dragon (6.72) and Rule 30 (6.65) have the widest spread -- their embedded points scatter across the disk far from each other. Logistic Period-2 scores 0.0 because its two-valued signal collapses to a single pair of points in the embedding.

### When it lights up

Hyperbolic geometry is most useful for separating hierarchical data from flat or periodic data. The centroid_offset metric uniquely identifies signals with heavy tails or asymmetric distributions -- Collatz gaps, rainfall, and prime gaps form a tight cluster at the top that no other geometry highlights the same way. Periodic signals collapse to near-zero on all three metrics.

---

## Mostow Rigidity

**What it measures:** Whether your signal's geometry is locked in place or free to deform -- the difference between a crystal and a liquid.

Embeds byte triples into the 3D Poincaré ball, forms tetrahedra, and then asks: if you shake the data slightly, does the hyperbolic geometry change? Mostow's rigidity theorem says that in dimension 3 and above, the shape of a hyperbolic manifold is completely determined by its topology -- there are no continuous deformations. This geometry tests the discrete analog: whether small perturbations (5% Gaussian noise) alter the distance structure, volume distribution, and spectral properties.

### Metrics

**distance_rigidity** — Correlation between the original and perturbed pairwise distance matrices. Rule 30 (0.985) and Symbolic Lorenz (0.984) are nearly rigid: shaking the input barely changes the distances. This means their structure is determined by the combinatorial pattern, not the exact byte values. Constants score 0.0 (no distances to correlate).

**volume_entropy** — Shannon entropy of the hyperbolic tetrahedron volume distribution. Beta Noise (4.52), XorShift32 (4.50), and Dice Rolls (4.50) produce the most varied volumes -- their points fill the ball uniformly, creating tetrahedra of many different sizes. Logistic Period-2 (0.0) produces identical tetrahedra.

**volume_rigidity** — How stable is the total hyperbolic volume under perturbation? AES Encrypted (0.97) and Gzip (0.96) are volume-rigid: noise barely changes their total volume. Lorenz and Rössler Attractors score 0.0 -- their attractor geometry is sensitive to perturbation, the volumes shift substantially.

**margulis_ratio** — Ratio of 5th-percentile to 95th-percentile pairwise distances. Measures the gap between thin and thick parts of the geometry. Logistic Period-2 (1.0) has perfectly uniform spacing. Pulse-Width Mod (0.91) is nearly uniform. Low values mean the signal has both tightly clustered and widely separated regions.

**spectral_rigidity** — Correlation between Laplacian eigenvalues before and after perturbation. L-System Dragon (1.0), Morse Code (1.0), and Symbolic Henon (1.0) have perfectly stable spectra -- their graph connectivity pattern is insensitive to small perturbations. Logistic Period-3 (0.33) has a fragile spectrum.

### When it lights up

Mostow Rigidity uniquely separates signals whose geometric structure is topologically determined from those that are metrically flexible. The volume_rigidity metric cleanly splits encrypted/compressed data (rigid, 0.96+) from dynamical attractors (flexible, near 0.0) -- a distinction no other geometry captures. The distance_rigidity metric at the top is dominated by cellular automata and symbolic dynamics, signals whose structure is entirely combinatorial.

---

## Spectral Graph

**What it measures:** The effective dimensionality your data's point cloud presents to a diffusing particle -- and whether that dimensionality follows known physical laws.

Delay-embeds the signal into 5 dimensions, builds an epsilon-neighborhood graph on the resulting point cloud, and computes the Laplacian spectrum. Two numbers come out: the spectral dimension (how fast heat spreads on the graph) and the Weyl exponent (how eigenvalues grow with index). For a regular d-dimensional lattice, the spectral dimension equals d and the Weyl exponent equals 2/d. Deviations from these relationships signal anomalous geometry -- fractals, clustered manifolds, or degenerate embeddings.

### Metrics

**spectral_dim** — The effective dimensionality seen by diffusion on the data manifold. Logistic periodic orbits (Period-2, Period-3, Period-4) all score 2.004 -- their delay embeddings land on clean low-dimensional manifolds (circles, figure-eights) where heat diffuses in exactly 2 dimensions. Tidal Gauge (0.10) has nearly zero spectral dimension, meaning diffusion is almost completely blocked -- the point cloud is fragmented or one-dimensional. The gap between 2.0 and 0.1 is the gap between a smooth manifold and a dust-like point set.

**weyl_exponent** — The eigenvalue growth rate. Wigner Semicircle (2.15) and Circle Map QP (2.01) have exponents near 2, matching the Weyl law prediction for a 1D manifold. Henon-Heiles (1.97) is close behind. Periodic logistic maps score near zero because their spectra have too few distinct eigenvalues for a meaningful power-law fit. High Weyl exponent means the Laplacian spectrum is rich and well-structured.

### When it lights up

Spectral Graph is the atlas's cleanest test for manifold dimensionality. It separates smooth delay embeddings (spectral_dim near 2.0) from fragmented point clouds (near 0) with almost no overlap. The spectral_dim and weyl_exponent pair together provide a two-parameter characterization that other topological geometries lack -- Persistent Homology counts features, Cayley measures graph growth, but Spectral Graph directly measures how space "feels" to diffusion.

---

## Persistent Homology

**What it measures:** How many holes and connected components survive as you zoom out -- the topological skeleton that persists across scales.

Delay-embeds the signal into 2D, deduplicates the point cloud, then builds a Vietoris-Rips filtration: imagine inflating a ball around every point and tracking when components merge (H0) and when loops form and fill in (H1). Features that persist across a wide range of scales are topologically robust. Features that blink in and out are noise. The distribution of lifetimes, their entropy, and the total persistence mass characterize the signal's topological complexity.

### Metrics

**n_significant_features** — Count of H0 features with lifetime above 0.1. Dice Rolls (34.1) produces the most significant components -- its random jumps create many well-separated clusters that merge at different scales. Intermittent Silence (24.9) and Collatz Stopping Times (20.8) follow. Rössler (1.0) has just one significant component -- its attractor is a single connected blob in the delay embedding.

**h1_total_persistence** — Total lifetime mass of all H1 (loop) features. Dice Rolls (1.83) dominates, with loops forming at many scales as its scattered point cloud creates ring-like structures during filtration. Collatz Gap Lengths (0.56) and Codon Usage (0.54) have modest loop structure. Logistic Chaos and Tent Map score 0.0 -- their dense, space-filling attractors leave no room for persistent loops.

**max_h1_lifetime** — Lifetime of the single most persistent loop. Rule 30 (0.414), Symbolic Lorenz (0.414), and Morse Code (0.414) share the top score -- their binary-valued signals create specific point-cloud geometries in delay embedding where one dominant loop persists across scales. This metric identifies signals with a single robust circular or toroidal structure.

**max_components** — Total number of H0 features (connected components at birth). Solar Wind IMF and Solar Wind Speed both score 50, meaning their delay embeddings start as 50 isolated clusters before merging. Quantum Walk scores 1.0 -- its encoding produces a nearly connected point cloud from the start.

**persistence_entropy** — Shannon entropy of the lifetime distribution. High entropy means the topological features have diverse lifetimes (no single scale dominates). Low entropy means one feature dominates all others. This separates multi-scale signals from those with a single characteristic scale.

### When it lights up

Persistent Homology is the atlas's most direct topological detector. The n_significant_features metric uniquely identifies signals with complex multi-component structure -- Dice Rolls, Intermittent Silence, and Collatz Stopping Times form a cluster that no other geometry highlights. The H1 metrics separate signals with genuine loop topology from those that are simply connected, a distinction only available through filtration-based analysis.

---

## Fractal (Mandelbrot)

**What it measures:** How your signal behaves when fed into the Mandelbrot iteration as a starting condition -- does it escape, get trapped, or linger at the boundary?

Takes consecutive byte pairs, maps them to the complex plane as the parameter c, and iterates z -> z^2 + c starting from z = 0. For each c, it records how many iterations before the orbit escapes (or whether it stays trapped). The distribution of escape times, the fraction of trapped orbits, and the entropy of the escape-time histogram characterize how the signal's byte-pair distribution intersects the Mandelbrot set's boundary structure.

### Metrics

**escape_entropy** — Shannon entropy of the escape-time histogram. Human Proteome (4.21) has the richest distribution -- its byte pairs map to c-values that scatter across the boundary region, producing many different escape speeds. Critical Circle Map (3.27) and Classical MIDI (3.23) are close behind. Fibonacci Word and Logistic Period-2 score near zero: their byte pairs map to a narrow region where everything escapes at the same speed or not at all.

**escape_time_variance** — How spread out are the escape times? Devil's Staircase (953.8) and ETH/BTC Ratio (951.8) have enormous variance -- their byte pairs hit both deep interior and far exterior of the Mandelbrot set, creating a bimodal distribution. Weierstrass (910.1) follows. Constants and Fibonacci Word score 0.0 -- all their byte pairs land in the same escape-time bin.

**interior_fraction** — Fraction of byte pairs whose orbits never escape (trapped inside the Mandelbrot set). Earthquake P-wave (0.97) and Bearing Inner (0.97) trap almost everything -- their byte-pair distributions map overwhelmingly to the Mandelbrot interior. Divisor Count (0.0) and Prime Gaps (0.0) trap nothing -- their byte pairs all map to the exterior, where orbits escape quickly.

### When it lights up

Fractal (Mandelbrot) is most discriminative for signals whose byte-pair distributions straddle the Mandelbrot set boundary. The escape_entropy metric uniquely identifies Human Proteome at the top -- its amino-acid-derived byte distribution intersects the cardioid boundary in a way no other natural signal does. The interior_fraction metric cleanly separates physical sensor data (seismic, bearing vibration -- high interior fraction) from number-theoretic sequences (prime gaps, divisor counts -- zero interior), reflecting how their value distributions cluster in the complex plane.

---

## Visibility Graph

**What it measures:** What kind of network your time series becomes when you connect every pair of points that can "see" each other over the intervening values.

Treats the signal as a landscape and draws an edge between two time points whenever no intermediate value blocks the line of sight between them. Tall peaks see far; valleys are hidden. The resulting graph's degree distribution, clustering, and assortativity encode the signal's dynamical class. Periodic signals produce regular graphs. Chaotic signals produce scale-free networks. Random signals produce specific power-law exponents predicted by theory.

### Metrics

**assortativity** — Do high-degree nodes connect to other high-degree nodes (positive), or to low-degree ones (negative)? Rössler Hyperchaos (0.76) is strongly assortative: its tall peaks cluster together. LIGO Livingston (0.54) also shows this pattern. Lotka-Volterra (-0.57) is strongly disassortative: its predator-prey oscillations create peaks that connect to valleys, not to each other. Logistic Period-2 (-0.50) alternates high-low, producing the same disassortative pattern.

**avg_clustering_coeff** — How interconnected are each node's neighbors? Rainfall (0.85) has the highest clustering -- its rain events create local clusters of mutually visible points. Neural Net Pruned (0.83) and Sensor Event Stream (0.81) show similar cliquish structure. Devil's Staircase (0.0) has zero clustering because its flat plateaus followed by jumps create a graph where neighbors never see each other.

**degree_exponent_gamma** — Power-law exponent of the degree distribution. DNA SARS-CoV-2 (2.94) has the steepest decay -- most nodes have low degree, with rare high-degree hubs. Circle Map QP (2.78) and Rule 30 (2.77) follow. For iid random data, the NVG degree distribution is exponential (Lacasa et al. 2008), so the power-law exponent from a forced fit is not theoretically meaningful in that regime — but deviations across signals still reveal structural differences.

**max_degree** — The most-connected node in the graph. Lotka-Volterra (690) has an extreme hub -- its largest predator-prey peak can see across 690 other time points. Rainfall (587) and Van der Pol (540) also produce high-visibility peaks. Devil's Staircase has max_degree = 2, meaning no point can see beyond its immediate neighbors.

### When it lights up

Visibility Graph is the atlas's best tool for separating oscillatory dynamics by their waveform shape. The assortativity metric spans from -0.57 to +0.76 across the atlas, a range no other metric covers, and it cleanly separates predator-prey / period-doubling dynamics (negative) from hyperchaotic / bursty dynamics (positive). The clustering coefficient uniquely identifies bursty event streams and rainfall as topologically cliquish -- a property invisible to spectral or distributional geometries.

---

## Klein Bottle

**What it measures:** Whether your signal has hidden linear structure over GF(2) -- the telltale signature of XorShift generators, LFSRs, and bit-level linear algebra.

The Klein bottle's first homology group contains a Z/2Z torsion factor -- algebraically, that is GF(2), the two-element field. This geometry exploits that correspondence. It runs Berlekamp-Massey on each bit channel to find the shortest linear feedback shift register that reproduces the sequence, forms binary matrices and computes their rank over GF(2) via Gaussian elimination, and maps byte pairs onto the Klein bottle's twisted surface to measure orientation coherence. Together, these three metrics detect whether a signal was generated by XOR-based operations.

### Metrics

**linear_complexity** — Berlekamp-Massey LFSR length, averaged across 8 bit channels, normalized by n/2. True random bits need an LFSR of length ~n/2 (score ~1.0). LIGO Hanford (1.045) and Thue-Morse (1.04) score just above 1.0 -- they are maximally complex, requiring more bits than expected. Random Steps (1.02) is similarly complex. Quantum Walk (0.0) has zero complexity -- its encoding produces trivially reproducible bit patterns. XorShift generators score low (~0.06) because their 32-bit state fully determines the sequence.

**orientation_coherence** — How consistently does the trajectory respect local orientation on the Klein bottle's surface? Sine Wave (0.995) is nearly perfectly coherent -- its smooth, monotonic segments maintain consistent orientation. Exponential Chirp (0.994) and Van der Pol (0.993) follow. Logistic Period-4 (0.00003) is nearly incoherent -- its jumps between four values constantly reverse the local orientation, mirroring the non-orientability of the Klein bottle itself.

**rank_deficit** — Normalized rank deficit of 16x16 binary matrices over GF(2). Quantum Walk (1.0) is maximally rank-deficient -- its bit patterns are heavily dependent. Square Wave (0.96) and Pulse-Width Mod (0.95) are nearly so, because their binary-valued signals create matrices with many identical rows. Ikeda Map (0.047) has almost full rank -- its chaotic bit patterns are nearly independent over GF(2).

### When it lights up

Klein Bottle is the atlas's only geometry targeting GF(2) algebraic structure. The linear_complexity metric is the definitive LFSR detector -- XorShift32 scores 0.06 while true random scores 1.0, a 16x separation with no overlap. The rank_deficit metric uniquely identifies signals with degenerate binary structure (Quantum Walk, Square Wave) that other topological geometries miss entirely. The orientation_coherence metric separates smooth waveforms from jump processes, complementing the algebraic metrics with a genuinely topological perspective.

## E8 Lattice

**What it measures:** How constrained the data's 8-byte structure is relative to the densest sphere packing in 8 dimensions.

Takes each group of 8 consecutive bytes, normalizes them, and finds the closest of the 240 root vectors of E8. These roots come in two families: 112 "integer" roots (pairs of coordinate axes) and 128 "half-integer" roots (all coordinates ±1/2 with even sign parity). The geometry also snaps each 8-byte window to the nearest actual E8 lattice point via parity correction, measuring the residual distance. Data with algebraic constraints concentrates on a few roots; unconstrained data spreads evenly across all 240.

### Metrics

**std_profile** — Variation in root alignment quality across windows, capturing how consistently the data's 8-byte structure matches E8 directions. High for signals with a mix of aligned and unaligned windows.

**coset_transition** — Rate at which consecutive windows switch between integer and half-integer root cosets. Measures temporal dynamics of parity structure. *Evolved via ShinkaEvolve.*

**diversity_ratio** — Fraction of the 240 E8 roots actually used. High diversity means the data explores the full root system; low means it's confined to a subspace.

**normalized_entropy** — Shannon entropy of root usage, normalized by maximum. Uniform distributions across 240 roots score high; periodic orbits score near 0.

### When it lights up

E8 Lattice is the highest-dimensional root system geometry in the framework (window size 8), complementing G2 (2), H3 (3), D4 and H4 (4). Its 240-root system is the densest lattice sphere packing in 8D, and the coset_transition metric captures temporal structure in how the data moves between the integer and half-integer sublattices — a signature that lower-dimensional root systems cannot detect.

---

## G2 Root System

**What it measures:** Hexagonal symmetry in consecutive byte pairs.

Projects each pair of adjacent bytes onto the 12 roots of G2 — 6 short roots at 60-degree intervals, and 6 long roots at 30-degree offsets with length proportional to the square root of 3. This 2D root system captures the simplest non-trivial Lie algebra structure, operating at the smallest possible window size. Because it works on pairs, it acts as a fast correlation probe: any directional preference in the (byte_t, byte_{t+1}) scatterplot shows up as root concentration.

### Metrics

**alignment_std** — Variation in how tightly byte pairs align with the nearest G2 root. Bursty signals (rainfall, forest fire) score highest — some pairs align perfectly, others scatter wildly. Periodic orbits score 0.0 (every pair identical).

**short_long_ratio** — Fraction of windows assigned to the 6 short roots (out of all 12). Binary-valued signals score 1.0 (exclusively short-root, coordinate-aligned); baker map prefers long roots (30-degree offset). Separates axial from diagonal pair correlations.

**alignment_mean** — Average alignment quality across all byte pairs. Constrained dynamics (Fibonacci word, logistic period-4) score highest; heavy-tailed distributions (rainfall) score lowest.

**normalized_entropy** — Uniformity of root usage across all 12 G2 roots.

**kurtosis_angular** — Fisher kurtosis of angular alignments (projections onto root directions). G2's 12 roots at uniform 30-degree spacing produce unimodal angular distributions for most data.

**kurtosis_raw** — Fisher kurtosis of raw alignment values (Euclidean distance to nearest root). G2's bimodal root lengths (1 vs sqrt(3)) make raw alignments bimodal — negative kurtosis.

**kurtosis_differential** — kurtosis_angular minus kurtosis_raw. Large for data where G2's specific root-length ratio matters; near zero when it doesn't. This is the key D1 metric: Thomson control (equal-length roots) has no bimodality, so kurtosis_raw is unimodal and the differential collapses. *Evolved via ShinkaEvolve atlas v1.*

### When it lights up

G2 is the complement to E8: where E8 examines 8-byte windows, G2 probes the shortest temporal structure (adjacent pairs). The kurtosis_differential metric exploits G2's unique two-length root structure (short:long = 1:sqrt(3)) — the only simple Lie algebra where root lengths differ. Thomson-6 control has equal-length directions, so the bimodality signal vanishes. The short_long_ratio metric remains unique to G2: no other geometry separates axial from diagonal pair correlations.

---

## H3 Icosahedral

**What it measures:** Whether 3-byte patterns prefer icosahedral directions.

Projects each triple of consecutive bytes onto the 30 roots of H3 — the vertices of an icosidodecahedron. These roots split into 6 axis-aligned vectors (permutations of (1,0,0)) and 24 "golden" vectors involving the golden ratio (even permutations of (1/2, phi/2, 1/2phi)). H3 is the symmetry group of the icosahedron, featuring 5-fold rotational symmetry that no crystallographic lattice can achieve. The geometry asks whether the data's 3-byte structure has any preference for these non-crystallographic directions.

### Metrics

**nn_enrichment** — Nearest-neighbor enrichment: how much more likely are consecutive windows to share a root than expected by chance. Measures temporal locality in root space. *Evolved via ShinkaEvolve.*

**temporal_coherence** — Autocorrelation of the root assignment sequence. High for smooth signals where 3-byte windows change gradually; low for noise and chaos.

**path_closure** — How often the root-space trajectory returns near its starting point over various time horizons. Captures periodicity and recurrence in 3D root coordinates.

**normalized_entropy** — Uniformity of root usage across all 30 H3 roots. Periodic orbits score near 0 (all weight on one root); high-entropy sources spread across many roots.

### When it lights up

H3 is the only geometry in the framework probing 5-fold icosahedral symmetry at the 3-byte scale, between G2 (pairs) and D4/H4 (quadruples). The evolved metrics (nn_enrichment, temporal_coherence, path_closure) shift focus from static root statistics to temporal dynamics in root space — how the signal's trajectory through the 30 icosahedral directions evolves over time.

---

## H4 600-Cell

**What it measures:** Non-crystallographic symmetry in 4-byte windows.

Projects each group of 4 consecutive bytes onto the 120 roots of H4 — the vertices of a 600-cell, the most complex regular polytope in 4D. The roots come in three families: 8 axis-aligned, 16 half-integer (all coordinates ±1/2), and 96 "golden" vectors built from even permutations of (0, 1/2, 1/2phi, phi/2). H4 is the largest non-crystallographic Coxeter group, governing the symmetry of 4D polytopes with icosahedral cross-sections.

### Metrics

**diversity_ratio** — Fraction of the 120 roots actually used. Sunspot, Kepler non-planet, and speech "zero" all score 0.233 (28 of 120 roots). Logistic period-2 uses a single root (0.008). With 120 roots available, the diversity ceiling is much lower than for E8 (240 roots) — most data concentrates on a small fraction of the 600-cell's directions.

**normalized_entropy** — Uniformity of root usage. BTC returns (0.640) and neural net dense weights (0.638) have the most uniform distributions — their high-entropy byte structure genuinely samples the 4D root system. Periodic orbits and Morse code score near 0.0. The entropy range (0 to 0.64) is wider than H3's, reflecting the larger root system's greater capacity to differentiate sources.

### When it lights up

H4 shares its 4-byte window size with D4 Triality but probes a completely different symmetry: D4's 24 roots are crystallographic (they tile via lattice translations), while H4's 120 roots are non-crystallographic (they cannot). This means H4 detects structural preferences that D4 misses — specifically, whether the data's 4-byte patterns prefer golden-ratio-related directions. In practice, H4's two metrics provide a coarse separation: high normalized_entropy signals (financial, neural network weights, noise) vs. low (periodic, symbolic). Its unique role is completing the non-crystallographic Coxeter family alongside H3.

---

## Sol (Thurston)

**What it measures:** Exponential anisotropy — whether one direction stretches while another contracts.

Builds a path through Sol space using the group law (x1 + e^{z1} * x2, y1 + e^{-z1} * y2, z1 + z2). Each step from a triple of byte values moves through a geometry where the x-direction is exponentially inflated and the y-direction exponentially compressed by the z-coordinate. Data with directional bias accumulates a strong z-drift; isotropic data wanders near z=0. The Sol metric amplifies any asymmetry in the data's local structure into exponentially different path lengths.

### Metrics

**path_length** — Total distance traveled in the Sol metric. Symbolic Henon (30.2M) scores highest: its chaotic dynamics create large steps that the exponential Sol metric amplifies enormously. Fibonacci word (28.3M) is close behind — its substitution structure produces systematic z-accumulation that inflates path length. Logistic period-3 scores only 82,909 — its tight orbit keeps z near zero, suppressing the exponential amplification. The 3-order-of-magnitude range across non-degenerate sources makes this a powerful separator.

**z_drift** — Final z-coordinate after traversing the signal. Clamped to [-5, 5] to keep the exponential factors finite. Stern-Brocot walk, El Centro earthquake, and forest fire all hit the +5 ceiling — they have systematic upward drift in their byte-triple structure. Prime gaps, divisor count, and Collatz gap lengths hit -5 (systematic downward drift). Z-drift is a signed metric: its polarity reveals which direction the data's asymmetry favors.

**z_variance** — Variance of the z-coordinate along the path. Van der Pol (23.4) and Hilbert walk (23.1) score highest: their oscillatory dynamics push z back and forth across its full range. Rainfall (0.018) barely moves z at all — the exponential distribution's near-zero values produce tiny increments. High z_variance without high z_drift means the path oscillates in the exponential direction; high drift with low variance means it ramps monotonically.

### When it lights up

Sol is the only Thurston geometry where the metric itself is exponential — the other seven (Euclidean, spherical, hyperbolic, Nil, the two products, SL(2,R)) all have at most polynomial distortion. This makes Sol path_length the framework's most sensitive amplifier of structural asymmetry. In the symmetry view, Sol separates sources that have directional z-bias (natural signals that saturate z_drift at ±5) from symmetric sources (chaos, noise) that wander without trend. The three metrics together form a concise dynamical portrait: drift tells you the direction, variance tells you the oscillation, and path_length integrates both through the exponential lens.

---

## S² × ℝ (Thurston)

**What it measures:** Whether the data has directional structure that drifts over time.

Maps each byte triple to a point on the sphere (first two bytes give polar angle and azimuth) plus a height on the real line (third byte). This is the Thurston geometry of spherical layers: data with stable directional concentration has high sphere_concentration, while the height coordinate tracks temporal drift in the third-byte channel.

### Metrics

**sphere_concentration** — Norm of the mean direction vector on S². Logistic period-3 and Collatz gap lengths score 1.0 (all points cluster at a single direction on the sphere). L-System Dragon scores 0.0006 (nearly uniform coverage of the sphere — the binary symbolic dynamics maps to antipodal directions that cancel). Rainfall also scores 1.0 (its near-zero values all map to the same latitude). This is a directional statistic: 1.0 means the data has a single preferred direction, 0.0 means it covers the sphere uniformly.

**height_drift** — Difference between final and initial height values. Rule 30 scores +1.0 (maximal upward drift) and Morse code scores -1.0 (maximal downward drift). ECG fusion scores -0.60. This captures systematic trends in the third-byte channel that the spherical components do not see.

**sphere_height_corr** — Correlation between the z-coordinate on S² and the height in the ℝ component. L-System Dragon (0.50) has the strongest positive correlation: when its sphere position moves poleward, its height increases. Logistic period-2 (-1.0) has perfect negative correlation — its alternating values create a strict z-height anti-relationship. Double pendulum (-0.99) and Van der Pol (-0.99) also show strong negative correlation, reflecting their oscillatory dynamics coupling the spherical and linear components.

### When it lights up

S² x R is the only geometry that decomposes the signal into a directional and a scalar component simultaneously. The sphere_height_corr metric captures coupling between these two degrees of freedom: positive correlation means the signal's direction and amplitude co-vary; negative correlation means they oppose. This is distinct from what Heisenberg measures (pure correlation twist) and from what Sol measures (exponential anisotropy). In the symmetry view, sphere_concentration separates concentrated signals (periodic, near-constant) from diffuse ones (chaos, symbolic dynamics), while sphere_height_corr adds a second axis that distinguishes coupled from decoupled dynamics.

---

## H² × ℝ (Thurston)

**What it measures:** Whether hierarchical depth co-varies with a linear trend.

Maps each byte triple into the Poincare disk (first two bytes) crossed with the real line (third byte). The Poincare disk distorts distances: points near the boundary are exponentially far from the center in hyperbolic terms. Data that clusters near the disk's edge has large hyperbolic variance; the depth_height_corr metric then asks whether this hierarchical depth is coupled to the linear drift.

### Metrics

**hyperbolic_variance** — Variance of the hyperbolic distance from the disk's origin. ECG ventricular tachycardia (3.45), damped pendulum (3.35), and ECG supraventricular tachycardia (3.34) dominate. These signals oscillate between near-center and near-boundary in the Poincare disk — the heartbeat's QRS complex pushes points to the boundary, while the baseline returns them to center. Logistic period-2 scores near 0 (its two-value alternation maps to a fixed radius). The ECG dominance is notable: pathological heart rhythms explore the full radial range of the disk.

**depth_height_corr** — Correlation between hyperbolic distance from the origin and the height coordinate. Forest fire (0.68) and humidity (0.62) have the strongest positive correlation: when they move toward the boundary (high hierarchical depth), their height increases. Wind speed (-0.79) and wave height (-0.81) have strong negative correlation: high hierarchical depth coincides with decreasing height. Intermittent silence scores -0.83, the strongest negative value — its bursts of silence (near-center) alternate with active periods (boundary) in a way that anti-correlates with the height trend.

### When it lights up

H² x R is the hyperbolic counterpart to S² x R: where S² x R decomposes into direction + scalar, H² x R decomposes into hierarchical depth + scalar. The hyperbolic_variance metric's ECG dominance (top 3 are all cardiac signals) suggests it captures a specific physical phenomenon: the heartbeat's voltage range maps naturally to radial excursion in the Poincare disk, and pathological rhythms (tachycardia) explore this range more than normal sinus rhythm. The depth_height_corr metric then adds temporal coupling: environmental signals (humidity, forest fire) show positive coupling while ocean/atmospheric signals (wind, waves) show negative.

---

## SL(2,ℝ) (Thurston)

**What it measures:** The character of 2x2 matrix dynamics — whether the accumulated transformation rotates, shears, or stretches.

Converts each byte triple to an SL(2,R) matrix via the KAK decomposition: a rotation, a hyperbolic boost, and a second rotation. These three parameters span all three conjugacy classes of 2x2 matrices with determinant 1. The geometry classifies each matrix by its trace: elliptic (|trace| < 2, rotation-dominated), parabolic (|trace| = 2, shear), or hyperbolic (|trace| > 2, exponential stretch). A running product of matrices accumulates the signal's overall dynamical character.

### Metrics

**hyperbolic_fraction** — Fraction of matrices with |trace| > 2. L-System Dragon, pulse-width modulation, and Morse code all score 1.0 (every matrix is hyperbolic — their binary structure pushes the boost parameter to extreme values). Earthquake P-wave scores 0.014 (almost entirely elliptic — the smooth waveform creates small boosts). This separates signals with discontinuous jumps (hyperbolic) from smooth oscillations (elliptic).

**lyapunov_exponent** — Average rate of exponential growth in the running matrix product, computed over blocks of 50 matrices. Rainfall (1.98) and Collatz gap lengths (1.96) score highest — their bursty or irregular dynamics create matrices whose product grows rapidly. Earthquake P-wave (0.020) is the lowest non-degenerate value: its smooth oscillation produces near-identity matrices that barely grow. This is a genuine Lyapunov exponent of the matrix cocycle, not an ad-hoc estimate.

**mean_trace** — Average trace of all matrices. L-System Dragon, Morse code, and Rule 110 all score 7.52 (far into the hyperbolic regime). DNA sequences score around -2.6 (elliptic regime with negative trace — the base pair frequencies create matrices with trace near the elliptic boundary). Logistic period-3 scores -2.77 (deepest into the negative-trace elliptic region). The sign of mean_trace separates two flavors of rotational dynamics: positive trace means the rotations partially cancel; negative means they compound.

**parabolic_fraction** — Fraction of matrices near the elliptic/hyperbolic boundary (|trace²-4| <= 0.2). Ambient microseism (0.40) and bearing inner fault (0.39) score highest — their narrowband oscillations produce matrices that hover near the boundary. This is the rarest conjugacy class: most data is either clearly elliptic or clearly hyperbolic, with few matrices landing in the thin parabolic strip. High parabolic fraction signals a critical or transitional dynamical regime.

### When it lights up

SL(2,R) is the most algebraically rich Thurston geometry in the framework. Its conjugacy class decomposition (elliptic/parabolic/hyperbolic) provides a three-way classification that no other metric replicates. In the symmetry view, hyperbolic_fraction is the primary separator: it cleanly divides binary/symbolic sources (fraction near 1.0) from smooth continuous signals (fraction near 0). The lyapunov_exponent adds a growth-rate axis, and mean_trace adds a polarity axis. Together they place each source in a 3D space where binary chaos, smooth oscillation, and critical behavior occupy distinct regions.

---

## Projective ℙ²

**What it measures:** Scale-invariant relationships between byte triples.

Lifts each triple of consecutive bytes to projective 2-space — the space of lines through the origin in 3D, where (x,y,z) and (kx,ky,kz) represent the same point. Distances are measured by the Fubini-Study metric (arccos of the absolute inner product). The geometry then samples cross-ratios — the fundamental projective invariant of four collinear points — and checks for near-collinearity in ℙ².

### Metrics

**cross_ratio_std** — Standard deviation of the cross-ratio across sampled quadruples of points. Logistic edge-of-chaos (40.3) has the widest cross-ratio distribution: its dynamics create point configurations spanning the full range of the Mobius invariant. Euler totient (38.3) and critical circle map (34.4) are close behind. Logistic period-3 scores 0.0 (degenerate — too few distinct projective points). High cross_ratio_std means the data's projective geometry is wild and varied; low means it's constrained to a narrow family of configurations.

**collinearity** — Fraction of sampled triples that are nearly collinear in ℙ² (parallelepiped volume < 0.1). Solar wind speed, speech "nine", and tidal gauge all score 1.0 (every triple is nearly collinear — their smooth waveforms keep consecutive triples in a low-dimensional projective subspace). Poisson spacings score 0.30 (most triples span the full ℙ²). High collinearity means the signal's projective image is nearly one-dimensional — it traces a curve rather than filling the plane.

**mean_distance** — Average Fubini-Study distance between nearby projective points. Collatz parity (0.937) scores highest: its binary sequence creates maximally separated projective points. L-System Dragon (0.914) is close behind. Logistic period-3 scores 0.0 (all triples project to the same point). High mean distance means successive triples jump across projective space; low means they cluster.

### When it lights up

Projective ℙ² is the framework's scale-invariance detector. The cross-ratio is invariant under all projective transformations (scaling, rotation, shear, perspective) — so cross_ratio_std measures how much the data's structure varies beyond what any single projective transformation could explain. In the symmetry view, the collinearity metric separates smooth waveforms (collinearity near 1.0, data traces a projective curve) from complex or binary sources (low collinearity, data fills ℙ²). This is complementary to the Fubini-Study distance metrics in Spherical S², which operate on the same underlying point-on-sphere representation but measure different invariants.

---

## Symplectic

**What it measures:** Phase-space area and trajectory stationarity.

Pairs each value with its discrete derivative (the difference to the next value) to construct a phase portrait — the (position, momentum) representation that Hamiltonian mechanics uses. The symplectic area form (shoelace formula on the phase portrait) measures how much area the trajectory encloses. Windowed area CV then checks whether this area is stable across the signal or varies in bursts.

### Metrics

**total_area** — Absolute area enclosed by the phase-space trajectory (shoelace formula, then abs). L-System Dragon (3,072) and Thue-Morse (2,730) score highest: their binary dynamics create phase portraits with large, non-self-canceling loops. Rule 110 (2,291) is close behind. Logistic period-2 scores 0.0 (its phase portrait collapses to a line segment — alternating values have constant derivative, so q and p are locked together). High total area means the trajectory genuinely sweeps through phase space; zero means it's confined to a lower-dimensional manifold.

**windowed_area_cv** — Coefficient of variation of the shoelace area computed over non-overlapping windows. Devil's staircase (11.1) dominates: its long constant plateaus (zero area) punctuated by sudden jumps (large area) create extreme variability. Accelerometer sitting (5.57) and rainfall (5.57) tie for second — both are bursty signals with long quiet intervals. Logistic period-2 scores near 0 (constant zero area in every window). This is the framework's most direct stationarity test: low CV means the dynamics look the same throughout; high CV means there are transient events or regime changes.

**q_spread** — Standard deviation of the position coordinate in phase space. Thue-Morse and L-System Dragon both score 0.50 (maximum spread for binary data — their values span the full [0,1] range uniformly). Collatz gap lengths scores 0.005 (nearly zero — the gap lengths are confined to a narrow range). This is a simple amplitude measure, but it gains meaning in the symplectic context: data with high q_spread but low total_area has oscillatory dynamics (the trajectory retraces itself), while data with both high q_spread and high total_area has genuinely 2D phase-space dynamics.

### When it lights up

Symplectic's windowed_area_cv was one of the key metrics in the negative re-evaluation study (2026-03-05), contributing to the reclassification of Standard Map and Arnold Cat as positive detections. Its power is in detecting non-stationarity: bursty signals (devil's staircase, rainfall, earthquake) have high CV because their quiet periods produce near-zero windowed area while their active periods produce large area. This is a different kind of non-stationarity detection than Nonstationarity geometry's vol_of_vol (which measures descriptor trajectory burstiness) — Symplectic's version is rooted in the physical concept of phase-space area, making it particularly natural for signals with Hamiltonian or conservative dynamics.

## Higher-Order Statistics

**What it measures:** The statistical fingerprint beyond mean and variance — skewness, kurtosis, and nonlinear frequency coupling.

Projects pairs of consecutive values onto 20 directions in 2D and computes the skewness and kurtosis along each projection. Also computes permutation entropy (how many ordinal patterns appear) and bicoherence (whether triplets of frequencies are phase-locked). Together these detect non-Gaussianity from four independent angles: directional asymmetry, tail heaviness, ordinal structure, and nonlinear coupling.

### Metrics

**bicoherence_max** — Are any frequency triplets phase-locked? Measures whether the phases of frequencies f1, f2, and f1+f2 are correlated across segments, normalized by sqrt(P1*P2*P3) — a non-standard normalization that is unbounded above 1.0 (standard bicoherence uses a different denominator). Devil's Staircase scores 7.92: its self-similar staircase structure creates strong quadratic phase coupling between harmonics. Quantum Walk (5.54) is next — interference creates phase-locked frequency relationships. Linear systems and noise score near zero because their phases are independent.

**kurt_max** — Maximum projection kurtosis across 20 directions. Heavy tails in any direction push this up. Rainfall dominates at 1373: most hours are dry (near zero), but rare downpours create extreme outliers. Forest Fire (886) has similar bursty dynamics. Gaussian noise scores near 0 (the excess kurtosis baseline — Fisher's definition, not Pearson's). Constants and periodic orbits score 0.0 (degenerate single-point projections).

**perm_entropy** — What fraction of ordinal patterns are observed, weighted by their frequencies? Pi Digits and XorShift32 score 0.999 (all 120 possible 5-element rank orderings appear, roughly equally). Forest Fire scores 0.008: its avalanche dynamics visit almost no ordinal patterns. This is encoding-invariant — it ignores absolute values entirely.

**perm_forbidden** — What fraction of ordinal patterns never appear? Logistic Period-2 scores 0.983 (only 2 of 120 patterns observed — alternating between two values constrains the rank structure almost completely). Noise, prime gaps, and pink noise score 0.0 (all patterns appear). This is a zero-training determinism detector: forbidden patterns are the fingerprint of low-dimensional dynamics.

**skew_mean** — Average absolute skewness across projections. Rainfall (20.8) and Forest Fire (8.32) have the most asymmetric distributions: rare large events create long right tails in nearly every direction. OpenBSD ELF (6.28) is skewed because binary executables have non-uniform byte distributions. Symmetric distributions (Gaussian noise, sine waves) score near zero.

### When it lights up

Higher-Order Statistics separates signals that look identical under mean-and-variance analysis. Two sources with the same entropy and autocorrelation can have completely different kurtosis profiles if one has heavy tails and the other doesn't. Bicoherence is the only metric in the framework that detects nonlinear frequency coupling — linear filters destroy it, so high bicoherence is a direct signature of nonlinear dynamics. In the atlas, the combination of kurt_max and perm_forbidden separates bursty processes (high kurtosis, low forbidden) from low-dimensional chaos (moderate kurtosis, high forbidden).

---

## Attractor Reconstruction

**What it measures:** The dimension and divergence rate of the signal's phase-space attractor.

Delay-embeds the time series (at dimensions 2 through 8 for correlation dimension, up to 10 for Lyapunov exponent) using the first zero-crossing of the autocorrelation as the lag. In this reconstructed space, applies Grassberger-Procaccia to estimate the correlation dimension D2 (how many dimensions the attractor fills) and Rosenstein's method to estimate the maximum Lyapunov exponent (how fast nearby trajectories diverge).

### Metrics

**correlation_dimension** — How many effective dimensions does the attractor fill? Collatz Stopping Times leads at 4.22: its complex branching dynamics fill a roughly 4D manifold. Neural Net Dense (4.02) and ECG Supraventricular (4.01) are similarly high-dimensional. The Lorenz attractor sits around 2.05 (textbook D2 for the Lorenz system). Constants and Fibonacci Word score 0.0 — degenerate point or 1D attractors.

**d2_saturation** — Does the dimension estimate converge as you increase the embedding dimension? Champernowne (0.997) and Triangle Wave (0.997) saturate immediately — their low intrinsic dimension is captured at the lowest embedding. Collatz Parity scores 0.0 (dimension never converges, suggesting the signal doesn't live on a finite-dimensional manifold). High saturation means you can trust the D2 estimate; low saturation means the attractor is higher-dimensional than the embedding can capture.

**filling_ratio** — What fraction of the embedding space does the trajectory actually visit? Dice Rolls (0.994) and XorShift32 (0.982) fill almost all of it — they're space-filling in delay coordinates. Logistic Period-2 scores 0.002 (the trajectory visits only two points in any embedding). This separates low-dimensional attractors from space-filling noise.

**lyapunov_max** — The maximum Lyapunov exponent: how fast do nearby trajectories diverge? Positive means chaos (exponential separation), zero means periodic or quasiperiodic, negative means contracting. Henon Near-Crisis leads at 0.106: it's on the edge of destruction, with maximum divergence. Financial returns (Nikkei -0.003, NYSE -0.0003) are slightly negative — they're mean-reverting on short timescales.

### When it lights up

Attractor Reconstruction provides the classic chaos diagnostic: positive Lyapunov with finite correlation dimension means deterministic chaos. The framework uses it alongside Gottwald-Melbourne (which doesn't need embedding) as a cross-check. In the atlas, correlation_dimension separates the dynamical view's low-dimensional chaos cluster (D2 = 2-4: Lorenz, Rossler, Henon) from noise (D2 saturates at embedding dimension) and periodicity (D2 = 1). The filling_ratio metric complements this by detecting whether the trajectory is confined to a manifold or fills the space uniformly.

---

## Gottwald-Melbourne

**What it measures:** Is the signal chaotic or regular? A binary classification with a confidence estimate.

Constructs 2D translation variables p(n) and q(n) by accumulating the signal modulated by cosine and sine at random frequencies. For chaotic signals, (p, q) performs a random walk whose mean square displacement grows linearly, giving K = 1. For regular signals, (p, q) stays bounded, giving K = 0. The median over 10 random frequencies provides robustness. No embedding dimension or lag estimation required.

### Metrics

**k_statistic** — The chaos indicator: 0 = regular, 1 = chaotic. Rossler Hyperchaos scores 1.0 (textbook chaos). Ocean Wind (0.999) and Gzip (0.999) score nearly 1.0 — both appear chaotic to this test. All logistic periodic orbits score exactly 0.0 (perfectly regular). The key limitation: noise also gives K = 1 because uncorrelated random walks have linearly growing MSD. This test distinguishes chaos from periodicity, not chaos from noise.

**k_variance** — Interquartile range of K across the 10 random frequencies. Low variance means the classification is confident. L-System Dragon Curve scores 0.884 (highest variance — the test can't make up its mind because the signal is at the boundary between order and chaos). Champernowne (0.831) and De Bruijn (0.769) are similarly ambiguous. Constants and periodic orbits have zero variance (confidently regular).

### When it lights up

Gottwald-Melbourne is the framework's only embedding-free chaos test. Its strength is that it requires no parameter tuning — no embedding dimension, no lag, no threshold selection. Its weakness is that it cannot distinguish chaos from noise (both give K = 1). In practice it's most useful paired with Attractor Reconstruction: K = 1 with finite D2 means chaos; K = 1 with unsaturated D2 means noise. The k_variance metric is uniquely informative for edge-of-chaos signals — high variance at K = 0.5 detects intermittency and crisis transitions that the binary K statistic misses.

---

## Nonstationarity

**What it measures:** How the signal's local geometric character changes over time.

Computes a 5D descriptor (entropy, lag-1 autocorrelation, variance, kurtosis, permutation entropy) on sliding windows and tracks its trajectory through descriptor space. The other geometries compute static, full-sequence summaries. This one measures the derivative: how fast the local character is changing, how bursty that change is, and how much of the descriptor space the trajectory explores.

### Metrics

**metric_volatility** — Mean speed of movement through descriptor space (z-scored). Triangle Wave (3.31) and Clipped Sine (3.27) score highest because their periodic structure creates rapid, repeated transitions between distinct geometric regimes. Logistic Period-5 (2.95) is similar. Devil's Staircase scores 0.0: its local geometry is constant within each plateau, and the jumps between plateaus are too rare to raise the mean speed.

**regime_persistence** — How long do geometric regimes last? Measured by the autocorrelation decay time of the descriptor trajectory. Rossler Hyperchaos, Quantum Walk, and Lotka-Volterra all score 1.0 (maximum persistence — once they enter a geometric regime, they stay). Rossler Attractor scores 0.033 (regimes change rapidly as the trajectory spirals between lobes). High persistence signals piecewise-stationary dynamics.

**trajectory_dim** — PCA participation ratio of the descriptor cloud, normalized by 5. How many independent descriptor axes does the trajectory use? Zipf Distribution (0.904) and Noisy Sine (0.896) explore nearly the full 5D space. Devil's Staircase scores 0.0 (the trajectory is confined to a single point in descriptor space). High trajectory_dim means the signal's local geometry changes in multiple independent ways simultaneously.

**vol_of_vol** — Coefficient of variation of the descriptor speed. Is the rate of geometric change itself stable or bursty? Gaussian Collatz (2.08) and Thue-Morse (2.01) score highest: their geometric changes come in bursts separated by calmer intervals. This is the actual "volatility of volatility" — a second-order nonstationarity measure. Van der Pol (1.74) scores high because its relaxation oscillations create alternating fast and slow geometric evolution.

### When it lights up

Nonstationarity detects regime switching and concatenation that static metrics miss entirely. A signal made by splicing together two different sources will score high on vol_of_vol (bursty regime changes) and trajectory_dim (multiple descriptors change) while possibly looking unremarkable to any single static geometry. In the atlas, regime_persistence separates the dynamical view's "coherent chaos" cluster (Rossler Hyperchaos, Lotka-Volterra: chaotic but geometrically stable) from "incoherent chaos" (Rossler Attractor: chaotic and geometrically unstable).

---

# Scale Geometries

---

## Logarithmic Spiral

**What it measures:** How uniformly the signal's growth structure fills angular space.

Maps the time series onto a logarithmic spiral in polar coordinates: each sample advances the angle by a data-dependent step (larger values rotate faster) while the radius grows exponentially. The resulting spiral path reflects multiplicative structure in the data.

### Metrics

**angular_uniformity** — How uniform are the angular step sizes along the spiral path? Computed as 1 minus the coefficient of variation of consecutive angular increments. 1.0 means all steps are the same size. Collatz Gap Lengths (0.984) scores highest: its angular increments are nearly constant in magnitude. Accelerometer sitting (0.968) and BTC Returns (0.961) are close behind — both have broad amplitude distributions that produce consistent angular steps. Collatz Parity scores 0.292 (strongly non-uniform — the binary values create only two angular step sizes). Constants score 0.0.

### When it lights up

Logarithmic Spiral's angular_uniformity is a proxy for how well the signal's amplitude distribution fills the dynamic range. Signals with uniform or symmetric distributions achieve high uniformity; signals with degenerate or heavily skewed distributions concentrate in narrow angular sectors. It complements the distributional view by providing a geometric (rather than entropic) measure of amplitude spread. In the atlas, it separates continuous-valued signals with broad distributions (financial returns, accelerometer data) from binary or symbolic signals (Collatz parity, Morse code) along the scale view's angular axis.

---

## p-Variation

**What it measures:** How rough is the signal's path, as characterized by the critical exponent where cumulative variation transitions from finite to infinite.

Computes the sum of |increments|^p for multiple values of p. For smooth signals, all p-variations are finite. For Brownian motion, the 2-variation is finite but the 1-variation diverges. For rougher paths (Levy flights), even the 2-variation diverges. The variation index — the critical p at the transition — is a fundamental path invariant that characterizes roughness independently of amplitude.

### Metrics

**var_p2** — The 2-variation (sum of squared increments, normalized). Fibonacci Word (0.764) scores highest: its binary substitution sequence produces increments of size 1 at almost every step, maximizing the normalized quadratic variation. Thue-Morse (0.667) and Collatz Parity (0.663) are similar — all are binary-valued with frequent transitions. Persistent fBm scores 0.000005 (nearly zero — its smooth, correlated increments have tiny squared variation). This metric separates "jumpy" binary signals from "smooth" continuous ones.

**variation_index** — The estimated critical p. Accelerometer Jog, Accelerometer Walk, and Wind Speed all hit 4.0 (maximum — their paths are extremely rough, with variation diverging even at high p). Mu-law Sine scores 1.0 (smooth path with finite 1-variation — the slope transition happens at p=1). The variation index is invariant under affine transformations (shift and scale) of the signal but not under general monotone transforms.

### When it lights up

p-Variation captures path roughness in a way that's complementary to Holder Regularity. Holder measures local smoothness point by point; p-Variation measures global path roughness as a summability property. In the atlas, var_p2 strongly separates binary/symbolic signals (Fibonacci, Thue-Morse, Collatz Parity: high var_p2) from continuous oscillations (fBm, sine waves: near-zero var_p2). The variation_index provides the roughness invariant that connects the framework to stochastic analysis: Brownian motion has index 2, and deviations from 2 indicate either smoother-than-Brownian (index < 2) or rougher-than-Brownian (index > 2) dynamics.

---

## Multi-Scale Wasserstein

**What it measures:** How the signal's distribution changes across scales — is it self-similar, drifting, or scale-free?

Splits the signal into blocks at multiple scales (2, 4, 8, 16, ... blocks), computes the histogram of each block, and measures the Wasserstein (earth mover's) distance between adjacent blocks at each scale. Self-similar signals have constant Wasserstein distance across scales. Non-stationary signals show drift. Scale-free processes follow a power law.

### Metrics

**w_mean** — Mean Wasserstein distance averaged across all scales. Captures the overall level of distributional heterogeneity regardless of scale structure.

**w_std** — Standard deviation of Wasserstein distances across scales. High values mean the signal's distributional contrast is strongly scale-dependent; low values mean it's roughly constant.

**w_fine** — Average Wasserstein distance between adjacent blocks at the finest scale. Morse Code (0.394) dominates: its on/off structure creates maximally different local distributions between signal and silence segments. Symbolic Lorenz (0.260) and Sprott-B (0.194) are next — chaotic dynamics with occasional regime switches produce large local distributional variation. Periodic orbits and constants score 0.0 (adjacent blocks have identical distributions).

**w_max_ratio** — Dynamic range of Wasserstein distances across scales (max/min). Triangle Wave scores 1521: at the finest scale, adjacent blocks have similar distributions (both are local ramps), but at the coarsest scale, one half is ascending and the other descending, creating maximum distributional contrast. Standard Map Mixed (1269) and Logistic Period-3 (1118) have similarly extreme ratios because their periodic structure creates scale-dependent distributional asymmetry.

**w_slope** — Log-log slope of Wasserstein distance vs. number of blocks. Positive slope means distributional differences grow at finer scales (more blocks = more local contrast between adjacent windows). Negative slope means they grow at coarser scales (fewer, larger blocks reveal structure that fine divisions miss). De Bruijn (6.38) has the steepest positive slope: adjacent small windows differ sharply because the de Bruijn sequence cycles through all byte patterns locally. Devil's Staircase (-0.74) has the most negative slope: its long constant plateaus make adjacent fine-scale windows identical, but coarse-scale blocks straddle plateau boundaries, creating distributional contrast.

### When it lights up

Multi-Scale Wasserstein measures distributional self-similarity — something no entropy metric captures directly. Two signals with the same entropy can have completely different scale profiles: one might have constant distributions at all scales (self-similar), while the other has scale-dependent drift. In the atlas, w_slope separates sources with fine-scale local variation (De Bruijn, Fibonacci: positive slope, adjacent small windows differ) from sources with coarse-scale structure (Devil's Staircase, fBm: negative slope, distributional contrast only emerges in large blocks).

---

# Quasicrystal Geometries

---

## Penrose (Quasicrystal)

**What it measures:** Does the signal have fivefold diffraction symmetry — the hallmark of Penrose quasicrystalline order?

Computes the power spectrum and autocorrelation, then tests whether their peak structure is invariant under scaling by the golden ratio (1.618...). True Penrose/Fibonacci quasicrystals have spectral peaks at positions related by powers of the golden ratio, creating self-similar diffraction patterns with discrete Bragg peaks — the defining property that separates quasicrystals from both periodic crystals and amorphous matter.

### Metrics

**long_range_order** — Autocorrelation self-similarity at golden-ratio-scaled lags. 1.0 means the autocorrelation pattern repeats at lags related by powers of phi. Circle Map QP, Phyllotaxis, and Fibonacci QC all score 1.0. Chaotic systems score 0.0 — no long-range autocorrelation structure, let alone ratio-scaled self-similarity.

**algebraic_tower** — Multi-scale weighted coherence of detrended spectral residuals under phi scaling, with phi-specificity gap (target R² minus best null-ratio R²), cross-scale stability, and residual fit quality. Rewards signals where the phi self-similarity model is both the best fit and a good fit. Produces wide source spread for atlas discrimination. *Evolved via ShinkaEvolve atlas v1.*

### When it lights up

Penrose is the primary golden-ratio quasicrystal detector. The algebraic_tower metric provides atlas discrimination (score 0.924) while maintaining ratio specificity: the coherence gap penalizes signals that score equally well at non-phi ratios. Sources governed by the golden ratio (Fibonacci QC, phyllotaxis, critical circle map at golden-mean rotation) stand out on both metrics simultaneously.

---

## Ammann-Beenker (Octagonal)

**What it measures:** Does the signal have eightfold diffraction symmetry — the signature of Ammann-Beenker quasicrystals?

Uses the same spectral self-similarity machinery as Penrose but tests for the silver ratio (1 + sqrt(2) = 2.414...) instead of the golden ratio. Ammann-Beenker tilings fill the plane with squares and 45-degree rhombi, creating 8-fold rotational symmetry that is forbidden in periodic crystals.

### Metrics

**convergent_resonance** — Geometric mean of spectral self-coherence at Pell convergent ratios (2/1, 5/2, 12/5, 29/12) of the silver ratio. Tests whether the power spectrum repeats under scaling by the continued-fraction approximants of 1+sqrt(2). *Evolved via ShinkaEvolve.*

**convergent_profile** — Coherence cascade shape across convergent scales with quadratic log-log detrending. Returns mean - std + 0.2*slope of coherences. The quadratic detrending (vs linear or moving-average) is key: it removes the spectral slope better, letting ratio-scale correlations emerge. IQR=0.615 — strong source discrimination, from Lorenz (0.95) to noise (~0). *Evolved via ShinkaEvolve atlas v1.*

### When it lights up

Ammann-Beenker complements Penrose: where Penrose tests for golden-ratio (phi) self-similarity, AB tests for silver-ratio (1+sqrt(2)) self-similarity. The convergent_profile metric provides genuine atlas discrimination (IQR=0.615), separating signals by how self-similar their detrended spectra are under Pell-convergent scaling. Sources with smooth spectral structure (Lorenz, Sine, Van der Pol) score high; white noise and chaos score near zero. The silver ratio's continued fraction [2; 2,2,2,...] means all convergents are ratios of Pell numbers — an algebraic constraint that only true octagonal QC structure would satisfy at all scales simultaneously.

---

## Einstein (Hat Monotile)

**What it measures:** Does the signal's local structure match the Hat aperiodic monotile, and does it show chirality?

Maps the data onto a hexagonal grid path and cross-correlates the turn sequence with the projected boundary of the Hat tile (Smith et al., 2023). The Hat is the first known single tile that forces aperiodic tiling — it cannot tile the plane periodically. Critically, the Hat and its mirror image are distinct (chirality), which this geometry tests separately.

### Metrics

**chirality** — Asymmetry between left and right turns on the hex path. Wigner Semicircle (0.526) is the most left-turning signal in the atlas: its symmetric distribution creates a systematic directional bias when mapped to hex directions. Logistic Edge-of-Chaos (-0.406) turns right. Circle Map QP (-0.204) also turns right. The sign matters: positive = left bias (hex turns 1,2 where sin > 0), negative = right bias (hex turns 4,5 where sin < 0). Signals near zero (most noise, most chaos) have no directional preference. This is a genuine chirality test — it detects handedness in the data's local structure.

**hat_boundary_match** — Fraction of hex-path windows with >0.9 correlation to the Hat's boundary kernel, scaled by kernel length. Scores are universally low across the atlas: Henon-Heiles (0.019), Rossler (0.017), and Sine Wave (0.015) lead. The Hat boundary has 90-degree angles that the hexagonal grid cannot represent exactly (hex steps are 60-degree), so this metric is an approximate motif detector. The low scores indicate no source in the atlas genuinely resembles the Hat boundary in its turn structure.

### When it lights up

Einstein Hat is the newest geometry in the quasicrystal lens, added to detect the 2023 discovery of aperiodic monotiles. Its hat_boundary_match metric is limited by the hex-grid approximation (the true Hat lives on a 12-direction kite grid), but chirality provides genuine signal: it detects left/right asymmetry that no other geometry in the framework measures. In investigations, chirality discriminated between ECG lead orientations — the heart's electrical axis creates a physical handedness that the hex-path chirality captures. The metric's range from -0.406 to +0.526 in the atlas spans zero symmetrically, confirming it's not biased.

---

## Dodecagonal (Stampfli)

**What it measures:** Does the signal have twelvefold diffraction symmetry — the signature of square-triangle Stampfli tilings?

Tests for spectral self-similarity at the dodecagonal ratio (2 + sqrt(3) = 3.732...) and also checks sqrt(3) scaling, which relates to the triangle height in square-triangle tilings. Dodecagonal quasicrystals (found in Ta-Te and V-Ni-Si alloys) have the highest rotational symmetry known in real quasicrystals.

### Metrics

**dodec_phase_coherence** — Phase-aware spectral self-coherence at the dodecagonal ratio (2 + sqrt(3) ~ 3.732). Tests whether complex spectral structure (not just power) repeats under scaling by the inflation factor of square-triangle Stampfli tilings.

**z_sqrt3_coherence** — Spectral self-coherence at sqrt(3) scaling, corresponding to the triangle height/base ratio in the tiling. EEG signals show moderate scores — brain oscillation bands have frequency ratios near sqrt(3) between adjacent bands (theta/alpha/beta).

### When it lights up

Dodecagonal tests for the highest rotational symmetry known in real quasicrystals (12-fold, found in Ta-Te and V-Ni-Si alloys). The sqrt(3) coherence in EEG signals is an interesting cross-domain finding, though likely reflects harmonic relationships in neural oscillation bands rather than genuine dodecagonal order.

---

## Septagonal (Danzer)

**What it measures:** Does the signal have sevenfold diffraction symmetry — the rarest rotational order in known quasicrystals?

Tests spectral self-similarity at the septagonal ratio rho = 1 + 2*cos(2*pi/7) ~ 2.247, the largest root of x^3 - 2x^2 - x + 1 = 0. The v2 evolution discovered a triple-conjugate ensemble approach: the three roots of the minimal polynomial (rho, its conjugate, and its reciprocal) provide independent spectral probes. Sevenfold symmetry is crystallographically forbidden and extremely rare even among quasicrystals.

### Metrics

**cubic_coherence** — Combined coherence across all three roots of the septagonal minimal polynomial. Tests the algebraic identity x^3 = 2x^2 + x - 1 in spectral space. *Evolved via ShinkaEvolve.*

**z_primary** — Spectral self-coherence at the primary septagonal ratio rho ~ 2.247.

**z_conjugate** — Spectral self-coherence at the conjugate root ~ 1.802. Independent probe of the same algebraic structure.

**z_reciprocal** — Spectral self-coherence at the reciprocal root ~ 0.445. Completes the triple-conjugate ensemble.

**ratio_symmetry** — Generic spectral self-similarity under septagonal-ratio scaling. Shared QC spectral machinery.

### When it lights up

Septagonal is the most exotic geometry in the quasicrystal lens. The triple-conjugate ensemble (z_primary, z_conjugate, z_reciprocal) provides stronger specificity than single-ratio testing: all three roots must show coherence simultaneously for genuine 7-fold order. Septagonal completes the quasicrystal lens's coverage of crystallographically forbidden symmetries (5, 7, 8, 12-fold).

