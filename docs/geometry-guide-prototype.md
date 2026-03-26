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

## Holder Regularity

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

## Heisenberg (centered)

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

**triality_invariance** — How symmetric is the root distribution under the triality automorphism? 1.0 means the distribution is unchanged by the permutation. Critical circle map scores 0.63 (highest in the atlas): its dynamics at the golden-mean rotation number create 4-byte patterns with near-perfect triality. Circle map quasiperiodic (0.62) and phyllotaxis (0.62) are close behind — both driven by irrational rotations. White noise scores 0.49 (close to the theoretical expectation for uniform random root assignment). Logistic period-2 scores exactly 0.0: alternating between two values creates a degenerate root pattern with no triality.

**diversity_ratio** — What fraction of the 24 roots are actually used? Solar wind, speech, and PRNG outputs reach 0.50 (12 of 24 roots). Logistic period-4 uses just 1 root (0.04). High diversity means the data explores the full D4 geometry; low diversity means the dynamics are confined to a low-dimensional subspace of the root system.

**normalized_entropy** — Shannon entropy of the root usage distribution, normalized by the maximum. Nikkei returns (0.64) and DNA (0.64) have the most uniform root distributions — their 4-byte structure genuinely samples the full D4 polytope. Periodic orbits score 0.0 (all weight on one root).

**alignment_mean** — How well do the 4-byte windows align with the nearest root? Langton's Ant (1.90) and circle map (1.89) produce windows that sit almost exactly on a root direction. Constants score 0.0 — degenerate windows that don't align with any root. This measures how "crystalline" the data's 4-byte structure is in D4 coordinates.

### When it lights up

D4 triality is encoding-invariant: the root projection depends on the relative order of the 4 bytes, not their absolute values. Its distinctive contribution to the ordinal view is detecting 4-byte rotational symmetry that the other ordinal geometries (which operate on pairs or 5-grams) miss. The critical circle map / quasiperiodic cluster at the top of the triality ranking connects to the golden ratio: irrational rotations produce 4-byte patterns that are maximally balanced across the triality orbits. This is a structural echo of the fact that D4 triality and quasicrystalline order are both connected to the exceptional algebra structure of Spin(8).

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

# Tier 2 — High-Discrimination Geometries

These 8 geometries dominate the atlas's strongest scientific findings. They are not encoding-invariant — their metrics depend on the actual byte values, not just rank order.

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

**causal_order_preserved** — Fraction of event pairs that are timelike-separated (causally ordered). Tidal gauge, projectile, and damped pendulum score 1.0 — smooth, slow-varying signals where each value follows causally from the previous one. Logistic edge-of-chaos scores 0.0 (every successive value is an acausal jump). This is the framework's most direct "smoothness vs. jumpiness" metric.

**spacelike_fraction** — Fraction of pairs that are acausally separated. Logistic period-3 (0.18) and Chirikov standard map (0.16) score highest — their dynamics make large jumps between successive values. Lorenz and Rössler score 0.0: despite being chaotic, their continuous-time dynamics produce smooth trajectories in which consecutive samples are always causally close.

**lightlike_fraction** — Fraction of pairs exactly on the light cone (|Δx| = |Δt|). Almost always near zero because exact equality is measure-zero for continuous data. Prime gaps (0.006) score highest: many consecutive prime gaps differ by exactly 1, landing on the light cone. This is a curiosity more than a useful discriminator.

### When it lights up

Lorentzian geometry captures something subtler than amplitude variance or entropy: the "speed" of the signal relative to its sampling rate. A high-frequency oscillation sampled slowly appears spacelike (large jumps); the same oscillation sampled fast appears timelike (smooth evolution). This makes Lorentzian metrics sensitive to the relationship between the signal's characteristic timescale and the observation timescale — a property relevant to aliasing detection and sampling adequacy. In the atlas, causal_order_preserved separates the distributional view's smooth oscillator cluster (C1) from the chaotic/noise clusters (C4, C5) along a "causality axis" orthogonal to the entropy axis.
