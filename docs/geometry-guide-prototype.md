# Geometry Guide — Prototype

Draft content for 6 encoding-invariant geometries. Each entry explains what the geometry detects through contrast and atlas examples, not mathematical definitions.

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
