# Negative Results

What the framework cannot detect, and why. These results are as important as the positive findings — they define the boundaries of geometric analysis and prevent misapplication.

## The Fundamental Limit: Random vs AES-CTR

**AES-CTR output is indistinguishable from true randomness across every approach we tested.**

This was verified exhaustively:
- All 24 individual geometries (static metrics)
- 6 block-averaging scales (multiscale)
- 36 temporal features (sliding window derivatives)
- 51 cross-geometry features (correlations, products, mutual information)
- Ollivier-Ricci curvature (graph-based)
- Higher-order statistics (3rd and 4th moments)
- 8 bit planes (bitplane decomposition)
- FFT spectral analysis

Zero significant differences survived Bonferroni correction in any of these tests. This is the correct theoretical result for a secure block cipher in CTR mode, and it validates the framework's methodology: we detect real structure but don't hallucinate structure in properly randomized data.

## PRNGs That Pass (or Nearly Pass)

A systematic study of 10 generators (`rng.py`) confirms that modern PRNGs are geometrically indistinguishable from os.urandom, while historical weak generators are massively detectable.

**Completely indistinguishable (0 sig):**

| PRNG | Category | Why It Passes |
|------|----------|--------------|
| **PCG64** | GOOD | NumPy default, passes BigCrush; 0 sig across all 233 metrics |
| **SFC64** | GOOD | Small Fast Chaotic; 0 sig |
| **SHA-256 CTR** | CRYPTO | Hash-based; cryptographically indistinguishable by design |

**Borderline (1 sig — essentially passes):**

| PRNG | Category | Detection |
|------|----------|-----------|
| **MT19937** | GOOD | 1 metric (S² × ℝ sphere_concentration, d=-1.27). Not reproducible at other sequence lengths |
| **MINSTD** | WEAK | 1 geometric metric; standard chi2 and entropy catch 2/11 — distributional bias is the weakness |

**Undetected raw but revealed by delay embedding:**

| PRNG | Category | Raw → DE2 |
|------|----------|-----------|
| **XorShift128** | WEAK | 0 → 2 sig. Thurston height_drift at both S² × ℝ and H² × ℝ (d=+1.10). Delay embedding creates phase-space pairs that expose linear correlations invisible in raw bytes |

Note: The earlier `prng.py` tested XorShift32 (which passes); `rng.py` tests the 128-bit variant which is borderline. Our framework operates at the byte level and cannot access the high-dimensional structure where many PRNG weaknesses live (e.g., MT19937's 623-dimensional equidistribution failures).

## Chaotic Maps That Look Random

| Map | Why It Passes |
|-----|--------------|
| **Standard map** | Uniformly mixing on the torus — ergodic measure equals Lebesgue measure |
| **Arnold cat map** | Same: uniformly mixing, Anosov diffeomorphism |

These maps are genuinely chaotic but produce uniform distributions in their phase space. Their Lyapunov exponents are positive (they ARE chaotic), but their invariant measure is uniform, so byte-level statistics see them as random. This is mathematically correct.

## Steganography: What Works and What Doesn't

Six embedding techniques tested across two carriers (`stego_deep.py`):

**Detectable at raw byte level:** PVD (42 sig metrics on texture, detectable at 10% rate) and spread spectrum (25 sig, detectable at 25% rate). These techniques modify pixel values aggressively enough to shift geometric signatures. Delay embedding (τ=2) amplifies the signal further.

**Barely detectable:** LSB replacement and LSB matching show 1-6 significant metrics at 100% embedding rate only. LSBMR is similarly weak. The ±1 byte-level perturbations are near the noise floor for most geometries. Fisher Information via LSB-correlation extraction catches LSB replacement at d=1.06, but only at full rate.

**Completely invisible:** Matrix embedding (Hamming(7,3) syndrome coding) produces 0 significant metrics at all rates on both carriers. By flipping at most 1 LSB per 7-pixel block, it stays below the detection floor of every geometry.

**Bitplane extraction does NOT help:** Contrary to initial claims, `bitplane_extract(data, 0)` yields 0 significant detections for ALL techniques including LSB replacement. The 8x sample size reduction (4000→500 bytes) destroys statistical power. The previously reported d=1166 was not reproducible. See `investigations/1d/stego_deep.py`.

## Financial Models

| Model | vs IID Normal | Why |
|-------|:------------:|-----|
| **GBM** (Geometric Brownian Motion) | d = 0 | GBM returns ARE IID normal by construction |
| **Ornstein-Uhlenbeck** | d = 0 | Mean-reversion is too subtle after byte quantization |

Other financial models (GARCH, regime-switching, jump-diffusion) ARE detectable, but all are geometrically closer to random than to chaos. Markets are not chaotic systems — they're noisy with occasional structure.

## Near-Identical Rules

Some systems that differ in definition produce indistinguishable geometric signatures:

| Pair | Why Indistinguishable |
|------|--------------------|
| **Game of Life ≈ HighLife** | Differ only in birth rule B6; statistically identical emergent behavior |
| **Kruskal ≈ Aldous-Broder** | Both produce uniform spanning trees (same distribution, different algorithms) |
| **Sandpile 10k ≈ 50k iterations** | Both have reached SOC steady state; additional iterations don't change the statistics |

These "failures" are actually correct: the framework detects that these systems produce the same statistical structure, which is the ground truth.

## EEG Golden Ratio Architecture (φⁿ Hypothesis)

**The claim that EEG spectral peaks follow f₀ × φⁿ organization anchored at f₀ = 7.5 Hz (Pletzer et al. 2010, Lacy 2026) is not supported by our independent analysis.**

Data: PhysioNet eegmmidb v1.0.0, 109 subjects, 64 channels each, eyes-closed resting baseline. 86,337 spectral peaks extracted via Welch PSD + median-filter 1/f subtraction.

Key methodological improvement: **phase-rotation permutation null**. The raw enrichment score is biased by alpha-peak dominance. Adding a random offset δ ~ U(0,1) to all lattice phases preserves the peak distribution shape but destroys alignment with f₀, giving a proper baseline.

| Test | Result | Implication |
|------|--------|-------------|
| **D1: Lattice enrichment** | obs=+0.309, null=-0.003±0.267, **p=0.213** | Not significant after null correction |
| **D2: Ratio specificity** | φ ranks 5th of 12; π, 2, e, 5/3 all higher excess | φ is not uniquely preferred |
| **D3: f₀ anchoring** | f₀=7.5 does not reach 95% null threshold; best f₀=3.04 Hz | Claimed anchor is not optimal |
| **D4: 2D (f₀, r) heatmap** | Global optimum at (4.94, 3.92); claimed (7.5, φ) in mediocre region | Joint parameter space doesn't peak at claim |
| **D5: Per-band** | Only gamma significant (p=0.024); alpha fails (p=0.277) | Effect is band-specific, not universal |
| **D6: Per-subject** | +0.305 excess, t=13.89, p<0.0001, 101/109 positive | Real signal, but not φ-specific |
| **D8: Surrogate** | p=0.127, enrichment survives IAAFT | Spectral property, not nonlinear |

**What IS real:** EEG spectral peaks show reproducible geometric lattice alignment across subjects — the per-subject test is overwhelming (t=13.89). There IS non-trivial spectral organization in resting-state EEG.

**What ISN'T supported:** The specific claim that this organization follows a golden ratio lattice anchored at 7.5 Hz with zero free parameters. The same data is equally or better described by ratios 2, e, π, or 5/3, and the optimal anchor frequency is not 7.5 Hz. The 2D heatmap shows the claimed (f₀, r) pair sitting in a mediocre region of the parameter landscape.

**Why the original result may have appeared stronger:** The original analysis tests enrichment/depletion at φⁿ positions without comparing against alternative ratios or using a phase-rotation null. Without these controls, the alpha peak's proximity to a φ attractor (9.6 Hz) produces apparent enrichment. The "< 2% error" claim reflects the goodness-of-fit of a geometric sequence to broadly-spaced peaks — any ratio in [1.5, 1.7] achieves comparable fit with appropriately chosen f₀.

See `investigations/1d/eeg_phi.py`. Figure: `docs/figures/eeg_phi.png`.

## Byte Quantization Limits

Converting continuous data to uint8 (0-255) loses information. Some structures are destroyed:

| Structure | Why Lost |
|-----------|---------|
| GARCH volatility clustering | Variance-of-variance is a 4th-order effect; uint8 preserves only 1st and 2nd order well |
| Cepstral features | Double-log transformation amplifies quantization noise |
| IEEE 754 float structure | Raw float64 bytes have exponent/mantissa structure that confounds temporal analysis |

**Mitigation**: Use `data_mode='auto'` for float data, or `encode_float_to_unit()` for explicit control over the quantization.
