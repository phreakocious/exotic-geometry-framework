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

## PRNGs That Pass

| PRNG | Why It Passes |
|------|--------------|
| **MT19937** (Mersenne Twister) | 623-dimensional equidistribution; byte-level analysis can't access the high-dimensional correlations |
| **XorShift32** | Simple but sufficient scrambling for byte-level randomness |
| **MINSTD** (Park-Miller) | Multiplicative LCG with good constant; high bits are well-mixed |

Note: These PRNGs have known weaknesses detectable by other means (e.g., MT19937 fails TestU01's BigCrush). Our framework operates at the byte level and cannot access the dimensional structure where these weaknesses live.

## Chaotic Maps That Look Random

| Map | Why It Passes |
|-----|--------------|
| **Standard map** | Uniformly mixing on the torus — ergodic measure equals Lebesgue measure |
| **Arnold cat map** | Same: uniformly mixing, Anosov diffeomorphism |

These maps are genuinely chaotic but produce uniform distributions in their phase space. Their Lyapunov exponents are positive (they ARE chaotic), but their invariant measure is uniform, so byte-level statistics see them as random. This is mathematically correct.

## Steganography at Byte Level

LSB steganography (flipping the least significant bit to embed messages) is **completely invisible** when analyzing full bytes. Changing the LSB changes a byte value by at most 1 (e.g., 127 → 126 or 128). This ±1 perturbation is far below the noise floor of any geometric metric operating on 0-255 byte values.

**The fix**: Use `bitplane_extract(data, plane=0)` to isolate the LSB plane, then analyze the resulting bit stream. This transforms stego from undetectable (d ≈ 0) to massively detectable (d = 1166). See `investigations/1d/stego.py`.

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

## Byte Quantization Limits

Converting continuous data to uint8 (0-255) loses information. Some structures are destroyed:

| Structure | Why Lost |
|-----------|---------|
| GARCH volatility clustering | Variance-of-variance is a 4th-order effect; uint8 preserves only 1st and 2nd order well |
| Cepstral features | Double-log transformation amplifies quantization noise |
| IEEE 754 float structure | Raw float64 bytes have exponent/mantissa structure that confounds temporal analysis |

**Mitigation**: Use `data_mode='auto'` for float data, or `encode_float_to_unit()` for explicit control over the quantization.
