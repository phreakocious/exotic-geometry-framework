# Proposal: Structure Atlas Geometry Guide & Encyclopedia

**Date:** 2026-03-25
**Status:** Tier 1 complete (8/54 geometries drafted)
**Prototype:** `docs/geometry-guide-prototype.md`

## The Opportunity

Nobody has run 54 exotic geometries against 211 data sources from 16 domains. The atlas JSON contains cross-domain structural connections that don't exist anywhere else on the internet — earthquake P-waves clustering with ECG arrhythmias, DNA looking like chaos maps, financial returns sharing ordinal structure with viral genomes. This is original scientific content, not Wikipedia summaries.

The atlas viewer is a single-page JS app. Search engines see an empty body tag. The SEO injection (`tools/inject_seo.py`) adds crawlable source lists, but the real content gap is **explanatory pages** that turn the atlas from a visualization into a reference.

## Content Strategy

### What NOT to do

- Don't write Wikipedia-style mathematical definitions. The audience can find those.
- Don't rename things to sound fancier ("Atmospheric Velocity" instead of "Wind Speed"). Keep common names, add science after.
- Don't create thin domain pages (16 pages with 10-15 bullet points each). The interesting content is cross-domain.

### What TO do

**Explain through contrast, not definition.** Every geometry is a question asked of data. The answer is a gradient from low to high, with specific atlas sources as landmarks. The atlas results *are* the explanation.

**Format per geometry (prototyped and validated):**

1. **What it measures** — One sentence, no math, just the question it asks of data
2. **How it works** — One paragraph, conceptual mechanism. No formulas. Analogies to physical intuition.
3. **Metrics as gradients** — Each metric is a spectrum from low to high, with named atlas sources and their scores as landmarks. "Fibonacci word scores 1.0 (maximally frustrated)" is more informative than Ising model exposition.
4. **"When it lights up"** — Which investigations found this geometry most discriminating, why, and what the effect size was. Ties the geometry to real scientific findings.

**Example (Boltzmann `frustration`):**
> The Fibonacci word scores 1.0 (maximally frustrated): the golden ratio creates irreconcilable correlations at every scale. Pink noise scores 0.0 (all correlations agree). White noise scores ~0.5 (random constraints, half frustrated by chance). Quantum walk probability amplitudes also hit 1.0 — a different mechanism (interference) producing the same geometric signature.

This paragraph contains no formulas but communicates exactly what frustration detects, how to interpret values, and reveals a non-obvious connection (Fibonacci ↔ quantum walk). Nobody else can write this because nobody else has the atlas data.

### Content tiers

**Tier 1 — 8 encoding-invariant geometries (ordinal view)** ✓
Priority because they're the ones whose results are physically meaningful regardless of data encoding. All 8 drafted: Boltzmann, Hölder Regularity, Recurrence Quantification, Ordinal Partition, Heisenberg (centered), Spectral Analysis, D4 Triality, Inflation (Substitution).

**Tier 2 — High-discrimination geometries**
The ones that dominate investigation results: Multifractal Spectrum, Predictability, Information Theory, Cayley, Tropical, Zariski, Julia Set, Lorentzian. These geometries carry the atlas's strongest scientific findings (seismic P-wave d=9.25, eclipse VLF d=1.19).

**Tier 3 — Remaining geometries**
Distributional (Wasserstein, Zipf, Cantor), topological (Hyperbolic, Persistent Homology, Mandelbrot), symmetry (E8, G2, Spherical, SL(2,R)), etc. Important for completeness but lower SEO priority.

### Cross-domain connection pages

The highest-value content is the cross-domain connections. Format:

> **Impulsive Transients** — Earthquake P-waves, ECG ventricular beats, bearing inner race faults. Three domains, one ordinal signature: sudden recurrence structure collapse in a quasi-stationary background. Top discriminators: Hölder hurst_exponent (d=9.06), Recurrence determinism, Ordinal Partition forbidden_transitions.

These pages don't exist anywhere. They emerge naturally from the ordinal nearest-neighbor analysis.

### Investigation summaries

76+ investigation writeups in `docs/investigations/` as markdown. Each is a complete scientific analysis. These should be formatted as standalone pages with links to the atlas source positions and the relevant geometry guide entries. They are the "proof" that the geometry descriptions are grounded in real results.

## Technical Approach

### Generation

Python script (not Node.js — same toolchain as the rest of the project). Reads `atlas/structure_atlas_data.json` and geometry guide source content. Outputs static HTML.

**Input:**
- `atlas/structure_atlas_data.json` — source profiles, metric extremes, neighbors, clusters
- `docs/geometry-guide/` — hand-written geometry descriptions (markdown, one per geometry)
- `docs/investigations/` — existing investigation writeups
- `exotic_geometry_framework.py` — geometry catalog metadata

**Output:**
- `atlas/guide/` — static HTML pages
- `atlas/guide/index.html` — master index
- `atlas/guide/geometries/<name>.html` — one page per geometry
- `atlas/guide/connections/<topic>.html` — cross-domain connection pages

### Deep links

Every atlas source mentioned in a geometry page links to `?source=Name` in the atlas viewer. The atlas already parses URL parameters for view selection; adding source focus is a small patch.

Every geometry page links to the ordinal view when relevant, showing which geometries survive re-encoding.

### Automation

The geometry descriptions are hand-written (the "gradient" examples need human judgment). But the metric extremes, source rankings, and cross-domain neighbor lists are auto-generated from the JSON. The build script merges hand-written content with auto-generated data tables.

After each atlas rebuild:
1. `python tools/inject_seo.py` — updates meta tags and crawlable content block
2. `python tools/build_guide.py` — regenerates guide pages from JSON + markdown sources

### Design

Match the atlas aesthetic (dark theme, Exo 2 + JetBrains Mono, glass cards, neon accents). The guide pages are static HTML — they load instantly and are fully crawlable. No JS required for content.

## What's Already Done

- [x] Ordinal view implemented (8 encoding-invariant geometries, `encoding_invariant` property)
- [x] 211 sources including 5 transitional gap-fillers
- [x] SEO injection script (`tools/inject_seo.py`) — crawlable content block in atlas HTML
- [x] Meta tags updated with current counts and compelling descriptions
- [x] JSON-LD structured data added
- [x] Prototype content for 8 geometries (`docs/geometry-guide-prototype.md`)
- [x] Metric extremes data extracted for all target geometries

## What's Next

1. ~~**Finish Tier 1 content**~~ ✓ — All 8 encoding-invariant geometries drafted
2. **Atlas `?source=` deep link** — Patch atlas viewer to focus on a named source from URL
3. **Build script skeleton** — `tools/build_guide.py` reading markdown + JSON
4. **Tier 2 content** — Write 8 high-discrimination geometry descriptions
5. **Cross-domain connection pages** — Extract from ordinal NN analysis
6. **Investigation page formatting** — Convert markdown investigations to guide HTML
7. **Tier 3 content** — Remaining geometries

## Key Principles

- **Common names, add the science.** "Wind Speed (Jena)" not "Atmospheric Velocity (Jena)."
- **Explain through atlas examples, not definitions.** The atlas data IS the explanation.
- **Every claim has a source score.** "Fibonacci scores 1.0" not "quasicrystals have high frustration."
- **Cross-domain connections are the differentiator.** Nobody else has this content.
- **Encoding invariance is the quality bar.** Highlight which results survive re-encoding.
- **No sycophancy, no hype.** State what the framework detects and what it doesn't. Include caveats.
