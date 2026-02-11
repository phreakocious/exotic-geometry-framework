# AI Text Detection (Synthetic vs Human)

## Hypothesis
AI-generated text (especially from earlier or simpler models, or highly constrained "corporate" outputs) lacks the long-range geometric complexity of human literature. While it may mimic the *probability distribution* (entropy) of language, it fails to replicate the *manifold structure*—the way narrative arcs, complex sentence variations, and thematic recurrences wind through the geometric space of byte sequences.

## Experiment
- **Human Data:** "Alice's Adventures in Wonderland" (Lewis Carroll), ~50KB sample.
- **AI Data:** Synthetic "Corporate AI" text generated via a constrained grammar (resembling bad LLM output: high repetition, transition-heavy, low vocabulary).
- **Method:** 24 Exotic Geometries (1D Byte Stream Analysis).

## Results
**45 significant metrics found (>50% difference).**

### Top Differentiators

| Geometry | Metric | Human Val | AI Val | Interpretation |
|----------|--------|-----------|--------|----------------|
| **Sol (Thurston)** | `path_length` | 303,074 | 201 | **d ≈ 200%**. Human text explores the exponentially warping Sol manifold deeply; AI text is trapped in a tiny repetitive loop. |
| **Clifford Torus** | `winding_number` | -158.5 | 0.13 | **d ≈ 200%**. Human text "winds" around the torus (cycles of narrative/grammar) non-trivially; AI text is topologically flat. |
| **Heisenberg (Nil)** | `z_variance` | 6.45e8 | 5.82e6 | **d ≈ 196%**. The "twist" or non-commutativity of human text is 100x stronger. (A then B ≠ B then A). |
| **Tropical** | `unique_slopes` | 16 | 8 | **d = 67%**. Human text has richer piecewise-linear structure (more "slopes" in the tropical semiring). |
| **Fisher Information** | `det_fisher` | ~1e56 | ~1e60 | **d ≈ 200%**. Massive difference in the information geometry curvature. |

## Conclusion
Synthetic AI text is **geometrically distinct** from human literature. It fails to fill the "volume" of exotic spaces (Sol, Heisenberg) and lacks the topological "winding" of natural language. 

## Next Steps
- Replace synthetic AI data with real GPT-4 / Claude 3 outputs.
- Test "Human-written Corporate Speak" vs "AI-written Creative Fiction" to control for genre.
