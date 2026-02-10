#!/usr/bin/env bash
#
# spawn_gemini.sh — Spawn Gemini CLI to write an investigation.
#
# Runs Gemini in the project directory (trusted, so write_file works).
# Gemini writes ONLY to investigations/{mode}/ — the prompt constrains it.
#
# Usage:
#   ./tools/spawn_gemini.sh "topic description"
#   ./tools/spawn_gemini.sh "music theory: MIDI byte stream key signatures"
#   ./tools/spawn_gemini.sh                          # Gemini picks the topic
#   ./tools/spawn_gemini.sh --2d "reaction-diffusion patterns"
#
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="1d"

# Parse flags
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --2d) MODE="2d"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

TOPIC="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================"
echo "  Exotic Geometry — Gemini Investigation Spawner"
echo "================================================"
echo "  Topic:    ${TOPIC:-<Gemini will choose>}"
echo "  Mode:     ${MODE}"
echo "  Project:  ${PROJ_ROOT}"
echo "================================================"

# Build the prompt
if [[ -n "$TOPIC" ]]; then
    PROMPT="You are writing a new investigation for the Exotic Geometry Framework.

READ the file GEMINI.md FIRST — it contains all the rules, the Runner API, and a complete template.

You can also read existing investigations in investigations/ for reference.

Topic: ${TOPIC}
Mode: ${MODE} (use Runner(name, mode='${MODE}'))

Write the investigation script to: investigations/${MODE}/<descriptive_name>.py

CRITICAL RULES (from GEMINI.md):
- Every direction must produce 3+ data points for the figure. NO single-bar panels.
- All generators must use rng for independent trials (no deterministic outputs).
- Use runner.compare() for all statistics. 25 trials always.
- Design directions as: taxonomy (heatmap), multi-condition bars, parameter sweeps (line), or scale tests (line).

ONLY write files under investigations/${MODE}/. Do not modify any other files.

Write the file, then stop. Do not run it."
else
    PROMPT="You are writing a new investigation for the Exotic Geometry Framework.

READ the file GEMINI.md FIRST — it contains all the rules, the Runner API, and a complete template.

Check existing investigations in investigations/ to avoid duplicates, then choose an interesting new topic.

Mode: ${MODE} (use Runner(name, mode='${MODE}'))

Write the investigation script to: investigations/${MODE}/<descriptive_name>.py

CRITICAL RULES (from GEMINI.md):
- Every direction must produce 3+ data points for the figure. NO single-bar panels.
- All generators must use rng for independent trials (no deterministic outputs).
- Use runner.compare() for all statistics. 25 trials always.
- Design directions as: taxonomy (heatmap), multi-condition bars, parameter sweeps (line), or scale tests (line).

ONLY write files under investigations/${MODE}/. Do not modify any other files.

Write the file, then stop. Do not run it."
fi

LOGFILE="${PROJ_ROOT}/tools/gemini_${TIMESTAMP}.log"

echo ""
echo "Launching Gemini CLI..."
echo "Log: ${LOGFILE}"
echo ""

cd "$PROJ_ROOT"
gemini \
    --yolo \
    --prompt "$PROMPT" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "================================================"
echo "  Gemini finished"
echo "================================================"

# List any new investigation files
echo "New investigation files:"
for f in investigations/${MODE}/*.py; do
    [[ -f "$f" ]] || continue
    fname=$(basename "$f")
    [[ "$fname" == "__init__.py" ]] && continue
    echo "  $f"
done

echo ""
echo "Log: ${LOGFILE}"
