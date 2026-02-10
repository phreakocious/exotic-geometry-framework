#!/usr/bin/env bash
#
# spawn_gemini.sh — Spawn Gemini CLI to write an investigation.
#
# Uses a temp workspace with --include-directories to give Gemini
# read access to the project but write access only to a sandbox.
# The investigation file is copied back after completion.
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
SLUG=$(echo "$TOPIC" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | head -c 40)
SLUG="${SLUG:-autopick}"

# Sandbox: Gemini writes here, we copy results back
SANDBOX="/tmp/egf-sandbox/${SLUG}_${TIMESTAMP}"
mkdir -p "${SANDBOX}/investigations/${MODE}"
mkdir -p "${SANDBOX}/tools"

# Copy only what Gemini needs to write into the sandbox
# (it can READ the main project via --include-directories)
cp "${PROJ_ROOT}/tools/investigation_runner.py" "${SANDBOX}/tools/"
cp "${PROJ_ROOT}/tools/__init__.py" "${SANDBOX}/tools/"
touch "${SANDBOX}/investigations/__init__.py"
touch "${SANDBOX}/investigations/${MODE}/__init__.py"

echo "================================================"
echo "  Exotic Geometry — Gemini Investigation Spawner"
echo "================================================"
echo "  Topic:    ${TOPIC:-<Gemini will choose>}"
echo "  Mode:     ${MODE}"
echo "  Sandbox:  ${SANDBOX}"
echo "  Project:  ${PROJ_ROOT} (read-only via --include-directories)"
echo "================================================"

# Build the prompt
if [[ -n "$TOPIC" ]]; then
    PROMPT="You are writing a new investigation for the Exotic Geometry Framework.

READ the file GEMINI.md (located in the included directory ${PROJ_ROOT}) FIRST — it contains all the rules, the Runner API, and a complete template.

You can also read existing investigations in ${PROJ_ROOT}/investigations/ for reference.

Topic: ${TOPIC}
Mode: ${MODE} (use Runner(name, mode='${MODE}'))

Write the investigation script to: investigations/${MODE}/<descriptive_name>.py
(This is inside your current working directory: ${SANDBOX})

CRITICAL RULES (from GEMINI.md):
- Every direction must produce 3+ data points for the figure. NO single-bar panels.
- All generators must use rng for independent trials (no deterministic outputs).
- Use runner.compare() for all statistics. 25 trials always.
- Design directions as: taxonomy (heatmap), multi-condition bars, parameter sweeps (line), or scale tests (line).
- Import the Runner: sys.path.insert(0, '${PROJ_ROOT}') then from tools.investigation_runner import Runner

Write the file, then stop. Do not run it."
else
    PROMPT="You are writing a new investigation for the Exotic Geometry Framework.

READ the file GEMINI.md (located in the included directory ${PROJ_ROOT}) FIRST — it contains all the rules, the Runner API, and a complete template.

Check existing investigations in ${PROJ_ROOT}/investigations/ to avoid duplicates, then choose an interesting new topic.

Mode: ${MODE} (use Runner(name, mode='${MODE}'))

Write the investigation script to: investigations/${MODE}/<descriptive_name>.py
(This is inside your current working directory: ${SANDBOX})

CRITICAL RULES (from GEMINI.md):
- Every direction must produce 3+ data points for the figure. NO single-bar panels.
- All generators must use rng for independent trials (no deterministic outputs).
- Use runner.compare() for all statistics. 25 trials always.
- Design directions as: taxonomy (heatmap), multi-condition bars, parameter sweeps (line), or scale tests (line).
- Import the Runner: sys.path.insert(0, '${PROJ_ROOT}') then from tools.investigation_runner import Runner

Write the file, then stop. Do not run it."
fi

LOGFILE="${PROJ_ROOT}/tools/gemini_${TIMESTAMP}.log"

echo ""
echo "Launching Gemini CLI (sandbox + included project dir)..."
echo "Log: ${LOGFILE}"
echo ""

cd "$SANDBOX"
gemini \
    --yolo \
    --include-directories "$PROJ_ROOT" \
    --prompt "$PROMPT" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "================================================"
echo "  Gemini finished"
echo "================================================"

# Copy investigation files back to main project
COPIED=0
for f in "${SANDBOX}"/investigations/${MODE}/*.py; do
    [[ -f "$f" ]] || continue
    fname=$(basename "$f")
    # Skip __init__.py
    [[ "$fname" == "__init__.py" ]] && continue
    echo "Copying: $fname → ${PROJ_ROOT}/investigations/${MODE}/"
    cp "$f" "${PROJ_ROOT}/investigations/${MODE}/${fname}"
    COPIED=$((COPIED + 1))
done

if [[ $COPIED -eq 0 ]]; then
    echo "No investigation files created. Check log: ${LOGFILE}"
else
    echo ""
    echo "$COPIED file(s) copied to project."
fi

echo ""
echo "Sandbox: ${SANDBOX}"
echo "Log: ${LOGFILE}"
