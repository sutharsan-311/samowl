#!/usr/bin/env bash
# End-to-end test: play the ROS bag through samowl and save per-frame debug images.
#
# Usage:
#   ./run_e2e_test.sh [--rate <playback_rate>] [--frames <max_frames>]
#
# Outputs are written to:
#   /tmp/samowl_e2e_test/
#     debug_frames/frame_NNNN/   -- per-frame mask/boundary/depth images
#     output/                    -- final scan-mode scene graph JSON
#     daemon.log                 -- daemon stdout/stderr
#     samowl.log                 -- node stdout/stderr
#     capture.log                -- debug frame capture log

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BAG_PATH="${SCRIPT_DIR}/rosbag2_2026_04_20-19_26_58"
INSTALL_DIR="/home/susan/nano/install"
TEST_DIR="/tmp/samowl_e2e_test"
WORK_DIR="/tmp/samowl"
BAG_RATE=1.0
MAX_FRAMES=0  # 0 = all frames

for arg in "$@"; do
  case "$arg" in
    --rate=*)   BAG_RATE="${arg#*=}" ;;
    --frames=*) MAX_FRAMES="${arg#*=}" ;;
  esac
done

# ── Derived paths ─────────────────────────────────────────────────────────────
DEBUG_DIR="${TEST_DIR}/debug_frames"
OUTPUT_DIR="${TEST_DIR}/output"
DAEMON_LOG="${TEST_DIR}/daemon.log"
NODE_LOG="${TEST_DIR}/samowl.log"
CAPTURE_LOG="${TEST_DIR}/capture.log"

DAEMON_SOCKET="${WORK_DIR}/daemon.sock"
HOTSPOTS_FILE="${WORK_DIR}/hotspots.json"
MASK_FILE="${WORK_DIR}/mask.png"
BOUNDARY_FILE="${WORK_DIR}/boundary.png"
DEPTH_MASK_FILE="${WORK_DIR}/masked_depth.png"
POINTS_FILE="${WORK_DIR}/object_points_map.pcd"

# ── ROS env ───────────────────────────────────────────────────────────────────
# colcon setup.bash uses unset variables internally; temporarily relax -u
set +u
# shellcheck source=/dev/null
source "${INSTALL_DIR}/setup.bash"
set -u

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[e2e]${NC} $*"; }
warn() { echo -e "${YELLOW}[e2e]${NC} $*"; }
err()  { echo -e "${RED}[e2e]${NC} $*"; }

PIDS=()
cleanup() {
  log "Shutting down processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  # Give ROS processes a moment to clean up
  sleep 1
  for pid in "${PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  log "Cleanup done."
}
trap cleanup EXIT INT TERM

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log "=== samowl end-to-end test ==="
log "Bag:       ${BAG_PATH}"
log "Test dir:  ${TEST_DIR}"
log "Bag rate:  ${BAG_RATE}x"

if [[ ! -d "${BAG_PATH}" ]]; then
  err "Bag not found: ${BAG_PATH}"
  exit 1
fi

if ! ros2 pkg executables samowl 2>/dev/null | grep -q samowl; then
  warn "samowl not found in install — rebuilding..."
  cd /home/susan/nano
  colcon build --packages-select samowl
  source "${INSTALL_DIR}/setup.bash"
fi

DATA_DIR="${INSTALL_DIR}/samowl/share/samowl/data"
for f in owl_image_encoder_patch32.engine mobile_sam_image_encoder.engine mobile_sam_mask_decoder.engine; do
  if [[ ! -f "${DATA_DIR}/${f}" ]]; then
    err "Model file missing: ${DATA_DIR}/${f}"
    exit 1
  fi
done
log "All model files present."

# ── Setup directories ─────────────────────────────────────────────────────────
rm -rf "${TEST_DIR}"
mkdir -p "${DEBUG_DIR}" "${OUTPUT_DIR}" "${WORK_DIR}"

# Clean any leftover daemon socket
rm -f "${DAEMON_SOCKET}" "${HOTSPOTS_FILE}"

# ── Step 1: Start daemon ───────────────────────────────────────────────────────
DAEMON_SCRIPT="${INSTALL_DIR}/samowl/share/samowl/scripts/samowl_daemon.py"
if [[ ! -f "${DAEMON_SCRIPT}" ]]; then
  DAEMON_SCRIPT="${SCRIPT_DIR}/samowl_daemon.py"
fi

PIPELINE_SCRIPT="${INSTALL_DIR}/samowl/share/samowl/scripts/samowl_pipeline.py"
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  PIPELINE_SCRIPT="${SCRIPT_DIR}/samowl_pipeline.py"
fi

log "Starting daemon (socket: ${DAEMON_SOCKET})..."
SAMOWL_PIPELINE_SCRIPT="${PIPELINE_SCRIPT}" \
python3 "${DAEMON_SCRIPT}" \
  --socket "${DAEMON_SOCKET}" \
  --config "${REPO_ROOT}/config/samowl.yaml" \
  >> "${DAEMON_LOG}" 2>&1 &
DAEMON_PID=$!
PIDS+=("${DAEMON_PID}")
log "Daemon PID: ${DAEMON_PID}"

# Wait for socket to appear (up to 60s — TRT engine load takes time)
log "Waiting for daemon to be ready (may take up to 60s for TRT)..."
for i in $(seq 1 120); do
  if [[ -S "${DAEMON_SOCKET}" ]]; then
    log "Daemon ready after ~${i}s"
    break
  fi
  if ! kill -0 "${DAEMON_PID}" 2>/dev/null; then
    err "Daemon died. Check ${DAEMON_LOG}"
    tail -30 "${DAEMON_LOG}"
    exit 1
  fi
  sleep 0.5
done

if [[ ! -S "${DAEMON_SOCKET}" ]]; then
  err "Daemon socket never appeared. Check ${DAEMON_LOG}"
  tail -30 "${DAEMON_LOG}"
  exit 1
fi

# ── Step 2: Start debug frame capture watcher ─────────────────────────────────
log "Starting debug frame capture watcher..."
python3 "${SCRIPT_DIR}/scripts/debug_frame_capture.py" \
  --work-dir "${WORK_DIR}" \
  --debug-dir "${DEBUG_DIR}" \
  --hotspots "hotspots.json" \
  --poll-interval 0.1 \
  >> "${CAPTURE_LOG}" 2>&1 &
CAPTURE_PID=$!
PIDS+=("${CAPTURE_PID}")
log "Capture watcher PID: ${CAPTURE_PID}"

# ── Step 3: Start samowl node ─────────────────────────────────────────────────
log "Starting samowl node (scan mode)..."
FRAME_SAMPLE=1
if [[ "${MAX_FRAMES}" -gt 0 ]]; then
  warn "Note: --frames is a guidance hint; samowl processes all bag frames in scan mode."
fi

ros2 run samowl samowl \
  --config "${SCRIPT_DIR}/config/samowl.yaml" \
  --rgb-topic  "/front_depth/rgb0/image" \
  --depth-topic "/front_depth/depth0/image_raw" \
  --camera-info-topic "/front_depth/rgb0/camera_info" \
  --map-frame "odom" \
  --work-dir "${WORK_DIR}" \
  --output-mask "${MASK_FILE}" \
  --output-boundary "${BOUNDARY_FILE}" \
  --output-depth-mask "${DEPTH_MASK_FILE}" \
  --output-points "${POINTS_FILE}" \
  --output-hotspots "${HOTSPOTS_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --daemon-socket "${DAEMON_SOCKET}" \
  --mode scan \
  --debug \
  >> "${NODE_LOG}" 2>&1 &
NODE_PID=$!
PIDS+=("${NODE_PID}")
log "samowl node PID: ${NODE_PID}"

# Give node time to subscribe
sleep 2

# ── Step 4: Play bag ──────────────────────────────────────────────────────────
log "Playing bag at ${BAG_RATE}x speed..."
log "Bag duration: ~4.5 min — estimated test time: $(echo "scale=1; 266 / ${BAG_RATE}" | bc)s"

ros2 bag play \
  "${BAG_PATH}" \
  --rate "${BAG_RATE}" \
  --clock \
  2>&1 | tee "${TEST_DIR}/bag.log"

log "Bag playback complete."

# ── Step 5: Wait for samowl scan finalization ─────────────────────────────────
log "Waiting for samowl scan to finalize (idle timeout ~5s)..."
FINALIZE_WAIT=20
for i in $(seq 1 "${FINALIZE_WAIT}"); do
  if ls "${OUTPUT_DIR}"/*.json 2>/dev/null | grep -q .; then
    log "Scan finalized — output JSON found in ${OUTPUT_DIR}"
    break
  fi
  if ! kill -0 "${NODE_PID}" 2>/dev/null; then
    log "samowl node exited (expected in scan mode)"
    break
  fi
  sleep 1
done

# ── Step 6: Stop watcher ──────────────────────────────────────────────────────
kill "${CAPTURE_PID}" 2>/dev/null || true
wait "${CAPTURE_PID}" 2>/dev/null || true

# ── Step 7: Summary ───────────────────────────────────────────────────────────
# Disable errexit for the summary so partial failures don't abort reporting
set +e

echo ""
log "=== TEST SUMMARY ==="

FRAME_COUNT=$(ls -d "${DEBUG_DIR}"/frame_* 2>/dev/null | wc -l)
log "Debug frames captured:  ${FRAME_COUNT}"

if [[ "${FRAME_COUNT}" -gt 0 ]]; then
  FIRST_FRAME=$(ls -d "${DEBUG_DIR}"/frame_* 2>/dev/null | head -1)
  log "Files in first frame:   $(ls "${FIRST_FRAME}" 2>/dev/null | tr '\n' ' ')"
  LAST_FRAME=$(ls -d "${DEBUG_DIR}"/frame_* 2>/dev/null | tail -1)
  log "Files in last frame:    $(ls "${LAST_FRAME}" 2>/dev/null | tr '\n' ' ')"
fi

JSON=$(ls "${OUTPUT_DIR}"/*.json 2>/dev/null | head -1)
if [[ -n "${JSON}" ]]; then
  log "Final scene graph JSON: ${JSON}"
  HOTSPOT_COUNT=$(python3 -c "
import json
d = json.load(open('${JSON}'))
hs = d.get('hotspots', d.get('objects', []))
print(len(hs))
" 2>/dev/null || echo "?")
  log "Hotspots/objects:       ${HOTSPOT_COUNT}"
fi

if [[ -f "${HOTSPOTS_FILE}" ]]; then
  HS_COUNT=$(python3 -c "
import json
d = json.load(open('${HOTSPOTS_FILE}'))
print(len(d.get('hotspots', [])))
" 2>/dev/null || echo "?")
  log "Hotspot map entries:    ${HS_COUNT}"
fi

FRAMES_PROC=$(grep -c "\[scan\] frame=" "${NODE_LOG}" 2>/dev/null || echo "?")
log "Frames processed (log): ${FRAMES_PROC}"

FRAMES_DROP=$(grep -c "dropping\|dropped" "${NODE_LOG}" 2>/dev/null || echo "0")
log "Frames dropped (log):   ${FRAMES_DROP}"

echo ""
log "Logs:         ${TEST_DIR}/"
log "Debug frames: ${DEBUG_DIR}/"
log "Outputs:      ${OUTPUT_DIR}/"
echo ""

if [[ "${FRAME_COUNT}" -eq 0 ]]; then
  err "No frames captured — check ${NODE_LOG} and ${DAEMON_LOG}"
  exit 1
fi

log "Test PASSED"
