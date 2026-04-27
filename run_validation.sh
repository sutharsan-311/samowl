#!/bin/bash
set -e

# 1. Setup Environment
source /opt/ros/humble/setup.bash
source /home/susan/nano/install/setup.bash

# 2. Clean Outputs
rm -rf output_test
rm -f hotspots.json

# 3. Locate Bag
BAG_FILE=$(find . -name "*.mcap" -o -name "*.db3" | head -n 1)
if [ -z "$BAG_FILE" ]; then
  echo "Error: No ROS bag found."
  exit 1
fi
echo "Using bag: $BAG_FILE"

# 4. Run samowl in background
samowl \
  --mode scan \
  --config config/samowl.yaml \
  --output-dir output_test \
  --text "chair,bed" \
  --frame-sample 3 \
  --scan-idle-timeout 120 > samowl_output.log 2>&1 &
SAMOWL_PID=$!

# 5. Wait for daemon socket to be ready
echo "Waiting for daemon..."
until grep -q "Daemon socket ready" samowl_output.log; do
  sleep 1
  if ! kill -0 $SAMOWL_PID 2>/dev/null; then
    echo "samowl process died prematurely."
    cat samowl_output.log
    exit 1
  fi
done
echo "Daemon ready."

# 6. Play bag
echo "Playing bag..."
ros2 bag play "$BAG_FILE" --rate 0.5 --clock

# 7. Wait for completion
echo "Waiting for scan to complete..."
until grep -qE "Saved scene_graph.json|scan_complete: true" samowl_output.log; do
  sleep 2
  if ! kill -0 $SAMOWL_PID 2>/dev/null; then
    echo "samowl process exited."
    break
  fi
done

kill $SAMOWL_PID 2>/dev/null || true
echo "Validation run complete."
