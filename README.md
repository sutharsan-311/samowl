# samowl

`samowl` is a vision system that detects objects in 3D space using OWL-ViT (open-vocabulary detection) and SAM (segmentation), with spatial projection into a robot's map frame.

## Architecture

samowl is a **two-process system with a hard language boundary**:

```
ROS2 C++ Executive (samowl.cpp)
├─ subscribes to RGB/depth topics
├─ looks up camera pose in TF
├─ saves frame data to /tmp/samowl
├─ fork/exec Python daemon per frame
└─ publishes detections as hotspots

Python Daemon (samowl_daemon.py)
├─ loads models once (OWL-ViT, SAM)
├─ listens on Unix socket
├─ runs inference per frame
└─ returns results via file
```

**Why two languages?**
- C++ handles ROS2, TF, topic subscription, lifecycle
- Python owns all ML: OWL-ViT (Transformers), SAM (TensorRT)
- Clean separation prevents Python's GIL from blocking ROS callbacks

**The bottleneck:** Each frame spawns a new Python process via fork/execvp. This is necessary because model reloading is expensive, but process creation adds latency. The long-term target is a persistent daemon that stays alive between frames (documented in CLAUDE.md).

### Core Components

| Component | Role | Connections |
|-----------|------|-------------|
| `main()` (C++) | Executive; coordinates all subsystems | 14 edges |
| `SceneGraphNode` (ROS2) | Maintains scene graph of detected objects | 11 edges |
| `SceneQueryNode` (ROS2) | Query interface for scene (spatial queries) | 8 edges |
| `ModelBundle` (Python) | Holds loaded OWL-ViT + SAM models | 6 edges |
| `Persistent Daemon` | Target arch for eliminating per-frame fork overhead | design doc |

See `graphify-out/graph.html` for the full dependency graph (120 nodes, 8 communities).

### Operating Modes

**File mode** (`--image`): single inference, exit
**Topic mode** (`--rgb-topic`): subscribe to synchronized streams
- Frame drops during inference (process overhead)
- Saves hotspots (label, 3D centroid, normal, confidence)
- Optional `--continuous` to keep processing

## Build

```bash
cd /home/susan/nano  # workspace root
colcon build --packages-select samowl
source install/setup.bash
```

Run linting:
```bash
colcon test --packages-select samowl
colcon test-result --verbose
```

## Run

From an image file:

```bash
ros2 run samowl samowl \
  --image path/to/image.jpg \
  --text "a person" \
  --output-boundary boundary.png \
  --output-mask mask.png
```

From synchronized RGB and depth topics:

```bash
ros2 run samowl samowl \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw \
  --camera-info-topic /camera/color/camera_info \
  --map-frame map \
  --room-id simulation_room \
  --text "a person" \
  --output-boundary boundary.png \
  --output-mask mask.png \
  --output-depth-mask masked_depth.png \
  --output-points object_points_map.pcd \
  --output-hotspots hotspots.json \
  --merge-radius 0.10
```

Topic mode saves the latest synchronized RGB/depth/camera-info set into `/tmp/samowl`, looks up the camera pose in TF, runs OWL and SAM on the RGB image, saves the binary mask, and saves a depth image where pixels outside the mask are set to zero. It also projects the masked depth pixels into 3D map-frame points and writes:

- `object_points_map.pcd`: masked object point cloud in the `map` frame
- `hotspots.json`: first semantic hotspot entry with label, centroid, normal, confidence, and point count

It processes one synchronized set and exits by default; add `--continuous` to keep processing new frames. When `--continuous` is used with the same `--output-hotspots` file, detections with the same label within `--merge-radius` meters are fused into one hotspot with an updated centroid, confidence, point count, and `detection_count`.

For simulation or recorded scans, record the data first:

```bash
ros2 bag record -o room_scan_bag \
  /camera/color/image_raw \
  /camera/depth/image_raw \
  /camera/color/camera_info \
  /tf \
  /tf_static \
  /odom \
  /scan \
  /clock
```

Then replay it and run `samowl` in another terminal:

```bash
ros2 bag play room_scan_bag --clock
```

```bash
ros2 run samowl samowl \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw \
  --camera-info-topic /camera/color/camera_info \
  --map-frame map \
  --text "door handle" \
  --output-points door_handle_points_map.pcd \
  --output-hotspots room_hotspots.json
```

The depth topic must be aligned to the RGB camera for the saved `CameraInfo` intrinsics to be correct.

The package includes its model files under `samowl/data` and uses these defaults:

- `data/owlvit-base-patch32`
- `data/resnet18_image_encoder.engine`
- `data/mobile_sam_mask_decoder.engine`

No extra model path arguments are needed for the default package layout.

## Models

All models are stored in `data/`:

- `owlvit-base-patch32/` — OWL-ViT base model from Hugging Face
- `resnet18_image_encoder.engine` — TensorRT optimized ResNet18 (vision encoder)
- `mobile_sam_mask_decoder.engine` — TensorRT optimized SAM decoder

**⚠️ TensorRT engines are hardware-specific** — they require an exact match of:
- GPU architecture (e.g., RTX 4090)
- CUDA version
- TensorRT version

Engines are not regenerated in this repo. If models don't load, check the engine version against your runtime.

## Understanding the Codebase

The `graphify-out/` directory contains an automatically-generated knowledge graph of the codebase:

- **graph.html** — Interactive visualization (open in browser)
- **GRAPH_REPORT.md** — Detailed analysis of god nodes, communities, and surprising connections
- **graph.json** — Raw graph data (120 nodes, 177 edges, 8 communities)

**Key insights from the graph:**

1. **`main()` is a god node** (14 connections) — touches all subsystems. Consider breaking up long-running operations to prevent blocking ROS callbacks.

2. **Daemon system is fragmented** — Three separate communities (1, 2, 7) handle daemon concerns but lack cohesion. The persistent daemon is currently a design document (CLAUDE.md), not implemented.

3. **Scene Graph + Query form a coherent subsystem** — These two ROS2 nodes are tightly coupled and handle spatial queries well.

4. **Bridge node: `Predictor`** (high centrality) — Connects C++ Executive to ML inference. This is where architecture bottleneck manifests (per-frame process spawn).

**To update the graph after code changes:**
```bash
/graphify .
```

## Design Decisions

See `CLAUDE.md` for:
- Why fork/exec per frame is the current bottleneck
- Target refactor toward persistent daemon
- Data contracts (JSON IPC format)
- Dependency constraints
