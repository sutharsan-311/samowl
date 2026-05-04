# samowl

Vision system that detects objects in 3D space using OWLv2 (open-vocabulary detection) and SAM (segmentation), with spatial projection and scene graph persistence.

## Architecture

**Two-process system with hard language boundary:**

```
C++ ROS2 Executive (samowl.cpp)
├─ subscribes to RGB/depth topics
├─ fork/exec Python pipeline per frame ⚠️ BOTTLENECK
└─ manages scene graph deduplication

Python Pipeline (samowl_pipeline.py)
├─ OWLv2 detection + SAM segmentation
├─ 3D projection & hotspot fusion
└─ JSON IPC via /tmp/samowl

ROS2 Scene Graph (scene_graph_node.py)
├─ persistent object tracking across frames
├─ per-label match/create thresholds
└─ spatial query interface (scene_query_node.py)
```

**Why two languages:**
- C++ owns ROS2 lifecycle, TF lookups, topic subscription
- Python owns ML (OWLv2, SAM TensorRT)
- Separation prevents Python's GIL from blocking C++ callbacks

**Critical bottleneck:** Each frame spawns a new Python process. Process creation overhead is significant; long-term fix is persistent daemon (see CLAUDE.md).

### Core Components

| Component | Role |
|-----------|------|
| `samowl` (main) | C++ ROS2 node; frame orchestration and deduplication |
| `samowl_pipeline.py` | ML inference: OWLv2 + SAM + 3D projection |
| `scene_graph_node.py` | Persistent object tracking across frames |
| `scene_query_node.py` | Spatial query interface (nearest, in-radius, by-label) |
| `object_registry_node.py` | Object lifecycle tracking |
| `nav_mission_node.py` | Downstream consumer (mission planning over detections) |

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

- `data/owlv2-base-patch16/`
- `data/owlv2_image_encoder_patch16.engine`
- `data/resnet18_image_encoder.engine`
- `data/mobile_sam_mask_decoder.engine`

No extra model path arguments are needed for the default package layout.

## Configuration

Configuration is loaded from `config/samowl.yaml` (see that file for all options). Key parameters:

### Detection Pipeline
- `detection.threshold` — OWLv2 confidence score threshold (0.0–1.0). Lower = more detections, higher recall but lower precision
- `detection.max_detections` — Maximum objects to segment per frame after NMS
- `detection.merge_radius` — 3D distance (meters) for deduplicating same-label hotspots within a frame

### Scene Graph Merging
The scene graph node uses label-specific match and create thresholds:

```yaml
graph:
  match_threshold: 0.95    # default: merge detections into existing nodes if centroid distance < this (meters)
  create_threshold: 0.85   # default: create a new node if no existing node is within this distance
  label_thresholds:
    chair:
      match_threshold: 2.0   # chairs have ~2m variation in centroid across views
      create_threshold: 1.8
    hospital bed:
      match_threshold: 1.5
      create_threshold: 1.3
  label_merge_radii:
    hospital bed: 0.95       # beds have ~0.87m max intra-detection spread
    chair: 0.35              # 2 physical chairs are 0.44m apart; keep distinct
```

Per-label thresholds override defaults and help prevent spurious merges across multiple views. For example, chairs cluster with large position variance, so a 2.0m match threshold prevents treating every view of the same chair as a new object.

## Models

All models are stored in `data/`:

- `owlv2-base-patch16/` — OWLv2 base model from Hugging Face (HF format with tokenizer)
- `owlv2_image_encoder_patch16.engine` — TensorRT optimized OWLv2 image encoder
- `resnet18_image_encoder.engine` — TensorRT optimized ResNet18 (SAM vision encoder)
- `mobile_sam_mask_decoder.engine` — TensorRT optimized SAM decoder

**⚠️ TensorRT engines are hardware-specific** — they require an exact match of:
- GPU architecture (e.g., RTX 4090)
- CUDA version
- TensorRT version

Engines are not regenerated in this repo. To build them for your hardware:

```bash
python3 scripts/build_owl_engine.py \
  --owl-dir data/owlv2-base-patch16 \
  --output data/owlv2_image_encoder_patch16.engine
```

If models fail to load, verify the engine version matches your CUDA and TensorRT runtime.

## Scene Graph and Deduplication

Detections are merged across frames using **per-label thresholds** to prevent spurious re-detections while maintaining distinctness.

**Per-frame flow:**
1. Detections in frame → C++ dedup (within-frame, merge_radius) → hotspot JSON
2. Hotspot → scene_graph_node (merge with persistent state using match_threshold)
3. Updated objects → scene_query_node (spatial queries)

**Why per-label?** Different object classes have different position variance:
- **Chairs**: high variance (~2m), need relaxed match_threshold (2.0m) to avoid treating same chair as new object
- **Beds**: low variance (~0.95m), tighter thresholds prevent spurious merges

## Key Architecture Decisions

**`main()` is the orchestration point** — all frame processing flows through it. This makes it a critical bottleneck when per-frame overhead is high (currently fork/exec adds ~200-500ms per frame).

**File-based IPC** (`/tmp/samowl`) is temporary. C++ writes frame data, Python reads/processes, writes results. This is replaced by persistent daemon + bidirectional JSON socket in the roadmap.

**Per-label thresholds are critical** — different object classes (chairs vs. beds) have different detection variance across views. Generic thresholds cause either spurious duplicates or missed distinct objects.

**Model loading is per-frame** because Python subprocess exits after processing. Cached models in persistent daemon would eliminate this overhead.

## Design Decisions

See `CLAUDE.md` for:
- Why fork/exec per frame is the current bottleneck
- Target refactor toward persistent daemon
- Data contracts (JSON IPC format)
- Dependency constraints
