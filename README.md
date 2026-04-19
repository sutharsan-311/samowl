# samowl

`samowl` provides a C++ command line executable that accepts an input image and a text prompt, detects the prompt with OWL, draws the selected boundary box, sends that box to SAM, and saves a binary mask.

The C++ executable launches the bundled Python model bridge because the model runtime uses Python, PyTorch, TensorRT, and Transformers APIs.

## Build

```bash
colcon build --packages-select samowl
source install/setup.bash
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
# samowl
