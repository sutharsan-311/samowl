#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
from PIL import Image, ImageDraw, ImageFont
from torch2trt import TRTModule
from transformers import OwlViTForObjectDetection, OwlViTProcessor


class OwlVit:
    def __init__(self, threshold=0.1, model_name="google/owlvit-base-patch32"):
        self.processor = OwlViTProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name, local_files_only=True)
        self.model = self.model.cuda().half()
        self.threshold = threshold

    def predict(self, image, texts):
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        if hasattr(self.processor, "post_process_object_detection"):
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.threshold,
            )
        else:
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.threshold,
                text_labels=[list(texts)],
            )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            label = int(label)
            detections.append(
                {
                    "bbox": box.tolist(),
                    "score": float(score.detach()),
                    "label": label,
                    "text": texts[label],
                }
            )
        return detections


def load_mask_decoder_engine(path):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

    return TRTModule(
        engine=engine,
        input_names=[
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
        ],
        output_names=["iou_predictions", "low_res_masks"],
    )


def load_image_encoder_engine(path):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

    return TRTModule(
        engine=engine,
        input_names=["image"],
        output_names=["image_embeddings"],
    )


def preprocess_image(image, size=512):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
    image_std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

    aspect_ratio = image.width / image.height
    if aspect_ratio >= 1:
        resize_width = size
        resize_height = int(size / aspect_ratio)
    else:
        resize_height = size
        resize_width = int(size * aspect_ratio)

    image_resized = image.resize((resize_width, resize_height))
    image_np = np.array(image_resized, copy=True)
    image_torch = torch.from_numpy(image_np).permute(2, 0, 1)
    image_torch = (image_torch.float() - image_mean) / image_std
    image_tensor = torch.zeros((1, 3, size, size))
    image_tensor[0, :, :resize_height, :resize_width] = image_torch
    return image_tensor.cuda()


def preprocess_points(points, image_size, size=1024):
    scale = size / max(*image_size)
    return points * scale


def run_mask_decoder(mask_decoder_engine, features, points, point_labels, mask_input=None):
    image_point_coords = torch.from_numpy(np.array([points], dtype=np.float32)).cuda()
    image_point_labels = torch.from_numpy(np.array([point_labels], dtype=np.float32)).cuda()

    if mask_input is None:
        mask_input = torch.zeros(1, 1, 256, 256).float().cuda()
        has_mask_input = torch.tensor([0]).float().cuda()
    else:
        has_mask_input = torch.tensor([1]).float().cuda()

    return mask_decoder_engine(
        features,
        image_point_coords,
        image_point_labels,
        mask_input,
        has_mask_input,
    )


def upscale_mask(mask, image_shape, size=256):
    if image_shape[1] > image_shape[0]:
        lim_x = size
        lim_y = int(size * image_shape[0] / image_shape[1])
    else:
        lim_x = int(size * image_shape[1] / image_shape[0])
        lim_y = size

    return F.interpolate(mask[:, :, :lim_y, :lim_x], image_shape, mode="bilinear")


class Predictor:
    def __init__(self, image_encoder_engine, mask_decoder_engine, image_encoder_size=1024):
        self.image_encoder_engine = load_image_encoder_engine(image_encoder_engine)
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size

    def set_image(self, image):
        self.image = image
        self.image_tensor = preprocess_image(image, self.image_encoder_size)
        self.features = self.image_encoder_engine(self.image_tensor)

    def predict(self, points, point_labels, mask_input=None):
        points = preprocess_points(points, (self.image.height, self.image.width), self.image_encoder_size)
        mask_iou, low_res_mask = run_mask_decoder(
            self.mask_decoder_engine,
            self.features,
            points,
            point_labels,
            mask_input,
        )
        hi_res_mask = upscale_mask(low_res_mask, (self.image.height, self.image.width))
        return hi_res_mask, mask_iou, low_res_mask


def resolve_existing_path(path, description):
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate

    script_path = Path(__file__).resolve()
    search_roots = [
        Path.cwd(),
        script_path.parents[1],
    ]
    for root in search_roots:
        fallback = root / path
        if fallback.exists():
            return fallback
        fallback = root / "data" / candidate.name
        if fallback.exists():
            return fallback

    searched = "\n  ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        f"{description} not found: {path}\n"
        f"Searched relative to:\n  {searched}\n"
        "Pass the full path or put the file under samowl/data."
    )


def bbox_to_points(bbox):
    bbox = np.array(bbox, dtype=np.float32)
    return (
        np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32),
        np.array([2, 3], dtype=np.float32),
    )


def draw_boundary(image, detection, output_path):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    bbox = [int(round(v)) for v in detection["bbox"]]
    label = f'{detection["text"]} {detection["score"]:.2f}'
    draw.rectangle(bbox, outline=(0, 255, 0), width=4)
    try:
        font = ImageFont.load_default()
        text_box = draw.textbbox((bbox[0], bbox[1]), label, font=font)
        draw.rectangle(text_box, fill=(0, 255, 0))
        draw.text((bbox[0], bbox[1]), label, fill=(0, 0, 0), font=font)
    except Exception:
        draw.text((bbox[0], bbox[1]), label, fill=(0, 255, 0))
    canvas.save(output_path)


def save_mask(mask_tensor, output_path, threshold):
    mask = mask_tensor[0, 0].detach().cpu().numpy()
    mask_image = ((mask > threshold).astype(np.uint8) * 255)
    Image.fromarray(mask_image, mode="L").save(output_path)
    return mask_image


def save_masked_depth(depth_path, mask_image, output_path):
    depth = np.array(Image.open(depth_path))
    if depth.shape[:2] != mask_image.shape[:2]:
        raise RuntimeError(
            f"Depth image shape {depth.shape[:2]} does not match RGB/mask shape {mask_image.shape[:2]}"
        )
    masked_depth = np.where(mask_image > 0, depth, 0).astype(depth.dtype)
    Image.fromarray(masked_depth).save(output_path)


def quaternion_xyzw_to_matrix(quat):
    x, y, z, w = quat
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def estimate_normal(points):
    if len(points) < 3:
        return [0.0, 0.0, 1.0]
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm == 0:
        return [0.0, 0.0, 1.0]
    return (normal / norm).tolist()


def write_pcd(path, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# .PCD v0.7 - Point Cloud Data file format\n")
        handle.write("VERSION 0.7\n")
        handle.write("FIELDS x y z\n")
        handle.write("SIZE 4 4 4\n")
        handle.write("TYPE F F F\n")
        handle.write("COUNT 1 1 1\n")
        handle.write(f"WIDTH {len(points)}\n")
        handle.write("HEIGHT 1\n")
        handle.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        handle.write(f"POINTS {len(points)}\n")
        handle.write("DATA ascii\n")
        for point in points:
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def project_mask_to_map_points(depth_path, mask_image, camera_model_path, max_points):
    camera_model = json.loads(Path(camera_model_path).read_text(encoding="utf-8"))
    depth = np.array(Image.open(depth_path))
    if depth.shape[:2] != mask_image.shape[:2]:
        raise RuntimeError(
            f"Depth image shape {depth.shape[:2]} does not match mask shape {mask_image.shape[:2]}"
        )

    valid = (mask_image > 0) & (depth > 0)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float64), camera_model

    if len(rows) > max_points:
        sample = np.linspace(0, len(rows) - 1, max_points).astype(np.int64)
        rows = rows[sample]
        cols = cols[sample]

    z = depth[rows, cols].astype(np.float64) * float(camera_model.get("depth_scale", 0.001))
    fx = float(camera_model["fx"])
    fy = float(camera_model["fy"])
    cx = float(camera_model["cx"])
    cy = float(camera_model["cy"])
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    camera_points = np.column_stack([x, y, z])

    rotation = quaternion_xyzw_to_matrix(camera_model["rotation_xyzw"])
    translation = np.array(camera_model["translation"], dtype=np.float64)
    map_points = camera_points @ rotation.T + translation
    return map_points, camera_model


def next_hotspot_id(hotspots):
    return f"hs_{len(hotspots) + 1:03d}"


def weighted_average(old_value, old_weight, new_value, new_weight):
    old_array = np.array(old_value, dtype=np.float64)
    new_array = np.array(new_value, dtype=np.float64)
    total = max(old_weight + new_weight, 1)
    return ((old_array * old_weight + new_array * new_weight) / total).tolist()


def write_hotspot_json(path, args, detection, mask_iou, map_points, normal, camera_model):
    centroid = map_points.mean(axis=0).tolist() if len(map_points) else [0.0, 0.0, 0.0]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        hotspot_map = json.loads(output.read_text(encoding="utf-8"))
    else:
        hotspot_map = {
            "room_id": args.room_id,
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "merge_radius_m": args.merge_radius,
            "hotspots": [],
        }

    hotspots = hotspot_map.setdefault("hotspots", [])
    matched = None
    for hotspot in hotspots:
        if hotspot.get("label") != args.text:
            continue
        distance = float(np.linalg.norm(np.array(hotspot["position_3d"], dtype=np.float64) - np.array(centroid)))
        if distance <= args.merge_radius:
            matched = hotspot
            break

    observation = {
        "position_3d": centroid,
        "confidence": detection["score"],
        "point_count": int(len(map_points)),
        "bbox": detection["bbox"],
        "points_file": args.output_points,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if matched is None:
        matched = {
            "id": next_hotspot_id(hotspots),
            "label": args.text,
            "position_3d": centroid,
            "normal_vector": normal,
            "confidence": detection["score"],
            "detection_count": 1,
            "point_count": int(len(map_points)),
            "source_frame": camera_model.get("camera_frame", ""),
            "map_frame": camera_model.get("map_frame", "map"),
            "bbox": detection["bbox"],
            "mask_iou": mask_iou.detach().cpu().numpy().tolist(),
            "points_file": args.output_points,
            "observations": [observation],
        }
        hotspots.append(matched)
    else:
        old_count = int(matched.get("detection_count", 1))
        new_weight = max(int(len(map_points)), 1)
        old_weight = max(int(matched.get("point_count", old_count)), 1)
        matched["position_3d"] = weighted_average(matched["position_3d"], old_weight, centroid, new_weight)
        matched["normal_vector"] = weighted_average(matched.get("normal_vector", normal), old_count, normal, 1)
        matched["confidence"] = max(float(matched.get("confidence", 0.0)), float(detection["score"]))
        matched["detection_count"] = old_count + 1
        matched["point_count"] = int(matched.get("point_count", 0)) + int(len(map_points))
        matched["bbox"] = detection["bbox"]
        matched["mask_iou"] = mask_iou.detach().cpu().numpy().tolist()
        matched["points_file"] = args.output_points
        matched.setdefault("observations", []).append(observation)

    hotspot_map["updated_timestamp"] = datetime.now(timezone.utc).isoformat()
    tmp_path = output.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(hotspot_map, indent=2), encoding="utf-8")
    os.replace(tmp_path, output)
    return hotspot_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a text prompt with OWL, prompt SAM with the box, and save the mask."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--depth-image", default="", help="Optional synchronized depth image path.")
    parser.add_argument("--text", required=True, help="Text prompt for OWL, for example 'a person'.")
    parser.add_argument("--output-mask", default="mask.png", help="Output binary mask path.")
    parser.add_argument("--output-boundary", default="boundary.png", help="Output image with OWL boundary.")
    parser.add_argument("--output-depth-mask", default="", help="Optional masked depth output path.")
    parser.add_argument("--camera-model", default="", help="Camera intrinsics and camera-to-map transform JSON.")
    parser.add_argument("--output-points", default="", help="Optional output PCD for masked 3D points in map frame.")
    parser.add_argument("--output-hotspots", default="", help="Optional hotspot map JSON output path.")
    parser.add_argument("--room-id", default="simulation_room", help="Room id stored in hotspot JSON.")
    parser.add_argument("--max-points", type=int, default=80000, help="Maximum masked points to save to PCD.")
    parser.add_argument("--merge-radius", type=float, default=0.10, help="3D radius for merging same-label hotspots.")
    parser.add_argument("--owl-model", default="data/owlvit-base-patch32", help="Package-local OWL-ViT model directory.")
    parser.add_argument("--image-encoder", default="data/resnet18_image_encoder.engine")
    parser.add_argument("--mask-decoder", default="data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--threshold", type=float, default=0.1, help="OWL score threshold.")
    parser.add_argument("--mask-threshold", type=float, default=0.0, help="SAM logits threshold.")
    parser.add_argument(
        "--metadata",
        default="",
        help="Optional JSON path for the selected detection metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    output_mask = Path(args.output_mask)
    output_boundary = Path(args.output_boundary)
    owl_model = resolve_existing_path(args.owl_model, "OWL model directory")
    image_encoder = resolve_existing_path(args.image_encoder, "SAM image encoder engine")
    mask_decoder = resolve_existing_path(args.mask_decoder, "SAM mask decoder engine")
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    output_boundary.parent.mkdir(parents=True, exist_ok=True)
    if args.output_depth_mask:
        Path(args.output_depth_mask).parent.mkdir(parents=True, exist_ok=True)
    if args.output_points:
        Path(args.output_points).parent.mkdir(parents=True, exist_ok=True)
    if args.output_hotspots:
        Path(args.output_hotspots).parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    detector = OwlVit(args.threshold, str(owl_model))
    detections = detector.predict(image, texts=[args.text])
    if not detections:
        raise RuntimeError(f"No OWL detections found for prompt '{args.text}' at threshold {args.threshold}")

    detection = max(detections, key=lambda item: item["score"])
    detection["bbox"] = [
        max(0.0, min(float(image.width - 1), float(detection["bbox"][0]))),
        max(0.0, min(float(image.height - 1), float(detection["bbox"][1]))),
        max(0.0, min(float(image.width - 1), float(detection["bbox"][2]))),
        max(0.0, min(float(image.height - 1), float(detection["bbox"][3]))),
    ]
    draw_boundary(image, detection, output_boundary)

    points, point_labels = bbox_to_points(detection["bbox"])
    sam_predictor = Predictor(str(image_encoder), str(mask_decoder))
    sam_predictor.set_image(image)
    mask, mask_iou, _ = sam_predictor.predict(points, point_labels)
    mask_image = save_mask(mask, output_mask, args.mask_threshold)
    if args.depth_image and args.output_depth_mask:
        save_masked_depth(args.depth_image, mask_image, args.output_depth_mask)

    hotspot_map = {}
    points_count = 0
    if args.depth_image and args.camera_model and args.output_points:
        map_points, camera_model = project_mask_to_map_points(
            args.depth_image,
            mask_image,
            args.camera_model,
            args.max_points,
        )
        points_count = int(len(map_points))
        write_pcd(args.output_points, map_points)
        normal = estimate_normal(map_points)
        if args.output_hotspots:
            hotspot_map = write_hotspot_json(
                args.output_hotspots,
                args,
                detection,
                mask_iou,
                map_points,
                normal,
                camera_model,
            )

    metadata = {
        "image": str(image_path),
        "depth_image": args.depth_image,
        "text": args.text,
        "bbox": detection["bbox"],
        "score": detection["score"],
        "label": detection["label"],
        "mask_iou": mask_iou.detach().cpu().numpy().tolist(),
        "output_boundary": str(output_boundary),
        "output_mask": str(output_mask),
        "output_depth_mask": args.output_depth_mask,
        "camera_model": args.camera_model,
        "output_points": args.output_points,
        "output_hotspots": args.output_hotspots,
        "points_count": points_count,
        "hotspot_map": hotspot_map,
    }
    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
