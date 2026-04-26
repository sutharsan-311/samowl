#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
from PIL import Image, ImageDraw, ImageFont
from torch2trt import TRTModule
from transformers import OwlViTProcessor, OwlViTModel
from nanoowl.owl_predictor import OwlPredictor as _NanoOwlRef


def _load_nanoowl_encoder(path: str) -> "TRTModule":
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as fh:
            engine = runtime.deserialize_cuda_engine(fh.read())
    return TRTModule(
        engine=engine,
        input_names=["image"],
        output_names=["image_embeds", "image_class_embeds", "logit_shift", "logit_scale", "pred_boxes"],
    )


class NanoOwlPredictor:
    """OWL-ViT with TRT vision encoder and cached PyTorch text encoder."""

    def __init__(self, model_name: str, image_encoder_engine: str, threshold: float = 0.1):
        self.processor = OwlViTProcessor.from_pretrained(model_name, local_files_only=True)
        self.text_model = OwlViTModel.from_pretrained(model_name, local_files_only=True)
        self.text_model = self.text_model.cuda().half()
        self.text_model.train(False)
        self.image_encoder = _load_nanoowl_encoder(image_encoder_engine)
        self.threshold = threshold
        self._text_cache: dict = {}
        # Use NanoOWL's image preprocessor and ROI extractor for correct square-pad resize
        self._nano = _NanoOwlRef(image_encoder_engine=image_encoder_engine)

    def _encode_text(self, texts: tuple) -> torch.Tensor:
        if texts not in self._text_cache:
            inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                out = self.text_model.text_model(**inputs)
                embeds = self.text_model.text_projection(out.pooler_output).half()  # (Q, 512)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            self._text_cache[texts] = embeds
        return self._text_cache[texts]

    def predict(self, image: "Image.Image", texts) -> list:
        texts = tuple(texts)
        text_embeds = self._encode_text(texts)                      # (Q, 512)

        W, H = image.size
        img_t = self._nano.image_preprocessor.preprocess_pil_image(image)
        roi = torch.tensor([[0, 0, W, H]], dtype=img_t.dtype, device=img_t.device)
        roi_img, _ = self._nano.extract_rois(img_t, roi, pad_square=True)
        with torch.no_grad():
            _, image_class_embeds, logit_shift, logit_scale, pred_boxes = self.image_encoder(roi_img)
        image_class_embeds = image_class_embeds.half()
        image_class_embeds = image_class_embeds / (image_class_embeds.norm(dim=-1, keepdim=True) + 1e-6)
        logit_shift = logit_shift.half()
        logit_scale = logit_scale.half()

        logits = torch.einsum("bpd,qd->bpq", image_class_embeds, text_embeds)  # (1, P, Q)
        logits = (logits + logit_shift) * logit_scale
        scores_all = torch.sigmoid(logits)[0]                       # (P, Q)

        detections = []
        for q_idx, text in enumerate(texts):
            q_scores = scores_all[:, q_idx]                         # (P,)
            keep = q_scores >= self.threshold
            if not keep.any():
                continue
            boxes_norm = pred_boxes[0][keep]                        # (K, 4) cx,cy,w,h in [0,1]
            cx, cy, bw, bh = boxes_norm.unbind(-1)
            x1 = ((cx - bw / 2) * W).clamp(0, W)
            y1 = ((cy - bh / 2) * H).clamp(0, H)
            x2 = ((cx + bw / 2) * W).clamp(0, W)
            y2 = ((cy + bh / 2) * H).clamp(0, H)
            for i, score in enumerate(q_scores[keep]):
                detections.append({
                    "bbox": [x1[i].item(), y1[i].item(), x2[i].item(), y2[i].item()],
                    "score": float(score),
                    "label": q_idx,
                    "text": text,
                })
        return detections


# Backward-compatible alias.
OwlVit = NanoOwlPredictor


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
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


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
    depth_min = float(camera_model.get("depth_min", 0.1))
    depth_max = float(camera_model.get("depth_max", 10.0))
    valid_z = np.isfinite(z) & (z >= depth_min) & (z <= depth_max)
    if not valid_z.any():
        return np.empty((0, 3), dtype=np.float64), camera_model
    rows = rows[valid_z]
    cols = cols[valid_z]
    z    = z[valid_z]
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


def _iou(box_a, box_b) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_detections(detections: list, iou_threshold: float = 0.5) -> list:
    """Per-class greedy NMS: keep highest-scoring non-overlapping boxes within each label.

    Returns all kept detections sorted by score descending.
    """
    by_text: dict = {}
    for det in detections:
        by_text.setdefault(det["text"], []).append(det)
    result = []
    for text_dets in by_text.values():
        sorted_dets = sorted(text_dets, key=lambda d: d["score"], reverse=True)
        kept: list = []
        for det in sorted_dets:
            if not any(_iou(det["bbox"], k["bbox"]) > iou_threshold for k in kept):
                kept.append(det)
        result.extend(kept)
    return sorted(result, key=lambda d: d["score"], reverse=True)


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
    parser.add_argument("--owl-encoder", default="data/owl_image_encoder_patch32.engine", help="NanoOWL TRT vision encoder engine.")
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
    owl_encoder = resolve_existing_path(args.owl_encoder, "NanoOWL TRT encoder engine")
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

    detector = OwlVit(str(owl_model), str(owl_encoder), args.threshold)
    texts = [t.strip() for t in args.text.split(",") if t.strip()]
    print(f"[samowl] Prompt split into {len(texts)} class(es): {texts}", file=sys.stderr)
    detections = detector.predict(image, texts=texts)
    if not detections:
        raise RuntimeError(f"No OWL detections found for prompt '{args.text}' at threshold {args.threshold}")

    # Keep best detection per label; clamp bbox to image bounds.
    by_label: dict = {}
    for det in detections:
        lbl = det["text"]
        if lbl not in by_label or det["score"] > by_label[lbl]["score"]:
            by_label[lbl] = det
    for det in by_label.values():
        det["bbox"] = [
            max(0.0, min(float(image.width - 1), float(det["bbox"][0]))),
            max(0.0, min(float(image.height - 1), float(det["bbox"][1]))),
            max(0.0, min(float(image.width - 1), float(det["bbox"][2]))),
            max(0.0, min(float(image.height - 1), float(det["bbox"][3]))),
        ]

    label_order = sorted(by_label, key=lambda lbl: by_label[lbl]["score"], reverse=True)
    draw_boundary(image, by_label[label_order[0]], output_boundary)

    base_mask = output_mask
    base_pcd = Path(args.output_points) if args.output_points else None
    sam_predictor = Predictor(str(image_encoder), str(mask_decoder))
    sam_predictor.set_image(image)

    hotspot_map = {}
    results_per_label = []
    primary_mask_iou = None

    for idx, label in enumerate(label_order):
        det = by_label[label]
        pts, pt_labels_sam = bbox_to_points(det["bbox"])
        mask, mask_iou, _ = sam_predictor.predict(pts, pt_labels_sam)
        if idx == 0:
            primary_mask_iou = mask_iou
        mask_path = Path(str(base_mask.with_suffix("")) + f"_{idx}.png")
        mask_image = save_mask(mask, mask_path, args.mask_threshold)

        if idx == 0 and args.depth_image and args.output_depth_mask:
            save_masked_depth(args.depth_image, mask_image, args.output_depth_mask)

        pcd_path = str(base_pcd.with_suffix("")) + f"_{idx}.pcd" if base_pcd else ""
        centroid = [0.0, 0.0, 0.0]
        points_count = 0

        if args.depth_image and args.camera_model and pcd_path:
            map_points, camera_model = project_mask_to_map_points(
                args.depth_image,
                mask_image,
                args.camera_model,
                args.max_points,
            )
            points_count = int(len(map_points))
            write_pcd(pcd_path, map_points)
            if points_count > 0:
                centroid = map_points.mean(axis=0).tolist()
            normal = estimate_normal(map_points)
            if args.output_hotspots:
                hotspot_map = write_hotspot_json(
                    args.output_hotspots,
                    types.SimpleNamespace(
                        room_id=args.room_id,
                        merge_radius=args.merge_radius,
                        text=label,
                        output_points=pcd_path,
                    ),
                    det,
                    mask_iou,
                    map_points,
                    normal,
                    camera_model,
                )

        results_per_label.append({
            "label": label,
            "score": float(det["score"]),
            "bbox": det["bbox"],
            "centroid": centroid,
            "points_count": points_count,
            "output_mask": str(mask_path),
            "output_points": pcd_path,
        })

    best = results_per_label[0]
    metadata = {
        "image": str(image_path),
        "depth_image": args.depth_image,
        "text": args.text,
        "bbox": best["bbox"],
        "score": best["score"],
        "label": best["label"],
        "mask_iou": primary_mask_iou.detach().cpu().numpy().tolist(),
        "output_boundary": str(output_boundary),
        "output_mask": best["output_mask"],
        "output_depth_mask": args.output_depth_mask,
        "camera_model": args.camera_model,
        "output_points": best["output_points"],
        "output_hotspots": args.output_hotspots,
        "points_count": best["points_count"],
        "hotspot_map": hotspot_map,
        "detections": results_per_label,
    }
    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
