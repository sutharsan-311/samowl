#!/usr/bin/env python3
"""Persistent Unix-socket daemon that loads ML models once and serves inference requests.

Protocol
--------
Each request is a single UTF-8 JSON object followed by a newline.
Each response is a single UTF-8 JSON object followed by a newline.

Request fields
--------------
  image_path           : str   (required)
  depth_image_path     : str   (optional, default "")
  camera_model_path    : str   (optional, default "")
  text                 : str   (required)
  threshold            : float (optional, default 0.1)
  mask_threshold       : float (optional, default 0.0)
  output_mask          : str   (optional, default "mask.png")
  output_boundary      : str   (optional, default "boundary.png")
  output_depth_mask    : str   (optional, default "")
  output_points        : str   (optional, default "")
  output_hotspots      : str   (optional, default "")
  room_id              : str   (optional, default "simulation_room")
  merge_radius         : float (optional, default 0.10)
  max_points           : int   (optional, default 80000)

Response fields (on success)
-----------------------------
  success              : true
  image                : str
  depth_image          : str
  text                 : str
  bbox                 : list[float]
  score                : float
  label                : int
  mask_iou             : list
  output_boundary      : str
  output_mask          : str
  output_depth_mask    : str
  camera_model         : str
  output_points        : str
  output_hotspots      : str
  points_count         : int
  hotspot_map          : dict

Response fields (on failure)
-----------------------------
  success              : false
  error                : str
"""

import argparse
import json
import logging
import os
import socket
import sys
import types
from pathlib import Path

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from PIL import Image

# ---------------------------------------------------------------------------
# Import inference helpers from samowl_pipeline, regardless of install layout
# ---------------------------------------------------------------------------
def _import_pipeline():
    """Return the samowl_pipeline module, searching the usual locations."""
    import importlib.util

    candidates = [
        Path(__file__).resolve().parent / "samowl_pipeline.py",
    ]
    env_script = os.environ.get("SAMOWL_PIPELINE_SCRIPT")
    if env_script:
        candidates.insert(0, Path(env_script))

    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("samowl_pipeline", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    # Fall back to a plain import (works when installed as a package).
    import samowl_pipeline as mod  # type: ignore
    return mod


_pipeline = _import_pipeline()

NanoOwlPredictor = _pipeline.NanoOwlPredictor
OwlVit = _pipeline.OwlVit  # alias kept for any external references
Predictor = _pipeline.Predictor
bbox_to_points = _pipeline.bbox_to_points
draw_boundary = _pipeline.draw_boundary
save_mask = _pipeline.save_mask
save_masked_depth = _pipeline.save_masked_depth
project_mask_to_map_points = _pipeline.project_mask_to_map_points
estimate_normal = _pipeline.estimate_normal
write_pcd = _pipeline.write_pcd
write_hotspot_json = _pipeline.write_hotspot_json
resolve_existing_path = _pipeline.resolve_existing_path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [samowl_daemon] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("samowl_daemon")


# ---------------------------------------------------------------------------
# Model holder — loaded once at startup
# ---------------------------------------------------------------------------
class ModelBundle:
    def __init__(self, owl_model: str, owl_encoder: str, image_encoder: str, mask_decoder: str, default_threshold: float):
        log.info("Loading NanoOWL — text model: %s  TRT encoder: %s", owl_model, owl_encoder)
        self.owl = NanoOwlPredictor(model_name=owl_model, image_encoder_engine=owl_encoder, threshold=default_threshold)
        log.info("Loading SAM TensorRT engines: encoder=%s  decoder=%s", image_encoder, mask_decoder)
        self.sam = Predictor(image_encoder, mask_decoder)
        log.info("Models loaded — daemon ready")


# ---------------------------------------------------------------------------
# Per-request inference
# ---------------------------------------------------------------------------
def get_param(req: dict, config: dict, section: str, key: str, default=None, cast=None):
    """Resolve a parameter with precedence: request JSON > YAML config > hardcoded default.

    Null/None values in req are treated as absent so YAML can fill in.
    If cast is provided and the resolved value fails conversion, default is returned.
    """
    if key in req and req[key] is not None:
        val = req[key]
    elif key in config.get(section, {}):
        val = config[section][key]
    else:
        val = default
    if cast is not None:
        try:
            return cast(val)
        except Exception:
            return default
    return val


def _run_inference(req: dict, bundle: ModelBundle, config: dict) -> dict:
    """Run the full OWL-ViT + SAM pipeline for one request.

    Returns a dict with key 'success' plus result fields or an 'error' string.
    """
    # --- required fields ---
    image_path = req.get("image_path", "")
    text = req.get("text", "")
    if not image_path:
        return {"success": False, "error": "request missing image_path"}
    if not text:
        return {"success": False, "error": "request missing text"}

    # --- optional fields: request > YAML config > hardcoded default ---
    depth_image_path = req.get("depth_image_path", "")
    camera_model_path = req.get("camera_model_path", "")
    threshold = get_param(req, config, "detection", "threshold", 0.1, float)
    mask_threshold = get_param(req, config, "detection", "mask_threshold", 0.0, float)
    output_mask = get_param(req, config, "outputs", "output_mask", "mask.png")
    output_boundary = get_param(req, config, "outputs", "output_boundary", "boundary.png")
    output_depth_mask = get_param(req, config, "outputs", "output_depth_mask", "")
    output_points = get_param(req, config, "outputs", "output_points", "")
    output_hotspots = get_param(req, config, "outputs", "output_hotspots", "")
    room_id = get_param(req, config, "system", "room_id", "simulation_room")
    merge_radius = get_param(req, config, "detection", "merge_radius", 0.10, float)
    max_points = get_param(req, config, "detection", "max_points", 80000, int)

    if config.get("system", {}).get("debug", False):
        log.debug(
            "[effective] threshold=%.3f mask_threshold=%.3f merge_radius=%.3f "
            "max_points=%d room_id=%s",
            threshold, mask_threshold, merge_radius, max_points, room_id,
        )

    # Ensure output directories exist.
    for p in [output_mask, output_boundary, output_depth_mask, output_points, output_hotspots]:
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    # OWL-ViT: update threshold dynamically without reloading the model.
    bundle.owl.threshold = threshold
    raw_detections = bundle.owl.predict(image, texts=[text])
    if not raw_detections:
        return {
            "success": False,
            "error": f"No OWL detections for prompt '{text}' at threshold {threshold}",
        }

    # Keep the best-scoring box per label and clamp to image bounds.
    by_label: dict = {}
    for det in raw_detections:
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

    # Order by score descending so the primary (highest-score) label is first.
    label_order = sorted(by_label, key=lambda l: by_label[l]["score"], reverse=True)
    primary = by_label[label_order[0]]
    draw_boundary(image, primary, output_boundary)

    # Encode image once; decode mask per label.
    bundle.sam.set_image(image)

    base_pcd = Path(output_points) if output_points else None
    hotspot_map: dict = {}
    results_per_label = []
    primary_mask_iou = None

    for idx, label in enumerate(label_order):
        det = by_label[label]
        pts, pt_labels_sam = bbox_to_points(det["bbox"])
        mask, mask_iou, _ = bundle.sam.predict(pts, pt_labels_sam)
        if idx == 0:
            primary_mask_iou = mask_iou
        mask_image = save_mask(mask, output_mask, mask_threshold)

        if idx == 0 and depth_image_path and output_depth_mask:
            save_masked_depth(depth_image_path, mask_image, output_depth_mask)

        pcd_path = str(base_pcd.with_suffix("")) + f"_{idx}.pcd" if base_pcd else ""
        centroid = [0.0, 0.0, 0.0]
        pts_count = 0

        if depth_image_path and camera_model_path and pcd_path:
            map_points, camera_model = project_mask_to_map_points(
                depth_image_path, mask_image, camera_model_path, max_points)
            pts_count = int(len(map_points))
            write_pcd(pcd_path, map_points)
            if pts_count > 0:
                centroid = map_points.mean(axis=0).tolist()
            normal = estimate_normal(map_points)
            if output_hotspots:
                args_ns = types.SimpleNamespace(
                    room_id=room_id,
                    merge_radius=merge_radius,
                    text=label,
                    output_points=pcd_path,
                )
                hotspot_map = write_hotspot_json(
                    output_hotspots, args_ns, det, mask_iou, map_points, normal, camera_model)

        results_per_label.append({
            "label": label,
            "score": float(det["score"]),
            "bbox": det["bbox"],
            "centroid": centroid,
            "points_count": pts_count,
            "output_points": pcd_path,
        })

    best = results_per_label[0]
    return {
        "success": True,
        "image": str(image_path),
        "depth_image": depth_image_path,
        "text": text,
        # Legacy single-detection fields kept for backward compatibility.
        "bbox": best["bbox"],
        "score": best["score"],
        "label": best["label"],
        "mask_iou": primary_mask_iou.detach().cpu().numpy().tolist(),
        "output_boundary": str(output_boundary),
        "output_mask": str(output_mask),
        "output_depth_mask": output_depth_mask,
        "camera_model": camera_model_path,
        "output_points": best["output_points"],
        "output_hotspots": output_hotspots,
        "points_count": best["points_count"],
        "hotspot_map": hotspot_map,
        # All per-label results — C++ uses these directly.
        "detections": results_per_label,
    }


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------
def _handle_client(conn: socket.socket, bundle: ModelBundle, config: dict) -> None:
    """Read one newline-delimited JSON request; write one JSON response."""
    try:
        data = b""
        while b"\n" not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return
            data += chunk

        line, _ = data.split(b"\n", 1)
        try:
            req = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            resp = {"success": False, "error": f"JSON parse error: {exc}"}
            conn.sendall(json.dumps(resp).encode("utf-8") + b"\n")
            return

        log.info("Request: image=%s text=%s", req.get("image_path", "?"), req.get("text", "?"))
        try:
            resp = _run_inference(req, bundle, config)
        except Exception as exc:  # noqa: BLE001
            log.exception("Inference error")
            resp = {"success": False, "error": str(exc)}

        if resp.get("success"):
            dets = resp.get("detections", [])
            if dets:
                for d in dets:
                    log.info("Done: label=%s score=%.3f centroid=(%.2f,%.2f,%.2f) points=%d",
                        d["label"], d["score"],
                        d["centroid"][0], d["centroid"][1], d["centroid"][2],
                        d["points_count"])
            else:
                log.info("Done: score=%.3f points=%d", resp.get("score", 0.0), resp.get("points_count", 0))
        else:
            log.error("Failed: %s", resp.get("error", "unknown"))

        conn.sendall(json.dumps(resp).encode("utf-8") + b"\n")

    finally:
        conn.close()


def serve(socket_path: str, bundle: ModelBundle, config: dict) -> None:
    """Accept connections on a Unix socket until the process is killed."""
    sock_file = Path(socket_path)
    sock_file.parent.mkdir(parents=True, exist_ok=True)
    if sock_file.exists():
        sock_file.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    log.info("Listening on %s", socket_path)

    try:
        while True:
            conn, _ = server.accept()
            _handle_client(conn, bundle, config)
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    finally:
        server.close()
        if sock_file.exists():
            sock_file.unlink()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def validate_config(config: dict) -> None:
    """Warn on obviously invalid config values; never raises so startup is not blocked."""
    detection = config.get("detection", {})
    threshold = detection.get("threshold")
    if threshold is not None and not (0.0 < threshold <= 1.0):
        log.warning("config detection.threshold=%s is outside (0, 1] — check your YAML", threshold)
    merge_radius = detection.get("merge_radius")
    if merge_radius is not None and merge_radius <= 0:
        log.warning("config detection.merge_radius=%s must be positive", merge_radius)
    max_points = detection.get("max_points")
    if max_points is not None and max_points <= 0:
        log.warning("config detection.max_points=%s must be positive", max_points)


def load_config(path: str) -> dict:
    """Load YAML config file, returning empty dict on any error or if YAML unavailable."""
    if not path:
        return {}
    if not _YAML_AVAILABLE:
        log.warning("PyYAML not installed — skipping config file %s", path)
        return {}
    try:
        with open(path) as f:
            return _yaml.safe_load(f) or {}
    except Exception as exc:
        log.warning("Could not load config %s: %s", path, exc)
        return {}


def parse_args(config: dict) -> argparse.Namespace:
    models = config.get("models", {})
    detection = config.get("detection", {})
    daemon_cfg = config.get("daemon", {})

    parser = argparse.ArgumentParser(description="samowl persistent inference daemon")
    parser.add_argument(
        "--config",
        default="",
        help="Path to YAML config file (samowl.yaml)",
    )
    parser.add_argument(
        "--socket",
        default=daemon_cfg.get("socket", "/tmp/samowl/daemon.sock"),
        help="Unix socket path to listen on",
    )
    parser.add_argument(
        "--owl-model",
        default=models.get("owl", "data/owlvit-base-patch32"),
        help="OWL-ViT model directory",
    )
    parser.add_argument(
        "--owl-encoder",
        default=models.get("owl_encoder", "data/owl_image_encoder_patch32.engine"),
        help="NanoOWL TRT vision encoder engine",
    )
    parser.add_argument(
        "--image-encoder",
        default=models.get("image_encoder", "data/mobile_sam_image_encoder.engine"),
        help="SAM image encoder TensorRT engine",
    )
    parser.add_argument(
        "--mask-decoder",
        default=models.get("mask_decoder", "data/mobile_sam_mask_decoder.engine"),
        help="SAM mask decoder TensorRT engine",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=detection.get("threshold", 0.1),
        help="Default OWL detection threshold; overridden per-request",
    )
    return parser.parse_args()


def main() -> None:
    # Phase 1: extract --config path before full parse so YAML can back defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    pre_args, _ = pre_parser.parse_known_args()

    config = load_config(pre_args.config)
    validate_config(config)
    args = parse_args(config)

    owl_model = str(resolve_existing_path(args.owl_model, "OWL model directory"))
    owl_encoder = str(resolve_existing_path(args.owl_encoder, "NanoOWL TRT encoder"))
    image_encoder = str(resolve_existing_path(args.image_encoder, "SAM image encoder engine"))
    mask_decoder = str(resolve_existing_path(args.mask_decoder, "SAM mask decoder engine"))

    bundle = ModelBundle(owl_model, owl_encoder, image_encoder, mask_decoder, args.threshold)
    serve(args.socket, bundle, config)


if __name__ == "__main__":
    main()
