#!/usr/bin/env python3
"""Build a TensorRT image-encoder engine for OWLv2 (or OWL-ViT).

Usage:
  python3 scripts/build_owl_engine.py \
      data/owlv2_image_encoder_patch16.engine \
      --model data/owlv2-base-patch16

Five output names match samowl_pipeline.py's _load_owl_image_encoder_engine:
  image_embeds, image_class_embeds, logit_shift, logit_scale, pred_boxes
"""

import argparse
import os
import shutil
import tempfile

import torch
from transformers import Owlv2ForObjectDetection
from transformers.models.owlv2.modeling_owlv2 import center_to_corners_format

_OUTPUT_NAMES = [
    "image_embeds",
    "image_class_embeds",
    "logit_shift",
    "logit_scale",
    "pred_boxes",
]

_IMAGE_SIZES = {
    "owlv2-base-patch16": 960,
    "owlv2-large-patch14": 960,
    "owlvit-base-patch32": 768,
    "owlvit-base-patch16": 768,
    "owlvit-large-patch14": 840,
}


def _get_image_size(model_path: str) -> int:
    name = os.path.basename(model_path.rstrip("/"))
    for key, size in _IMAGE_SIZES.items():
        if key in name:
            return size
    raise ValueError(
        f"Cannot infer image size from '{model_path}'. "
        f"Known suffixes: {list(_IMAGE_SIZES.keys())}"
    )


class _VisionEncoderWrapper(torch.nn.Module):
    """Wraps Owlv2ForObjectDetection to export the 5 TRT outputs.

    Uses image_embedder() which applies the class-token merge step that
    class_head and box_head expect. Skipping this produces incorrect outputs.
    """

    def __init__(self, model: Owlv2ForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        # image_feats: (B, H, W, D) — class-token-weighted patch embeddings
        image_feats, _ = self.model.image_embedder(pixel_values=image)
        feature_map = image_feats
        b, h, w, d = image_feats.shape
        image_feats_flat = image_feats.reshape(b, h * w, d)

        # Projected class token for image_embeds output
        vision_out = self.model.owlv2.vision_model(pixel_values=image)
        last_hidden = vision_out.last_hidden_state
        img_emb = self.model.owlv2.vision_model.post_layernorm(last_hidden[:, :1, :])
        img_emb = self.model.owlv2.visual_projection(img_emb)

        image_class_embeds = self.model.class_head.dense0(image_feats_flat)
        logit_shift = self.model.class_head.logit_shift(image_feats_flat)
        logit_scale = self.model.class_head.logit_scale(image_feats_flat)
        logit_scale = self.model.class_head.elu(logit_scale) + 1

        pred_boxes = self.model.box_predictor(image_feats_flat, feature_map)
        pred_boxes = center_to_corners_format(pred_boxes)

        return img_emb, image_class_embeds, logit_shift, logit_scale, pred_boxes


def export_onnx(model_path: str, onnx_path: str, image_size: int, opset: int = 17):
    print(f"Loading {model_path} ...")
    model = Owlv2ForObjectDetection.from_pretrained(model_path, local_files_only=True)
    model = model.cuda()
    model.train(False)

    wrapper = _VisionEncoderWrapper(model).cuda()
    wrapper.train(False)
    dummy = torch.randn(1, 3, image_size, image_size, device="cuda")

    print(f"Exporting ONNX (input {image_size}x{image_size}, opset {opset}) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["image"],
            output_names=_OUTPUT_NAMES,
            dynamic_axes={"image": {0: "batch"}, **{n: {0: "batch"} for n in _OUTPUT_NAMES}},
            opset_version=opset,
        )
    print(f"ONNX saved: {onnx_path}")


def build_engine(onnx_path: str, engine_path: str, fp16: bool, trtexec: str, image_size: int):
    args = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=image:1x3x{image_size}x{image_size}",
        f"--optShapes=image:1x3x{image_size}x{image_size}",
        f"--maxShapes=image:1x3x{image_size}x{image_size}",
    ]
    if fp16:
        args.append("--fp16")
    cmd = " ".join(args)
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"trtexec exited with code {ret}")
    print(f"Engine saved: {engine_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", help="Destination .engine file")
    parser.add_argument("--model", default="data/owlv2-base-patch16", help="HF weights dir")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--trtexec", default=None)
    args = parser.parse_args()

    trtexec = args.trtexec or shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec):
        raise FileNotFoundError(
            f"trtexec not found at '{trtexec}'. Install TensorRT or pass --trtexec."
        )

    image_size = _get_image_size(args.model)
    tmpdir = tempfile.mkdtemp(prefix="samowl_owl_onnx_")
    onnx_path = os.path.join(tmpdir, "image_encoder.onnx")

    try:
        export_onnx(args.model, onnx_path, image_size, opset=args.opset)
        build_engine(onnx_path, args.output_path, fp16=args.fp16, trtexec=trtexec, image_size=image_size)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
