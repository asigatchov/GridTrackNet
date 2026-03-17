#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tf2onnx

from model.GridTrackNet import GridTrackNet


WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5


def configure_tensorflow():
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def resolve_weights_path(model_path):
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"TensorFlow weights not found: {path}")
        return path

    candidates = sorted(
        Path("outputs").glob("GridTrackNet_seq5_*/checkpoints/best.weights.h5"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No default best.weights.h5 checkpoint found in outputs/")
    return candidates[0]


def export_onnx(weights_path, onnx_path, opset):
    configure_tensorflow()
    model = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)
    model.load_weights(weights_path)
    signature = (
        tf.TensorSpec(
            shape=(None, HEIGHT, WIDTH, IMGS_PER_INSTANCE * 3),
            dtype=tf.float32,
            name="input",
        ),
    )
    tf2onnx.convert.from_keras(
        model,
        input_signature=signature,
        opset=opset,
        output_path=str(onnx_path),
    )


def main():
    parser = argparse.ArgumentParser(description="Export src GridTrackNet TensorFlow weights to ONNX")
    parser.add_argument("--model_path", required=False, default=None, type=str, help="Path to TensorFlow .weights.h5 file")
    parser.add_argument("--output_path", required=False, default="gridtracknet_src.onnx", type=str, help="Output ONNX model path")
    parser.add_argument("--opset", required=False, default=17, type=int, help="ONNX opset version")
    args = parser.parse_args()

    weights_path = resolve_weights_path(args.model_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(weights_path, output_path, args.opset)
    print(f"Weights: {weights_path}")
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
