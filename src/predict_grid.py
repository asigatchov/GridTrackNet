#!/usr/bin/env python3

import argparse
import csv
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model.GridTrackNet import GridTrackNet, GridTrackNetLegacy


INPUT_WIDTH = 768
INPUT_HEIGHT = 432
GRID_COLS = 48
GRID_ROWS = 27
SEQ = 5


def parse_args():
    parser = argparse.ArgumentParser(description="GridTrackNet TensorFlow inference")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_path", type=str, default=None, help="Path to .weights.h5 or .h5 weights")
    parser.add_argument("--track_length", type=int, default=8, help="Track length")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, gpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--backend",
        choices=["tensorflow", "pytorch_example"],
        default="tensorflow",
        help="Use tensorflow for real inference. pytorch_example is a compatibility placeholder.",
    )
    return parser.parse_args()


def configure_runtime(device_name):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if device_name == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "/CPU:0"
    if device_name == "gpu":
        if not gpus:
            raise RuntimeError("Requested GPU device, but TensorFlow GPU is unavailable")
        return "/GPU:0"
    return "/GPU:0" if gpus else "/CPU:0"


def resolve_model_path(model_path):
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found: {path}")
        return path

    candidates = sorted(
        Path("outputs").glob("GridTrackNet_seq5_*/checkpoints/best.weights.h5"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        default_h5 = Path("model_weights.h5")
        if default_h5.exists():
            return default_h5
        raise FileNotFoundError("No default TensorFlow weights found in outputs/ or model_weights.h5")
    return candidates[0]


def load_model(model_path, device):
    last_error = None
    with tf.device(device):
        for layout, factory in (("nhwc", GridTrackNet), ("legacy_nchw", GridTrackNetLegacy)):
            try:
                model = factory(SEQ, INPUT_HEIGHT, INPUT_WIDTH)
                model.load_weights(model_path)
                return model, layout
            except Exception as error:
                last_error = error
    raise RuntimeError(f"Could not load weights from {model_path}: {last_error}")


def initialize_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames


def setup_output_writer(video_basename, output_dir, frame_width, frame_height, fps, only_csv):
    if output_dir is None or only_csv:
        return None, None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_basename}_predict.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    return writer, output_path


def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{video_basename}_predict_ball.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frame", "Visibility", "X", "Y"])
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([result["Frame"], result["Visibility"], result["X"], result["Y"]])


def preprocess_frame(frame, layout):
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    if layout == "legacy_nchw":
        return np.transpose(rgb, (2, 0, 1))
    return rgb


def make_input_tensor(frame_buffer, layout):
    axis = 0 if layout == "legacy_nchw" else -1
    stacked = np.concatenate(list(frame_buffer), axis=axis)
    return np.expand_dims(stacked, axis=0).astype(np.float32)


def decode_predictions(output, threshold, layout):
    if layout == "legacy_nchw":
        output = output.reshape(SEQ, 3, GRID_ROWS, GRID_COLS)
    else:
        output = output.reshape(GRID_ROWS, GRID_COLS, SEQ, 3)
        output = np.transpose(output, (2, 3, 0, 1))
    results = []
    for frame_idx in range(SEQ):
        conf = output[frame_idx, 0]
        x_offset = output[frame_idx, 1]
        y_offset = output[frame_idx, 2]
        max_index = int(np.argmax(conf))
        row = max_index // GRID_COLS
        col = max_index % GRID_COLS
        conf_score = float(conf[row, col])
        if conf_score < threshold:
            results.append((0, -1, -1, conf_score))
            continue
        x = (col + float(x_offset[row, col])) * (INPUT_WIDTH / GRID_COLS)
        y = (row + float(y_offset[row, col])) * (INPUT_HEIGHT / GRID_ROWS)
        x = int(np.clip(x, 0, INPUT_WIDTH - 1))
        y = int(np.clip(y, 0, INPUT_HEIGHT - 1))
        results.append((1, x, y, conf_score))
    return results


def draw_track(frame, track_points):
    points = list(track_points)
    for point in points[:-1]:
        if point is not None:
            cv2.circle(frame, point, 5, (255, 0, 0), -1)
    if points and points[-1] is not None:
        cv2.circle(frame, points[-1], 8, (0, 0, 255), -1)
    return frame


def main():
    args = parse_args()
    if args.backend == "pytorch_example":
        raise RuntimeError("PyTorch path is left as an example placeholder. Use --backend tensorflow for execution.")

    device = configure_runtime(args.device)
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    model_path = resolve_model_path(args.model_path)

    model, layout = load_model(model_path, device)
    cap, frame_width, frame_height, fps, total_frames = initialize_video(video_path)
    basename = video_path.stem
    writer, output_path = setup_output_writer(basename, output_dir, frame_width, frame_height, fps, args.only_csv)
    csv_path = setup_csv_file(basename, output_dir)

    processed_buffer = deque(maxlen=SEQ)
    track = deque(maxlen=args.track_length)
    frame_index = 0
    predicted_frames = 0
    start_time = time.perf_counter()

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_buffer.append(preprocess_frame(frame, layout))

        if len(processed_buffer) < SEQ:
            result = {"Frame": frame_index, "Visibility": 0, "X": -1, "Y": -1}
            append_to_csv(result, csv_path)
            if writer or args.visualize:
                vis_frame = draw_track(frame.copy(), track)
                if writer:
                    writer.write(vis_frame)
                if args.visualize:
                    cv2.imshow("Grid Prediction", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frame_index += 1
            pbar.update(1)
            continue

        input_tensor = make_input_tensor(processed_buffer, layout)
        with tf.device(device):
            output = model(input_tensor, training=False).numpy()[0]
        predictions = decode_predictions(output, args.threshold, layout)
        visibility, x_resized, y_resized, _ = predictions[-1]
        predicted_frames += 1

        if visibility:
            x_orig = int(x_resized * frame_width / INPUT_WIDTH)
            y_orig = int(y_resized * frame_height / INPUT_HEIGHT)
            track.append((x_orig, y_orig))
        else:
            x_orig, y_orig = -1, -1
            if track:
                track.popleft()

        result = {"Frame": frame_index, "Visibility": visibility, "X": x_orig, "Y": y_orig}
        append_to_csv(result, csv_path)

        if writer or args.visualize:
            vis_frame = draw_track(frame.copy(), track)
            if visibility:
                cv2.circle(vis_frame, (x_orig, y_orig), 12, (0, 255, 0), 2)
            if writer:
                writer.write(vis_frame)
            if args.visualize:
                cv2.imshow("Grid Prediction", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_index += 1
        pbar.update(1)

    elapsed = time.perf_counter() - start_time
    pbar.close()
    cap.release()
    if writer:
        writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

    rps = predicted_frames / elapsed if elapsed > 0 else 0.0
    print(f"Model: {model_path}")
    print(f"Backend: tensorflow")
    print(f"Layout: {layout}")
    print(f"Device: {device}")
    print(f"Processed frames: {frame_index}")
    print(f"Inference windows: {predicted_frames}")
    print(f"Elapsed seconds: {elapsed:.4f}")
    print(f"RPS: {rps:.4f}")
    if output_path:
        print(f"Output video: {output_path}")
    if csv_path:
        print(f"Output csv: {csv_path}")


if __name__ == "__main__":
    main()
