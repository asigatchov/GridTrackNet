#!/usr/bin/env python3

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5
GRID_COLS = 48
GRID_ROWS = 27
GRID_SIZE_COL = WIDTH / GRID_COLS
GRID_SIZE_ROW = HEIGHT / GRID_ROWS


def is_headless():
    return not any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY"))


def build_session(onnx_path, provider):
    available = ort.get_available_providers()
    if provider == "gpu":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in onnxruntime")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    return session, session.get_inputs()[0].name, session.get_providers()


def resolve_model_path(model_path):
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return path

    default_path = Path("gridtracknet_src.onnx")
    if default_path.exists():
        return default_path
    raise FileNotFoundError("No default ONNX model found. Pass --model_path or create gridtracknet_src.onnx")


def get_predictions(session, input_name, frames, is_bgr_format=False):
    output_height = frames[0].shape[0]
    output_width = frames[0].shape[1]

    batches = []
    for i in range(0, len(frames), IMGS_PER_INSTANCE):
        batch = frames[i : i + IMGS_PER_INSTANCE]
        if len(batch) == IMGS_PER_INSTANCE:
            batches.append(batch)

    units = []
    for batch in batches:
        unit = []
        for frame in batch:
            if is_bgr_format:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            unit.append(frame.astype(np.float32) / 255.0)
        units.append(np.concatenate(unit, axis=-1))

    units = np.asarray(units, dtype=np.float32)
    y_pred = session.run(None, {input_name: units})[0]

    y_pred = y_pred.reshape((-1, GRID_ROWS, GRID_COLS, IMGS_PER_INSTANCE, 3))
    y_pred = np.transpose(y_pred, (0, 3, 4, 1, 2))

    conf_grid = y_pred[:, :, 0]
    x_offset_grid = y_pred[:, :, 1]
    y_offset_grid = y_pred[:, :, 2]

    ball_coordinates = []
    for i in range(conf_grid.shape[0]):
        for j in range(conf_grid.shape[1]):
            curr_conf_grid = conf_grid[i][j]
            curr_x_offset_grid = x_offset_grid[i][j]
            curr_y_offset_grid = y_offset_grid[i][j]

            max_conf_val = np.max(curr_conf_grid)
            pred_row, pred_col = np.unravel_index(np.argmax(curr_conf_grid), curr_conf_grid.shape)
            pred_has_ball = max_conf_val >= 0.5

            x_offset = curr_x_offset_grid[pred_row][pred_col]
            y_offset = curr_y_offset_grid[pred_row][pred_col]

            x_pred = int((x_offset + pred_col) * GRID_SIZE_COL)
            y_pred = int((y_offset + pred_row) * GRID_SIZE_ROW)

            if pred_has_ball:
                ball_coordinates.append((int((x_pred / WIDTH) * output_width), int((y_pred / HEIGHT) * output_height)))
            else:
                ball_coordinates.append((0, 0))

    return ball_coordinates


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX inference for src GridTrackNet")
    parser.add_argument("--video_path", required=True, type=str, help="Path to input .mp4 video")
    parser.add_argument("--model_path", required=False, default=None, type=str, help="Path to ONNX model file")
    parser.add_argument("--display_trail", required=False, default=1, type=int, help="Output a visible trail of the ball trajectory. Default = 1")
    parser.add_argument("--output_dir", required=False, default=None, type=str, help="Directory to save output video and CSV")
    parser.add_argument("--only_csv", action="store_true", default=False, help="Save only CSV, skip video output")
    parser.add_argument("--chunk_size", required=False, default=5, type=int, help="Number of frames buffered before inference. Use multiples of 5. Default = 5")
    parser.add_argument("--provider", choices=("cpu", "gpu"), default="gpu", help="Execution provider for ONNX Runtime")
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = args.video_path
    output_dir = args.output_dir
    model_path = resolve_model_path(args.model_path)
    display_trail = bool(args.display_trail)
    chunk_size = max(IMGS_PER_INSTANCE, args.chunk_size)

    if chunk_size % IMGS_PER_INSTANCE != 0:
        raise ValueError(f"--chunk_size must be a multiple of {IMGS_PER_INSTANCE}")

    session, input_name, providers = build_session(model_path, args.provider)
    print(f"ONNX providers: {providers}")

    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        raise RuntimeError("Error opening video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps >= 57 and fps <= 62:
        num_frames_skip = 2
    elif fps >= 22 and fps <= 32:
        num_frames_skip = 1
    else:
        raise RuntimeError("ERROR: Video is not 30FPS or 60FPS")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // num_frames_skip
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    directory, filename = os.path.split(video_dir)
    name, extension = os.path.splitext(filename)
    if output_dir:
        output_base_dir = Path(output_dir) / name
        output_base_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_base_dir = Path(directory)

    csv_path = output_base_dir / "ball.csv"
    csv_file = csv_path.open("w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Visibility", "X", "Y"])
    csv_file.flush()

    video_writer = None
    if not args.only_csv:
        output_path = output_base_dir / "predict_onnx.mp4"
        output_fps = fps if num_frames_skip == 1 else 30
        video_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (frame_width, frame_height),
        )
    else:
        output_path = None

    frames = []
    index = 0
    ball_coordinates_history = []
    predicted_frame_index = 0
    start_time = time.perf_counter()
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    def write_predicted_frames(predicted_frames, ball_coordinates):
        nonlocal_predicted_frame_index = predicted_frame_index
        history_start_idx = len(ball_coordinates_history) - len(ball_coordinates)
        for i, frame in enumerate(predicted_frames):
            if i < len(ball_coordinates):
                x, y = ball_coordinates[i]
                visibility = 0 if (x == 0 and y == 0) else 1
                csv_writer.writerow([nonlocal_predicted_frame_index, visibility, x, y])
                nonlocal_predicted_frame_index += 1

                if display_trail:
                    current_history_idx = history_start_idx + i
                    for trail_offset in range(7, 0, -2):
                        idx = current_history_idx - trail_offset
                        if idx >= 0:
                            cv2.circle(frame, ball_coordinates_history[idx], 4, (0, 255, 255), -1)
                else:
                    cv2.circle(frame, ball_coordinates[i], 8, (0, 0, 255), 4)

            if video_writer is not None:
                video_writer.write(frame)

        csv_file.flush()
        return nonlocal_predicted_frame_index

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if index % num_frames_skip == 0:
                frames.append(frame)
                pbar.update(1)
                if len(frames) == chunk_size:
                    ball_coordinates = get_predictions(session, input_name, frames, True)
                    ball_coordinates_history.extend(ball_coordinates)
                    predicted_frame_index = write_predicted_frames(frames, ball_coordinates)
                    frames = []

            index += 1

        if len(frames) >= IMGS_PER_INSTANCE:
            processable_frames = frames[: len(frames) - (len(frames) % IMGS_PER_INSTANCE)]
            if processable_frames:
                ball_coordinates = get_predictions(session, input_name, processable_frames, True)
                ball_coordinates_history.extend(ball_coordinates)
                predicted_frame_index = write_predicted_frames(processable_frames, ball_coordinates)
    finally:
        elapsed = time.perf_counter() - start_time
        pbar.close()
        cap.release()
        csv_file.close()
        if video_writer is not None:
            video_writer.release()
        if not is_headless():
            cv2.destroyAllWindows()

    fps_result = predicted_frame_index / elapsed if elapsed > 0 else 0.0
    print(f"Model: {model_path}")
    print(f"Processed frames: {predicted_frame_index}")
    print(f"Elapsed seconds: {elapsed:.4f}")
    print(f"FPS: {fps_result:.4f}")
    if output_path:
        print(f"Output video: {output_path}")
    print(f"Output csv: {csv_path}")


if __name__ == "__main__":
    main()
