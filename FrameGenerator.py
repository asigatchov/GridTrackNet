import argparse
import csv
import os
from pathlib import Path

import cv2


WIDTH = 1280
HEIGHT = 720
CSV_HEADER = ["Frame", "Visibility", "X", "Y"]


def valid_match_dir_name(directory: str) -> bool:
    suffix = os.path.basename(os.path.normpath(directory))
    return suffix.startswith("match") and suffix[5:].isdigit()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_video_mode(cap: cv2.VideoCapture) -> tuple[int, int]:
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if 57 <= fps <= 61:
        return fps, 2
    if 23 <= fps <= 31:
        return fps, 1
    raise ValueError(f"Unsupported FPS: {fps}. Expected 30 or 60 FPS.")


def export_video_frames(video_path: Path, frames_dir: Path) -> int:
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Video is not an .mp4 file: {video_path}")

    ensure_dir(frames_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    _, step = detect_video_mode(cap)

    frame_index = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if step == 2 and frame_index % 2 == 1:
            frame_index += 1
            continue

        img = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.imwrite(str(frames_dir / f"{saved_count}.png"), img)
        saved_count += 1
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    return saved_count


def parse_int_like(value: str) -> int:
    return int(float(str(value).strip()))


def parse_float_like(value: str) -> float:
    return float(str(value).strip())


def normalize_label_row(row: dict[str, str]) -> dict[str, str]:
    frame = parse_int_like(row["Frame"])
    visibility = 1 if parse_int_like(row["Visibility"]) else 0
    x = parse_float_like(row["X"])
    y = parse_float_like(row["Y"])

    # DataGen treats invisible frames correctly only with 0,0 coordinates.
    if visibility == 0 or x < 0 or y < 0:
        x = 0.0
        y = 0.0
        visibility = 0

    x = min(max(x, 0.0), float(WIDTH - 1))
    y = min(max(y, 0.0), float(HEIGHT - 1))

    return {
        "Frame": str(frame),
        "Visibility": str(visibility),
        "X": str(int(round(x))),
        "Y": str(int(round(y))),
    }


def normalize_csv(csv_path: Path, export_csv_path: Path, max_frames: int | None = None) -> int:
    with csv_path.open("r", newline="") as src:
        reader = csv.DictReader(src)
        missing = set(CSV_HEADER) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV {csv_path} is missing columns: {sorted(missing)}")

        rows = [normalize_label_row(row) for row in reader]

    if max_frames is not None:
        rows = [row for row in rows if int(row["Frame"]) < max_frames]

    rows.sort(key=lambda row: int(row["Frame"]))

    with export_csv_path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def csv_name_candidates(csv_path: Path) -> list[str]:
    stem = csv_path.stem
    candidates = [stem]
    for suffix in ("_ball", "_predict_ball"):
        if stem.endswith(suffix):
            candidates.append(stem[: -len(suffix)])
    if stem.endswith("_ball") and stem[:-5].endswith("_predict"):
        candidates.append(stem[:-13])
    return list(dict.fromkeys(candidates))


def find_video_for_csv(csv_path: Path) -> Path | None:
    parent = csv_path.parent.parent
    video_dir = parent / "video"
    if not video_dir.exists():
        return None

    for candidate in csv_name_candidates(csv_path):
        exact = video_dir / f"{candidate}.mp4"
        if exact.exists():
            return exact

    available = {video.stem: video for video in video_dir.glob("*.mp4")}
    for candidate in csv_name_candidates(csv_path):
        if candidate in available:
            return available[candidate]
    return None


def collect_pairs(input_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for csv_path in sorted(input_root.rglob("*.csv")):
        if csv_path.parent.name not in {"csv", "corrected_csv"}:
            continue
        video_path = find_video_for_csv(csv_path)
        if video_path is None:
            print(f"Skipping CSV without matching video: {csv_path}")
            continue
        pairs.append((video_path, csv_path))
    return pairs


def prepare_dataset(input_root: Path, output_root: Path) -> None:
    pairs = collect_pairs(input_root)
    if not pairs:
        raise ValueError(f"No mp4/csv pairs found under: {input_root}")

    ensure_dir(output_root)
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["match", "video", "csv", "saved_frames", "saved_labels"])

        for index, (video_path, csv_path) in enumerate(pairs, start=1):
            match_dir = output_root / f"match{index}"
            frames_dir = match_dir / "frames"
            labels_path = match_dir / "Labels.csv"

            ensure_dir(match_dir)
            print(f"[{index}/{len(pairs)}] Preparing {video_path.name}")
            saved_frames = export_video_frames(video_path, frames_dir)
            saved_labels = normalize_csv(csv_path, labels_path, max_frames=saved_frames)
            writer.writerow(
                [
                    f"match{index}",
                    str(video_path.resolve()),
                    str(csv_path.resolve()),
                    saved_frames,
                    saved_labels,
                ]
            )

    print(f"Prepared {len(pairs)} matches in {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Argument Parser for GridTrackNet")
    parser.add_argument("--video_dir", type=str, help="Path to .mp4 video file.")
    parser.add_argument("--export_dir", type=str, help="Export directory for a single matchX folder.")
    parser.add_argument(
        "--input_root",
        type=str,
        help="Root directory containing nested video/csv folders to convert into match1..matchN.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Output directory where prepared match folders will be stored in batch mode.",
    )
    args = parser.parse_args()

    single_mode = bool(args.video_dir or args.export_dir)
    batch_mode = bool(args.input_root or args.output_root)

    if single_mode and batch_mode:
        raise ValueError("Use either single-file mode (--video_dir/--export_dir) or batch mode (--input_root/--output_root).")
    if not single_mode and not batch_mode:
        raise ValueError("No mode selected. Use single-file mode or batch mode.")

    if single_mode:
        if not args.video_dir or not args.export_dir:
            raise ValueError("Single-file mode requires both --video_dir and --export_dir.")
        if not valid_match_dir_name(args.export_dir):
            raise ValueError("Specified export folder must start with 'match' followed by an index.")
        export_dir = Path(args.export_dir)
        frames_dir = export_dir / "frames"
        ensure_dir(export_dir)
        saved_frames = export_video_frames(Path(args.video_dir), frames_dir)
        print(f"Done. Exported {saved_frames} frames to {frames_dir}")
        return

    if not args.input_root or not args.output_root:
        raise ValueError("Batch mode requires both --input_root and --output_root.")

    prepare_dataset(Path(args.input_root), Path(args.output_root))


if __name__ == "__main__":
    main()
