import argparse
import csv
import math
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.python.ops.numpy_ops import np_config

from GridTrackNet import GridTrackNet

np_config.enable_numpy_behavior()


IMGS_PER_INSTANCE = 5
DATA_HEIGHT = 720
DATA_WIDTH = 1280
HEIGHT = 432
WIDTH = 768
GRID_COLS = 48
GRID_ROWS = 27
GRID_SIZE_COL = DATA_WIDTH / GRID_COLS
GRID_SIZE_ROW = DATA_HEIGHT / GRID_ROWS


parser = argparse.ArgumentParser(description="Argument Parser for GridTrackNet")
parser.add_argument("--data_dir", required=True, type=str, help="Directory with TFRecords or prepared match folders.")
parser.add_argument("--load_weights", required=False, type=str, help="Directory to load pre-trained weights.")
parser.add_argument("--save_weights", required=True, type=str, help="Directory to store model weights and training metrics.")
parser.add_argument("--epochs", required=True, type=int, help="Number of epochs.")
parser.add_argument("--tol", required=False, type=int, default=4, help="Pixel tolerance. Default = 4")
parser.add_argument("--batch_size", required=False, type=int, default=3, help="Batch size. Default = 3")
parser.add_argument("--val_split", required=False, type=float, default=0.2, help="Validation split for direct mode.")
parser.add_argument(
    "--next_img_index",
    required=False,
    type=int,
    default=5,
    choices=range(1, IMGS_PER_INSTANCE + 1),
    help="Instance stride for direct mode. Default = 5",
)
parser.add_argument("--seed", required=False, type=int, default=42, help="Random seed for direct mode split.")

args = parser.parse_args()

DATA_DIR = args.data_dir
LOAD_WEIGHTS = args.load_weights
SAVE_WEIGHTS = args.save_weights
EPOCHS = args.epochs
TOL = args.tol
BATCH_SIZE = args.batch_size
VAL_SPLIT = args.val_split
NEXT_IMG_INDEX = args.next_img_index
SEED = args.seed

if VAL_SPLIT <= 0.0 or VAL_SPLIT >= 1.0:
    raise ValueError("Input argument 'val_split' must be greater than 0.0 and less than 1.0.")


def calcOutcomeStats(y_pred, y_true):
    TP = TN = FP1 = FP2 = FN = 0

    y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)

    confGridTrue, xOffsetGridTrue, yOffsetGridTrue = np.split(y_true, 3, axis=-1)
    confGridPred, xOffsetGridPred, yOffsetGridPred = np.split(y_pred, 3, axis=-1)

    confGridTrue = np.squeeze(confGridTrue, axis=-1)
    xOffsetGridTrue = np.squeeze(xOffsetGridTrue, axis=-1)
    yOffsetGridTrue = np.squeeze(yOffsetGridTrue, axis=-1)
    confGridPred = np.squeeze(confGridPred, axis=-1)
    xOffsetGridPred = np.squeeze(xOffsetGridPred, axis=-1)
    yOffsetGridPred = np.squeeze(yOffsetGridPred, axis=-1)

    for i in range(0, confGridTrue.shape[0]):
        for j in range(0, confGridTrue.shape[1]):
            currConfGridTrue = confGridTrue[i][j]
            currXOffsetGridTrue = xOffsetGridTrue[i][j]
            currYOffsetGridTrue = yOffsetGridTrue[i][j]

            currConfGridPred = confGridPred[i][j]
            currXOffsetGridPred = xOffsetGridPred[i][j]
            currYOffsetGridPred = yOffsetGridPred[i][j]

            maxConfValTrue = np.max(currConfGridTrue)
            trueRow, trueCol = np.unravel_index(np.argmax(currConfGridTrue), currConfGridTrue.shape)

            maxConfValPred = np.max(currConfGridPred)
            predRow, predCol = np.unravel_index(np.argmax(currConfGridPred), currConfGridPred.shape)

            threshold = 0.5
            trueHasBall = maxConfValTrue >= threshold
            predHasBall = maxConfValPred >= threshold

            xOffsetTrue = currXOffsetGridTrue[trueRow][trueCol]
            yOffsetTrue = currYOffsetGridTrue[trueRow][trueCol]
            xOffsetPred = currXOffsetGridPred[predRow][predCol]
            yOffsetPred = currYOffsetGridPred[predRow][predCol]

            grid_size_col = WIDTH / GRID_COLS
            grid_size_row = HEIGHT / GRID_ROWS

            xTrue = int((xOffsetTrue + trueCol) * grid_size_col)
            yTrue = int((yOffsetTrue + trueRow) * grid_size_row)
            xPred = int((xOffsetPred + predCol) * grid_size_col)
            yPred = int((yOffsetPred + predRow) * grid_size_row)

            if (not predHasBall) and (not trueHasBall):
                TN += 1
            elif predHasBall and (not trueHasBall):
                FP2 += 1
            elif (not predHasBall) and trueHasBall:
                FN += 1
            elif predHasBall and trueHasBall:
                dist = int(((xPred - xTrue) ** 2 + (yPred - yTrue) ** 2) ** 0.5)
                if dist > TOL:
                    FP1 += 1
                else:
                    TP += 1

    return TP, TN, FP1, FP2, FN


GLOBAL_ACCURACY = 0
GLOBAL_PRECISION = 0
GLOBAL_RECALL = 0
GLOBAL_F1 = 0


def accuracy(y_true, y_pred):
    global GLOBAL_ACCURACY, GLOBAL_PRECISION, GLOBAL_RECALL, GLOBAL_F1
    TP, TN, FP1, FP2, FN = calcOutcomeStats(y_pred, y_true)
    try:
        GLOBAL_ACCURACY = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except Exception:
        GLOBAL_ACCURACY = 0.0
    try:
        GLOBAL_PRECISION = TP / (TP + FP1 + FP2)
    except Exception:
        GLOBAL_PRECISION = 0.0
    try:
        GLOBAL_RECALL = TP / (TP + FN)
    except Exception:
        GLOBAL_RECALL = 0.0
    try:
        GLOBAL_F1 = (2 * (GLOBAL_PRECISION * GLOBAL_RECALL)) / (GLOBAL_PRECISION + GLOBAL_RECALL)
    except Exception:
        GLOBAL_F1 = 0.0
    return GLOBAL_ACCURACY


def precision(y_true, y_pred):
    return GLOBAL_PRECISION


def recall(y_true, y_pred):
    return GLOBAL_RECALL


def f1(y_true, y_pred):
    return GLOBAL_F1


def custom_loss(y_true, y_pred):
    confWeight = 1
    offsetWeight = 0.001

    y_pred = tf.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = tf.stack(y_pred, axis=2)
    y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 4, 1])

    confGridTrue, xOffsetGridTrue, yOffsetGridTrue = tf.split(y_true, 3, axis=-1)
    confGridPred, xOffsetGridPred, yOffsetGridPred = tf.split(y_pred, 3, axis=-1)

    yTrueOffset = tf.concat([xOffsetGridTrue, yOffsetGridTrue], axis=-1)
    yPredOffset = tf.concat([xOffsetGridPred, yOffsetGridPred], axis=-1)

    diff = tf.abs(yTrueOffset - yPredOffset)
    sum_diff = tf.reduce_sum(diff, axis=-1, keepdims=True)
    masked_sum_diff = confGridTrue * sum_diff
    sum_offset = tf.reduce_sum(masked_sum_diff, axis=[2, 3, 4])
    offset = tf.reduce_mean(sum_offset, axis=[1])

    alpha = 0.75
    gamma = 2
    positiveConfLoss = alpha * confGridTrue * tf.pow(1 - confGridPred, gamma) * tf.math.log(
        tf.clip_by_value(confGridPred, tf.keras.backend.epsilon(), 1)
    )
    negativeConfLoss = (1 - alpha) * (1 - confGridTrue) * tf.pow(confGridPred, gamma) * tf.math.log(
        tf.clip_by_value(1 - confGridPred, tf.keras.backend.epsilon(), 1)
    )
    confidence = tf.reduce_mean((-1) * (positiveConfLoss + negativeConfLoss), axis=[1, 2, 3, 4])
    loss = offsetWeight * offset + confWeight * confidence
    return tf.reduce_sum(loss)


def parseInstance(rawData):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(rawData, feature_description)
    image = tf.io.decode_raw(example["image"], tf.float32)
    label = tf.io.decode_raw(example["label"], tf.float32)
    image = tf.reshape(image, (IMGS_PER_INSTANCE * 3, HEIGHT, WIDTH))
    label = tf.reshape(label, (IMGS_PER_INSTANCE, GRID_ROWS, GRID_COLS, 3))
    return image, label


def loadSubDataset(file):
    subdataset = tf.data.TFRecordDataset(file)
    subdataset = subdataset.map(parseInstance, num_parallel_calls=tf.data.AUTOTUNE)
    return subdataset


def createEpochDataset(tfRecordFile, bufferSize, batch_size, numParallelCalls):
    filenames = tf.data.Dataset.list_files(tfRecordFile, shuffle=True)
    interleavedDataset = filenames.interleave(
        loadSubDataset,
        cycle_length=numParallelCalls,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    dataset = interleavedDataset.shuffle(bufferSize)
    dataset = dataset.batch(batch_size)
    return dataset


def read_labels(labels_path: Path):
    with labels_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def build_instance(match_path: Path, rows: list[dict[str, str]]):
    dataEntry = []
    labelEntry = []

    for row in rows:
        frame_idx = int(float(row["Frame"]))
        visibility = 1 if int(float(row["Visibility"])) else 0
        x = float(row["X"])
        y = float(row["Y"])
        if visibility == 0:
            x = 0.0
            y = 0.0

        confGrid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        xOffsetGrid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        yOffsetGrid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

        if visibility == 1:
            xPos = x / GRID_SIZE_COL
            yPos = y / GRID_SIZE_ROW
            xCellIndex = min(math.floor(xPos), GRID_COLS - 1)
            yCellIndex = min(math.floor(yPos), GRID_ROWS - 1)
            xOffset = xPos - xCellIndex
            yOffset = yPos - yCellIndex
            confGrid[yCellIndex, xCellIndex] = 1
            xOffsetGrid[yCellIndex, xCellIndex] = xOffset
            yOffsetGrid[yCellIndex, xCellIndex] = yOffset

        labelEntry.append(np.stack((confGrid, xOffsetGrid, yOffsetGrid), axis=-1))

        img_path = match_path / "frames" / f"{frame_idx}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = np.moveaxis(img, -1, 0)
        dataEntry.extend([img[0], img[1], img[2]])

    dataEntry = np.asarray(dataEntry, dtype=np.float32)
    labelEntry = np.asarray(labelEntry, dtype=np.float32)
    return dataEntry, labelEntry


def collect_direct_instances(data_dir: str, val_split: float, next_img_index: int, seed: int):
    rng = random.Random(seed)
    train_instances = []
    val_instances = []

    match_paths = sorted(
        path for path in Path(data_dir).glob("match*") if path.is_dir() and (path / "Labels.csv").exists() and (path / "frames").exists()
    )
    if not match_paths:
        raise ValueError(f"No prepared match folders found in {data_dir}")

    for match_path in match_paths:
        rows = read_labels(match_path / "Labels.csv")
        i = 0
        while i + IMGS_PER_INSTANCE - 1 < len(rows):
            window = rows[i : i + IMGS_PER_INSTANCE]
            if rng.random() < val_split:
                val_instances.append((match_path, window))
            else:
                train_instances.append((match_path, window))
            i += next_img_index

    return train_instances, val_instances


def create_direct_dataset(instances, batch_size: int, shuffle: bool):
    output_signature = (
        tf.TensorSpec(shape=(IMGS_PER_INSTANCE * 3, HEIGHT, WIDTH), dtype=tf.float32),
        tf.TensorSpec(shape=(IMGS_PER_INSTANCE, GRID_ROWS, GRID_COLS, 3), dtype=tf.float32),
    )

    def generator():
        for match_path, rows in instances:
            yield build_instance(match_path, rows)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(min(len(instances), 1024), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_training_sources():
    val_tfrecord = os.path.join(DATA_DIR, "val.tfrecord")
    if os.path.exists(val_tfrecord):
        rawValData = tf.data.TFRecordDataset(val_tfrecord)
        parsedValData = rawValData.map(parseInstance)
        valData = parsedValData.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        tfRecordFilePattern = os.path.join(DATA_DIR, "train*.tfrecord")
        numTrainFiles = len(glob(tfRecordFilePattern)) - 1
        numInstancesPerFile = 50
        bufferSize = numInstancesPerFile
        numParallelCalls = 8
        stepsPerEpoch = max(1, (numTrainFiles * numInstancesPerFile) // BATCH_SIZE)
        return {
            "mode": "tfrecord",
            "val_data": valData,
            "train_pattern": tfRecordFilePattern,
            "buffer_size": bufferSize,
            "parallel_calls": numParallelCalls,
            "steps_per_epoch": stepsPerEpoch,
        }

    train_instances, val_instances = collect_direct_instances(DATA_DIR, VAL_SPLIT, NEXT_IMG_INDEX, SEED)
    if not train_instances or not val_instances:
        raise ValueError("Direct mode requires both non-empty train and validation instance sets.")
    print(f"Direct mode: {len(train_instances)} train instances, {len(val_instances)} validation instances.")
    return {
        "mode": "direct",
        "train_instances": train_instances,
        "val_data": create_direct_dataset(val_instances, BATCH_SIZE, shuffle=False),
    }


ADADELTA = optimizers.Adadelta(learning_rate=1.0)
model = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)
model.compile(loss=custom_loss, optimizer=ADADELTA, metrics=[accuracy, precision, recall, f1], run_eagerly=True)

if LOAD_WEIGHTS is not None:
    model.load_weights(LOAD_WEIGHTS)

print(model.summary())

if not os.path.exists(SAVE_WEIGHTS):
    os.makedirs(SAVE_WEIGHTS)

sources = build_training_sources()

header = ["epoch", "loss", "val loss", "accuracy", "val_accuracy", "precision", "val_precision", "recall", "val_recall", "f1", "val_f1"]
csvPath = SAVE_WEIGHTS + "/Results.csv"
if not os.path.exists(csvPath):
    f = open(csvPath, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(header)
    f.flush()
else:
    f = open(csvPath, "a", newline="")
    writer = csv.writer(f)

for epoch in range(EPOCHS):
    print(f"\nTraining epoch {epoch + 1}/{EPOCHS}")

    if sources["mode"] == "tfrecord":
        dataset = createEpochDataset(
            sources["train_pattern"],
            sources["buffer_size"],
            BATCH_SIZE,
            sources["parallel_calls"],
        )
        history = model.fit(dataset, epochs=1, steps_per_epoch=sources["steps_per_epoch"], verbose=1)
    else:
        train_dataset = create_direct_dataset(sources["train_instances"], BATCH_SIZE, shuffle=True)
        history = model.fit(train_dataset, epochs=1, verbose=1)

    model_save_path = os.path.join(SAVE_WEIGHTS, "epoch_" + str(epoch + 1) + ".weights.h5")
    model.save_weights(model_save_path)

    values = list(history.history.values())

    print(f"\nEvaluating epoch {epoch + 1}/{EPOCHS}")
    valLoss, valAccuracy, valPrecision, valRecall, valF1 = model.evaluate(sources["val_data"], verbose=1)

    writer.writerow(
        [
            epoch + 1,
            round(values[0][0], 6),
            round(valLoss, 6),
            round(values[1][0], 6),
            round(valAccuracy, 6),
            round(values[2][0], 6),
            round(valPrecision, 6),
            round(values[3][0], 6),
            round(valRecall, 6),
            round(values[4][0], 6),
            round(valF1, 6),
        ]
    )
    f.flush()

print("\nDone.")
f.close()
