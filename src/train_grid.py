#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from model.GridTrackNet import GridTrackNet
from utils.grid_dataset import GridSequenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="GridTrackNet TensorFlow training")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to .weights.h5 checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0, help="Reserved for compatibility")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, gpu")
    parser.add_argument("--optimizer", choices=["Adadelta", "Adam", "AdamW", "SGD"], default="Adam")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["ReduceLROnPlateau", "None"], default="ReduceLROnPlateau")
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seq", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--grid_rows", type=int, default=27)
    parser.add_argument("--grid_cols", type=int, default=48)
    parser.add_argument("--tol", type=int, default=4, help="Reserved for compatibility")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--model_name", type=str, default="GridTrackNet")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Optional limit for training steps per epoch")
    parser.add_argument("--validation_steps", type=int, default=None, help="Optional limit for validation steps per epoch")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = {"Adadelta": 1.0, "Adam": 0.001, "AdamW": 0.001, "SGD": 0.01}[args.optimizer]
    if args.name is None:
        args.name = args.model_name
    return args


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


class KerasGridSequence(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        batch_x = []
        batch_y = []
        for dataset_idx in batch_indices:
            x, y = self.dataset[dataset_idx]
            batch_x.append(x)
            batch_y.append(y)
        return np.stack(batch_x), np.stack(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class GridTrackNetLoss(tf.keras.losses.Loss):
    def __init__(self, seq=5, conf_weight=1.0, offset_weight=0.001, alpha=0.75, gamma=2.0):
        super().__init__(name="gridtracknet_loss")
        self.seq = seq
        self.conf_weight = conf_weight
        self.offset_weight = offset_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-7

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        grid_rows = tf.shape(y_pred)[1]
        grid_cols = tf.shape(y_pred)[2]
        y_pred = tf.reshape(y_pred, [batch_size, grid_rows, grid_cols, self.seq, 3])
        y_true = tf.reshape(y_true, [batch_size, grid_rows, grid_cols, self.seq, 3])
        y_pred = tf.transpose(y_pred, [0, 3, 1, 2, 4])
        y_true = tf.transpose(y_true, [0, 3, 1, 2, 4])

        conf_true = y_true[..., 0:1]
        x_true = y_true[..., 1:2]
        y_true_offset = y_true[..., 2:3]
        conf_pred = tf.clip_by_value(y_pred[..., 0:1], self.eps, 1.0 - self.eps)
        x_pred = y_pred[..., 1:2]
        y_pred_offset = y_pred[..., 2:3]

        offset_true = tf.concat([x_true, y_true_offset], axis=-1)
        offset_pred = tf.concat([x_pred, y_pred_offset], axis=-1)
        diff = tf.reduce_sum(tf.abs(offset_true - offset_pred), axis=-1, keepdims=True)
        offset_loss = tf.reduce_mean(tf.reduce_sum(conf_true * diff, axis=[2, 3, 4]), axis=1)

        positive = self.alpha * conf_true * tf.pow(1.0 - conf_pred, self.gamma) * tf.math.log(conf_pred)
        negative = (1.0 - self.alpha) * (1.0 - conf_true) * tf.pow(conf_pred, self.gamma) * tf.math.log(1.0 - conf_pred)
        confidence_loss = -tf.reduce_mean(positive + negative, axis=[1, 2, 3, 4])

        loss = self.offset_weight * offset_loss + self.conf_weight * confidence_loss
        return tf.reduce_mean(loss)


def create_optimizer(args):
    if args.optimizer == "Adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=args.lr)
    if args.optimizer == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=args.lr)
    if args.optimizer == "AdamW":
        return tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.wd)
    return tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)


def tensorboard_available():
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


class ArtifactCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.ckpt_dir = save_dir / "checkpoints"
        self.metrics_csv = save_dir / "metrics.csv"
        self.best_val_loss = float("inf")
        with self.metrics_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "loss", "val_loss", "learning_rate"])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        latest_path = self.ckpt_dir / "latest.weights.h5"
        best_path = self.ckpt_dir / "best.weights.h5"
        self.model.save_weights(latest_path)

        val_loss = float(logs.get("val_loss", np.inf))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save_weights(best_path)

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        with self.metrics_csv.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch + 1,
                    round(float(logs.get("loss", np.nan)), 6),
                    round(val_loss, 6),
                    round(lr, 8),
                ]
            )


def main():
    args = parse_args()
    device = configure_runtime(args.device)

    train_ds = GridSequenceDataset(
        args.data,
        seq=args.seq,
        stride=args.stride,
        height=args.height,
        width=args.width,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        augment=True,
    )
    val_ds = GridSequenceDataset(
        args.val_data,
        seq=args.seq,
        stride=args.stride,
        height=args.height,
        width=args.width,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        augment=False,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"Empty training dataset: {args.data}")
    if len(val_ds) == 0:
        raise RuntimeError(f"Empty validation dataset: {args.val_data}")

    train_loader = KerasGridSequence(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = KerasGridSequence(val_ds, batch_size=args.batch, shuffle=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.out) / f"{args.name}_seq{args.seq}_{timestamp}"
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "tensorboard").mkdir(exist_ok=True)
    with (save_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Artifacts: {save_dir}")

    with tf.device(device):
        model = GridTrackNet(args.seq, args.height, args.width)
        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            model.load_weights(resume_path)
            print(f"Resumed weights: {resume_path}")

        model.compile(
            optimizer=create_optimizer(args),
            loss=GridTrackNetLoss(seq=args.seq),
            run_eagerly=False,
        )

        callbacks = [
            ArtifactCallback(save_dir),
        ]
        if tensorboard_available():
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(save_dir / "tensorboard")))
        else:
            print("TensorBoard is not installed in the active environment; skipping TensorBoard callback.")
        if args.scheduler == "ReduceLROnPlateau":
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=args.factor,
                    patience=args.patience,
                    min_lr=args.min_lr,
                    verbose=1,
                )
            )

        model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=args.epochs,
            callbacks=callbacks,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps,
            verbose=1,
        )

    print(f"Artifacts saved to {save_dir}")


if __name__ == "__main__":
    main()
