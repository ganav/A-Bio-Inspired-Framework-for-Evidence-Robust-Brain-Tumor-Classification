#!/usr/bin/env python3
"""Measure single-image classifier inference latency on Jetson TX2 (or desktop).

Uses the actual exported inference-only classifier from train_brain_mri_framework.py.
Does not load brain masks or run auxiliary training losses. Times include the
TensorFlow model call and output synchronization, but exclude disk image I/O,
image decoding, and image preprocessing.

Example:
 python3 benchmark_jetson_tx2.py \
   --model outputs/densenet121_proposed/brain_mri_classifier.h5 \
   --images Brain-Tumor-MRI-Dataset/Testing \
   --warmup 30 --iterations 200 --output tx2_densenet_proposed.json
"""
import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def image_paths(root):
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError('No MRI image files found under {}'.format(root))
    return paths


def read_grayscale(path, height, width):
    # TensorFlow decoding matches the provided training script: one channel,
    # bilinear resize, float32 range [0, 1]. No random augmentation at inference.
    raw = tf.io.read_file(str(path))
    x = tf.io.decode_image(raw, channels=1, expand_animations=False)
    x.set_shape((None, None, 1))
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = tf.image.resize(x, (height, width), method='bilinear')
    x = tf.clip_by_value(x, 0., 1.)
    return x.numpy()


def sync_predict(model, batch):
    y = model(batch, training=False)
    if isinstance(y, (tuple, list)):
        y = y[0]
    elif isinstance(y, dict):
        y = next(iter(y.values()))
    # Force the GPU work to complete before stopping timer.
    return y.numpy()


def try_command(*argv):
    try:
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=5, check=False)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='Exported inference-only .h5 or .keras model')
    ap.add_argument('--images', help='Test MRI image file or directory; otherwise synthetic inputs')
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--iterations', type=int, default=200)
    ap.add_argument('--output', default='jetson_tx2_benchmark.json')
    ap.add_argument('--per-sample-csv', help='Optional file of individual measured times')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.warmup < 0 or args.iterations < 2:
        ap.error('warmup must be >= 0 and iterations must be >= 2')

    model_path = Path(args.model)
    if not model_path.is_file():
        ap.error('Model does not exist: {}'.format(model_path))
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:
        raise RuntimeError('Cannot load this model with the installed TensorFlow/Keras. '
                           'Use a version-compatible export; do not report benchmark '
                           'measurements until it loads successfully. Original error: {}'.format(exc))
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        raise ValueError('Expected a single-input inference classifier')
    _, height, width, channels = input_shape
    if None in (height, width) or channels != 1:
        raise ValueError('Expected fixed-shape single-channel input, found {}'.format(input_shape))

    if args.images:
        paths = image_paths(args.images)
        arrays = [read_grayscale(p, height, width) for p in paths[:min(len(paths), args.iterations)]]
        source = 'real_test_mri'
    else:
        rng = np.random.RandomState(args.seed)
        arrays = [rng.uniform(0., 1., (height, width, 1)).astype('float32')]
        source = 'synthetic_random_input'
    batches = [tf.convert_to_tensor(a[None, ...], dtype=tf.float32) for a in arrays]
    for i in range(args.warmup):
        sync_predict(model, batches[i % len(batches)])

    ms = []
    for i in range(args.iterations):
        t0 = time.perf_counter()
        sync_predict(model, batches[i % len(batches)])
        ms.append((time.perf_counter() - t0) * 1000.)

    mean_ms = statistics.mean(ms)
    sorted_ms = np.asarray(ms)
    summary = {
        'model': str(model_path), 'platform': platform.platform(),
        'python': platform.python_version(), 'tensorflow': tf.__version__,
        'gpus': [d.name for d in tf.config.list_physical_devices('GPU')],
        'jetson_release': try_command('cat', '/etc/nv_tegra_release'),
        'input_shape': list(input_shape), 'source': source,
        'input_images_loaded': len(arrays), 'batch_size': 1,
        'warmup_iterations': args.warmup, 'timed_iterations': args.iterations,
        'latency_mean_ms': mean_ms, 'latency_median_ms': statistics.median(ms),
        'latency_std_ms': statistics.stdev(ms),
        'latency_p95_ms': float(np.percentile(sorted_ms, 95)),
        'fps_from_mean_latency': 1000. / mean_ms,
        'timing_scope': 'model forward pass plus output synchronization; preprocessing and file I/O excluded',
        'note': 'Measured latency is not a historical result; compare using identical device settings and model input shapes.'
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if args.per_sample_csv:
        csv_path = Path(args.per_sample_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['iteration', 'inference_ms'])
            w.writerows((i + 1, '{:.6f}'.format(v)) for i, v in enumerate(ms))
    print(json.dumps(summary, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    main()
