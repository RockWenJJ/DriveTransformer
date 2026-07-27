#!/usr/bin/env python3
"""
Interactive visualizer for DriveTransformer image-token 3D positional encoding.

This script mirrors the geometric part of:

  adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_head.py
  - prepare_location():              lines 737-745
  - img_3d_position_embedding():     lines 704-734
  - forward() image-token path:      lines 541-551

It produces a self-contained HTML file with:

  1. the H x W feature-grid token selected on the image plane;
  2. the LID (linear-increasing discretization) depth samples;
  3. [u*d, v*d, d, 1] inverse-projected into the LiDAR frame;
  4. the point-cloud-range normalization and inverse-sigmoid input;
  5. optional PCA maps of the learned raw/final PE captured from a real model.

Quick start (no dataset or checkpoint required):

  python tools/visualize_3d_position_embedding.py \
      --output outputs/drive_transformer_3d_pe.html

Use real calibration:

  python tools/visualize_3d_position_embedding.py \
      --input-npz /tmp/pe_capture.npz \
      --output outputs/drive_transformer_3d_pe_real.html

Capture the learned tensors during an existing model forward:

  from tools.visualize_3d_position_embedding import PositionEmbeddingCapture

  head = model.pts_bbox_head
  capture = PositionEmbeddingCapture(head)
  with torch.no_grad():
      _ = model(...)
  capture.save(
      "/tmp/pe_capture.npz",
      lidar2img=data["lidar2img"],
      feat_hw=(12, 33),
      pad_hw=(384, 1056),
      position_range=head.position_range,
  )
  capture.close()

The capture hooks observe:

  - img_position_encoder input/output: normalized ray coordinates before/after
    the learned MLP (line 730);
  - featurized_pe output: the final image-conditioned PE (line 551).

Only NumPy is required to generate the HTML. PyTorch is required only when
PositionEmbeddingCapture is used inside a model run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_POSITION_RANGE = np.asarray(
    [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0], dtype=np.float64
)
DEFAULT_CAMERA_NAMES = (
    "front",
    "front-left",
    "back-left",
    "back",
    "back-right",
    "front-right",
)


def lid_depth_samples(
    depth_num: int,
    depth_start: float,
    position_range: np.ndarray,
) -> np.ndarray:
    """Reproduce drivetransformer_head.py lines 202-206."""
    index = np.arange(depth_num, dtype=np.float64)
    bin_size = (
        float(position_range[3]) - depth_start
    ) / (depth_num * (1 + depth_num))
    return depth_start + bin_size * index * (index + 1)


def synthetic_lidar2img(
    pad_hw: Tuple[int, int],
    camera_yaws_deg: Sequence[float],
    focal_px: float = 720.0,
    camera_height: float = 1.55,
) -> np.ndarray:
    """
    Build simple six-camera pinhole matrices for the calibration-free demo.

    LiDAR axes: x forward, y left, z up.
    Camera axes: x right, y down, z forward.
    """
    pad_h, pad_w = pad_hw
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = focal_px
    intrinsic[1, 1] = focal_px
    intrinsic[0, 2] = pad_w / 2.0
    intrinsic[1, 2] = pad_h / 2.0

    matrices = []
    for yaw_deg in camera_yaws_deg:
        yaw = math.radians(float(yaw_deg))
        forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.asarray([math.sin(yaw), -math.cos(yaw), 0.0])
        down = np.asarray([0.0, 0.0, -1.0])

        camera_to_lidar = np.eye(4, dtype=np.float64)
        camera_to_lidar[:3, :3] = np.stack([right, down, forward], axis=1)
        camera_to_lidar[:3, 3] = np.asarray([0.0, 0.0, camera_height])
        lidar_to_camera = np.linalg.inv(camera_to_lidar)
        matrices.append(intrinsic @ lidar_to_camera)

    return np.stack(matrices, axis=0)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _first_batch(array: np.ndarray, expected_ndim_without_batch: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == expected_ndim_without_batch + 1:
        return array[0]
    return array


def randomized_pca_rgb(
    features: np.ndarray,
    seed: int = 7,
) -> Optional[np.ndarray]:
    """Reduce [tokens, channels] to three robustly scaled PCA-like channels."""
    if features is None:
        return None
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 3:
        features = features[0]
    if features.ndim != 2 or min(features.shape) < 3:
        return None

    finite = np.isfinite(features)
    if not finite.all():
        features = np.where(finite, features, 0.0)
    centered = features - features.mean(axis=0, keepdims=True)

    # Randomized SVD keeps capture visualization fast for [2376, 512].
    rng = np.random.default_rng(seed)
    projection_dim = min(12, centered.shape[1])
    omega = rng.standard_normal((centered.shape[1], projection_dim))
    q, _ = np.linalg.qr(centered @ omega, mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
    reduced = centered @ vt[:3].T

    low = np.percentile(reduced, 1.0, axis=0, keepdims=True)
    high = np.percentile(reduced, 99.0, axis=0, keepdims=True)
    reduced = np.clip((reduced - low) / np.maximum(high - low, 1e-8), 0.0, 1.0)
    return np.round(reduced * 255.0).astype(np.uint8)


class PositionEmbeddingCapture:
    """
    Non-invasive forward hooks for the learned part of the 3D PE.

    Import this class from an existing inference/debug script, construct it with
    ``model.pts_bbox_head``, run one forward, call ``save()``, then ``close()``.
    """

    def __init__(self, head: Any):
        self.head = head
        self.inverse_sigmoid_coords = None
        self.raw_position_embedding = None
        self.final_position_embedding = None
        self._handles = [
            head.img_position_encoder.register_forward_hook(
                self._capture_position_encoder
            ),
            head.featurized_pe.register_forward_hook(self._capture_final_pe),
        ]

    def _capture_position_encoder(self, module: Any, inputs: Any, output: Any) -> None:
        del module
        self.inverse_sigmoid_coords = _to_numpy(inputs[0])
        self.raw_position_embedding = _to_numpy(output)

    def _capture_final_pe(self, module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        self.final_position_embedding = _to_numpy(output)

    def save(
        self,
        path: Union[str, Path],
        *,
        lidar2img: Any,
        feat_hw: Tuple[int, int],
        pad_hw: Tuple[int, int],
        position_range: Any,
        stride: int = 32,
        camera_names: Optional[Sequence[str]] = None,
    ) -> Path:
        if self.inverse_sigmoid_coords is None:
            raise RuntimeError("No PE was captured. Run one model forward before save().")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        matrices = _to_numpy(lidar2img)
        if matrices.ndim == 4:
            matrices = matrices[0]
        names = tuple(camera_names or DEFAULT_CAMERA_NAMES[: matrices.shape[0]])

        arrays = {
            "lidar2img": matrices,
            "inverse_sigmoid_coords": self.inverse_sigmoid_coords,
            "position_range": _to_numpy(position_range).reshape(-1),
            "feat_h": np.asarray(feat_hw[0]),
            "feat_w": np.asarray(feat_hw[1]),
            "pad_h": np.asarray(pad_hw[0]),
            "pad_w": np.asarray(pad_hw[1]),
            "stride": np.asarray(stride),
            "camera_names": np.asarray(names),
        }
        if self.raw_position_embedding is not None:
            arrays["raw_position_embedding"] = self.raw_position_embedding
        if self.final_position_embedding is not None:
            arrays["final_position_embedding"] = self.final_position_embedding
        np.savez_compressed(destination, **arrays)
        return destination

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "PositionEmbeddingCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def load_payload(args: argparse.Namespace) -> Dict[str, Any]:
    position_range = DEFAULT_POSITION_RANGE.copy()
    feat_h, feat_w = args.feat_h, args.feat_w
    pad_h, pad_w = args.pad_h, args.pad_w
    stride = args.stride
    raw_pe = None
    final_pe = None

    if args.input_npz:
        with np.load(args.input_npz, allow_pickle=False) as capture:
            if "lidar2img" not in capture:
                raise KeyError(
                    "The NPZ must contain 'lidar2img'. Use PositionEmbeddingCapture.save()."
                )
            lidar2img = _first_batch(capture["lidar2img"], 3).astype(np.float64)
            if "position_range" in capture:
                position_range = capture["position_range"].reshape(-1).astype(
                    np.float64
                )
            feat_h = int(capture["feat_h"]) if "feat_h" in capture else feat_h
            feat_w = int(capture["feat_w"]) if "feat_w" in capture else feat_w
            pad_h = int(capture["pad_h"]) if "pad_h" in capture else pad_h
            pad_w = int(capture["pad_w"]) if "pad_w" in capture else pad_w
            stride = int(capture["stride"]) if "stride" in capture else stride
            if "camera_names" in capture:
                camera_names = [str(value) for value in capture["camera_names"]]
            else:
                camera_names = [
                    f"camera-{index}" for index in range(lidar2img.shape[0])
                ]
            if "raw_position_embedding" in capture:
                raw_pe = capture["raw_position_embedding"]
            if "final_position_embedding" in capture:
                final_pe = capture["final_position_embedding"]
            source_mode = "real capture"
    else:
        yaws = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
        lidar2img = synthetic_lidar2img(
            (pad_h, pad_w),
            yaws,
            focal_px=args.synthetic_focal,
            camera_height=args.synthetic_camera_height,
        )
        camera_names = list(DEFAULT_CAMERA_NAMES)
        source_mode = "synthetic calibration"

    if lidar2img.ndim != 3 or lidar2img.shape[-2:] != (4, 4):
        raise ValueError(
            f"lidar2img must have shape [Ncam,4,4], got {lidar2img.shape}"
        )

    expected_tokens = lidar2img.shape[0] * feat_h * feat_w
    raw_rgb = randomized_pca_rgb(raw_pe)
    final_rgb = randomized_pca_rgb(final_pe)
    if raw_rgb is not None and raw_rgb.shape[0] != expected_tokens:
        raise ValueError(
            f"raw PE contains {raw_rgb.shape[0]} tokens, expected {expected_tokens}"
        )
    if final_rgb is not None and final_rgb.shape[0] != expected_tokens:
        raise ValueError(
            f"final PE contains {final_rgb.shape[0]} tokens, expected {expected_tokens}"
        )

    depths = lid_depth_samples(args.depth_num, args.depth_start, position_range)
    img2lidar = np.linalg.inv(lidar2img)
    payload = {
        "sourceMode": source_mode,
        "cameraNames": camera_names,
        "img2lidar": np.round(img2lidar, 9).tolist(),
        "positionRange": np.round(position_range, 6).tolist(),
        "depths": np.round(depths, 6).tolist(),
        "featH": feat_h,
        "featW": feat_w,
        "padH": pad_h,
        "padW": pad_w,
        "stride": stride,
        "rawPeRgb": raw_rgb.tolist() if raw_rgb is not None else None,
        "finalPeRgb": final_rgb.tolist() if final_rgb is not None else None,
    }
    return payload


FRAGMENT_TEMPLATE = r"""
<div id="dt-pe-viz">
  <style>
    #dt-pe-viz {
      color: var(--foreground);
      background: transparent;
      width: 100%;
    }
    #dt-pe-viz .pe-steps {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }
    #dt-pe-viz .pe-controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.75rem;
      margin-bottom: 1rem;
    }
    #dt-pe-viz .pe-control {
      min-width: 0;
    }
    #dt-pe-viz .pe-control output {
      color: var(--muted-foreground);
      margin-left: 0.35rem;
    }
    #dt-pe-viz .pe-stage {
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(15rem, 0.8fr);
      gap: 1rem;
      align-items: stretch;
    }
    #dt-pe-viz .pe-main {
      position: relative;
      min-width: 0;
    }
    #dt-pe-viz canvas {
      display: block;
      width: 100%;
      color: var(--foreground);
    }
    #dt-pe-viz #pe-main-canvas {
      height: 25rem;
      touch-action: none;
      cursor: grab;
    }
    #dt-pe-viz #pe-main-canvas:active {
      cursor: grabbing;
    }
    #dt-pe-viz .pe-side {
      display: grid;
      grid-template-rows: repeat(3, minmax(7rem, 1fr));
      gap: 0.75rem;
      min-width: 0;
    }
    #dt-pe-viz .pe-side canvas {
      height: 100%;
      min-height: 7rem;
    }
    #dt-pe-viz .pe-detail {
      margin-top: 0.8rem;
      color: var(--muted-foreground);
      overflow-wrap: anywhere;
    }
    #dt-pe-viz .pe-detail code {
      color: var(--foreground);
    }
    #dt-pe-viz .pe-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-top: 0.5rem;
      color: var(--muted-foreground);
    }
    #dt-pe-viz .pe-legend span::before {
      content: "";
      display: inline-block;
      width: 0.7rem;
      height: 0.7rem;
      margin-right: 0.35rem;
      border-radius: 999px;
      background: var(--viz-series-1);
    }
    #dt-pe-viz .pe-legend span:nth-child(2)::before {
      background: var(--viz-series-2);
    }
    #dt-pe-viz .pe-legend span:nth-child(3)::before {
      background: var(--muted-foreground);
    }
    @media (max-width: 720px) {
      #dt-pe-viz .pe-controls {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      #dt-pe-viz .pe-stage {
        grid-template-columns: 1fr;
      }
      #dt-pe-viz .pe-side {
        grid-template-columns: 1fr;
        grid-template-rows: repeat(3, 9rem);
      }
      #dt-pe-viz #pe-main-canvas {
        height: 21rem;
      }
    }
    @media (max-width: 420px) {
      #dt-pe-viz .pe-controls {
        grid-template-columns: 1fr;
      }
    }
  </style>

  <div class="pe-steps" role="group" aria-label="3D position encoding stages">
    <button type="button" class="btn btn-primary" data-step="grid" aria-pressed="true">1 · Feature grid</button>
    <button type="button" class="btn" data-step="depth" aria-pressed="false">2 · LID depth</button>
    <button type="button" class="btn" data-step="homogeneous" aria-pressed="false">3 · [ud,vd,d,1]</button>
    <button type="button" class="btn" data-step="inverse" aria-pressed="false">4 · inverse(lidar2img)</button>
    <button type="button" class="btn" data-step="normalize" aria-pressed="false">5 · normalize + logit</button>
    <button type="button" class="btn" data-step="mlp" aria-pressed="false">6 · learned MLP</button>
  </div>

  <div class="pe-controls viz-controls">
    <label class="pe-control form-label">
      Camera
      <select id="pe-camera" class="form-select"></select>
    </label>
    <label class="pe-control form-label">
      Feature row <output id="pe-row-value"></output>
      <input id="pe-row" class="form-range" type="range" min="0" step="1">
    </label>
    <label class="pe-control form-label">
      Feature column <output id="pe-col-value"></output>
      <input id="pe-col" class="form-range" type="range" min="0" step="1">
    </label>
    <label class="pe-control form-label">
      Depth sample <output id="pe-depth-value"></output>
      <input id="pe-depth" class="form-range" type="range" min="0" step="1">
    </label>
  </div>

  <div class="pe-stage">
    <div class="pe-main">
      <canvas id="pe-main-canvas" role="img" aria-label="Interactive LiDAR-frame camera rays. Drag to rotate."></canvas>
      <div class="pe-legend text-small">
        <span>Selected ray</span>
        <span>Selected depth point</span>
        <span>Other cameras</span>
      </div>
    </div>
    <div class="pe-side">
      <canvas id="pe-grid-canvas" role="img" aria-label="Image feature grid and selected token"></canvas>
      <canvas id="pe-depth-canvas" role="img" aria-label="LID depth sample distribution"></canvas>
      <canvas id="pe-heat-canvas" role="img" aria-label="The selected ray's 3 by 64 inverse-sigmoid MLP input"></canvas>
    </div>
  </div>

  <div id="pe-detail" class="pe-detail text-small" aria-live="polite"></div>

  <script>
    (function () {
      "use strict";
      const DATA = __PAYLOAD__;
      const root = document.getElementById("dt-pe-viz");
      const cameraSelect = root.querySelector("#pe-camera");
      const rowInput = root.querySelector("#pe-row");
      const colInput = root.querySelector("#pe-col");
      const depthInput = root.querySelector("#pe-depth");
      const rowValue = root.querySelector("#pe-row-value");
      const colValue = root.querySelector("#pe-col-value");
      const depthValue = root.querySelector("#pe-depth-value");
      const detail = root.querySelector("#pe-detail");
      const mainCanvas = root.querySelector("#pe-main-canvas");
      const gridCanvas = root.querySelector("#pe-grid-canvas");
      const depthCanvas = root.querySelector("#pe-depth-canvas");
      const heatCanvas = root.querySelector("#pe-heat-canvas");

      let activeStep = "grid";
      let viewYaw = -0.62;
      let viewPitch = 0.72;
      let dragging = false;
      let pointerX = 0;
      let pointerY = 0;

      function css(name, fallback) {
        const value = getComputedStyle(root).getPropertyValue(name).trim();
        return value || fallback;
      }

      function fitCanvas(canvas) {
        const ratio = Math.max(1, window.devicePixelRatio || 1);
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width * ratio));
        const height = Math.max(1, Math.round(rect.height * ratio));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        const context = canvas.getContext("2d");
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        return { context: context, width: rect.width, height: rect.height };
      }

      function matVec(matrix, vector) {
        return matrix.map(function (row) {
          return row.reduce(function (sum, value, index) {
            return sum + value * vector[index];
          }, 0);
        });
      }

      function pixelCenter(row, column) {
        return [
          column * DATA.stride + Math.floor(DATA.stride / 2),
          row * DATA.stride + Math.floor(DATA.stride / 2)
        ];
      }

      function rayPoint(camera, row, column, depthIndex) {
        const uv = pixelCenter(row, column);
        const depth = DATA.depths[depthIndex];
        const point = matVec(
          DATA.img2lidar[camera],
          [uv[0] * depth, uv[1] * depth, depth, 1]
        );
        return point.slice(0, 3);
      }

      function cameraCenter(camera) {
        return matVec(DATA.img2lidar[camera], [0, 0, 0, 1]).slice(0, 3);
      }

      function normalizedPoint(point) {
        return point.map(function (value, axis) {
          return (
            (value - DATA.positionRange[axis]) /
            (DATA.positionRange[axis + 3] - DATA.positionRange[axis])
          );
        });
      }

      function logit(value) {
        const clipped = Math.min(1 - 1e-5, Math.max(1e-5, value));
        return Math.log(clipped / (1 - clipped));
      }

      function project(point, width, height) {
        const cosYaw = Math.cos(viewYaw);
        const sinYaw = Math.sin(viewYaw);
        const x1 = cosYaw * point[0] - sinYaw * point[1];
        const y1 = sinYaw * point[0] + cosYaw * point[1];
        const cosPitch = Math.cos(viewPitch);
        const sinPitch = Math.sin(viewPitch);
        const y2 = cosPitch * y1 - sinPitch * point[2];
        const z2 = sinPitch * y1 + cosPitch * point[2];
        const span = Math.max(
          DATA.positionRange[3] - DATA.positionRange[0],
          DATA.positionRange[4] - DATA.positionRange[1]
        );
        const scale = Math.min(width, height) * 0.72 / span;
        return [width * 0.5 + x1 * scale, height * 0.56 - y2 * scale, z2];
      }

      function line3d(context, points, width, height, color, lineWidth, alpha) {
        const projected = points.map(function (point) {
          return project(point, width, height);
        });
        context.save();
        context.strokeStyle = color;
        context.globalAlpha = alpha;
        context.lineWidth = lineWidth;
        context.beginPath();
        projected.forEach(function (point, index) {
          if (index === 0) context.moveTo(point[0], point[1]);
          else context.lineTo(point[0], point[1]);
        });
        context.stroke();
        context.restore();
      }

      function drawRangeBox(context, width, height) {
        const range = DATA.positionRange;
        const x0 = range[0], y0 = range[1], z0 = range[2];
        const x1 = range[3], y1 = range[4], z1 = range[5];
        const corners = [
          [x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0],
          [x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1]
        ];
        const edges = [
          [0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4],
          [0,4],[1,5],[2,6],[3,7]
        ];
        edges.forEach(function (edge) {
          line3d(
            context,
            [corners[edge[0]], corners[edge[1]]],
            width,
            height,
            css("--border", "#667085"),
            activeStep === "normalize" ? 1.5 : 1,
            activeStep === "normalize" ? 0.85 : 0.35
          );
        });
      }

      function drawMain() {
        const fitted = fitCanvas(mainCanvas);
        const context = fitted.context;
        const width = fitted.width;
        const height = fitted.height;
        context.clearRect(0, 0, width, height);
        drawRangeBox(context, width, height);

        const camera = Number(cameraSelect.value);
        const row = Number(rowInput.value);
        const column = Number(colInput.value);
        const selectedDepth = Number(depthInput.value);
        const neutral = css("--muted-foreground", "#667085");
        const selected = css("--viz-series-1", "#2563eb");
        const pointColor = css("--viz-series-2", "#f97316");

        DATA.cameraNames.forEach(function (_, cameraIndex) {
          const points = [cameraCenter(cameraIndex)];
          DATA.depths.forEach(function (_, depthIndex) {
            points.push(rayPoint(cameraIndex, row, column, depthIndex));
          });
          line3d(
            context,
            points,
            width,
            height,
            cameraIndex === camera ? selected : neutral,
            cameraIndex === camera ? 2.5 : 1,
            cameraIndex === camera ? 0.95 : 0.22
          );
        });

        if (activeStep === "inverse" || activeStep === "homogeneous") {
          [-2, 0, 2].forEach(function (rowDelta) {
            [-2, 0, 2].forEach(function (columnDelta) {
              const neighborRow = Math.max(
                0, Math.min(DATA.featH - 1, row + rowDelta)
              );
              const neighborColumn = Math.max(
                0, Math.min(DATA.featW - 1, column + columnDelta)
              );
              const points = [cameraCenter(camera)];
              DATA.depths.forEach(function (_, depthIndex) {
                points.push(
                  rayPoint(camera, neighborRow, neighborColumn, depthIndex)
                );
              });
              line3d(
                context, points, width, height, selected, 1, 0.18
              );
            });
          });
        }

        const selectedPoint = rayPoint(camera, row, column, selectedDepth);
        const projected = project(selectedPoint, width, height);
        context.save();
        context.fillStyle = pointColor;
        context.beginPath();
        context.arc(projected[0], projected[1], 5, 0, Math.PI * 2);
        context.fill();
        context.restore();

        context.save();
        context.fillStyle = css("--muted-foreground", "#667085");
        context.font = "12px sans-serif";
        context.fillText("LiDAR x / y / z · drag to rotate", 10, 18);
        context.restore();
      }

      function drawGrid() {
        const fitted = fitCanvas(gridCanvas);
        const context = fitted.context;
        const width = fitted.width;
        const height = fitted.height;
        context.clearRect(0, 0, width, height);
        const cellWidth = width / DATA.featW;
        const cellHeight = Math.max(1, (height - 22) / DATA.featH);
        const camera = Number(cameraSelect.value);
        const selectedRow = Number(rowInput.value);
        const selectedColumn = Number(colInput.value);
        const mapMode = activeStep === "mlp"
          ? (DATA.finalPeRgb ? "final" : DATA.rawPeRgb ? "raw" : "none")
          : "none";
        const map = mapMode === "final" ? DATA.finalPeRgb : DATA.rawPeRgb;

        for (let row = 0; row < DATA.featH; row += 1) {
          for (let column = 0; column < DATA.featW; column += 1) {
            const token = camera * DATA.featH * DATA.featW
              + row * DATA.featW + column;
            if (map) {
              const rgb = map[token];
              context.fillStyle = "rgb(" + rgb.join(",") + ")";
              context.globalAlpha = 0.82;
            } else {
              context.fillStyle = css("--muted", "#d1d5db");
              context.globalAlpha = 0.45;
            }
            context.fillRect(
              column * cellWidth,
              row * cellHeight + 22,
              Math.max(1, cellWidth - 0.6),
              Math.max(1, cellHeight - 0.6)
            );
          }
        }
        context.globalAlpha = 1;
        context.strokeStyle = css("--viz-series-2", "#f97316");
        context.lineWidth = 2;
        context.strokeRect(
          selectedColumn * cellWidth,
          selectedRow * cellHeight + 22,
          cellWidth,
          cellHeight
        );
        context.fillStyle = css("--muted-foreground", "#667085");
        context.font = "12px sans-serif";
        const label = mapMode === "none"
          ? "feature tokens · " + DATA.featH + "×" + DATA.featW
          : mapMode + " PE · PCA→RGB";
        context.fillText(label, 4, 14);
      }

      function drawDepths() {
        const fitted = fitCanvas(depthCanvas);
        const context = fitted.context;
        const width = fitted.width;
        const height = fitted.height;
        context.clearRect(0, 0, width, height);
        const pad = 20;
        const selectedDepth = Number(depthInput.value);
        const maxDepth = DATA.depths[DATA.depths.length - 1];

        context.strokeStyle = css("--border", "#667085");
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(pad, height - pad);
        context.lineTo(width - 8, height - pad);
        context.stroke();

        context.strokeStyle = css("--viz-series-1", "#2563eb");
        context.lineWidth = 2;
        context.beginPath();
        DATA.depths.forEach(function (depth, index) {
          const x = pad + index / (DATA.depths.length - 1) * (width - pad - 10);
          const y = height - pad - depth / maxDepth * (height - 2 * pad);
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.stroke();

        const sx = pad + selectedDepth / (DATA.depths.length - 1)
          * (width - pad - 10);
        const sy = height - pad - DATA.depths[selectedDepth] / maxDepth
          * (height - 2 * pad);
        context.fillStyle = css("--viz-series-2", "#f97316");
        context.beginPath();
        context.arc(sx, sy, 4, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = css("--muted-foreground", "#667085");
        context.font = "12px sans-serif";
        context.fillText("LID depth · meters", 4, 14);
      }

      function drawHeatmap() {
        const fitted = fitCanvas(heatCanvas);
        const context = fitted.context;
        const width = fitted.width;
        const height = fitted.height;
        context.clearRect(0, 0, width, height);
        const camera = Number(cameraSelect.value);
        const row = Number(rowInput.value);
        const column = Number(colInput.value);
        const top = 22;
        const left = 16;
        const cellWidth = (width - left) / DATA.depths.length;
        const cellHeight = (height - top - 5) / 3;
        const positive = css("--viz-series-1", "#2563eb");
        const negative = css("--viz-series-2", "#f97316");

        for (let depthIndex = 0; depthIndex < DATA.depths.length; depthIndex += 1) {
          const point = rayPoint(camera, row, column, depthIndex);
          const normalized = normalizedPoint(point);
          normalized.forEach(function (value, axis) {
            const encoded = logit(value);
            context.globalAlpha = Math.min(0.95, 0.12 + Math.abs(encoded) / 8);
            context.fillStyle = encoded >= 0 ? positive : negative;
            context.fillRect(
              left + depthIndex * cellWidth,
              top + axis * cellHeight,
              Math.ceil(cellWidth),
              Math.ceil(cellHeight)
            );
          });
        }
        context.globalAlpha = 1;
        context.fillStyle = css("--muted-foreground", "#667085");
        context.font = "12px sans-serif";
        context.fillText("inverse_sigmoid(coords3d) · 3×64", 4, 14);
        context.fillText("x", 3, top + cellHeight * 0.7);
        context.fillText("y", 3, top + cellHeight * 1.7);
        context.fillText("z", 3, top + cellHeight * 2.7);
      }

      function updateDetail() {
        const camera = Number(cameraSelect.value);
        const row = Number(rowInput.value);
        const column = Number(colInput.value);
        const depthIndex = Number(depthInput.value);
        const depth = DATA.depths[depthIndex];
        const uv = pixelCenter(row, column);
        const homogeneous = [uv[0] * depth, uv[1] * depth, depth, 1];
        const point = rayPoint(camera, row, column, depthIndex);
        const normalized = normalizedPoint(point);
        const encoded = normalized.map(logit);
        const token = camera * DATA.featH * DATA.featW
          + row * DATA.featW + column;
        const values = {
          grid: "token " + token + " · pixel center (u,v)=(" +
            uv.map(function (v) { return v.toFixed(1); }).join(", ") + ")",
          depth: "d[" + depthIndex + "]=" + depth.toFixed(3) +
            " m · LID spacing grows quadratically",
          homogeneous: "[u·d,v·d,d,1]=[" +
            homogeneous.map(function (v) { return v.toFixed(3); }).join(", ") +
            "]",
          inverse: "LiDAR xyz=[" +
            point.map(function (v) { return v.toFixed(3); }).join(", ") + "] m",
          normalize: "normalized xyz=[" +
            normalized.map(function (v) { return v.toFixed(3); }).join(", ") +
            "] · logit=[" +
            encoded.map(function (v) { return v.toFixed(3); }).join(", ") + "]",
          mlp: DATA.finalPeRgb
            ? "img_position_encoder → 512-D · featurized_pe → final 512-D · grid shows final PE PCA→RGB"
            : DATA.rawPeRgb
              ? "img_position_encoder → captured raw PE · grid shows raw PE PCA→RGB"
              : "geometry input is 64×3=192-D · load a capture NPZ to visualize the learned 512-D PE"
        };
        detail.innerHTML = "<code>" + values[activeStep] + "</code>" +
          " · " + DATA.cameraNames[camera] + " · " + DATA.sourceMode;
        mainCanvas.setAttribute(
          "aria-label",
          "Selected camera " + DATA.cameraNames[camera] +
          ", feature row " + row + ", column " + column +
          ", depth " + depth.toFixed(3) + " meters."
        );
      }

      function render() {
        rowValue.value = rowInput.value + " / " + (DATA.featH - 1);
        colValue.value = colInput.value + " / " + (DATA.featW - 1);
        depthValue.value = Number(DATA.depths[Number(depthInput.value)]).toFixed(2)
          + " m";
        drawMain();
        drawGrid();
        drawDepths();
        drawHeatmap();
        updateDetail();
      }

      DATA.cameraNames.forEach(function (name, index) {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = index + " · " + name;
        cameraSelect.appendChild(option);
      });
      rowInput.max = String(DATA.featH - 1);
      rowInput.value = String(Math.floor(DATA.featH / 2));
      colInput.max = String(DATA.featW - 1);
      colInput.value = String(Math.floor(DATA.featW / 2));
      depthInput.max = String(DATA.depths.length - 1);
      depthInput.value = String(Math.floor(DATA.depths.length * 0.6));

      [cameraSelect, rowInput, colInput, depthInput].forEach(function (control) {
        control.addEventListener("input", render);
        control.addEventListener("change", render);
      });

      root.querySelectorAll("[data-step]").forEach(function (button) {
        button.addEventListener("click", function () {
          activeStep = button.dataset.step;
          root.querySelectorAll("[data-step]").forEach(function (candidate) {
            const selected = candidate === button;
            candidate.setAttribute("aria-pressed", String(selected));
            candidate.classList.toggle("btn-primary", selected);
          });
          render();
        });
      });

      mainCanvas.addEventListener("pointerdown", function (event) {
        dragging = true;
        pointerX = event.clientX;
        pointerY = event.clientY;
        mainCanvas.setPointerCapture(event.pointerId);
      });
      mainCanvas.addEventListener("pointermove", function (event) {
        if (!dragging) return;
        viewYaw += (event.clientX - pointerX) * 0.008;
        viewPitch = Math.max(
          -1.25,
          Math.min(1.25, viewPitch + (event.clientY - pointerY) * 0.008)
        );
        pointerX = event.clientX;
        pointerY = event.clientY;
        drawMain();
      });
      mainCanvas.addEventListener("pointerup", function () {
        dragging = false;
      });
      mainCanvas.addEventListener("pointercancel", function () {
        dragging = false;
      });

      const observer = new ResizeObserver(render);
      observer.observe(root);
      render();
    })();
  </script>
</div>
"""


STANDALONE_PREFIX = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DriveTransformer 3D Position Encoding Lab</title>
  <style>
    :root {
      --background: #f7f7f5;
      --foreground: #172033;
      --card: #ffffff;
      --card-foreground: #172033;
      --primary: #2657d9;
      --primary-foreground: #ffffff;
      --secondary: #eceff5;
      --secondary-foreground: #172033;
      --muted: #e5e7eb;
      --muted-foreground: #667085;
      --accent: #e6ecff;
      --accent-foreground: #173da8;
      --border: #cfd4dc;
      --input: #cfd4dc;
      --ring: #2657d9;
      --viz-series-1: #2657d9;
      --viz-series-2: #f2762e;
      --viz-series-3: #2a9d6f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 2rem;
      color: var(--foreground);
      background: var(--background);
      font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    }
    main { width: min(76rem, 100%); margin: 0 auto; }
    h1 { margin: 0 0 0.25rem; font-weight: 500; }
    .intro { color: var(--muted-foreground); margin: 0 0 1.5rem; }
    .btn {
      appearance: none;
      border: 1px solid var(--border);
      border-radius: 0.55rem;
      padding: 0.55rem 0.75rem;
      color: var(--foreground);
      background: var(--card);
      cursor: pointer;
    }
    .btn-primary {
      color: var(--primary-foreground);
      background: var(--primary);
      border-color: var(--primary);
    }
    .form-label { display: block; }
    .form-select {
      display: block;
      width: 100%;
      margin-top: 0.35rem;
      border: 1px solid var(--input);
      border-radius: 0.45rem;
      padding: 0.45rem;
      color: var(--foreground);
      background: var(--card);
    }
    .form-range { display: block; width: 100%; margin-top: 0.55rem; }
    .text-small { font-size: 0.82rem; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    @media (max-width: 540px) {
      body { padding: 1rem; }
    }
  </style>
</head>
<body>
  <main>
    <h1>DriveTransformer 3D Position Encoding Lab</h1>
    <p class="intro">对应 drivetransformer_head.py 第 541–551、704–734 行</p>
"""

STANDALONE_SUFFIX = """
  </main>
</body>
</html>
"""


def render_html(payload: Dict[str, Any], standalone: bool) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    fragment = FRAGMENT_TEMPLATE.replace("__PAYLOAD__", payload_json).strip() + "\n"
    if standalone:
        return STANDALONE_PREFIX + fragment + STANDALONE_SUFFIX
    return fragment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DriveTransformer's image-token 3D positional encoding."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        help="Optional capture/calibration NPZ. Default: synthetic six-camera rig.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/drive_transformer_3d_pe.html"),
        help="HTML destination.",
    )
    parser.add_argument(
        "--fragment",
        action="store_true",
        help="Write an embeddable HTML fragment instead of a standalone document.",
    )
    parser.add_argument("--feat-h", type=int, default=12)
    parser.add_argument("--feat-w", type=int, default=33)
    parser.add_argument("--pad-h", type=int, default=384)
    parser.add_argument("--pad-w", type=int, default=1056)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--depth-num", type=int, default=64)
    parser.add_argument("--depth-start", type=float, default=1.0)
    parser.add_argument("--synthetic-focal", type=float, default=720.0)
    parser.add_argument("--synthetic-camera-height", type=float, default=1.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_payload(args)
    html = render_html(payload, standalone=not args.fragment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output.resolve())
    print(
        f"{payload['sourceMode']} | cameras={len(payload['cameraNames'])} | "
        f"tokens={len(payload['cameraNames']) * payload['featH'] * payload['featW']} | "
        f"depth_samples={len(payload['depths'])}"
    )


if __name__ == "__main__":
    main()
