#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight WebSocket policy server (msgpack) aligned with actibot serve_policy."""

from __future__ import annotations

import argparse
import asyncio
import logging
import numpy as np
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

try:
    import msgpack_numpy
    import msgpack
    import websockets
    import websockets.asyncio.server as ws_server
    import websockets.frames
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guardrail
    missing = exc.name
    msg = (
        f"Missing dependency '{missing}'. Install msgpack-numpy and websockets to run lerobot-serve.\n"
        "Example: pip install msgpack-numpy websockets"
    )
    raise SystemExit(msg) from exc

from lerobot.configs import parser as cfg_parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve a LeRobot policy over WebSocket (msgpack format).")
    p.add_argument("--policy.path", required=True, dest="policy_path", help="Local dir or Hub repo id.")
    p.add_argument(
        "--policy.device",
        dest="policy_device",
        default=None,
        help="Device to load policy on (cuda, cpu, mps...).",
    )
    p.add_argument(
        "--task",
        dest="default_task",
        default=None,
        help="Optional default task/prompt string if client payload does not provide one.",
    )
    p.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    p.add_argument("--port", type=int, default=8000, help="Port to bind.")
    p.add_argument(
        "--record-dir",
        default=None,
        help="If set, save request/response msgpack frames for debugging (files under this directory).",
    )
    p.add_argument("--max-msg-size", type=int, default=None, help="Max WS message size in bytes (None = unlimited).")
    return p.parse_args()


def load_policy(policy_path: str, device: str | None) -> tuple[PreTrainedPolicy, Any, Any, dict[str, Any]]:
    # Allow CLI overrides for policy.* if provided (draccus parser)
    cli_overrides = cfg_parser.get_cli_overrides("policy")
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
    cfg.pretrained_path = policy_path
    if device is not None:
        cfg.device = device

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=policy_path, config=cfg)
    policy.to(cfg.device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )

    metadata = {
        "policy_name": getattr(cfg, "type", None),
        "device": policy.device if hasattr(policy, "device") else str(device),
        "obs_keys": list(cfg.input_features) if hasattr(cfg, "input_features") else None,
        "action_dim": getattr(cfg, "action_dim", None),
    }
    # Ensure metadata is available to the server handler
    setattr(policy, "metadata", metadata)
    return policy, preprocessor, postprocessor, metadata


def _piper_obs_from_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Parse PiperEffortEnvironment observation into flat observation.* keys."""
    obs = payload.get("obs")
    if not isinstance(obs, dict):
        raise ValueError("Expected payload['obs'] to be a dict from PiperEffortEnvironment.")

    task_val = obs.get("prompt") or obs.get("task")

    obs_flat: dict[str, Any] = {}
    if "state" in obs:
        obs_flat[f"{OBS_STR}.state"] = obs["state"]
    if "effort" in obs:
        eff = obs["effort"]
        if isinstance(eff, (list, tuple)):
            eff = eff[-1]
        else:
            eff = np.asarray(eff)
            if eff.ndim == 2:
                eff = eff[-1]
        obs_flat[f"{OBS_STR}.effort"] = eff
    if "velocity" in obs:
        vel = obs["velocity"]
        if isinstance(vel, (list, tuple)):
            vel = vel[-1]
        else:
            vel = np.asarray(vel)
            if vel.ndim == 2:
                vel = vel[-1]
        obs_flat[f"{OBS_STR}.velocity"] = vel

    imgs = obs.get("images")
    if isinstance(imgs, dict):
        if "cam_high" in imgs:
            obs_flat[f"{OBS_STR}.images.cam_top"] = imgs["cam_high"]
        if "cam_left_wrist" in imgs:
            obs_flat[f"{OBS_STR}.images.cam_left"] = imgs["cam_left_wrist"]
        if "cam_right_wrist" in imgs:
            obs_flat[f"{OBS_STR}.images.cam_right"] = imgs["cam_right_wrist"]
        if "cam_left" in imgs:
            obs_flat[f"{OBS_STR}.images.cam_left"] = imgs["cam_left"]
        if "cam_right" in imgs:
            obs_flat[f"{OBS_STR}.images.cam_right"] = imgs["cam_right"]

    return obs_flat, task_val


def _normalize_payload(obj: Any) -> Any:
    """Decode bytes keys/values into str and normalize nested dict/list structures."""
    if isinstance(obj, dict):
        # Handle msgpack-numpy encoded arrays (from openpi_client)
        if b"__ndarray__" in obj or "__ndarray__" in obj:
            data = obj.get(b"data") if b"data" in obj else obj.get("data")
            dtype = obj.get(b"dtype") if b"dtype" in obj else obj.get("dtype")
            shape = obj.get(b"shape") if b"shape" in obj else obj.get("shape")
            return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=tuple(shape))
        if b"__npgeneric__" in obj or "__npgeneric__" in obj:
            data = obj.get(b"data") if b"data" in obj else obj.get("data")
            dtype = obj.get(b"dtype") if b"dtype" in obj else obj.get("dtype")
            return np.dtype(dtype).type(data)

        normalized: dict[str, Any] = {}
        for key, val in obj.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="ignore")
            normalized[key] = _normalize_payload(val)
        return normalized
    if isinstance(obj, list):
        return [_normalize_payload(item) for item in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    return obj


def _describe_value(val: Any) -> str:
    if torch.is_tensor(val):
        return f"tensor shape={tuple(val.shape)} dtype={val.dtype}"
    if isinstance(val, np.ndarray):
        return f"ndarray shape={val.shape} dtype={val.dtype}"
    if isinstance(val, list):
        return f"list len={len(val)}"
    if isinstance(val, dict):
        keys = list(val.keys())[:6]
        return f"dict keys={keys}"
    if isinstance(val, str):
        return f"str len={len(val)}"
    if isinstance(val, bytes):
        return f"bytes len={len(val)}"
    return type(val).__name__


def _summarize_mapping(data: dict[str, Any], max_keys: int = 8) -> dict[str, str]:
    summary = {}
    for idx, (key, val) in enumerate(data.items()):
        if idx >= max_keys:
            summary["..."] = f"+{len(data) - max_keys} more"
            break
        summary[str(key)] = _describe_value(val)
    return summary


def _format_array_head(value: Any, max_items: int = 14) -> str:
    arr = np.asarray(value)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return f"shape={arr.shape} dtype={arr.dtype} empty"
    head = flat[:max_items]
    suffix = " ..." if flat.size > max_items else ""
    return f"shape={arr.shape} dtype={arr.dtype} head={np.array2string(head, precision=4, separator=', ')}{suffix}"


def _format_image_stats(value: Any) -> str:
    arr = np.asarray(value)
    return (
        f"shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min():.1f} max={arr.max():.1f} mean={arr.mean():.2f}"
    )


def _is_image_key(key: str) -> bool:
    return key.startswith(f"{OBS_STR}.images.")


def _to_image_tensor(value: Any, key: str) -> torch.Tensor:
    arr = np.asarray(value)
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4):
            arr = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(f"Image key '{key}' has unsupported shape {arr.shape}")

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    t = torch.as_tensor(arr)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def _pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _to_rtc_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


async def handle_connection(
    websocket: ws_server.ServerConnection,
    *,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    packer,
    record_dir: Path | None,
    default_task: str | None,
) -> None:
    LOG.info("Connection from %s opened", websocket.remote_address)
    prev_total_time = None

    # Send metadata once on connect
    policy_meta = getattr(policy, "metadata", {}) if hasattr(policy, "metadata") else {}
    await websocket.send(packer.pack(policy_meta))

    step = 0
    while True:
        try:
            start_t = time.monotonic()
            raw = await websocket.recv()
            raw_len = len(raw) if hasattr(raw, "__len__") else "n/a"
            LOG.info("Step %s: recv type=%s size=%s", step, type(raw).__name__, raw_len)
            payload = msgpack_numpy.unpackb(raw)
            if isinstance(payload, dict):
                payload = _normalize_payload(payload)
            LOG.info("Step %s: payload keys=%s", step, list(payload.keys()))
            # Fields kept for format parity with actibot serve_policy
            _ = payload.get("inference_delay", 0.0)
            _ = payload.get("prev_chunk_left_over", [])

            obs_in, task_val = _piper_obs_from_payload(payload)
            if isinstance(obs_in, dict):
                obs_in = _normalize_payload(obs_in)
            LOG.info("Step %s: obs keys=%s", step, list(obs_in.keys()))
            LOG.info("Step %s: obs summary=%s", step, _summarize_mapping(obs_in))
            for key in ("observation.state", "observation.effort", "observation.velocity"):
                if key in obs_in:
                    LOG.info("Step %s: %s=%s", step, key, _format_array_head(obs_in[key]))
            for key in (
                "observation.images.cam_top",
                "observation.images.cam_left",
                "observation.images.cam_right",
            ):
                if key in obs_in:
                    LOG.info("Step %s: %s=%s", step, key, _format_image_stats(obs_in[key]))

            obs_tensor: dict[str, Any] = {}
            for key, val in obs_in.items():
                if isinstance(val, dict):
                    val = _normalize_payload(val)
                    if isinstance(val, dict):
                        raise ValueError(f"Unsupported dict for observation key '{key}'")
                if _is_image_key(key):
                    obs_tensor[key] = _to_image_tensor(val, key)
                    continue

                if isinstance(val, np.ndarray):
                    t = torch.as_tensor(val.copy())
                else:
                    t = torch.as_tensor(val)
                if t.is_floating_point() and t.dtype != torch.float32:
                    t = t.float()
                if t.dim() == len(getattr(val, "shape", ())) or t.dim() == len(np.shape(val)):
                    t = t.unsqueeze(0)
                obs_tensor[key] = t

            batch: dict[str, Any] = dict(obs_tensor)
            LOG.info("Step %s: batch keys=%s", step, list(batch.keys()))
            LOG.info("Step %s: batch summary=%s", step, _summarize_mapping(batch))

            # Complementary data: task/prompt
            if task_val is None and default_task is not None:
                b = next(iter(obs_tensor.values())).shape[0] if obs_tensor else 1
                task_val = [default_task] * b
            if task_val is not None:
                if isinstance(task_val, str):
                    b = next(iter(obs_tensor.values())).shape[0] if obs_tensor else 1
                    task_val = [task_val] * b
                batch["task"] = task_val

            # Preprocessor expects batch dict with flat observation.* keys (+ optional task)
            obs = preprocessor(batch)
            if isinstance(obs, dict):
                LOG.info("Step %s: preprocessed keys=%s", step, list(obs.keys()))
                LOG.info("Step %s: preprocessed summary=%s", step, _summarize_mapping(obs))

            infer_start = time.monotonic()
            with torch.inference_mode():
                if hasattr(policy, "predict_action_chunk"):
                    device = next(policy.parameters()).device
                    inference_delay = payload.get("inference_delay")
                    if inference_delay is not None:
                        inference_delay = int(inference_delay)
                    prev_chunk_left_over = _to_rtc_tensor(payload.get("prev_chunk_left_over"), device)
                    action = policy.predict_action_chunk(
                        obs,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_chunk_left_over,
                    )
                else:
                    action = policy.select_action(obs)
            infer_ms = (time.monotonic() - infer_start) * 1000.0

            action = postprocessor(action)
            # Normalize action payload for msgpack
            response: dict[str, Any]
            if torch.is_tensor(action):
                action_np = action.detach().cpu().numpy()
                if action_np.ndim == 3 and action_np.shape[0] == 1:
                    action_np = action_np[0]
                response = {"actions": action_np}
            elif isinstance(action, dict):
                resp = {}
                for k, v in action.items():
                    if torch.is_tensor(v):
                        resp[k] = v.detach().cpu().numpy()
                    else:
                        resp[k] = v
                response = resp
            else:
                response = {"actions": action}
            response["policy_timing"] = {"infer_ms": infer_ms}
            response["server_timing"] = {"infer_ms": infer_ms}
            if prev_total_time is not None:
                response["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
            LOG.info("Step %s: response summary=%s", step, _summarize_mapping(response))
            if "actions" in response:
                LOG.info("Step %s: actions=%s", step, _format_array_head(response["actions"], max_items=21))

            packed = packer.pack(response)
            await websocket.send(packed)

            total_time = time.monotonic() - start_t
            prev_total_time = total_time

            if record_dir:
                (record_dir / f"req_{step}.msgpack").write_bytes(raw)
                (record_dir / f"resp_{step}.msgpack").write_bytes(packed)
            step += 1

        except websockets.ConnectionClosed:  # type: ignore[attr-defined]
            LOG.info("Connection from %s closed", websocket.remote_address)
            break
        except Exception:
            tb = traceback.format_exc()
            LOG.error("Internal server error:\n%s", tb)
            try:
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,  # type: ignore[attr-defined]
                    reason="Internal server error. Traceback included in previous frame.",
                )
            finally:
                raise


async def run_server(
    *,
    host: str,
    port: int,
    max_msg_size: int | None,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    record_dir: Path | None,
    default_task: str | None,
) -> None:
    packer = msgpack.Packer(default=_pack_array, use_bin_type=True)

    async def _handler(websocket: ws_server.ServerConnection):
        await handle_connection(
            websocket,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            packer=packer,
            record_dir=record_dir,
            default_task=default_task,
        )

    async with ws_server.serve(
        _handler, host, port, compression=None, max_size=max_msg_size, process_request=None
    ) as server:
        LOG.info("Serving on %s:%s", host, port)
        await server.serve_forever()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    record_dir = Path(args.record_dir) if args.record_dir else None
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)

    register_third_party_plugins()

    try:
        policy, preprocessor, postprocessor, meta = load_policy(args.policy_path, args.policy_device)
    except Exception as exc:
        LOG.error("Failed to load policy: %s", exc)
        raise

    LOG.info("Policy loaded: %s", meta)

    try:
        asyncio.run(
            run_server(
                host=args.host,
                port=args.port,
                max_msg_size=args.max_msg_size,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                record_dir=record_dir,
                default_task=args.default_task,
            )
        )
    except KeyboardInterrupt:
        LOG.info("Shutting down")


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required.")
    main()
