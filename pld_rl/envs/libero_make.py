"""LIBERO Environment Factory."""

import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class LiberoGymnasiumWrapper(gym.Env):
    """
    Wrapper to convert LIBERO environment to Gymnasium API.

    Handles:
    - Old Gym API (obs, reward, done, info) -> New Gymnasium API (obs, reward, terminated, truncated, info)
    - Observation key normalization
    - Action space normalization
    """

    def __init__(
        self,
        env,
        max_episode_steps: int = 500,
        task_description: str = "",
    ):
        super().__init__()
        self._env = env
        self.max_episode_steps = max_episode_steps
        self.task_description = task_description
        self._step_count = 0

        # Get action and observation spaces from underlying env
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space
        else:
            # Default 7-DOF action space
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(7,), dtype=np.float32
            )

        # Build observation space based on what env provides
        self.observation_space = self._build_observation_space()

    def _build_observation_space(self) -> gym.spaces.Dict:
        """Build observation space from environment."""
        # Try to get a sample observation
        try:
            if hasattr(self._env, 'observation_space'):
                base_space = self._env.observation_space
                if isinstance(base_space, gym.spaces.Dict):
                    return base_space
        except Exception:
            pass

        # Default observation space
        return gym.spaces.Dict({
            "observation.images.image": gym.spaces.Box(
                low=0, high=1, shape=(3, 224, 224), dtype=np.float32
            ),
            "observation.images.image2": gym.spaces.Box(
                low=0, high=1, shape=(3, 224, 224), dtype=np.float32
            ),
            "observation.state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            ),
        })

    def _normalize_obs(self, obs: dict) -> dict:
        """Normalize observation keys to standard format.

        Reference: lerobot/src/lerobot/processor/env_processor.py (LiberoProcessorStep)
        """
        normalized = {}

        # Image key mapping
        image_key_map = {
            "agentview_image": "observation.images.image",
            "robot0_agentview_image": "observation.images.image",
            "eye_in_hand_image": "observation.images.image2",
            "robot0_eye_in_hand_image": "observation.images.image2",
        }

        for src_key, dst_key in image_key_map.items():
            if src_key in obs:
                img = obs[src_key]
                # Convert HWC uint8 to CHW float
                if isinstance(img, np.ndarray):
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0
                    if img.ndim == 3 and img.shape[-1] == 3:
                        img = np.transpose(img, (2, 0, 1))
                    # Flip 180 degrees (flip H and W) - lerobot convention for LIBERO
                    img = np.flip(img, axis=(1, 2)).copy()
                normalized[dst_key] = img

        # State: concatenate robot state components
        # Reference: lerobot uses eef_pos(3) + axis_angle(3) + gripper_qpos(2) = 8
        eef_pos = obs.get("robot0_eef_pos")
        eef_quat = obs.get("robot0_eef_quat")
        gripper_qpos = obs.get("robot0_gripper_qpos")

        if eef_pos is not None and eef_quat is not None and gripper_qpos is not None:
            eef_pos = np.asarray(eef_pos).flatten()
            eef_quat = np.asarray(eef_quat).flatten()
            gripper_qpos = np.asarray(gripper_qpos).flatten()

            # Convert quaternion to axis-angle (lerobot convention)
            axis_angle = self._quat2axisangle(eef_quat)

            normalized["observation.state"] = np.concatenate([
                eef_pos,      # (3,)
                axis_angle,   # (3,)
                gripper_qpos, # (2,)
            ]).astype(np.float32)
        elif "observation.state" in obs:
            normalized["observation.state"] = obs["observation.state"]

        # Pass through any already-normalized keys
        for key in obs:
            if key.startswith("observation.") and key not in normalized:
                normalized[key] = obs[key]

        return normalized

    def _quat2axisangle(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to axis-angle format.

        Reference: lerobot/src/lerobot/processor/env_processor.py

        Args:
            quat: (4,) quaternion in (x, y, z, w) format

        Returns:
            (3,) axis-angle vector
        """
        quat = np.asarray(quat, dtype=np.float32)
        w = np.clip(quat[3], -1.0, 1.0)
        den = np.sqrt(np.maximum(1.0 - w * w, 0.0))

        if den > 1e-10:
            angle = 2.0 * np.arccos(w)
            axis = quat[:3] / den
            return axis * angle
        else:
            return np.zeros(3, dtype=np.float32)

    def reset(self, **kwargs) -> tuple[dict, dict]:
        """Reset environment (Gymnasium API)."""
        self._step_count = 0

        # Call underlying reset
        result = self._env.reset(**kwargs)

        # Handle different return formats
        if isinstance(result, tuple):
            obs = result[0]
            info = result[1] if len(result) > 1 else {}
        else:
            obs = result
            info = {}

        # Normalize observation
        obs = self._normalize_obs(obs)

        # Add task to info
        info["task"] = self.task_description

        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Step environment (Gymnasium API)."""
        self._step_count += 1

        # Call underlying step
        result = self._env.step(action)

        # Handle old Gym API (obs, reward, done, info) vs new Gymnasium API
        if len(result) == 4:
            # Old Gym API
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            # New Gymnasium API
            obs, reward, terminated, truncated, info = result

        # Check success using LIBERO's check_success() method
        # Reference: lerobot/src/lerobot/envs/libero.py line 311
        is_success = False
        if hasattr(self._env, 'check_success'):
            is_success = self._env.check_success()
        elif "success" in info:
            is_success = info["success"]

        # Update info with is_success (lerobot convention)
        info["is_success"] = is_success
        info["success"] = is_success  # Keep for backwards compatibility

        # Determine terminated/truncated
        terminated = bool(terminated) or is_success
        truncated = bool(truncated)

        # Check for max steps truncation
        if self._step_count >= self.max_episode_steps and not (terminated or truncated):
            truncated = True

        # Normalize observation
        obs = self._normalize_obs(obs)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render environment."""
        if hasattr(self._env, 'render'):
            return self._env.render()
        return None

    def close(self):
        """Close environment."""
        if hasattr(self._env, 'close'):
            self._env.close()


def make_libero_env(
    task_name: str = "libero_10",
    task_id: int = 0,
    max_episode_steps: int = 500,
    render_mode: str | None = None,
    **kwargs: Any,
) -> gym.Env:
    """
    Create LIBERO environment with Gymnasium API.

    Args:
        task_name: LIBERO task suite name (e.g., "libero_10", "libero_90")
        task_id: Task ID within the suite
        max_episode_steps: Maximum steps per episode
        render_mode: Rendering mode
        **kwargs: Additional arguments for environment

    Returns:
        Gymnasium-compatible environment
    """
    try:
        # Try to import LIBERO
        import os
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        # Get task suite - NOTE: benchmark_dict[name] returns a class, must call it!
        # Reference: lerobot/src/lerobot/envs/libero.py
        benchmark_dict = benchmark.get_benchmark_dict()
        if task_name not in benchmark_dict:
            available = list(benchmark_dict.keys())
            raise ValueError(f"Unknown LIBERO suite '{task_name}'. Available: {available}")

        # IMPORTANT: Call as function to instantiate the suite!
        task_suite = benchmark_dict[task_name]()

        # Verify suite has tasks
        if not getattr(task_suite, "tasks", None):
            raise ValueError(f"Suite '{task_name}' has no tasks.")

        # Get task by ID
        task = task_suite.get_task(task_id)

        # Get task description
        if hasattr(task, 'language'):
            task_description = task.language
        elif hasattr(task, 'name'):
            task_description = task.name
        else:
            task_description = task_name

        # Get full task bddl file path for OffScreenRenderEnv
        # Reference: lerobot/src/lerobot/envs/libero.py
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file
        )

        # Create LIBERO environment
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            **kwargs,
        )

        # Wrap with Gymnasium compatibility layer
        env = LiberoGymnasiumWrapper(
            env,
            max_episode_steps=max_episode_steps,
            task_description=task_description,
        )

        logger.info(f"Created LIBERO environment: {task_name} task {task_id}")
        logger.info(f"  Task: {task_description}")

        return env

    except ImportError as e:
        # Fallback: create a dummy env for testing
        logger.warning(f"LIBERO not installed ({e}). Using dummy environment.")
        return DummyLiberoEnv(max_episode_steps=max_episode_steps)
    except Exception as e:
        logger.error(f"Failed to create LIBERO environment: {e}")
        logger.warning("Falling back to dummy environment.")
        return DummyLiberoEnv(max_episode_steps=max_episode_steps)


class DummyLiberoEnv(gym.Env):
    """
    Dummy environment for testing when LIBERO is not available.

    Simulates LIBERO-like observations and random success.
    """

    def __init__(
        self,
        max_episode_steps: int = 500,
        obs_image_shape: tuple[int, int, int] = (3, 224, 224),
        state_dim: int = 8,  # lerobot convention: pos:3 + axis_angle:3 + gripper:2
        action_dim: int = 7,
        success_prob: float = 0.1,
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.obs_image_shape = obs_image_shape
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.success_prob = success_prob

        # Define spaces
        self.observation_space = gym.spaces.Dict({
            "observation.images.image": gym.spaces.Box(
                low=0, high=1, shape=obs_image_shape, dtype=np.float32
            ),
            "observation.images.image2": gym.spaces.Box(
                low=0, high=1, shape=obs_image_shape, dtype=np.float32
            ),
            "observation.state": gym.spaces.Box(
                low=-1, high=1, shape=(state_dim,), dtype=np.float32
            ),
        })
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._current_state = None

    def reset(self, **kwargs) -> tuple[dict, dict]:
        """Reset environment."""
        self._step_count = 0
        self._current_state = np.random.rand(self.state_dim).astype(np.float32) * 2 - 1
        obs = self._get_obs()
        return obs, {"task": "dummy_task: pick up the object"}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Take a step."""
        self._step_count += 1

        # Simulate state dynamics (simple random walk)
        # Use action to influence state (handle dimension mismatch)
        action_effect = np.zeros(self.state_dim, dtype=np.float32)
        action_len = min(len(action), self.state_dim)
        action_effect[:action_len] = action[:action_len]

        self._current_state = np.clip(
            self._current_state + action_effect * 0.1 + np.random.randn(self.state_dim).astype(np.float32) * 0.01,
            -1, 1
        ).astype(np.float32)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        # Random success (simulating task completion)
        success = np.random.random() < self.success_prob

        info = {
            "success": success,
            "step": self._step_count,
        }

        if success:
            reward = 1.0
            terminated = True

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        """Generate observation."""
        return {
            "observation.images.image": np.random.rand(*self.obs_image_shape).astype(np.float32),
            "observation.images.image2": np.random.rand(*self.obs_image_shape).astype(np.float32),
            "observation.state": self._current_state.copy(),
        }

    def render(self) -> np.ndarray | None:
        """Render environment."""
        return None

    def close(self):
        """Close environment."""
        pass
