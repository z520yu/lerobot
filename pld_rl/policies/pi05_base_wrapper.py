"""PI05 Base Policy Wrapper for PLD RL."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

logger = logging.getLogger(__name__)


class PI05BaseWrapper:
    """
    Wrapper for frozen PI05 base policy with action chunking support.

    Handles:
    - Loading and freezing PI05 model
    - Action chunking cache management
    - Observation preprocessing with proper tokenization
    - Dataset stats loading for normalization
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        dataset_stats: dict[str, Any] | None = None,
        device: str = "cuda",
        chunk_size: int = 50,
        n_action_steps: int = 50,
    ):
        self.device = device
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps

        # Action cache
        self.action_cache: np.ndarray | None = None
        self.cache_step: int = 0
        self.task_text: str = ""

        # Load model
        self._load_model(checkpoint_path, dataset_stats)

    def _load_model(
        self,
        checkpoint_path: str | Path,
        dataset_stats: dict[str, Any] | None,
    ):
        """Load and freeze PI05 model."""
        checkpoint_path = Path(checkpoint_path)

        # Load policy
        self.policy = PI05Policy.from_pretrained(
            str(checkpoint_path),
            strict=False,
        )
        self.policy.to(self.device)
        self.policy.eval()

        # Freeze all parameters
        for param in self.policy.parameters():
            param.requires_grad = False

        # Get config
        self.config = self.policy.config

        # Update chunk settings from config
        self.chunk_size = self.config.chunk_size
        self.n_action_steps = self.config.n_action_steps

        # Try to load dataset_stats from checkpoint if not provided
        if dataset_stats is None:
            dataset_stats = self._load_dataset_stats(checkpoint_path)

        # Create processors
        if dataset_stats is not None:
            try:
                from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
                self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(
                    self.config,
                    dataset_stats=dataset_stats,
                )
                logger.info("Created PI05 preprocessor and postprocessor with dataset_stats")
            except Exception as e:
                logger.warning(f"Failed to create processors: {e}. Using manual processing.")
                self.preprocessor = None
                self.postprocessor = None
        else:
            logger.warning("No dataset_stats available. Actions may not be properly normalized.")
            self.preprocessor = None
            self.postprocessor = None

        # Get actual action dimension from config
        self._actual_action_dim = None
        if hasattr(self.config, 'output_features') and 'action' in self.config.output_features:
            self._actual_action_dim = self.config.output_features['action'].shape[0]
        else:
            self._actual_action_dim = self.config.max_action_dim

        logger.info(f"Loaded PI05 policy from {checkpoint_path}")
        logger.info(f"  chunk_size={self.chunk_size}, n_action_steps={self.n_action_steps}")
        logger.info(f"  action_dim={self._actual_action_dim}")

    def _load_dataset_stats(self, checkpoint_path: Path) -> dict[str, Any] | None:
        """Try to load dataset_stats from checkpoint directory."""
        # Check for stats.json or dataset_stats.json in checkpoint
        possible_paths = [
            checkpoint_path / "stats.json",
            checkpoint_path / "dataset_stats.json",
            checkpoint_path / "normalization_stats.json",
            checkpoint_path.parent / "stats.json",
        ]

        for stats_path in possible_paths:
            if stats_path.exists():
                try:
                    with open(stats_path) as f:
                        stats = json.load(f)
                    # Convert lists to tensors if needed
                    stats = self._convert_stats_to_tensors(stats)
                    logger.info(f"Loaded dataset_stats from {stats_path}")
                    return stats
                except Exception as e:
                    logger.warning(f"Failed to load stats from {stats_path}: {e}")

        # Try to load from safetensors metadata or model config
        try:
            config_path = checkpoint_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
                if "dataset_stats" in config_data:
                    stats = self._convert_stats_to_tensors(config_data["dataset_stats"])
                    logger.info("Loaded dataset_stats from config.json")
                    return stats
        except Exception as e:
            logger.debug(f"No stats in config.json: {e}")

        logger.warning(f"Could not find dataset_stats in {checkpoint_path}")
        return None

    def _convert_stats_to_tensors(self, stats: dict) -> dict:
        """Convert stats values to tensors if they are lists."""
        converted = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                converted[key] = {}
                for k, v in value.items():
                    if isinstance(v, list):
                        converted[key][k] = torch.tensor(v)
                    elif isinstance(v, (int, float)):
                        converted[key][k] = torch.tensor([v])
                    else:
                        converted[key][k] = v
            else:
                converted[key] = value
        return converted

    def reset(self, task_text: str = "") -> None:
        """Reset action cache and task text."""
        self.action_cache = None
        self.cache_step = 0
        self.task_text = task_text
        self.policy.reset()

    def act(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """
        Get action for given observation.

        Implements action chunking:
        - Every n_action_steps, run full inference
        - Otherwise, return cached action

        Args:
            batch: preprocessed observation batch

        Returns:
            action: (action_dim,) numpy array
        """
        if self.action_cache is None or self.cache_step >= self.n_action_steps:
            # Need to run inference
            self._run_inference(batch)

        action = self.action_cache[self.cache_step]
        self.cache_step += 1
        return action

    def _run_inference(self, batch: dict[str, torch.Tensor]) -> None:
        """Run PI05 inference and cache action chunk."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Ensure task is present
        if "task" not in batch:
            batch["task"] = [self.task_text]

        # Apply preprocessor if available (handles tokenization, normalization, etc.)
        if self.preprocessor is not None:
            try:
                batch = self.preprocessor(batch)
            except Exception as e:
                logger.warning(f"Preprocessor failed: {e}. Trying manual processing.")
                batch = self._manual_preprocess(batch)
        else:
            batch = self._manual_preprocess(batch)

        with torch.no_grad():
            # Use select_action which handles action chunking internally
            action = self.policy.select_action(batch)

            # Handle different output shapes
            if action.dim() == 3:  # (1, chunk_size, action_dim)
                action_chunk = action[0]  # (chunk_size, action_dim)
            elif action.dim() == 2:  # (1, action_dim) - single action
                # Need to get full chunk
                action_chunk = self.policy.predict_action_chunk(batch)
                if action_chunk.dim() == 3:
                    action_chunk = action_chunk[0]
            else:
                action_chunk = action.unsqueeze(0)

        # Postprocess if available (handles unnormalization)
        if self.postprocessor is not None:
            try:
                processed = self.postprocessor({"action": action_chunk.unsqueeze(0)})
                action_chunk = processed["action"].squeeze(0)
            except Exception as e:
                logger.warning(f"Postprocessor failed: {e}")

        # Truncate to actual action dimension
        if self._actual_action_dim and action_chunk.shape[-1] > self._actual_action_dim:
            action_chunk = action_chunk[..., :self._actual_action_dim]

        self.action_cache = action_chunk.cpu().numpy()
        self.cache_step = 0

    def _manual_preprocess(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Manual preprocessing when processor is not available.

        This attempts to create the required tokens and masks for PI05.
        Reference: lerobot/src/lerobot/policies/pi05/processor_pi05.py
        """
        try:
            from transformers import AutoTokenizer

            # Check if tokenization is needed
            if "observation.language.tokens" not in batch:
                tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

                # Get task text
                task_texts = batch.get("task", [self.task_text])
                if isinstance(task_texts, str):
                    task_texts = [task_texts]

                batch_size = len(task_texts)
                max_state_dim = getattr(self.config, 'max_state_dim', 32)

                # Prepare state for tokenization
                state = batch.get("observation.state")
                if state is not None:
                    state_np = state.cpu().numpy()
                    # Ensure 2D
                    if state_np.ndim == 1:
                        state_np = state_np.reshape(1, -1)

                    # Pad to max_state_dim (reference: lerobot pad_vector)
                    current_dim = state_np.shape[-1]
                    if current_dim < max_state_dim:
                        padding = np.zeros((state_np.shape[0], max_state_dim - current_dim))
                        state_np = np.concatenate([state_np, padding], axis=-1)
                    elif current_dim > max_state_dim:
                        state_np = state_np[..., :max_state_dim]

                    # Clip to [-1, 1] and discretize to 256 bins
                    # Reference: lerobot Pi05PrepareStateTokenizerProcessorStep
                    state_np = np.clip(state_np, -1, 1)
                    discretized = np.digitize(
                        state_np,
                        bins=np.linspace(-1, 1, 256 + 1)[:-1]
                    ) - 1
                    discretized = np.clip(discretized, 0, 255).astype(np.int32)
                else:
                    discretized = np.zeros((batch_size, max_state_dim), dtype=np.int32)

                # Build full prompts (reference: lerobot format)
                full_prompts = []
                for i, task in enumerate(task_texts):
                    cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
                    state_str = " ".join(map(str, discretized[i]))
                    full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
                    full_prompts.append(full_prompt)

                # Tokenize
                tokens = tokenizer(
                    full_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=getattr(self.config, 'tokenizer_max_length', 200),
                    truncation=True,
                )

                batch["observation.language.tokens"] = tokens.input_ids.to(self.device)
                # PI05 expects boolean attention mask
                batch["observation.language.attention_mask"] = tokens.attention_mask.bool().to(self.device)

        except ImportError:
            logger.error("transformers not installed. Cannot do manual tokenization.")
        except Exception as e:
            logger.error(f"Manual preprocessing failed: {e}")
            import traceback
            traceback.print_exc()

        return batch

    def act_single(
        self,
        obs_dict: dict[str, np.ndarray | torch.Tensor],
        task_text: str | None = None,
    ) -> np.ndarray:
        """
        Get action from raw observation dictionary.

        Args:
            obs_dict: raw observation dictionary
            task_text: optional task text override

        Returns:
            action: (action_dim,) numpy array
        """
        if task_text is not None:
            self.task_text = task_text

        # Convert to batch format
        batch = self._prepare_batch(obs_dict)
        return self.act(batch)

    def _prepare_batch(
        self,
        obs_dict: dict[str, np.ndarray | torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert observation dict to batch format."""
        batch = {}

        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.tensor(value, dtype=torch.float32)

            # Add batch dimension if needed
            if "image" in key:
                if value.dim() == 3:  # (H, W, C) or (C, H, W)
                    # Check if channels-last and convert to channels-first
                    if value.shape[-1] == 3:
                        value = value.permute(2, 0, 1)
                    value = value.unsqueeze(0)  # (1, C, H, W)
            else:
                if value.dim() == 1:
                    value = value.unsqueeze(0)  # (1, D)

            batch[key] = value

        batch["task"] = [self.task_text]

        return batch

    @property
    def action_dim(self) -> int:
        """Get actual action dimension."""
        return self._actual_action_dim or self.config.max_action_dim
