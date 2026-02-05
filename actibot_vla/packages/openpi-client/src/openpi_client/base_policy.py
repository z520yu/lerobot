import abc
from typing import Dict

class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict, inference_delay, prev_chunk_left_over) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
