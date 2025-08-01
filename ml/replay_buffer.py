import torch
import numpy as np

# Create a random number generator for modern numpy random usage with seed for reproducibility
rng = np.random.default_rng(42)
from collections import deque
from typing import List, Optional, Any

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with proper sampling."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity) # Using deque to efficiently handle capacity
        self.priorities = np.zeros(capacity, dtype=np.float32) # Using numpy array for efficient updates
        self.position = 0
        self.max_priority = 1.0 # Initialize with high priority to ensure first samples are chosen

    def push(self, experience):
        """Add experience with maximum priority."""
        # When using deque, append handles maxlen; for numpy array of priorities, overwrite at position
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Optional[List[Any]], Optional[np.ndarray], Optional[torch.Tensor]]:
        """Sample batch with priority-based sampling."""
        current_size = len(self.buffer)
        if current_size < batch_size:
            return None, None, None

        # Get valid priorities for current buffer size
        valid_priorities = self.priorities[:current_size]

        # Convert to probabilities, with small epsilon for zeros and re-normalize
        probabilities = (valid_priorities + 1e-6) ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = rng.choice(current_size, batch_size, p=probabilities, replace=False) # replace=False for unique samples

        # Calculate importance sampling weights
        weights = (current_size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max() if weights.max() > 0 else 1.0 # Normalize to 1, avoids too high weights

        # Get experiences corresponding to sampled indices
        experiences = [self.buffer[i] for i in indices]

        return experiences, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = float(abs(td_error) + 1e-6)  # Add epsilon to prevent zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        """Return a dictionary containing the complete state of the buffer."""
        return {
            'buffer': list(self.buffer), # Convert deque to list
            'priorities': self.priorities.tolist(), # Convert numpy array to list
            'position': self.position,
            'max_priority': self.max_priority,
            'capacity': self.capacity, # Include for re-initialization if needed
            'alpha': self.alpha,
            'beta': self.beta
        }

    def load_state_dict(self, state_dict):
        """Load buffer state from a state dict."""
        # Re-initialize buffer to match capacity
        self.capacity = state_dict['capacity']
        self.alpha = state_dict['alpha']
        self.beta = state_dict['beta']
        self.buffer = deque(state_dict['buffer'], maxlen=self.capacity)

        self.priorities = np.array(state_dict['priorities'], dtype=np.float32)
        # Ensure priorities array is correct size and padded with zeros if loaded from smaller buffer
        if self.priorities.shape[0] < self.capacity:
             temp_priorities = np.zeros(self.capacity, dtype=np.float32)
             temp_priorities[:self.priorities.shape[0]] = self.priorities
             self.priorities = temp_priorities

        self.position = state_dict['position']
        self.max_priority = state_dict['max_priority']
