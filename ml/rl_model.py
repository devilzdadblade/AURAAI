import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Create a random number generator for modern numpy random usage
rng = np.random.default_rng(seed=42)
import copy
import logging
from collections import deque
import random
from typing import Tuple, List, Optional, Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

from src.ml.replay_buffer import PrioritizedReplayBuffer
from src.ml.noisy_linear import NoisyLinear
from src.ml.metrics import TrainingMetrics # Import TrainingMetrics
from src.ml.memory_management import MemoryMonitor, BoundedMetricsStorage, DeviceAwareTensorCache, CleanupLevel
from src.ml.numerical_stability import NumericalStabilityManager
from src.ml.gradient_validator import GradientValidator
from src.ml.parallel_component_updater import ParallelComponentUpdater

class DuelingNetwork(nn.Module):
    """Dueling network architecture for Q-value estimation with NoisyLinear exploration."""

    def __init__(self, input_size: int, hidden_size: int, action_size: int, use_noisy_layers: bool = True):
        super().__init__()
        self.use_noisy_layers = use_noisy_layers
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        # Value stream (no noise for value estimation)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Advantage stream with NoisyLinear layers for parameter space exploration
        if use_noisy_layers:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_size, hidden_size // 2, std_init=0.5),
                nn.ReLU(),
                NoisyLinear(hidden_size // 2, action_size, std_init=0.5)
            )
        else:
            # Fallback to regular linear layers
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from src.ml.numerical_stability import NumericalStabilityManager
        stability_manager = NumericalStabilityManager(epsilon=1e-10)
        
        # Add numerical stability check for input
        x = stability_manager.handle_numerical_instability(
            x, replacement_value=0.0, name="dueling_network_input"
        )
        
        features = self.feature_layer(x)
        
        # Add numerical stability check for features
        features = stability_manager.handle_numerical_instability(
            features, replacement_value=0.0, name="dueling_network_features"
        )
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Add numerical stability checks for value and advantage streams
        value = stability_manager.handle_numerical_instability(
            value, replacement_value=0.0, name="dueling_network_value"
        )
        advantage = stability_manager.handle_numerical_instability(
            advantage, replacement_value=0.0, name="dueling_network_advantage"
        )
        
        # Calculate advantage mean with numerical stability
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        advantage_mean = stability_manager.handle_numerical_instability(
            advantage_mean, replacement_value=0.0, name="dueling_network_advantage_mean"
        )

        # Combine value and advantage
        q_values = value + (advantage - advantage_mean)
        
        # Add final numerical stability check for Q-values
        q_values = stability_manager.handle_numerical_instability(
            q_values, replacement_value=0.0, name="dueling_network_q_values"
        )
        
        return q_values

class SelfImprovingRLModel(nn.Module):
    """
    Next-generation RL model with comprehensive self-improving capabilities:
    - Meta-gradient hyperparameter learning (optional)
    - Intrinsic curiosity for exploration (optional)
    - Uncertainty-aware decisions via ensemble (optional)
    - Auxiliary task learning for representation (optional, currently limited to next-state prediction)
    - Prioritized Replay Buffer with N-step returns
    - Improved network architecture and robust training
    """

    def __init__(self, config: dict):
        """
        Accepts a configuration dictionary containing all hyperparameters and settings.
        Example keys: input_size, hidden_size, action_size, config_size, learning_rate, gamma, tau, buffer_size,
        batch_size, n_step, use_meta_learning, use_curiosity, use_uncertainty, use_auxiliary, use_prioritized_replay,
        curiosity_weight, auxiliary_weight, device
        """
        input_size = config.get('input_size', 32)
        hidden_size = config.get('hidden_size', 128)
        action_size = config.get('action_size', 4)
        config_size = config.get('config_size', 4)
        learning_rate = config.get('learning_rate', 0.001)
        gamma = config.get('gamma', 0.99)
        tau = config.get('tau', 0.005)
        buffer_size = config.get('buffer_size', 100000)
        batch_size = config.get('batch_size', 32)
        n_step = config.get('n_step', 3)
        use_meta_learning = config.get('use_meta_learning', True)
        use_curiosity = config.get('use_curiosity', True)
        use_uncertainty = config.get('use_uncertainty', True)
        use_auxiliary = config.get('use_auxiliary', True)
        use_prioritized_replay = config.get('use_prioritized_replay', True)
        curiosity_weight = config.get('curiosity_weight', 0.01)
        auxiliary_weight = config.get('auxiliary_weight', 0.1)
        device = config.get('device', 'cpu')

        super().__init__()

        # Core model configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.config_size = config_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.curiosity_weight = curiosity_weight
        self.auxiliary_weight = auxiliary_weight
        self.device = torch.device(device)

        # Flags to enable/disable optional components. These are critical for save/load.
        self.use_meta_learning = use_meta_learning
        self.use_curiosity = use_curiosity
        self.use_uncertainty = use_uncertainty
        self.use_auxiliary = use_auxiliary
        self.use_prioritized_replay = use_prioritized_replay

        # 1. Build the main Q-network architecture (feature extractor, dueling heads, config head)
        self._build_network()

        # 2. Initialize optional self-improving components and their optimizers
        self._init_optional_components(learning_rate, gamma, tau, buffer_size)

        # Move the entire model to the specified device
        self.to(self.device)

        # Initialize device-aware tensor cache for efficient tensor management
        self.tensor_cache = DeviceAwareTensorCache(self.device, cache_size=1000)
        
        # Warm cache with frequently accessed tensors
        self._warm_tensor_cache()

        # Initialize memory monitoring with configurable thresholds
        self.memory_monitor = MemoryMonitor(
            alert_threshold_gb=8.0,
            warning_threshold=0.7,
            critical_threshold=0.9
        )

        # Initialize numerical stability manager for consistent stability checks
        self.stability_manager = NumericalStabilityManager(epsilon=1e-10, max_grad_norm=1.0)
        
        # Initialize gradient validator for safe gradient updates
        self.gradient_validator = GradientValidator(max_grad_norm=1.0)

        # Training state tracking (non-PyTorch parameters)
        self.training_metrics = BoundedMetricsStorage(maxlen=10000)
        
        # Register cleanup callbacks for memory management
        self._register_cleanup_callbacks()
        self.performance_history = deque(maxlen=1000) # Tracks episode total rewards for meta-learning/logging
        self.total_steps = 0
        self.episode_count = 0

    def _build_network(self):
        """Constructs the core neural network components."""
        # Shared feature extractor network
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size), # LayerNorm for stability
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU()
        )

        # Dueling network: Value stream
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1) # Outputs scalar value estimate
        )

        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.action_size) # Outputs advantage for each action
        )

        # Configuration head for predicting environment parameters
        self.config_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1), # Dropout for regularization in config prediction
            nn.Linear(self.hidden_size // 2, self.config_size),
            nn.Tanh() # Output values clamped between -1 and 1
        )

    def _init_optional_components(self, learning_rate: float, gamma: float, tau: float, buffer_size: int):
        """Initializes optional self-improving modules based on use_* flags."""

        # Meta-learning for hyperparameters
        if self.use_meta_learning:
            from src.ml.meta_learning import MetaLearningHyperparams
            self.meta_hyperparams = MetaLearningHyperparams(
                initial_lr=learning_rate, initial_gamma=gamma, initial_tau=tau,
                initial_epsilon=0.1) # Default initial epsilon (will be meta-learned)
            # When meta-learning, these fixed HPs are irrelevant for dynamic use
            self.gamma = None
            self.tau = None
        else:
            self.gamma = gamma # Fixed discount factor
            self.tau = tau     # Fixed soft update factor
            self.meta_hyperparams = None

        # Curiosity module for intrinsic motivation
        if self.use_curiosity:
            from src.ml.curiosity import ImprovedCuriosityModule
            self.curiosity_module = ImprovedCuriosityModule(
                self.input_size, self.action_size, self.hidden_size).to(self.device)
            self.curiosity_optimizer = optim.Adam(
                self.curiosity_module.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            self.curiosity_module = None
            self.curiosity_optimizer = None

        # Uncertainty estimator (ensemble-based for robust Q-value uncertainty)
        if self.use_uncertainty:
            from src.ml.risk_management_model import ImprovedUncertaintyEstimator
            # Factory function to create individual ensemble network members (predict Q-values from features)
            def ensemble_q_network_factory():
                return nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU(),
                    # Dropout is applied on feature_net's last layer. No dropout directly here as ensemble handles stochasticity.
                    nn.Linear(self.hidden_size // 2, self.action_size)
                ).to(self.device) # Ensure factory creates networks on device
            self.uncertainty_estimator = ImprovedUncertaintyEstimator(ensemble_q_network_factory, learning_rate=learning_rate).to(self.device)
        else:
            self.uncertainty_estimator = None

        # Auxiliary task heads for richer representation learning
        if self.use_auxiliary:
            from src.ml.auxiliary_tasks import AuxiliaryTaskHeads
            self.auxiliary_heads = AuxiliaryTaskHeads(self.hidden_size, self.input_size).to(self.device)
            self.auxiliary_optimizer = optim.Adam(
                self.auxiliary_heads.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            self.auxiliary_heads = None
            self.auxiliary_optimizer = None

        # Replay buffer (Prioritized Experience Replay or standard deque)
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = deque(maxlen=buffer_size) # Standard replay buffer (deque)

        # Main optimizer for the Q-network (features, value head, advantage head, config head)
        self.optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=1e-4) # AdamW provides stronger regularization

        # Target network: A copy of the main network, updated periodically or softly for stability
        self.target_net = copy.deepcopy(self).to(self.device)
        self.target_net.eval() # Target network should always be in evaluation mode

        # N-step return buffer: Temporarily holds experiences to compute N-step returns
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Loss function for Q-value estimation (SmoothL1Loss/Huber Loss, robust to large errors)
        self.q_criterion = nn.SmoothL1Loss(reduction='none') # reduction='none' for per-element loss before applying weights
        
        # Initialize ParallelComponentUpdater for concurrent optimization
        self.component_updater = None  # Will be initialized in update() when components are ready

    def _warm_tensor_cache(self):
        """
        Warm the tensor cache with frequently accessed tensors to improve performance.
        This reduces cache misses during training and inference.
        """
        # Common tensor patterns for warming
        common_tensors = []
        
        # Zero tensors for various operations
        common_tensors.append(("zero_scalar", np.array(0.0)))
        common_tensors.append(("zero_vector", np.zeros(self.action_size)))
        common_tensors.append(("zero_batch", np.zeros((self.batch_size, self.input_size))))
        
        # Identity and unit tensors
        common_tensors.append(("ones_batch", np.ones(self.batch_size)))
        common_tensors.append(("unit_vector", np.ones(self.action_size)))
        
        # Common batch size patterns
        common_tensors.append(("batch_indices", np.arange(self.batch_size)))
        common_tensors.append(("action_indices", np.arange(self.action_size)))
        
        # Small random tensors for initialization
        common_tensors.append(("small_random", rng.standard_normal((self.batch_size, self.input_size)) * 0.01))
        
        # Common masks for operations
        common_tensors.append(("done_mask_false", np.zeros(self.batch_size, dtype=bool)))
        common_tensors.append(("done_mask_true", np.ones(self.batch_size, dtype=bool)))
        
        # Common weights for prioritized replay
        common_tensors.append(("uniform_weights", np.ones(self.batch_size)))
        
        # Common action masks
        for action in range(min(5, self.action_size)):  # Cache first few actions for quick access
            action_mask = np.zeros(self.action_size)
            action_mask[action] = 1.0
            common_tensors.append((f"action_mask_{action}", action_mask))
        
        # Prefill cache with common patterns
        self.tensor_cache.prefill_cache(common_tensors, dtype=torch.float32)
        
        logger.info(f"Tensor cache warmed with {len(common_tensors)} common tensor patterns")
        
        # Pre-allocate tensors for batched operations
        self._preallocate_batch_tensors()
   
    def _preallocate_batch_tensors(self):
        """
        Pre-allocate tensors for batched operations to minimize memory allocation.
        This improves performance by reducing overhead during training.
        """
        # Pre-allocate common batch tensors used in training
        batch_shape = (self.batch_size, self.input_size)
        
        self.tensor_cache.preallocate_tensors({
            "batch_states": torch.zeros(batch_shape, device=self.device),
            "batch_actions": torch.zeros(self.batch_size, dtype=torch.long, device=self.device),
            "batch_rewards": torch.zeros(self.batch_size, device=self.device),
            "batch_next_states": torch.zeros(batch_shape, device=self.device),
            "batch_dones": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
        })
        
        logger.info("Pre-allocated batch tensors for efficient training.")

def _register_cleanup_callbacks(self):
    """Register cleanup callbacks for memory management."""
    # Register light cleanup (called frequently)
    self.memory_monitor.register_cleanup_callback(
        self.tensor_cache.clear_cache, 
        level=CleanupLevel.LIGHT
    )

    # Register moderate cleanup (called when memory usage is moderate)
    self.memory_monitor.register_cleanup_callback(
        lambda: self.training_metrics.flush_to_storage(None),
        level=CleanupLevel.MODERATE
    )

    # Register aggressive cleanup (called when memory usage is high)
    def aggressive_cleanup():
        self.tensor_cache.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    self.memory_monitor.register_cleanup_callback(
        aggressive_cleanup,
        level=CleanupLevel.AGGRESSIVE
    )

def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a forward pass: state -> features -> (Q-values, config_params).
    """
    # Validate input tensor for NaN/Inf
    x = self.stability_manager.handle_numerical_instability(
        x, replacement_value=0.0, name="forward_input"
    )
    
    features = self.feature_net(x) # State observation -> Latent features
    
    # Add numerical stability check for features
    features = self.stability_manager.handle_numerical_instability(
        features, replacement_value=0.0, name="features"
    )

    value = self.value_head(features)
    advantage = self.advantage_head(features)
    
    # Add NaN/Inf detection and handling for value and advantage streams
    value = self.stability_manager.handle_numerical_instability(
        value, replacement_value=0.0, name="value"
    )
    advantage = self.stability_manager.handle_numerical_instability(
        advantage, replacement_value=0.0, name="advantage"
    )

    # Compute advantage mean with numerical stability
    advantage_mean = advantage.mean(dim=1, keepdim=True)
    advantage_mean = self.stability_manager.handle_numerical_instability(
        advantage_mean, replacement_value=0.0, name="advantage_mean"
    )
    
    q_values = value + (advantage - advantage_mean)
    
    # Add final NaN/Inf check for Q-values
    q_values = self.stability_manager.handle_numerical_instability(
        q_values, replacement_value=0.0, name="forward_q_values"
    )

    # Config head output
    config_params = self.config_head(features)
    
    # Add numerical stability check for config parameters
    config_params = self.stability_manager.handle_numerical_instability(
        config_params, replacement_value=0.0, name="config_params"
    )

    return q_values, config_params

def act(self, state: np.ndarray, epsilon: float = 0.0,
        use_uncertainty: bool = True) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """
    Action selection mechanism: Epsilon-greedy, with an option to use
    uncertainty for more informed exploration.
    """
    cache_key = f"state_{hash(state.tobytes())}"
    state_tensor = self.tensor_cache.to_tensor(state, key=cache_key)

    with torch.no_grad():
        # Cache features for this state to avoid recomputation
        features_key = f"features_{cache_key}"
        if features_key not in self.tensor_cache.cache:
            features = self.feature_net(state_tensor) # Extract features
            # Cache the features for future use
            self.tensor_cache.cache[features_key] = features
        uncertainty_bonus = 0.0
        # If uncertainty estimation is enabled AND requested for act
        if self.use_uncertainty and use_uncertainty and self.uncertainty_estimator is not None:
            # Get mean Q-values and their uncertainty (standard deviation) from the uncertainty estimator
            q_values_mean, q_uncertainty_std = self.uncertainty_estimator(state_tensor)
            uncertainty_bonus = q_uncertainty_std.mean().item() # Average uncertainty across all actions
            q_values_for_action_selection = q_values_mean
        else:
            # Use the standard Q-network's predictions
            q_values_for_action_selection, _ = self.forward(state_tensor)

        # Epsilon-greedy exploration action
        if rng.random() < epsilon:
            action = rng.integers(0, self.action_size)
        else:
            # Add a small exploration bonus to Q-values for more robust exploration
            exploration_q_values = q_values_for_action_selection + 0.1 * uncertainty_bonus

            # Ensure numerical stability for exploration Q-values
            from src.ml.numerical_stability import NumericalStabilityManager
            stability_manager = NumericalStabilityManager(epsilon=1e-10)
            exploration_q_values = stability_manager.handle_numerical_instability(
                exploration_q_values, replacement_value=0.0, name="exploration_q_values"
            )

            # Apply softmax for action probability distribution and clamp for stability
            action_probs = F.softmax(exploration_q_values, dim=1)
            action_probs = stability_manager.clamp_probabilities(action_probs)

            # Select action based on clamped probabilities (argmax for greedy selection)
            action = torch.argmax(exploration_q_values, dim=1).item()

        # (If uncertainty estimator predicts only Qs, the config_head needs to be called)
        _, config_params_output = self.forward(state_tensor)
        config_params = config_params_output.cpu().numpy()

        # Calculate policy entropy for diagnostic monitoring
        stability_manager = NumericalStabilityManager(epsilon=1e-10)

        # Use stable softmax and log_softmax for entropy calculation
        probs = F.softmax(q_values_for_action_selection, dim=1)
        probs = stability_manager.clamp_probabilities(probs)  # Clamp and renormalize probabilities

        # Calculate entropy using log_softmax for numerical stability
        log_probs = F.log_softmax(q_values_for_action_selection, dim=1)
        policy_entropy = -torch.sum(probs * log_probs, dim=1)

        # Validate entropy computation results
        if torch.isnan(policy_entropy).any() or torch.isinf(policy_entropy).any():
            # Handle numerical instability with fallback value
            logger.warning("Numerical instability detected in entropy calculation")
            policy_entropy = torch.tensor(0.5, device=policy_entropy.device)  # Fallback to moderate entropy value

        policy_entropy = policy_entropy.mean().item()

        return action, config_params, {
            'uncertainty_bonus': uncertainty_bonus,
            'q_values': q_values_for_action_selection.squeeze().cpu().numpy(),
            'policy_entropy': policy_entropy
        }

def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
    """
    Adds an experience tuple to the N-step buffer. When N steps are collected,
    adds the N-step experience to the main replay buffer.
    """
    self.n_step_buffer.append((state, action, reward, next_state, done))

    if len(self.n_step_buffer) == self.n_step:
        n_step_experience = self._compute_n_step_experience()
        if self.use_prioritized_replay:
            self.replay_buffer.push(n_step_experience)
        else:
            self.replay_buffer.append(n_step_experience)
        if done:
            self.n_step_buffer.clear()

def _compute_n_step_experience(self):
    """
    Computes the N-step experience tuple from the buffer.
    Refactored for reduced cognitive complexity.
    """
    buffer = self.n_step_buffer

    initial_state, initial_action = self._get_initial_state_action(buffer)
    gamma_val = self._get_gamma_value()
    cumulative_return = self._get_cumulative_return(buffer, gamma_val)
    next_state, done_flag = self._get_final_next_state_and_done(buffer)

    return (
        initial_state,
        initial_action,
        cumulative_return,
        next_state,
        done_flag
    )

def _get_initial_state_action(self, buffer):
    """Get the initial state and action from the buffer."""
    return buffer[0][0], buffer[0][1]

def _get_gamma_value(self):
    """Get the gamma value, possibly from meta-learning."""
    if self.use_meta_learning and self.meta_hyperparams:
        return self.meta_hyperparams.get_hyperparams()['gamma']
    return self.gamma

def _get_cumulative_return(self, buffer, gamma_val):
    """Compute the cumulative discounted reward."""
    cumulative_return = 0.0
    for i, (_, _, reward, _, done) in enumerate(buffer):
        cumulative_return += (gamma_val ** i) * reward
        if done:
            break
    return cumulative_return

def _get_final_next_state_and_done(self, buffer):
    """
    Get the final next_state and done flag from the buffer.
    Returns the next_state and done flag of the first item where done==True,
    otherwise returns the last item's next_state and done flag.
    Reduced cognitive complexity using generator expression.
    """
    found = next(((item[3], item[4]) for item in buffer if item[4]), None)
    return found if found is not None else (buffer[-1][3], buffer[-1][4])

class SelfImprovingRLModel(nn.Module):
    # ... previous code ...

    # Add these methods inside the SelfImprovingRLModel class

    def batch_to_tensor(self, batch_data, weights=None):
        """
        Efficiently convert batch data to tensors using DeviceAwareTensorCache.
        Returns tensors for states, actions, rewards, next_states, dones, and weights.
        Robust to input shape and dtype issues.
        """
        # Validate batch_data
        if not batch_data or len(batch_data[0]) != 5:
            raise ValueError("batch_data must be a list of 5-tuples: (state, action, reward, next_state, done)")

        # Use np.asarray for robust conversion
        states      = np.asarray([item[0] for item in batch_data], dtype=np.float32)
        actions     = np.asarray([item[1] for item in batch_data], dtype=np.int64)
        rewards     = np.asarray([item[2] for item in batch_data], dtype=np.float32)
        next_states = np.asarray([item[3] for item in batch_data], dtype=np.float32)
        dones       = np.asarray([item[4] for item in batch_data], dtype=np.bool_)

        # Use tensor cache for device-aware conversion
        states_tensor      = self.tensor_cache.to_tensor(states,      key="batch_states",      dtype=torch.float32)
        actions_tensor     = self.tensor_cache.to_tensor(actions,     key="batch_actions",     dtype=torch.long)
        rewards_tensor     = self.tensor_cache.to_tensor(rewards,     key="batch_rewards",     dtype=torch.float32)
        next_states_tensor = self.tensor_cache.to_tensor(next_states, key="batch_next_states", dtype=torch.float32)
        dones_tensor       = self.tensor_cache.to_tensor(dones,       key="batch_dones",       dtype=torch.bool)

        # Weights tensor
        if weights is not None:
            weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        else:
            weights_tensor = torch.ones(len(batch_data), dtype=torch.float32, device=self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, weights_tensor

    def update(self, step_in_episode_count: int = 0) -> "TrainingMetrics":
        """
        Perform a training update step: sample from replay buffer, compute losses, update networks, and track metrics.
        Refactored to reduce cognitive complexity.
        """
        min_samples = max(self.batch_size * 4, self.batch_size)
        if len(self.replay_buffer) < min_samples:
            return self._create_empty_metrics()

        batch_data, indices, weights = self._sample_replay_buffer()
        if batch_data is None:
            return self._create_empty_metrics()

        states, actions, rewards, next_states, dones, weights_tensor = self.batch_to_tensor(batch_data, weights)
        hyperparams = self._get_current_hyperparams()
        current_lr, tau, gamma = hyperparams['lr'], hyperparams['tau'], hyperparams['gamma']

        self._update_learning_rates(current_lr)
        self._initialize_component_updater()

        q_values_info = self._compute_q_learning_losses(states, actions, rewards, next_states, dones, gamma, weights_tensor)
        weighted_loss, td_errors, current_q_values = q_values_info

        loss_fns = self._build_loss_functions(weighted_loss, states, next_states, actions, current_q_values)
        update_results = self.component_updater.update_all(loss_fns)

        self._soft_update_target_net(tau)
        self._update_priorities(indices, td_errors)

        metrics = self._build_metrics(update_results, current_q_values, td_errors, current_lr, gamma, tau, step_in_episode_count)
        self.training_metrics.append(metrics)
        self.total_steps += 1

        self._periodic_logging()
        return metrics

    def _sample_replay_buffer(self):
        """Sample a batch from the replay buffer."""
        if self.use_prioritized_replay:
            sample_result = self.replay_buffer.sample(self.batch_size)
            if sample_result is None or sample_result[0] is None:
                return None, None, None
            batch_data, indices, weights = sample_result
        else:
            if len(self.replay_buffer) < self.batch_size:
                return None, None, None
            batch_data = random.sample(self.replay_buffer, self.batch_size)
            indices = None
            weights = None
        return batch_data, indices, weights

    def _update_learning_rates(self, current_lr):
        """Update learning rates for all optimizers."""
        for opt in [self.optimizer, self.curiosity_optimizer, self.auxiliary_optimizer]:
            if opt:
                for param_group in opt.param_groups:
                    param_group['lr'] = current_lr
        if self.use_uncertainty and self.uncertainty_estimator and hasattr(self.uncertainty_estimator, 'optimizer'):
            for param_group in self.uncertainty_estimator.optimizer.param_groups:
                param_group['lr'] = current_lr

    def _initialize_component_updater(self):
        """Initialize ParallelComponentUpdater if needed."""
        if self.component_updater is None:
            components = {"main": (self, self.optimizer)}
            if self.use_auxiliary and self.auxiliary_heads and self.auxiliary_optimizer:
                components["auxiliary"] = (self.auxiliary_heads, self.auxiliary_optimizer)
            if self.use_curiosity and self.curiosity_module and self.curiosity_optimizer:
                components["curiosity"] = (self.curiosity_module, self.curiosity_optimizer)
            if self.use_uncertainty and self.uncertainty_estimator and hasattr(self.uncertainty_estimator, 'optimizer'):
                components["uncertainty"] = (self.uncertainty_estimator, self.uncertainty_estimator.optimizer)
            self.component_updater = ParallelComponentUpdater(components)

    def _compute_q_learning_losses(self, states, actions, rewards, next_states, dones, gamma, weights_tensor):
        """Compute Q-learning losses and TD errors."""
        current_q_values, _ = self.forward(states)
        state_action_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        stability_manager = NumericalStabilityManager(epsilon=1e-10)
        current_q_values = stability_manager.handle_numerical_instability(current_q_values, replacement_value=0.0, name="current_q_values")
        state_action_values = stability_manager.handle_numerical_instability(state_action_values, replacement_value=0.0, name="state_action_values")

        next_q_values_online, _ = self.forward(next_states)
        next_q_values_online = stability_manager.handle_numerical_instability(next_q_values_online, replacement_value=0.0, name="next_q_values_online")
        next_actions = next_q_values_online.argmax(dim=1)
        next_q_values_target, _ = self.target_net.forward(next_states)
        next_q_values_target = stability_manager.handle_numerical_instability(next_q_values_target, replacement_value=0.0, name="next_q_values_target")
        next_state_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        next_state_values[dones] = 0.0
        target_q_values = rewards + (gamma * next_state_values)
        target_q_values = stability_manager.handle_numerical_instability(target_q_values, replacement_value=0.0, name="target_q_values")

        td_errors = torch.abs(target_q_values - state_action_values)
        losses = self.q_criterion(state_action_values, target_q_values)
        weighted_loss = (losses * weights_tensor).mean() if weights_tensor is not None else losses.mean()
        return weighted_loss, td_errors, current_q_values

    def _build_loss_functions(self, weighted_loss, states, next_states, actions, current_q_values):
        """Build loss functions for each component."""
        loss_fns = {"main": lambda: weighted_loss}
        if self.use_auxiliary and self.auxiliary_heads:
            loss_fns["auxiliary"] = lambda: self.auxiliary_weight * F.mse_loss(states, next_states)
        if self.use_curiosity and self.curiosity_module:
            loss_fns["curiosity"] = lambda: self.curiosity_weight * sum(self.curiosity_module.compute_loss(states, actions, next_states))
        if self.use_uncertainty and self.uncertainty_estimator:
            loss_fns["uncertainty"] = lambda: self.uncertainty_estimator.compute_loss(self.feature_net(states), actions, current_q_values)
        return loss_fns

    def _soft_update_target_net(self, tau):
        """Soft update the target network."""
        with torch.no_grad():
            for target_param, local_param in zip(self.target_net.parameters(), self.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _update_priorities(self, indices, td_errors):
        """Update priorities in replay buffer."""
        if self.use_prioritized_replay and indices is not None:
            priorities = td_errors.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)

    def _build_metrics(self, update_results, current_q_values, td_errors, current_lr, gamma, tau, step_in_episode_count):
        """Build TrainingMetrics object."""
        return TrainingMetrics(
            loss=update_results.get('main', 0.0),
            q_value_mean=current_q_values.mean().item(),
            q_value_max=current_q_values.max().item(),
            td_error_mean=td_errors.mean().item(),
            replay_buffer_size=len(self.replay_buffer),
            learning_rate=current_lr,
            gamma=gamma,
            tau=tau,
            step_in_episode=step_in_episode_count,
            total_steps=self.total_steps,
        )

    def _periodic_logging(self):
        """Periodic memory/tensor cache logging."""
        if self.total_steps % 100 == 0:
            memory_status = self.memory_monitor.get_status()
            logger.debug(f"Memory status: {memory_status}")
            cache_stats = self.tensor_cache.get_stats()
            logger.debug(f"Tensor cache stats: {cache_stats}")

    def _get_current_hyperparams(self):
        """Get current hyperparameters from meta-learning module."""
        if self.use_meta_learning and self.meta_hyperparams:
            return self.meta_hyperparams.get_hyperparams()
        else:
            return {
                'lr': self.optimizer.param_groups[0]['lr'],
                'gamma': self.gamma,
                'tau': self.tau,
                'epsilon': 0.01  # Default epsilon if not meta-learning
            }