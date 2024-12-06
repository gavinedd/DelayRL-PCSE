import numpy as np
from typing import Tuple, List, Optional
from collections import deque

class KnowledgePool:
    """Manages shared knowledge and experiences between farms"""
    def __init__(self, learning_rate: float = 0.1, memory_size: int = 1000,
                 similarity_threshold: float = 0.2):
        self.shared_experiences = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.similarity_threshold = similarity_threshold
        self.state_means = None
        self.state_stds = None
        
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        """Add new experience to the knowledge pool with state normalization"""
        if self.state_means is None:
            self._initialize_normalizers(state)
        
        normalized_state = self._normalize_state(state)
        self.shared_experiences.append((normalized_state, action, reward))
        self._update_normalizers(state)
    
    def get_knowledge_bonus(self, state: np.ndarray, action: np.ndarray) -> float:
        """Calculate bonus based on similar successful experiences with adaptive similarity"""
        if not self.shared_experiences or self.state_means is None:
            return 0.0
            
        normalized_state = self._normalize_state(state)
        
        # Find similar experiences using adaptive thresholds
        similar_experiences = []
        total_weight = 0.0
        
        for exp_state, exp_action, exp_reward in self.shared_experiences:
            # Calculate similarities
            state_similarity = np.exp(-np.mean(np.square(normalized_state - exp_state)))
            action_similarity = np.exp(-np.mean(np.square(action - exp_action)))
            
            # Combined similarity with more weight on state
            similarity = 0.7 * state_similarity + 0.3 * action_similarity
            
            if similarity > self.similarity_threshold:
                similar_experiences.append((exp_reward, similarity))
                total_weight += similarity
        
        if not similar_experiences:
            return 0.0
            
        # Calculate weighted average of rewards
        weighted_reward = sum(reward * weight for reward, weight in similar_experiences) / total_weight
        return weighted_reward * self.learning_rate
    
    def _initialize_normalizers(self, state: np.ndarray) -> None:
        """Initialize state normalizers with first state"""
        self.state_means = state.copy()
        self.state_stds = np.ones_like(state)
        
    def _update_normalizers(self, state: np.ndarray) -> None:
        """Update running statistics for state normalization"""
        if len(self.shared_experiences) == 1:
            return
        
        # Update running mean and std using Welford's online algorithm with numerical stability fixes
        n = len(self.shared_experiences)
        delta = state - self.state_means
        self.state_means = self.state_means + delta / n
        
        # Use a more numerically stable method for variance calculation
        if n > 1:
            delta2 = state - self.state_means
            self.state_stds = np.sqrt(
                ((self.state_stds ** 2) * (n - 2) + delta * delta2) / (n - 1)
            )
        
        # Ensure minimum standard deviation to prevent division by zero
        self.state_stds = np.maximum(self.state_stds, 1e-8)
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics with clipping"""
        if self.state_means is None:
            return state
        
        # Clip input state to prevent extreme values
        state = np.clip(state, -1e6, 1e6)
        
        # Normalize with numerical stability
        normalized = np.zeros_like(state)
        mask = self.state_stds > 1e-8
        normalized[mask] = (state[mask] - self.state_means[mask]) / self.state_stds[mask]
        normalized[~mask] = 0.0
        
        # Clip normalized values to prevent extreme outputs
        return np.clip(normalized, -10.0, 10.0)
    
    def reset(self) -> None:
        """Reset knowledge pool and normalizers"""
        self.shared_experiences.clear()
        self.state_means = None
        self.state_stds = None 