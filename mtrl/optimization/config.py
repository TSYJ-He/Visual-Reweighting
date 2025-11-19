
"""
Optimization configuration
"""

from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    gamma: float = 1.0
    """discount factor for advantage estimator"""
    lam: float = 1.0
    """lambda value for advantage estimator"""
    advantage_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_divergence: bool = False
    """disable reference model"""
    use_divergence_loss: bool = False
    """use divergence loss instead of divergence in reward"""
    divergence_penalty: str = "kl"
    """divergence penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    divergence_coef: float = 1e-3
    """divergence coefficient"""
    divergence_type: str = "fixed"
    """divergence controller type, support `fixed`, `adaptive`"""
    divergence_horizon: float = 10000.0
    """divergence horizon for adaptive divergence controller"""
    divergence_target: float = 0.1
    """target divergence for adaptive divergence controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""

    enable_entropy_filtering: bool = False
    top_p_entropy_tokens: float = 0.2
    """enable token filtering based on entropy"""

    enable_perception_filtering: bool = False
    top_p_perception_tokens: float = 0.2
    """enable token filtering based on perception"""

    use_entropy_penalty: bool = False
    entropy_penalty_coef: float = 0.06
    """use entropy penalty for training"""

    enable_trajectory_reweighting: bool = False
    trajectory_scaling_min: float = 0.8

