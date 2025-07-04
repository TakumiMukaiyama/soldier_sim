from typing import Tuple


def kalman_update(
    belief: float, sigma_b: float, observation: float, sigma_o: float = 0.2
) -> Tuple[float, float]:
    """
    Simple Kalman filter update for scalar values.

    Args:
        belief: Current belief value (prior)
        sigma_b: Uncertainty in current belief (prior variance)
        observation: New observation value
        sigma_o: Observation noise/uncertainty (variance)

    Returns:
        Tuple of (updated_belief, updated_sigma)
    """
    # Calculate Kalman gain
    # K represents how much we should trust the observation vs prior belief
    K = sigma_b / (sigma_b + sigma_o)

    # Update belief (weighted average based on Kalman gain)
    updated_belief = K * observation + (1 - K) * belief

    # Update uncertainty (always decreases after an observation)
    updated_sigma = (1 - K) * sigma_b

    return updated_belief, updated_sigma
