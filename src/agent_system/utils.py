"""
Utility functions for agent system
"""

from typing import Any, Dict, Optional, TypeVar, Union

T = TypeVar('T', int, float)


def clamp_value(value: T, min_value: T, max_value: T) -> T:
    """Clamp a value between min and max
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(max_value, value))


def apply_personality_modifier(
    base_value: float,
    personality: Dict[str, float],
    trait: str,
    effect_strength: float = 0.4,
    random_factor: Optional[float] = None
) -> float:
    """Apply a personality trait-based modifier to a value
    
    Args:
        base_value: Base value to modify
        personality: Dictionary of personality traits
        trait: The personality trait to use
        effect_strength: Strength of the personality effect (default: 0.4)
        random_factor: Optional random factor to add variability
        
    Returns:
        Modified value
    """
    trait_value = personality.get(trait, 0.5)
    
    # Calculate modifier based on trait (centered around 0.5)
    modifier = (trait_value - 0.5) * effect_strength
    
    # Apply random factor if provided
    if random_factor is not None:
        modifier += random_factor
    
    return base_value * (1 + modifier)