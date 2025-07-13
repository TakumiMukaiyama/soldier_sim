"""
Needs management for agent system
"""

import random
from typing import Dict, List, Optional

from src.agent_system.utils import clamp_value


class NeedsManager:
    """Manages agent needs like energy, hunger, and social"""
    
    def __init__(self, 
                 personality: Dict[str, float],
                 initial_values: Optional[Dict[str, float]] = None):
        """Initialize needs manager with initial values
        
        Args:
            personality: Agent's personality traits
            initial_values: Initial need values (0-1 range)
        """
        self.personality = personality
        
        # Set default needs values (0-100 range)
        self._energy = 100.0
        self._hunger = 0.0
        self._social = 50.0
        
        # Apply initial values if provided (convert from 0-1 to 0-100 range)
        if initial_values:
            self._energy = initial_values.get("energy", 1.0) * 100.0
            self._hunger = initial_values.get("hunger", 0.0) * 100.0
            self._social = initial_values.get("social", 0.5) * 100.0
    
    def update_needs(self, time_delta: float = 1.0, current_hour: int = 12) -> None:
        """Update agent needs based on time passing and time of day
        
        Args:
            time_delta: Time multiplier for need changes
            current_hour: Current hour of the day (0-23)
        """
        # Time-based modifiers
        is_night_time = current_hour >= 22 or current_hour <= 5
        is_meal_time = current_hour in [12, 18]

        # Energy decreases over time (with slight randomization)
        base_energy_decay = 0.05 * time_delta
        # Energy drains faster at night if not sleeping
        if is_night_time:
            base_energy_decay *= 1.5
        energy_decay = base_energy_decay * random.uniform(0.8, 1.2)
        self._energy = clamp_value(self._energy - energy_decay, 0.0, 100.0)

        # Hunger increases over time (with slight randomization)
        base_hunger_increase = 0.08 * time_delta
        # Hunger increases faster approaching meal times
        if is_meal_time:
            base_hunger_increase *= 1.3
        hunger_increase = base_hunger_increase * random.uniform(0.8, 1.2)
        self._hunger = clamp_value(self._hunger + hunger_increase, 0.0, 100.0)

        # Social need changes based on personality and time
        social_base_change = 0.03 * time_delta
        # Night time reduces social needs
        if is_night_time:
            social_base_change *= 0.5

        # Extroverts lose social energy faster when alone
        if self.personality["extroversion"] > 0.5:
            social_change = social_base_change * (
                self.personality["extroversion"] + 0.5
            )
        else:
            # Introverts lose social energy more slowly
            social_change = social_base_change * self.personality["extroversion"]

        self._social = clamp_value(self._social - social_change, 0.0, 100.0)
    
    def apply_effect(self, need_type: str, effect: float) -> None:
        """Apply effect to a need attribute
        
        Args:
            need_type: The need to update ("energy", "hunger", "social")
            effect: The effect value (positive or negative)
        """
        if effect == 0:
            return
            
        # Map need_type to property
        property_map = {
            "energy": "_energy",
            "hunger": "_hunger",
            "social": "_social",
        }
        
        if need_type not in property_map:
            return
            
        property_name = property_map[need_type]
        current = getattr(self, property_name)
        
        # Apply effect
        new_value = current + effect
        
        # Ensure we stay within bounds
        new_value = clamp_value(new_value, 0.0, 100.0)
        
        # Update the attribute
        setattr(self, property_name, new_value)
    
    @property
    def energy(self) -> float:
        """Get energy value"""
        return self._energy
    
    @property
    def hunger(self) -> float:
        """Get hunger value"""
        return self._hunger
    
    @property
    def social(self) -> float:
        """Get social value"""
        return self._social
    
    def to_dict(self) -> Dict[str, float]:
        """Convert needs to dictionary for serialization"""
        return {
            "energy": self._energy,
            "hunger": self._hunger,
            "social": self._social,
        }