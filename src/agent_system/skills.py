"""
Skill management for agent system
"""

import random
from typing import Dict, List, Optional

from src.agent_system.archetypes import get_skill_multiplier
from src.agent_system.utils import clamp_value


class SkillManager:
    """Manages skill acquisition, decay, and caps for agents"""
    
    def __init__(self, 
                 agent_id: str, 
                 archetype: str, 
                 personality: Dict[str, float],
                 initial_values: Optional[Dict[str, float]] = None):
        """Initialize skill manager with initial values
        
        Args:
            agent_id: ID of the agent this manager belongs to
            archetype: Agent's archetype
            personality: Agent's personality traits
            initial_values: Initial skill values
        """
        self.agent_id = agent_id
        self.archetype = archetype
        self.personality = personality
        
        # Set default skill values
        self._weapon_strength = 0.5
        self._management_skill = 0.3
        self._sociability = 0.5
        self._power = 0.5
        
        # Apply initial values if provided
        if initial_values:
            self._weapon_strength = initial_values.get("weapon_strength", self._weapon_strength)
            self._management_skill = initial_values.get("management_skill", self._management_skill)
            self._sociability = initial_values.get("sociability", self._sociability)
            self._power = initial_values.get("power", self._power)
            
            # Convert from 0-1 range to 0-100 range
            self._weapon_strength *= 100.0
            self._management_skill *= 100.0
            self._sociability *= 100.0
            self._power *= 100.0
        
        # Training sessions tracking
        self._training_sessions = 0
        self._management_sessions = 0
        self._social_sessions = 0
        
        # Initialize skill caps with some randomization
        self._skill_caps = self._initialize_skill_caps()
        
        # Individual growth rate modifiers
        self._growth_modifiers = self._initialize_growth_modifiers()
        
        # Individual decay resistance modifiers
        self._decay_resistance = self._initialize_decay_resistance()
    
    def _initialize_skill_caps(self) -> Dict[str, float]:
        """Initialize skill caps with randomization"""
        return {
            "weapon_strength": 70.0 + random.uniform(-10.0, 10.0),
            "management_skill": 70.0 + random.uniform(-10.0, 10.0),
            "sociability": 80.0 + random.uniform(-10.0, 10.0),
            "power": 80.0 + random.uniform(-10.0, 10.0),
        }
    
    def _initialize_growth_modifiers(self) -> Dict[str, float]:
        """Initialize growth modifiers based on personality"""
        return {
            "weapon_strength": 1.0
            + (self.personality.get("conscientiousness", 0.5) - 0.5) * 0.4
            + random.uniform(-0.2, 0.2),
            "management_skill": 1.0
            + (self.personality.get("openness", 0.5) - 0.5) * 0.4
            + random.uniform(-0.2, 0.2),
            "sociability": 1.0
            + (self.personality.get("extroversion", 0.5) - 0.5) * 0.4
            + random.uniform(-0.2, 0.2),
            "power": 1.0
            + (self.personality.get("conscientiousness", 0.5) - 0.5) * 0.4
            + random.uniform(-0.2, 0.2),
        }
    
    def _initialize_decay_resistance(self) -> Dict[str, float]:
        """Initialize decay resistance modifiers"""
        conscientiousness = self.personality.get("conscientiousness", 0.5)
        return {
            "weapon_strength": 0.5
            + conscientiousness * 0.5
            + random.uniform(-0.1, 0.1),
            "management_skill": 0.5
            + conscientiousness * 0.5
            + random.uniform(-0.1, 0.1),
            "sociability": 0.7
            + self.personality.get("agreeableness", 0.5) * 0.3
            + random.uniform(-0.1, 0.1),
            "power": 0.5 + conscientiousness * 0.5 + random.uniform(-0.1, 0.1),
        }
    
    def update_skill_caps(self, age: int, rank: str) -> None:
        """Update skill caps based on rank, training sessions, age, and archetype
        
        Args:
            age: Agent's age
            rank: Agent's rank
        """
        rank_multipliers = {
            "private": 1.0,
            "corporal": 1.1,
            "sergeant": 1.2,
            "lieutenant": 1.3,
            "captain": 1.4,
        }

        # Get multiplier based on rank (default to private)
        rank_mult = rank_multipliers.get(rank.lower(), 1.0)

        # Age-based modifiers
        age_factor = 1.0
        if age > 35:
            age_factor = 1.1  # More experienced
        elif age < 25:
            age_factor = 0.95  # Less experienced but potentially faster learning

        # Calculate base caps with rank and experience influence
        weapon_cap = min(
            100.0, (70.0 + (self._training_sessions * 2.0)) * rank_mult * age_factor
        )
        management_cap = min(
            100.0, (70.0 + (self._management_sessions * 2.0)) * rank_mult * age_factor
        )
        sociability_cap = min(100.0, (80.0 + (self._social_sessions * 1.5)) * rank_mult)
        power_cap = min(
            100.0, (80.0 + (self._training_sessions * 1.0)) * rank_mult * age_factor
        )

        # Personality influences caps more strongly
        conscientiousness = self.personality.get("conscientiousness", 0.5)
        openness = self.personality.get("openness", 0.5)
        agreeableness = self.personality.get("agreeableness", 0.5)
        extroversion = self.personality.get("extroversion", 0.5)

        # Enhanced personality effects with more variation
        personality_weapon_mod = (conscientiousness - 0.5) * 0.4
        personality_mgmt_mod = (openness - 0.5) * 0.4
        personality_social_mod = (extroversion - 0.5) * 0.4
        personality_power_mod = (conscientiousness - 0.5) * 0.4

        # Update the skill caps with personality modifiers
        self._skill_caps["weapon_strength"] = min(
            100.0, weapon_cap * (1 + personality_weapon_mod)
        )
        self._skill_caps["management_skill"] = min(
            100.0, management_cap * (1 + personality_mgmt_mod)
        )
        self._skill_caps["power"] = min(
            100.0, power_cap * (1 + personality_power_mod)
        )

        # Sociability cap influenced by multiple personality traits
        personality_modifier = (
            (agreeableness + extroversion) / 2 - 0.5
        ) * 0.5
        self._skill_caps["sociability"] = min(
            100.0, sociability_cap * (1 + personality_modifier)
        )

        # Apply archetype-specific hard caps
        if self.archetype == "weapon_specialist":
            self._skill_caps["management_skill"] = min(
                self._skill_caps["management_skill"], 75.0
            )
        elif self.archetype == "scholar":
            self._skill_caps["weapon_strength"] = min(
                self._skill_caps["weapon_strength"], 80.0
            )
        elif self.archetype == "introvert":
            self._skill_caps["sociability"] = min(self._skill_caps["sociability"], 85.0)
        elif self.archetype == "fitness_enthusiast":
            self._skill_caps["management_skill"] = min(
                self._skill_caps["management_skill"], 80.0
            )
    
    def apply_effect(self, attribute: str, effect: float) -> None:
        """Apply effect to a skill attribute with diminishing returns
        
        Args:
            attribute: The attribute to update ("weapon_strength", "management_skill", etc.)
            effect: The effect value (positive or negative)
        """
        if effect == 0:
            return
            
        # Map attribute to property
        property_map = {
            "weapon_strength": "_weapon_strength",
            "management_skill": "_management_skill",
            "sociability": "_sociability",
            "power": "_power",
        }
        
        if attribute not in property_map:
            return
            
        property_name = property_map[attribute]
        current = getattr(self, property_name)
        cap = self._skill_caps.get(attribute, 100.0)

        # Apply individual growth modifiers for skill improvements
        if effect > 0 and attribute in self._growth_modifiers:
            growth_modifier = self._growth_modifiers[attribute]
            effect *= growth_modifier

        if effect > 0:
            # Positive effects have diminishing returns as we approach the cap
            distance_to_cap = cap - current
            if distance_to_cap <= 0:
                # Already at or above cap, minimal effect
                new_value = current + (effect * 0.1)
            else:
                # Effect diminishes as we approach the cap
                diminish_factor = distance_to_cap / cap
                # Add more randomness for varied growth patterns
                random_factor = random.uniform(0.6, 1.4)
                new_value = current + (effect * diminish_factor * random_factor)
        else:
            # Negative effects are more pronounced when at high values
            # and less pronounced when at low values
            diminish_factor = current / cap
            # Add randomness for more varied decrease
            random_factor = random.uniform(0.8, 1.2)
            new_value = current + (effect * diminish_factor * random_factor)

        # Ensure we stay within bounds
        new_value = clamp_value(new_value, 0.0, cap)

        # Update the attribute
        setattr(self, property_name, new_value)
    
    def apply_skill_decay(self, skill: str, amount: float) -> None:
        """Apply natural decay to a skill with personalized resistance
        
        Args:
            skill: The skill to decay
            amount: Base decay amount
        """
        property_map = {
            "weapon_strength": "_weapon_strength",
            "management_skill": "_management_skill",
            "sociability": "_sociability",
            "power": "_power",
        }
        
        if skill not in property_map:
            return
            
        property_name = property_map[skill]
        current_value = getattr(self, property_name)

        # Apply decay resistance based on personality
        resistance = self._decay_resistance.get(skill, 1.0)
        adjusted_amount = amount * (2.0 - resistance)  # Higher resistance = less decay

        # Skills decay slower as they approach baseline values
        baseline = 30.0  # Base skill level
        decay_factor = max(
            0.1, (current_value - baseline) / 70.0
        )  # Higher skills decay faster

        new_value = current_value - (adjusted_amount * decay_factor)
        # Don't decay below baseline
        new_value = max(baseline, new_value)
        setattr(self, property_name, new_value)
    
    def record_activity(self, activity: str) -> None:
        """Record an activity that affects skill development
        
        Args:
            activity: Activity name
        """
        if activity in ["train", "exercise", "outdoor_train"]:
            self._training_sessions += 1
        elif activity in ["manage", "study"]:
            self._management_sessions += 1
        elif activity == "socialize":
            self._social_sessions += 1
    
    @property
    def weapon_strength(self) -> float:
        """Get weapon strength value"""
        return self._weapon_strength
    
    @property
    def management_skill(self) -> float:
        """Get management skill value"""
        return self._management_skill
    
    @property
    def sociability(self) -> float:
        """Get sociability value"""
        return self._sociability
    
    @property
    def power(self) -> float:
        """Get power value"""
        return self._power
    
    def to_dict(self) -> Dict[str, float]:
        """Convert skills to dictionary for serialization"""
        return {
            "weapon_strength": self._weapon_strength,
            "management_skill": self._management_skill,
            "sociability": self._sociability,
            "power": self._power,
        }