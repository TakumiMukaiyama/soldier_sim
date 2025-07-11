import random
from typing import Dict, List, Optional
from uuid import uuid4


class Agent:
    def __init__(
        self,
        agent_id: Optional[str] = None,
        age: int = 30,
        personality: Dict[str, float] = None,
        rank: str = "private",
        energy: float = 1.0,
        social: float = 0.5,
        hunger: float = 0.0,
        weapon_strength: float = 0.5,
        management_skill: float = 0,
        sociability: float = 0.5,
    ):
        self.id = agent_id or str(uuid4())
        self.age = age
        self.personality = personality or {
            "extroversion": 0.5,
            "conscientiousness": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
            "openness": 0.5,
        }
        self.rank = rank

        # Current state values (0.0 to 1.0)
        self.energy = max(0.0, min(1.0, energy))
        self.social = max(0.0, min(1.0, social))
        self.hunger = max(0.0, min(1.0, hunger))
        self.weapon_strength = max(0.0, min(1.0, weapon_strength))
        self.management_skill = max(0.0, min(1.0, management_skill))
        self.sociability = max(0.0, min(1.0, sociability))

        # Current location (set when agent moves)
        self.current_poi_id = None
        self.location = [0.0, 0.0]

        # Track training sessions and activities
        self._training_sessions = 0
        self._management_sessions = 0
        self._social_sessions = 0
        self._daily_activities = []
        self._skill_caps = {
            "weapon_strength": 0.7,
            "management_skill": 0.7,
            "social": 0.7,
            "sociability": 0.8,
        }

    def update_needs(self, time_delta: float = 1.0) -> None:
        """Update agent needs based on time passing"""
        # Energy decreases over time (with slight randomization)
        energy_decay = 0.05 * time_delta * random.uniform(0.8, 1.2)
        self.energy = max(0.0, self.energy - energy_decay)

        # Hunger increases over time (with slight randomization)
        hunger_increase = 0.08 * time_delta * random.uniform(0.8, 1.2)
        self.hunger = min(1.0, self.hunger + hunger_increase)

        # Social need changes based on personality
        social_base_change = 0.03 * time_delta
        # Extroverts lose social energy faster when alone
        if self.personality["extroversion"] > 0.5:
            social_change = social_base_change * (
                self.personality["extroversion"] + 0.5
            )
        else:
            # Introverts lose social energy more slowly
            social_change = social_base_change * self.personality["extroversion"]

        self.social = max(0.0, min(1.0, self.social - social_change))

        # Apply natural decay to skills if they haven't been practiced recently
        if "train" not in self._daily_activities:
            self._apply_skill_decay("weapon_strength", 0.01 * time_delta)

        if "manage" not in self._daily_activities:
            self._apply_skill_decay("management_skill", 0.01 * time_delta)

        if "socialize" not in self._daily_activities:
            self._apply_skill_decay(
                "sociability", 0.005 * time_delta
            )  # Slower decay for social skills

        # Reset daily activities list if it's getting too long
        if len(self._daily_activities) > 10:
            self._daily_activities = self._daily_activities[-5:]

    def _apply_skill_decay(self, skill: str, amount: float) -> None:
        """Apply natural decay to a skill"""
        current_value = getattr(self, skill)
        # Skills decay slower as they approach baseline values
        baseline = 0.3  # Base skill level
        decay_factor = max(
            0.1, (current_value - baseline) / 0.7
        )  # Higher skills decay faster

        new_value = current_value - (amount * decay_factor)
        # Don't decay below baseline
        new_value = max(baseline, new_value)
        setattr(self, skill, new_value)

    def _update_skill_caps(self) -> None:
        """Update skill caps based on rank and training/management sessions"""
        rank_multipliers = {
            "private": 1.0,
            "corporal": 1.1,
            "sergeant": 1.2,
            "lieutenant": 1.3,
            "captain": 1.4,
        }

        # Get multiplier based on rank (default to private)
        rank_mult = rank_multipliers.get(self.rank.lower(), 1.0)

        # Calculate base caps with rank influence
        weapon_cap = min(1.0, 0.7 + (self._training_sessions * 0.02)) * rank_mult
        management_cap = min(1.0, 0.7 + (self._management_sessions * 0.02)) * rank_mult
        sociability_cap = min(1.0, 0.8 + (self._social_sessions * 0.015)) * rank_mult

        # Personality influences caps as well
        conscientiousness = self.personality.get("conscientiousness", 0.5)
        openness = self.personality.get("openness", 0.5)
        agreeableness = self.personality.get("agreeableness", 0.5)
        extroversion = self.personality.get("extroversion", 0.5)

        # Update the skill caps
        self._skill_caps["weapon_strength"] = min(
            1.0, weapon_cap * (1 + (conscientiousness - 0.5) * 0.2)
        )
        self._skill_caps["management_skill"] = min(
            1.0, management_cap * (1 + (openness - 0.5) * 0.2)
        )
        self._skill_caps["social"] = min(
            1.0, 0.7 + (self.personality.get("extroversion", 0.5) * 0.3)
        )

        # Sociability cap influenced by agreeableness and extroversion
        personality_modifier = ((agreeableness + extroversion) / 2 - 0.5) * 0.3
        self._skill_caps["sociability"] = min(
            1.0, sociability_cap * (1 + personality_modifier)
        )

    def apply_poi_effect(self, poi_effects: Dict[str, float]) -> None:
        """
        Apply effects from a POI visit

        Effects now have diminishing returns as values approach 1.0
        """
        # Track activity
        if "weapon_strength" in poi_effects and poi_effects["weapon_strength"] > 0:
            self._daily_activities.append("train")
            self._training_sessions += 1

        if "management_skill" in poi_effects and poi_effects["management_skill"] > 0:
            self._daily_activities.append("manage")
            self._management_sessions += 1

        if "sociability" in poi_effects and poi_effects["sociability"] > 0:
            self._daily_activities.append("socialize")
            self._social_sessions += 1

        # Update skill caps
        self._update_skill_caps()

        # Apply effects with diminishing returns
        self._apply_effect_with_diminishing_returns(
            "energy", poi_effects.get("energy", 0)
        )
        self._apply_effect_with_diminishing_returns(
            "hunger", poi_effects.get("hunger", 0)
        )
        self._apply_effect_with_diminishing_returns(
            "social", poi_effects.get("social", 0), self._skill_caps["social"]
        )
        self._apply_effect_with_diminishing_returns(
            "weapon_strength",
            poi_effects.get("weapon_strength", 0),
            self._skill_caps["weapon_strength"],
        )
        self._apply_effect_with_diminishing_returns(
            "management_skill",
            poi_effects.get("management_skill", 0),
            self._skill_caps["management_skill"],
        )
        self._apply_effect_with_diminishing_returns(
            "sociability",
            poi_effects.get("sociability", 0),
            self._skill_caps["sociability"],
        )

    def _apply_effect_with_diminishing_returns(
        self, attribute: str, effect: float, cap: float = 1.0
    ) -> None:
        """
        Apply effect to an attribute with diminishing returns

        Args:
            attribute: The attribute to update
            effect: The effect value (positive or negative)
            cap: Maximum value this attribute can reach (default 1.0)
        """
        if effect == 0:
            return

        current = getattr(self, attribute)

        if effect > 0:
            # Positive effects have diminishing returns as we approach the cap
            distance_to_cap = cap - current
            if distance_to_cap <= 0:
                # Already at or above cap, minimal effect
                new_value = current + (effect * 0.1)
            else:
                # Effect diminishes as we approach the cap
                diminish_factor = distance_to_cap / cap
                # Add randomness for more varied growth
                random_factor = random.uniform(0.8, 1.2)
                new_value = current + (effect * diminish_factor * random_factor)
        else:
            # Negative effects are more pronounced when at high values
            # and less pronounced when at low values
            diminish_factor = current / cap
            # Add randomness for more varied decrease
            random_factor = random.uniform(0.8, 1.2)
            new_value = current + (effect * diminish_factor * random_factor)

        # Ensure we stay within bounds
        new_value = max(0.0, min(cap, new_value))

        # Update the attribute
        setattr(self, attribute, new_value)

    def move_to(self, poi_id: str, location: List[float]) -> None:
        """Move agent to a POI"""
        self.current_poi_id = poi_id
        self.location = location

    def to_dict(self) -> Dict:
        """Convert agent to dictionary for serialization"""
        return {
            "id": self.id,
            "age": self.age,
            "personality": self.personality,
            "rank": self.rank,
            "energy": self.energy,
            "social": self.social,
            "hunger": self.hunger,
            "weapon_strength": self.weapon_strength,
            "management_skill": self.management_skill,
            "sociability": self.sociability,
            "current_poi_id": self.current_poi_id,
            "location": self.location,
            "skill_caps": self._skill_caps,
        }
