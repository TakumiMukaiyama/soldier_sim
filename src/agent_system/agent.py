import random
from typing import Dict, List, Optional
from uuid import uuid4

# Archetype definitions for different military personas
ARCHETYPES = {
    "weapon_specialist": {
        "description": "Weapon enthusiast who loves training and armory work",
        "skill_multipliers": {
            "weapon_strength": 1.5,
            "management_skill": 0.8,
            "sociability": 0.9,
            "power": 1.4,
        },
        "poi_preferences": {
            "training": 2.0,
            "armory": 2.5,
            "food": 1.0,
            "rest": 0.8,
            "sleep": 1.0,
            "office": 0.5,
            "recreation": 0.7,
            "medical": 0.8,
            "fitness": 1.4,
            "library": 0.6,
            "workshop": 1.8,
            "communications": 0.6,
            "maintenance": 1.3,
            "outdoor": 1.9,
            "spiritual": 0.7,
            "logistics": 0.8,
        },
    },
    "natural_leader": {
        "description": "Born leader with management and strategy focus",
        "skill_multipliers": {
            "weapon_strength": 1.0,
            "management_skill": 1.6,
            "sociability": 1.3,
            "power": 1.1,
        },
        "poi_preferences": {
            "training": 1.2,
            "armory": 1.0,
            "food": 1.1,
            "rest": 0.9,
            "sleep": 1.0,
            "office": 2.0,
            "recreation": 1.0,
            "medical": 1.0,
            "fitness": 1.1,
            "library": 1.6,
            "workshop": 1.2,
            "communications": 1.8,
            "maintenance": 1.4,
            "outdoor": 1.3,
            "spiritual": 1.1,
            "logistics": 1.9,
        },
    },
    "social_butterfly": {
        "description": "Highly social person who excels in communication",
        "skill_multipliers": {
            "weapon_strength": 0.9,
            "management_skill": 1.2,
            "sociability": 1.7,
            "power": 0.8,
        },
        "poi_preferences": {
            "training": 1.0,
            "armory": 0.6,
            "food": 1.8,
            "rest": 1.0,
            "sleep": 1.0,
            "office": 1.3,
            "recreation": 2.2,
            "medical": 1.2,
            "fitness": 1.3,
            "library": 0.8,
            "workshop": 0.9,
            "communications": 2.1,
            "maintenance": 1.0,
            "outdoor": 1.4,
            "spiritual": 1.5,
            "logistics": 1.2,
        },
    },
    "scholar": {
        "description": "Intellectual type who prefers study and strategic planning",
        "skill_multipliers": {
            "weapon_strength": 0.8,
            "management_skill": 1.4,
            "sociability": 1.0,
            "power": 0.7,
        },
        "poi_preferences": {
            "training": 0.8,
            "armory": 0.7,
            "food": 1.0,
            "rest": 1.2,
            "sleep": 1.1,
            "office": 2.2,
            "recreation": 0.9,
            "medical": 1.0,
            "fitness": 0.7,
            "library": 2.5,
            "workshop": 1.1,
            "communications": 1.7,
            "maintenance": 0.8,
            "outdoor": 0.6,
            "spiritual": 1.3,
            "logistics": 1.6,
        },
    },
    "fitness_enthusiast": {
        "description": "Physical fitness focused with high energy",
        "skill_multipliers": {
            "weapon_strength": 1.3,
            "management_skill": 0.9,
            "sociability": 1.1,
            "power": 1.6,
        },
        "poi_preferences": {
            "training": 2.3,
            "armory": 1.1,
            "food": 1.2,
            "rest": 0.6,
            "sleep": 0.8,
            "office": 0.4,
            "recreation": 1.5,
            "medical": 0.9,
            "fitness": 2.5,
            "library": 0.5,
            "workshop": 1.2,
            "communications": 0.7,
            "maintenance": 1.1,
            "outdoor": 2.4,
            "spiritual": 0.8,
            "logistics": 0.6,
        },
    },
    "introvert": {
        "description": "Quiet type who prefers individual work and rest",
        "skill_multipliers": {
            "weapon_strength": 1.1,
            "management_skill": 1.0,
            "sociability": 0.7,
            "power": 1.0,
        },
        "poi_preferences": {
            "training": 1.0,
            "armory": 1.3,
            "food": 0.7,
            "rest": 2.0,
            "sleep": 1.8,
            "office": 1.4,
            "recreation": 0.5,
            "medical": 1.1,
            "fitness": 0.8,
            "library": 1.9,
            "workshop": 1.6,
            "communications": 0.8,
            "maintenance": 1.5,
            "outdoor": 0.7,
            "spiritual": 1.7,
            "logistics": 1.2,
        },
    },
}


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
        power: float = 0.5,
        archetype: str = "fitness_enthusiast",
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
        self.archetype = archetype if archetype in ARCHETYPES else "fitness_enthusiast"

        # Current state values (0.0 to 100.0)
        self.energy = max(0.0, min(100.0, energy))
        self.social = max(0.0, min(100.0, social))
        self.hunger = max(0.0, min(100.0, hunger))
        self.weapon_strength = max(0.0, min(100.0, weapon_strength))
        self.management_skill = max(0.0, min(100.0, management_skill))
        self.sociability = max(0.0, min(100.0, sociability))
        self.power = max(0.0, min(100.0, power))

        # Current location (set when agent moves)
        self.current_poi_id = None
        self.location = [0.0, 0.0]

        # Track training sessions and activities
        self._training_sessions = 0
        self._management_sessions = 0
        self._social_sessions = 0
        self._daily_activities = []

        # Initialize base skill caps with some randomization (scale to 100)
        self._skill_caps = {
            "weapon_strength": 70.0 + random.uniform(-10.0, 10.0),
            "management_skill": 70.0 + random.uniform(-10.0, 10.0),
            "social": 70.0 + random.uniform(-10.0, 10.0),
            "sociability": 80.0 + random.uniform(-10.0, 10.0),
            "power": 80.0 + random.uniform(-10.0, 10.0),
        }

        # Individual growth rate modifiers (based on personality + randomness)
        self._growth_modifiers = {
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

        # Individual decay rate modifiers (lazy people lose skills faster)
        conscientiousness = self.personality.get("conscientiousness", 0.5)
        self._decay_resistance = {
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

        # Update skill caps based on archetype
        self._apply_archetype_modifiers()

    def _apply_archetype_modifiers(self) -> None:
        """Apply archetype-specific modifiers to skill caps and preferences"""
        if self.archetype not in ARCHETYPES:
            return

        archetype_data = ARCHETYPES[self.archetype]
        multipliers = archetype_data["skill_multipliers"]

        # Apply archetype multipliers to base skill caps
        self._skill_caps["weapon_strength"] *= multipliers.get("weapon_strength", 1.0)
        self._skill_caps["management_skill"] *= multipliers.get("management_skill", 1.0)
        self._skill_caps["sociability"] *= multipliers.get("sociability", 1.0)
        self._skill_caps["power"] *= multipliers.get("power", 1.0)

        # Ensure caps don't exceed 100.0
        for skill in self._skill_caps:
            self._skill_caps[skill] = min(100.0, self._skill_caps[skill])

    def get_poi_preference_multiplier(self, poi_category: str) -> float:
        """Get preference multiplier for a specific POI category based on archetype"""
        if self.archetype not in ARCHETYPES:
            return 1.0

        archetype_data = ARCHETYPES[self.archetype]
        return archetype_data["poi_preferences"].get(poi_category, 1.0)

    def update_needs(self, time_delta: float = 1.0, current_hour: int = 12) -> None:
        """Update agent needs based on time passing and time of day"""
        # Time-based modifiers
        is_night_time = current_hour >= 22 or current_hour <= 5
        is_meal_time = current_hour in [12, 18]

        # Energy decreases over time (with slight randomization)
        base_energy_decay = 0.05 * time_delta
        # Energy drains faster at night if not sleeping
        if is_night_time:
            base_energy_decay *= 1.5
        energy_decay = base_energy_decay * random.uniform(0.8, 1.2)
        self.energy = max(0.0, self.energy - energy_decay)

        # Hunger increases over time (with slight randomization)
        base_hunger_increase = 0.08 * time_delta
        # Hunger increases faster approaching meal times
        if is_meal_time:
            base_hunger_increase *= 1.3
        hunger_increase = base_hunger_increase * random.uniform(0.8, 1.2)
        self.hunger = min(100.0, self.hunger + hunger_increase)

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

        self.social = max(0.0, min(100.0, self.social - social_change))

        # Apply natural decay to skills if they haven't been practiced recently
        if (
            "train" not in self._daily_activities
            and "exercise" not in self._daily_activities
        ):
            self._apply_skill_decay("weapon_strength", 0.01 * time_delta)
            self._apply_skill_decay("power", 0.01 * time_delta)

        if (
            "manage" not in self._daily_activities
            and "study" not in self._daily_activities
        ):
            self._apply_skill_decay("management_skill", 0.01 * time_delta)

        if "socialize" not in self._daily_activities:
            self._apply_skill_decay("sociability", 0.005 * time_delta)

        # Reset daily activities list if it's getting too long
        if len(self._daily_activities) > 10:
            self._daily_activities = self._daily_activities[-5:]

    def _apply_skill_decay(self, skill: str, amount: float) -> None:
        """Apply natural decay to a skill with personalized resistance"""
        current_value = getattr(self, skill)

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
        setattr(self, skill, new_value)

    def _update_skill_caps(self) -> None:
        """Update skill caps based on rank, training/management sessions, age, and archetype"""
        rank_multipliers = {
            "private": 1.0,
            "corporal": 1.1,
            "sergeant": 1.2,
            "lieutenant": 1.3,
            "captain": 1.4,
        }

        # Get multiplier based on rank (default to private)
        rank_mult = rank_multipliers.get(self.rank.lower(), 1.0)

        # Age-based modifiers (older agents have higher potential but slower learning)
        age_factor = 1.0
        if self.age > 35:
            age_factor = 1.1  # More experienced
        elif self.age < 25:
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
        )  # New power cap

        # Personality influences caps more strongly
        conscientiousness = self.personality.get("conscientiousness", 0.5)
        openness = self.personality.get("openness", 0.5)
        agreeableness = self.personality.get("agreeableness", 0.5)
        extroversion = self.personality.get("extroversion", 0.5)

        # Enhanced personality effects with more variation
        personality_weapon_mod = (
            conscientiousness - 0.5
        ) * 0.4  # More pronounced effect
        personality_mgmt_mod = (openness - 0.5) * 0.4
        personality_social_mod = (extroversion - 0.5) * 0.4
        personality_power_mod = (conscientiousness - 0.5) * 0.4  # New power modifier

        # Update the skill caps with personality and archetype modifiers
        self._skill_caps["weapon_strength"] = min(
            100.0, weapon_cap * (1 + personality_weapon_mod)
        )
        self._skill_caps["management_skill"] = min(
            100.0, management_cap * (1 + personality_mgmt_mod)
        )
        self._skill_caps["social"] = min(
            100.0, 70.0 + (extroversion * 30.0) + personality_social_mod * 100.0
        )
        self._skill_caps["power"] = min(
            100.0, power_cap * (1 + personality_power_mod)
        )  # Update power cap

        # Sociability cap influenced by multiple personality traits
        personality_modifier = (
            (agreeableness + extroversion) / 2 - 0.5
        ) * 0.5  # Increased effect
        self._skill_caps["sociability"] = min(
            100.0, sociability_cap * (1 + personality_modifier)
        )

        # Apply archetype-specific hard caps (some archetypes have natural limits)
        # Archetype hard limits (prevents weapon specialists from becoming great managers, etc.)
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

    def apply_poi_effect(self, poi_effects: Dict[str, float]) -> None:
        """
        Apply effects from a POI visit

        Effects now have diminishing returns as values approach 1.0
        Archetype influences the effectiveness of skill improvements
        """
        # Track activity based on POI effects
        if "weapon_strength" in poi_effects and poi_effects["weapon_strength"] > 0:
            self._daily_activities.append("train")
            self._training_sessions += 1

        if "management_skill" in poi_effects and poi_effects["management_skill"] > 0:
            self._daily_activities.append("manage")
            self._management_sessions += 1

        if "sociability" in poi_effects and poi_effects["sociability"] > 0:
            self._daily_activities.append("socialize")
            self._social_sessions += 1

        # Track new activity types
        if (
            "energy" in poi_effects and poi_effects["energy"] > 0.3
        ):  # Significant energy gain indicates sleep/rest
            self._daily_activities.append("sleep")

        # Track fitness activities (high weapon_strength gain + energy loss)
        if (
            "weapon_strength" in poi_effects
            and poi_effects["weapon_strength"] > 0.1
            and "energy" in poi_effects
            and poi_effects["energy"] < 0
        ):
            self._daily_activities.append("exercise")

        # Track study/library activities (management_skill gain without high energy loss)
        if (
            "management_skill" in poi_effects
            and poi_effects["management_skill"] > 0.05
            and ("energy" not in poi_effects or poi_effects["energy"] > -0.05)
        ):
            self._daily_activities.append("study")

        # Update skill caps
        self._update_skill_caps()

        # Get archetype multipliers
        archetype_multipliers = ARCHETYPES.get(self.archetype, {}).get(
            "skill_multipliers", {}
        )

        # Apply effects with diminishing returns and archetype bonuses
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
            "power", poi_effects.get("power", 0), self._skill_caps["power"]
        )  # New power effect

        # Apply archetype bonuses to skill improvements
        weapon_effect = poi_effects.get(
            "weapon_strength", 0
        ) * archetype_multipliers.get("weapon_strength", 1.0)
        self._apply_effect_with_diminishing_returns(
            "weapon_strength",
            weapon_effect,
            self._skill_caps["weapon_strength"],
        )

        management_effect = poi_effects.get(
            "management_skill", 0
        ) * archetype_multipliers.get("management_skill", 1.0)
        self._apply_effect_with_diminishing_returns(
            "management_skill",
            management_effect,
            self._skill_caps["management_skill"],
        )

        sociability_effect = poi_effects.get(
            "sociability", 0
        ) * archetype_multipliers.get("sociability", 1.0)
        self._apply_effect_with_diminishing_returns(
            "sociability",
            sociability_effect,
            self._skill_caps["sociability"],
        )

        power_effect = poi_effects.get("power", 0) * archetype_multipliers.get(
            "power", 1.0
        )
        self._apply_effect_with_diminishing_returns(
            "power",
            power_effect,
            self._skill_caps["power"],
        )

    def _apply_effect_with_diminishing_returns(
        self, attribute: str, effect: float, cap: float = 100.0
    ) -> None:
        """
        Apply effect to an attribute with diminishing returns and personalized growth rates

        Args:
            attribute: The attribute to update
            effect: The effect value (positive or negative)
            cap: Maximum value this attribute can reach (default 1.0)
        """
        if effect == 0:
            return

        current = getattr(self, attribute)

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
                random_factor = random.uniform(0.6, 1.4)  # Increased variance
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
            "archetype": self.archetype,
            "energy": self.energy,
            "social": self.social,
            "hunger": self.hunger,
            "weapon_strength": self.weapon_strength,
            "management_skill": self.management_skill,
            "sociability": self.sociability,
            "power": self.power,
            "current_poi_id": self.current_poi_id,
            "location": self.location,
            "skill_caps": self._skill_caps,
        }
