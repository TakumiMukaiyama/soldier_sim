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
        management_skill: float = 0.3,
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

        # Current location (set when agent moves)
        self.current_poi_id = None
        self.location = [0.0, 0.0]

    def update_needs(self, time_delta: float = 1.0) -> None:
        """Update agent needs based on time passing"""
        # Energy decreases over time
        self.energy = max(0.0, self.energy - (0.05 * time_delta))

        # Hunger increases over time
        self.hunger = min(1.0, self.hunger + (0.08 * time_delta))

        # Social need increases for extroverted people
        social_change = 0.03 * self.personality["extroversion"] * time_delta
        self.social = max(0.0, min(1.0, self.social - social_change))

    def apply_poi_effect(self, poi_effects: Dict[str, float]) -> None:
        """Apply effects from a POI visit"""
        # Update state based on POI effects
        if "energy" in poi_effects:
            self.energy = max(0.0, min(1.0, self.energy + poi_effects["energy"]))

        if "hunger" in poi_effects:
            self.hunger = max(0.0, min(1.0, self.hunger + poi_effects["hunger"]))

        if "social" in poi_effects:
            self.social = max(0.0, min(1.0, self.social + poi_effects["social"]))

        if "weapon_strength" in poi_effects:
            self.weapon_strength = max(
                0.0, min(1.0, self.weapon_strength + poi_effects["weapon_strength"])
            )

        if "management_skill" in poi_effects:
            self.management_skill = max(
                0.0, min(1.0, self.management_skill + poi_effects["management_skill"])
            )

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
            "current_poi_id": self.current_poi_id,
            "location": self.location,
        }
