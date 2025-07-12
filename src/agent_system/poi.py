from typing import Dict, List, Optional
from uuid import uuid4

from .kalman import kalman_update


class POI:
    """Point of Interest with belief vector updated via Kalman filter"""

    def __init__(
        self,
        poi_id: Optional[str] = None,
        name: str = "",
        category: str = "generic",
        location: List[float] = None,
        belief: Dict[str, float] = None,
        belief_sigma: Dict[str, float] = None,
        effects: Dict[str, float] = None,
    ):
        self.id = poi_id or str(uuid4())
        self.name = name
        self.category = category
        self.location = location or [0.0, 0.0]

        # Belief vector with uncertainty
        self.belief = belief or {
            "satisfaction": 0.5,
            "price": 0.5,
            "convenience": 0.5,
            "atmosphere": 0.5,
        }

        # Sigma (uncertainty) values for each belief dimension
        self.belief_sigma = belief_sigma or {
            "satisfaction": 0.3,
            "price": 0.3,
            "convenience": 0.3,
            "atmosphere": 0.3,
        }

        # Effects on agent states when visited
        self.effects = effects or {
            "energy": 0.0,
            "hunger": 0.0,
            "social": 0.0,
            "weapon_strength": 0.0,
            "management_skill": 0.0,
            "power": 0.0,
        }

    def update_belief(
        self, observation: Dict[str, float], sigma_o: float = 0.2
    ) -> None:
        """Update belief using Kalman filter based on observation"""
        for key in observation:
            if key in self.belief and key in self.belief_sigma:
                self.belief[key], self.belief_sigma[key] = kalman_update(
                    belief=self.belief[key],
                    sigma_b=self.belief_sigma[key],
                    observation=observation[key],
                    sigma_o=sigma_o,
                )

    def get_effects(self) -> Dict[str, float]:
        """Get effects this POI applies to agents"""
        return self.effects

    def to_dict(self) -> Dict:
        """Convert POI to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "location": self.location,
            "belief": self.belief,
            "belief_sigma": self.belief_sigma,
            "effects": self.effects,
        }
