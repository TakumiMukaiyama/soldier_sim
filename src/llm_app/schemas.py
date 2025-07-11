from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Schema for agent state information used in planning"""

    id: str = Field(..., description="Agent's unique ID")
    age: int = Field(..., description="Agent's age")
    rank: str = Field(..., description="Military rank")

    # Current state parameters (0.0-1.0)
    energy: float = Field(..., ge=0.0, le=1.0, description="Current energy level (1.0=full energy, 0.0=exhausted)")
    social: float = Field(..., ge=0.0, le=1.0, description="Social need level (1.0=satisfied, 0.0=lonely/isolated)")
    hunger: float = Field(..., ge=0.0, le=1.0, description="Hunger level (0.0=well-fed, 1.0=very hungry)")

    # Skill parameters (0.0-1.0)
    weapon_strength: float = Field(..., ge=0.0, le=1.0, description="Weapon handling and combat skill")
    management_skill: float = Field(..., ge=0.0, le=1.0, description="Leadership and management capability")
    sociability: float = Field(..., ge=0.0, le=1.0, description="Communication skills and social competence")

    # Personality traits (0.0-1.0)
    personality: Dict[str, float] = Field(..., description="Big Five personality traits")

    # Current location
    current_poi_id: str = Field(None, description="Current POI location")
    location: List[float] = Field(..., description="Current coordinates [x, y]")


class PlanOutput(BaseModel):
    """Schema for LLM agent planning output"""

    agent_id: str = Field(..., description="Agent's unique ID")
    chosen_poi: str = Field(..., description="ID of the POI to visit next")
    activity: Literal["train", "eat", "rest", "manage", "arm", "socialize"] = Field(
        ..., description="Activity to perform at the POI"
    )
    expected_duration: int = Field(..., ge=1, le=8, description="Expected duration of the activity in hours")
    reason: str = Field(..., description="Brief reason for choosing this POI and activity")


class ReflectionOutput(BaseModel):
    """Schema for daily agent reflection output"""

    agent_id: str = Field(..., description="Agent's unique ID")
    date: str = Field(..., description="Date of reflection (YYYY-MM-DD)")
    energy_assessment: str = Field(..., description="Assessment of energy levels throughout the day")
    main_activities: List[str] = Field(..., description="Main activities performed during the day")
    social_interactions: str = Field(..., description="Assessment of social interactions")
    skill_progress: str = Field(..., description="Progress on skills during the day")
    sociability_development: str = Field(..., description="Assessment of communication and social skill development")
    mood: str = Field(..., description="Overall mood assessment")
    goals_for_tomorrow: List[str] = Field(..., description="Goals for the next day")
