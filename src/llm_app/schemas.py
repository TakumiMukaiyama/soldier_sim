from pydantic import BaseModel, Field
from typing import Literal, List


class PlanOutput(BaseModel):
    """Schema for LLM agent planning output"""

    agent_id: str = Field(..., description="Agent's unique ID")
    chosen_poi: str = Field(..., description="ID of the POI to visit next")
    activity: Literal["train", "eat", "rest", "manage", "arm"] = Field(
        ..., description="Activity to perform at the POI"
    )
    expected_duration: int = Field(
        ..., ge=1, le=8, description="Expected duration of the activity in hours"
    )
    reason: str = Field(
        ..., description="Brief reason for choosing this POI and activity"
    )


class ReflectionOutput(BaseModel):
    """Schema for daily agent reflection output"""

    agent_id: str = Field(..., description="Agent's unique ID")
    date: str = Field(..., description="Date of reflection (YYYY-MM-DD)")
    energy_assessment: str = Field(
        ..., description="Assessment of energy levels throughout the day"
    )
    main_activities: List[str] = Field(
        ..., description="Main activities performed during the day"
    )
    social_interactions: str = Field(
        ..., description="Assessment of social interactions"
    )
    skill_progress: str = Field(..., description="Progress on skills during the day")
    mood: str = Field(..., description="Overall mood assessment")
    goals_for_tomorrow: List[str] = Field(..., description="Goals for the next day")
