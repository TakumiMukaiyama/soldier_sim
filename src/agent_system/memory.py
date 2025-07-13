from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import polars as pl

from src.agent_system.agent import Agent
from src.agent_system.poi import POI


class Memory:
    """Memory system for agents including temporal, reflective, and spatial memory"""

    def __init__(self, buffer_size: int = 100):
        """Initialize empty memory dataframes
        
        Args:
            buffer_size: Size at which the buffer is automatically flushed
        """
        # Temporal memory - records of all agent actions and observations
        self.temporal_memory = pl.DataFrame(
            schema={
                "id": pl.Utf8,
                "time": pl.Utf8,
                "agent_id": pl.Utf8,
                "poi_id": pl.Utf8,
                "location": pl.List(pl.Float64),
                "activity_key": pl.Utf8,
                "observation": pl.Struct(
                    [
                        pl.Field("activity", pl.Utf8),
                        pl.Field("poi_id", pl.Utf8),
                        pl.Field("reason", pl.Utf8),
                        pl.Field("expected_duration", pl.Int64),
                    ]
                ),  # Nested structure for observation details
                "current_energy": pl.Float64,
                "current_social": pl.Float64,
                "current_weapon_strength": pl.Float64,
                "current_hunger": pl.Float64,
                "current_management_skill": pl.Float64,
                "current_sociability": pl.Float64,
                "current_power": pl.Float64,
            }
        )

        # Initialize temporal memory buffer for batch processing
        self._temporal_memory_buffer: List[Dict[str, Any]] = []
        self._buffer_size = buffer_size

        # Reflective memory - daily summaries by agent
        self.reflective_memory = {}  # agent_id -> date -> summary_dict

    def _append_to_buffer(self, record: Dict[str, Any]) -> None:
        """Append a record to the buffer and flush if needed
        
        Args:
            record: The record to append
        """
        self._temporal_memory_buffer.append(record)
        
        # Auto-flush if buffer gets too large
        if len(self._temporal_memory_buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush the buffer to the temporal memory DataFrame"""
        if not self._temporal_memory_buffer:
            return

        # Create DataFrame from buffer and concat with existing memory
        buffer_df = pl.DataFrame(self._temporal_memory_buffer)
        self.temporal_memory = pl.concat([self.temporal_memory, buffer_df])

        # Clear buffer
        self._temporal_memory_buffer = []

    def record_action(
        self,
        agent: Agent,
        time: datetime,
        poi: Optional[POI] = None,
        activity_key: str = "idle",
        observation: Dict[str, Any] = None,
    ) -> str:
        """
        Record an agent action to temporal memory

        Args:
            agent: Agent performing the action
            time: Timestamp of the action
            poi: POI where action occurred (or None)
            activity_key: Type of activity (e.g., "train", "eat", "rest")
            observation: Observation data from this action

        Returns:
            ID of the created memory node
        """
        node_id = str(uuid4())
        time_str = time.isoformat()
        poi_id = poi.id if poi else None
        location = poi.location if poi else agent.location

        # Ensure observation has required fields with default values
        if not observation:
            observation = {}
        observation = {
            "activity": observation.get("activity", activity_key),
            "poi_id": observation.get("poi_id", poi_id or ""),
            "reason": observation.get("reason", ""),
            "expected_duration": observation.get("expected_duration", 1),
        }

        # Create new record
        new_record = {
            "id": node_id,
            "time": time_str,
            "agent_id": agent.id,
            "poi_id": poi_id or "",
            "location": location,
            "activity_key": activity_key,
            "observation": observation,
            "current_energy": agent.energy,
            "current_social": agent.social,
            "current_weapon_strength": agent.weapon_strength,
            "current_hunger": agent.hunger,
            "current_management_skill": agent.management_skill,
            "current_sociability": agent.sociability,
            "current_power": agent.power,
        }

        # Append to temporal memory buffer
        self._append_to_buffer(new_record)

        return node_id

    def _calculate_statistics(self, daily_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate statistics from daily activities
        
        Args:
            daily_df: DataFrame with daily activities for an agent
            
        Returns:
            Dictionary with calculated statistics
        """
        return {
            "activity_count": len(daily_df),
            "activity_distribution": self._get_activity_distribution(daily_df),
            "most_visited_poi": self._get_most_visited_poi(daily_df),
            "energy_stats": {
                "start": daily_df["current_energy"].first(),
                "end": daily_df["current_energy"].last(),
                "avg": daily_df["current_energy"].mean(),
                "min": daily_df["current_energy"].min(),
                "max": daily_df["current_energy"].max(),
            },
            "hunger_stats": {
                "start": daily_df["current_hunger"].first(),
                "end": daily_df["current_hunger"].last(),
                "avg": daily_df["current_hunger"].mean(),
            },
            "weapon_strength_change": daily_df["current_weapon_strength"].last()
            - daily_df["current_weapon_strength"].first(),
            "management_skill_change": daily_df["current_management_skill"].last()
            - daily_df["current_management_skill"].first(),
            "power_change": daily_df["current_power"].last()
            - daily_df["current_power"].first(),
        }
    
    def _get_activity_distribution(self, df: pl.DataFrame) -> Dict:
        """Get distribution of activities
        
        Args:
            df: DataFrame with activities
            
        Returns:
            Dictionary with activity distribution
        """
        return df.group_by("activity_key").len().sort("len", descending=True).to_dict()
    
    def _get_most_visited_poi(self, df: pl.DataFrame) -> Dict:
        """Get most visited POI
        
        Args:
            df: DataFrame with POI visits
            
        Returns:
            Dictionary with most visited POI
        """
        return (
            df.group_by("poi_id")
            .len()
            .sort("len", descending=True)
            .head(1)
            .to_dict()
        )
    
    def generate_daily_summary(self, agent_id: str, date: str) -> Dict[str, Any]:
        """
        Generate a reflective summary for an agent for a specific day

        Args:
            agent_id: ID of the agent
            date: Date string in YYYY-MM-DD format

        Returns:
            Summary dictionary for reflective memory
        """
        # Flush buffer to ensure all records are in the DataFrame
        self._flush_buffer()

        # Filter memory for this agent on this date
        daily_df = self.temporal_memory.filter(
            (pl.col("agent_id") == agent_id) & (pl.col("time").str.contains(date))
        )

        if len(daily_df) == 0:
            return {"no_activity": True}

        # Generate summary statistics
        summary = {"date": date}
        summary.update(self._calculate_statistics(daily_df))

        # Store in reflective memory
        if agent_id not in self.reflective_memory:
            self.reflective_memory[agent_id] = {}

        self.reflective_memory[agent_id][date] = summary
        return summary

    def get_recent_memories(self, agent_id: str, limit: int = 10) -> pl.DataFrame:
        """Get recent temporal memories for an agent"""
        # Flush buffer to ensure all records are in the DataFrame
        self._flush_buffer()

        return (
            self.temporal_memory.filter(pl.col("agent_id") == agent_id)
            .sort("time", descending=True)
            .head(limit)
        )

    def get_reflective_memory(self, agent_id: str, date: Optional[str] = None) -> Dict:
        """Get reflective memory for an agent, optionally for a specific date"""
        if agent_id not in self.reflective_memory:
            return {}

        if date:
            return self.reflective_memory[agent_id].get(date, {})

        return self.reflective_memory[agent_id]
