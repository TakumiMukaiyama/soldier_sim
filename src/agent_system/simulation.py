from datetime import datetime, timedelta
from typing import Dict, Any


from .agent import Agent
from .memory import Memory
from .poi import POI


class Simulation:
    """Main simulation class for the multi-agent system"""

    def __init__(
        self,
        days: int = 100,
        time_steps_per_day: int = 8,
        planner=None,
    ):
        self.days = days
        self.time_steps_per_day = time_steps_per_day
        self.current_day = 0
        self.current_step = 0
        self.current_time = datetime.now().replace(
            hour=6, minute=0, second=0, microsecond=0
        )  # Start at 6am

        # Collections
        self.agents: Dict[str, Agent] = {}
        self.pois: Dict[str, POI] = {}

        # Memory and logs
        self.memory = Memory()
        self.logs = []

        # Optional LLM planner
        self.planner = planner

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the simulation"""
        self.agents[agent.id] = agent

    def add_poi(self, poi: POI) -> None:
        """Add a POI to the simulation"""
        self.pois[poi.id] = poi

    def step(self) -> Dict[str, Any]:
        """Run a single time step of the simulation"""
        step_stats = {
            "day": self.current_day,
            "step": self.current_step,
            "time": self.current_time.isoformat(),
            "agent_actions": [],
        }

        # For each agent, update needs and decide/execute action
        for agent_id, agent in self.agents.items():
            # Update agent needs based on time passing
            agent.update_needs(time_delta=1.0)

            # Choose action for this agent
            action = self._choose_action(agent)

            # Execute action
            self._execute_action(agent, action)

            # Record action
            step_stats["agent_actions"].append(
                {
                    "agent_id": agent_id,
                    "action": action,
                }
            )

        # Update time and step
        self.current_step += 1
        self.current_time += timedelta(hours=24 / self.time_steps_per_day)

        # Check if day is complete
        if self.current_step >= self.time_steps_per_day:
            self._end_day()

        self.logs.append(step_stats)
        return step_stats

    def _choose_action(self, agent: Agent) -> Dict[str, Any]:
        """Choose next action for an agent"""
        if self.planner:
            # Use LLM planner if available
            recent_memories = self.memory.get_recent_memories(agent.id, limit=10)
            reflective_memory = self.memory.get_reflective_memory(agent.id)

            # Convert POIs to dict for planner
            poi_list = [poi.to_dict() for poi in self.pois.values()]

            # Get plan from LLM
            plan = self.planner.plan_action(
                agent_state=agent.to_dict(),
                reflective_memory=reflective_memory,
                poi_list=poi_list,
            )

            return {
                "poi_id": plan.chosen_poi,
                "activity": plan.activity,
                "expected_duration": plan.expected_duration,
                "reason": plan.reason,
            }

        # Simple rule-based backup logic if no planner
        if agent.hunger > 0.7:
            # Find food POI
            food_pois = [poi for poi in self.pois.values() if poi.category == "food"]
            if food_pois:
                chosen_poi = food_pois[0]  # Just pick first one for now
                return {
                    "poi_id": chosen_poi.id,
                    "activity": "eat",
                    "expected_duration": 1,
                    "reason": "Hungry, need food",
                }

        if agent.energy < 0.3:
            # Find rest POI
            rest_pois = [poi for poi in self.pois.values() if poi.category == "rest"]
            if rest_pois:
                chosen_poi = rest_pois[0]
                return {
                    "poi_id": chosen_poi.id,
                    "activity": "rest",
                    "expected_duration": 2,
                    "reason": "Low energy, need rest",
                }

        # Default: training
        training_pois = [
            poi for poi in self.pois.values() if poi.category == "training"
        ]
        if training_pois:
            chosen_poi = training_pois[0]
            return {
                "poi_id": chosen_poi.id,
                "activity": "train",
                "expected_duration": 2,
                "reason": "Default training activity",
            }

        # Fallback if no matching POIs
        return {
            "poi_id": None,
            "activity": "idle",
            "expected_duration": 1,
            "reason": "No suitable POI found",
        }

    def _execute_action(self, agent: Agent, action: Dict[str, Any]) -> None:
        """Execute an action for an agent"""
        poi_id = action.get("poi_id")
        activity = action.get("activity", "idle")

        # Get POI if specified
        poi = self.pois.get(poi_id) if poi_id else None

        if poi:
            # Move agent to POI
            agent.move_to(poi_id, poi.location)

            # Apply POI effects to agent
            effects = poi.get_effects()
            agent.apply_poi_effect(effects)

            # Agent's observation of POI
            observation = {
                "satisfaction": effects.get("energy", 0) * 0.5
                + effects.get("hunger", 0) * 0.5,
                "price": 0.5,  # Placeholder
                "convenience": 0.5,  # Placeholder
                "atmosphere": 0.5,  # Placeholder
            }

            # Update POI belief based on observation
            poi.update_belief(observation)

        # Record action to memory
        observation_data = {
            "activity": activity,
            "poi_id": poi_id,
            "reason": action.get("reason", ""),
            "expected_duration": action.get("expected_duration", 1),
        }

        self.memory.record_action(
            agent=agent,
            time=self.current_time,
            poi=poi,
            activity_key=activity,
            observation=observation_data,
        )

    def _end_day(self) -> None:
        """Process end-of-day activities"""
        self.current_day += 1
        self.current_step = 0

        # Reset time to 6am next day
        self.current_time = self.current_time.replace(
            hour=6, minute=0, second=0, microsecond=0
        )
        self.current_time += timedelta(days=1)

        # Generate reflective memories for all agents
        date_str = self.current_time.strftime("%Y-%m-%d")
        for agent_id in self.agents:
            self.memory.generate_daily_summary(agent_id, date_str)

    def run(self) -> Dict[str, Any]:
        """Run full simulation for specified number of days"""
        results = {
            "days_completed": 0,
            "steps_completed": 0,
        }

        while self.current_day < self.days:
            self.step()
            results["steps_completed"] += 1

            if self.current_step == 0:
                results["days_completed"] += 1
                print(f"Day {self.current_day} of {self.days} completed")

        # Prepare results
        results["logs"] = self.logs
        results["temporal_memory"] = self.memory.temporal_memory
        results["reflective_memory"] = self.memory.reflective_memory

        return results
