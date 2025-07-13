import random
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.agent_system.agent import Agent
from src.agent_system.memory import Memory
from src.agent_system.poi import POI
from src.utils.get_logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Random event definitions
RANDOM_EVENTS = {
    "injury": {
        "description": "Agent suffers minor injury during training",
        "probability": 0.02,  # 2% chance per step
        "duration": 2,  # lasts 2 time steps
        "forced_poi_category": "medical",
        "effects": {"energy": -15.0, "weapon_strength": -5.0, "power": -8.0},
        "required_conditions": ["recent_training"],  # Only happens after training
    },
    "illness": {
        "description": "Agent gets sick and needs medical attention",
        "probability": 0.015,  # 1.5% chance per step
        "duration": 3,  # lasts 3 time steps
        "forced_poi_category": "medical",
        "effects": {"energy": -20.0, "social": -10.0, "power": -5.0},
        "required_conditions": [],  # Can happen anytime
    },
    "special_training": {
        "description": "Agent selected for special training exercise",
        "probability": 0.03,  # 3% chance per step
        "duration": 1,  # lasts 1 time step
        "forced_poi_category": "outdoor",
        "effects": {"weapon_strength": 15.0, "management_skill": 8.0, "power": 20.0},
        "required_conditions": [],
    },
    "equipment_maintenance": {
        "description": "Agent assigned to equipment maintenance duty",
        "probability": 0.025,  # 2.5% chance per step
        "duration": 1,
        "forced_poi_category": "workshop",
        "effects": {"weapon_strength": 10.0, "management_skill": 5.0, "power": 8.0},
        "required_conditions": [],
    },
    "communication_duty": {
        "description": "Agent assigned to communication center duty",
        "probability": 0.02,
        "duration": 1,
        "forced_poi_category": "communications",
        "effects": {"management_skill": 12.0, "sociability": 10.0},
        "required_conditions": [],
    },
    "stress_fatigue": {
        "description": "Agent experiences stress and fatigue",
        "probability": 0.025,
        "duration": 2,
        "forced_poi_category": "spiritual",
        "effects": {"energy": -10.0, "social": -5.0, "power": -3.0},
        "required_conditions": [
            "high_activity"
        ],  # Only when agent has been very active
    },
}


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

        # Random events tracking
        self.active_events: Dict[str, Dict] = {}  # agent_id -> event_data

    def _check_event_conditions(self, agent: Agent, event_data: Dict) -> bool:
        """Check if agent meets conditions for a specific event"""
        conditions = event_data.get("required_conditions", [])

        for condition in conditions:
            if condition == "recent_training":
                # Check if agent did training in last few activities
                if not any(
                    activity in ["train", "exercise", "outdoor_train"]
                    for activity in agent._daily_activities[-3:]
                ):
                    return False
            elif condition == "high_activity":
                # Check if agent has been very active
                if len(agent._daily_activities) < 5:
                    return False
            # Add more conditions as needed

        return True

    def _trigger_random_events(self) -> None:
        """Check and trigger random events for agents"""
        for agent_id, agent in self.agents.items():
            # Skip if agent already has an active event
            if agent_id in self.active_events:
                continue

            # Check each possible event
            for event_type, event_data in RANDOM_EVENTS.items():
                # Check probability
                if random.random() > event_data["probability"]:
                    continue

                # Check conditions
                if not self._check_event_conditions(agent, event_data):
                    continue

                # Trigger event
                self.active_events[agent_id] = {
                    "type": event_type,
                    "description": event_data["description"],
                    "remaining_duration": event_data["duration"],
                    "forced_poi_category": event_data["forced_poi_category"],
                    "effects": event_data["effects"],
                }

                # Apply immediate effects
                for effect, value in event_data["effects"].items():
                    if hasattr(agent, effect):
                        current_value = getattr(agent, effect)
                        new_value = max(0.0, min(100.0, current_value + value))
                        setattr(agent, effect, new_value)

                logger.info(
                    f"Event triggered for {agent_id}: {event_data['description']}"
                )
                break  # Only one event per agent per step

    def _update_active_events(self) -> None:
        """Update duration of active events and remove expired ones"""
        expired_events = []

        for agent_id, event_data in self.active_events.items():
            event_data["remaining_duration"] -= 1
            if event_data["remaining_duration"] <= 0:
                expired_events.append(agent_id)
                logger.info(f"Event ended for {agent_id}: {event_data['description']}")

        # Remove expired events
        for agent_id in expired_events:
            del self.active_events[agent_id]

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

        # Trigger random events for the current step
        self._trigger_random_events()

        # Update active events
        self._update_active_events()

        # For each agent, update needs and decide/execute action
        for agent_id, agent in self.agents.items():
            # Update agent needs based on time passing and current time
            agent.update_needs(time_delta=1.0, current_hour=self.current_time.hour)

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

    def _handle_active_event(self, agent: Agent) -> Optional[Dict[str, Any]]:
        """Handle agent actions when under an active event"""
        if agent.id not in self.active_events:
            return None

        event_data = self.active_events[agent.id]
        forced_category = event_data["forced_poi_category"]

        # Find POIs of the required category
        forced_pois = [
            poi for poi in self.pois.values() if poi.category == forced_category
        ]
        if not forced_pois:
            return None

        chosen_poi = random.choice(forced_pois)  # Random selection from available POIs
        activity_map = {
            "medical": "heal",
            "outdoor": "outdoor_train",
            "workshop": "craft",
            "communications": "communicate",
            "spiritual": "reflect",
        }
        activity = activity_map.get(forced_category, "idle")

        return {
            "poi_id": chosen_poi.id,
            "activity": activity,
            "expected_duration": 2,
            "reason": f"Event: {event_data['description']}",
        }

    def _handle_time_based_needs(
        self, agent: Agent, current_hour: int
    ) -> Optional[Dict[str, Any]]:
        """Handle time-based needs such as sleep and meals"""
        # Time-based behavior priorities
        is_night_time = current_hour >= 22 or current_hour <= 5  # 10pm to 5am
        is_meal_time = current_hour in [12, 18]  # 12pm and 6pm

        # SLEEP PRIORITY: Force sleep during night hours if energy is low
        if is_night_time and agent.energy < 60.0:
            rest_pois = [
                poi for poi in self.pois.values() if poi.category in ["rest", "sleep"]
            ]
            if rest_pois:
                # Prefer sleep POIs if available, otherwise use rest POIs
                sleep_pois = [poi for poi in rest_pois if poi.category == "sleep"]
                chosen_poi = sleep_pois[0] if sleep_pois else rest_pois[0]
                return {
                    "poi_id": chosen_poi.id,
                    "activity": "sleep",
                    "expected_duration": 3,
                    "reason": "Night time sleep schedule",
                }

        # MEAL PRIORITY: Strong preference for eating during meal times
        if is_meal_time or agent.hunger > 60.0:
            food_pois = [poi for poi in self.pois.values() if poi.category == "food"]
            if food_pois:
                # Apply archetype preferences to food POI selection
                food_poi_scores = []
                for poi in food_pois:
                    base_score = 1.0
                    # Boost score for meal times
                    if is_meal_time:
                        base_score *= 2.0
                    # Apply archetype preferences
                    archetype_multiplier = agent.get_poi_preference_multiplier(
                        poi.category
                    )
                    score = base_score * archetype_multiplier
                    food_poi_scores.append((poi, score))

                # Sort by score and pick the best
                food_poi_scores.sort(key=lambda x: x[1], reverse=True)
                chosen_poi = food_poi_scores[0][0]

                return {
                    "poi_id": chosen_poi.id,
                    "activity": "eat",
                    "expected_duration": 1,
                    "reason": "Meal time or high hunger",
                }

        return None

    def _handle_critical_needs(
        self, agent: Agent, current_hour: int
    ) -> Optional[Dict[str, Any]]:
        """Handle critical needs like extreme hunger or fatigue"""
        is_night_time = current_hour >= 22 or current_hour <= 5  # 10pm to 5am

        # Critical hunger
        if agent.hunger > 70.0:
            food_pois = [poi for poi in self.pois.values() if poi.category == "food"]
            if food_pois:
                chosen_poi = food_pois[0]
                return {
                    "poi_id": chosen_poi.id,
                    "activity": "eat",
                    "expected_duration": 1,
                    "reason": "Critical hunger level",
                }

        # Critical fatigue
        if agent.energy < 30.0:
            rest_pois = [
                poi for poi in self.pois.values() if poi.category in ["rest", "sleep"]
            ]
            if rest_pois:
                chosen_poi = rest_pois[0]
                activity = "sleep" if is_night_time else "rest"
                return {
                    "poi_id": chosen_poi.id,
                    "activity": activity,
                    "expected_duration": 2 if is_night_time else 1,
                    "reason": "Critical energy level",
                }

        return None

    def _choose_archetype_based_action(
        self, agent: Agent, current_hour: int
    ) -> Dict[str, Any]:
        """Choose action based on agent's archetype and time of day"""
        is_night_time = current_hour >= 22 or current_hour <= 5  # 10pm to 5am
        is_early_morning = current_hour in [6, 7, 8]  # 6am to 8am

        # Archetype-based POI selection for normal activities
        available_pois = list(self.pois.values())
        poi_scores = []

        for poi in available_pois:
            # Base score
            base_score = 1.0

            # Time-based modifiers
            if is_night_time and poi.category not in ["rest", "sleep"]:
                base_score *= 0.1  # Strong penalty for non-rest activities at night
            elif is_early_morning and poi.category == "training":
                base_score *= 1.5  # Morning training bonus
            elif poi.category == "recreation" and current_hour in [19, 20, 21]:
                base_score *= 1.3  # Evening recreation bonus

            # Apply archetype preferences
            archetype_multiplier = agent.get_poi_preference_multiplier(poi.category)
            score = base_score * archetype_multiplier
            poi_scores.append((poi, score))

        # Sort by score and pick the best
        poi_scores.sort(key=lambda x: x[1], reverse=True)
        if poi_scores:
            chosen_poi = poi_scores[0][0]

            # Determine activity based on POI category
            activity_map = {
                "training": "train",
                "armory": "arm",
                "office": "manage",
                "food": "eat",
                "rest": "rest",
                "sleep": "sleep",
                "recreation": "socialize",
                "medical": "heal",
                "fitness": "exercise",
                "library": "study",
                "workshop": "craft",
                "communications": "communicate",
                "maintenance": "maintain",
                "outdoor": "outdoor_train",
                "spiritual": "reflect",
                "logistics": "organize",
            }
            activity = activity_map.get(chosen_poi.category, "idle")

            return {
                "poi_id": chosen_poi.id,
                "activity": activity,
                "expected_duration": 2,
                "reason": f"Archetype-based choice for {agent.archetype} at {current_hour}:00",
            }

        # Fallback if no POIs available
        return {
            "poi_id": None,
            "activity": "idle",
            "expected_duration": 1,
            "reason": "No suitable POI found",
        }

    def _choose_action(self, agent: Agent) -> Dict[str, Any]:
        """Choose next action for an agent based on time of day, needs, and active events"""
        current_hour = self.current_time.hour

        # Check for active events first - these override normal behavior
        event_action = self._handle_active_event(agent)
        if event_action:
            return event_action

        # Handle time-based needs
        time_based_action = self._handle_time_based_needs(agent, current_hour)
        if time_based_action:
            return time_based_action

        # Handle critical needs
        critical_action = self._handle_critical_needs(agent, current_hour)
        if critical_action:
            return critical_action

        # Use LLM planner if available (for complex decisions)
        if (
            self.planner and current_hour > 5 and current_hour < 22
        ):  # Don't use LLM at night
            try:
                reflective_memory = self.memory.get_reflective_memory(agent.id)

                # Convert POIs to dict for planner, include time context
                poi_list = [poi.to_dict() for poi in self.pois.values()]

                # Add time context to agent state for LLM
                agent_state = agent.to_dict()
                agent_state["current_hour"] = current_hour
                agent_state["time_context"] = {
                    "is_night_time": current_hour >= 22 or current_hour <= 5,
                    "is_meal_time": current_hour in [12, 18],
                    "is_early_morning": current_hour in [6, 7, 8],
                }
                agent_state["active_event"] = self.active_events.get(agent.id, None)

                # Get plan from LLM
                plan = self.planner.plan_action(
                    agent_state=agent_state,
                    reflective_memory=reflective_memory,
                    poi_list=poi_list,
                )

                return {
                    "poi_id": plan.chosen_poi,
                    "activity": plan.activity,
                    "expected_duration": plan.expected_duration,
                    "reason": plan.reason,
                }
            except Exception as e:
                logger.warning(
                    f"LLM planner failed: {e}, falling back to rule-based planning"
                )

        # RULE-BASED BACKUP LOGIC: Use archetype-based selection
        return self._choose_archetype_based_action(agent, current_hour)

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

        # Get summary date before advancing to next day
        summary_date_str = self.current_time.strftime("%Y-%m-%d")

        # Reset time to 6am next day
        self.current_time = self.current_time.replace(
            hour=6, minute=0, second=0, microsecond=0
        )
        self.current_time += timedelta(days=1)

        # Generate reflective memories for all agents using the stored date
        for agent_id in self.agents:
            self.memory.generate_daily_summary(agent_id, summary_date_str)

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
                logger.info(f"Day {self.current_day} of {self.days} completed")

        # Prepare results
        results["logs"] = self.logs
        results["temporal_memory"] = self.memory.temporal_memory
        results["reflective_memory"] = self.memory.reflective_memory

        return results
