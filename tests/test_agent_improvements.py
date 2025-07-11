import random

from src.agent_system.agent import Agent
from src.agent_system.poi import POI


def test_agent_skill_caps():
    """Test that agent skill caps are properly calculated and applied"""
    # Set random seed for predictable tests
    random.seed(42)

    # Create agent with different ranks
    private = Agent(agent_id="private_agent", rank="private")
    sergeant = Agent(agent_id="sergeant_agent", rank="sergeant")
    captain = Agent(agent_id="captain_agent", rank="captain")

    # Higher ranks should have higher initial caps
    assert (
        captain._skill_caps["weapon_strength"] > sergeant._skill_caps["weapon_strength"]
    )
    assert (
        sergeant._skill_caps["weapon_strength"] > private._skill_caps["weapon_strength"]
    )

    # Initial caps should be based on rank
    assert private._skill_caps["weapon_strength"] == 0.7  # Base cap for private

    # Simulate training sessions for private
    weapon_effects = {"weapon_strength": 0.1}
    # Apply multiple training sessions
    for _ in range(5):
        private.apply_poi_effect(weapon_effects)

    # Cap should increase with training
    assert private._skill_caps["weapon_strength"] > 0.7
    assert private._training_sessions == 5


def test_diminishing_returns():
    """Test that effects have diminishing returns"""
    # Set random seed for predictable tests
    random.seed(42)

    agent = Agent(weapon_strength=0.3)  # Start low
    poi_effect = {"weapon_strength": 0.2}  # Significant effect

    # First application should have full effect
    agent.apply_poi_effect(poi_effect)
    first_value = agent.weapon_strength

    # Move close to the cap
    agent.weapon_strength = agent._skill_caps["weapon_strength"] - 0.05
    high_value = agent.weapon_strength

    # Second application should have diminishing returns
    agent.apply_poi_effect(poi_effect)
    second_value = agent.weapon_strength

    # The gain from the second application should be less
    first_gain = first_value - 0.3  # From 0.3 to first_value
    second_gain = second_value - high_value  # From high_value to second_value

    assert second_gain < first_gain
    assert second_gain > 0  # But still some gain


def test_skill_decay():
    """Test that skills decay when not practiced"""
    # Set random seed for predictable tests
    random.seed(42)

    agent = Agent(weapon_strength=0.7)  # Start high

    # Call update_needs which should apply decay
    agent.update_needs(time_delta=1.0)

    # Skill should decay
    assert agent.weapon_strength < 0.7

    # Apply training to reset decay
    agent.apply_poi_effect({"weapon_strength": 0.1})
    initial_value = agent.weapon_strength

    # Ensure "train" is in daily activities to prevent decay
    assert "train" in agent._daily_activities

    # Update needs should not decay the skill now
    agent.update_needs(time_delta=1.0)

    # Skill should not decay much if at all
    assert agent.weapon_strength >= initial_value - 0.01


def test_personality_effects():
    """Test that personality affects skill development"""
    # Create two agents with different personalities
    extrovert = Agent(
        personality={"extroversion": 0.9, "conscientiousness": 0.5, "openness": 0.5}
    )
    introvert = Agent(
        personality={"extroversion": 0.1, "conscientiousness": 0.5, "openness": 0.5}
    )

    # Social loss should be higher for extroverts when alone
    extrovert_social = extrovert.social
    introvert_social = introvert.social

    extrovert.update_needs()
    introvert.update_needs()

    extrovert_loss = extrovert_social - extrovert.social
    introvert_loss = introvert_social - introvert.social

    assert extrovert_loss > introvert_loss


def test_randomization():
    """Test that randomization creates variations in changes"""
    # Set random seed for predictable tests
    random.seed(42)

    # Create multiple agents with same properties
    agents = [Agent(energy=1.0, hunger=0.0) for _ in range(5)]

    # Update needs for all
    for agent in agents:
        agent.update_needs()

    # Check that energy and hunger values vary
    energy_values = [agent.energy for agent in agents]
    hunger_values = [agent.hunger for agent in agents]

    # Not all values should be identical
    assert len(set(energy_values)) > 1
    assert len(set(hunger_values)) > 1


def test_new_pois():
    """Test that new POIs have expected effects"""
    agent = Agent()
    initial_energy = agent.energy
    initial_social = agent.social

    # Test recreation POI
    recreation = POI(
        category="recreation",
        effects={
            "energy": 0.05,
            "hunger": 0.03,
            "social": 0.25,
            "weapon_strength": -0.01,
            "management_skill": 0.03,
        },
    )

    agent.apply_poi_effect(recreation.get_effects())

    # Social should increase significantly
    assert agent.social > initial_social

    # Energy should increase slightly
    assert agent.energy > initial_energy
