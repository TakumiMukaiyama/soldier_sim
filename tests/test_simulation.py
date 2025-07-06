import pytest
from datetime import datetime


from src.agent_system.agent import Agent
from src.agent_system.poi import POI
from src.agent_system.kalman import kalman_update
from src.agent_system.memory import Memory
from src.agent_system.simulation import Simulation


def test_agent_init():
    """Test agent initialization with default and custom values"""
    # Test with defaults
    agent = Agent()
    assert agent.id is not None
    assert agent.energy == 1.0
    assert agent.hunger == 0.0

    # Test with custom values
    agent = Agent(
        agent_id="test_agent", age=25, rank="sergeant", energy=0.8, hunger=0.2
    )
    assert agent.id == "test_agent"
    assert agent.age == 25
    assert agent.rank == "sergeant"
    assert agent.energy == 0.8
    assert agent.hunger == 0.2


def test_agent_update_needs():
    """Test agent needs update over time"""
    agent = Agent(energy=1.0, hunger=0.0)

    # Update with default time delta (1.0)
    agent.update_needs()
    assert agent.energy < 1.0
    assert agent.hunger > 0.0

    # Update with custom time delta
    initial_energy = agent.energy
    initial_hunger = agent.hunger
    agent.update_needs(time_delta=2.0)
    assert agent.energy < initial_energy
    assert agent.hunger > initial_hunger


def test_poi_init():
    """Test POI initialization"""
    poi = POI(
        poi_id="test_poi", name="Test POI", category="training", location=[10.0, 20.0]
    )
    assert poi.id == "test_poi"
    assert poi.name == "Test POI"
    assert poi.category == "training"
    assert poi.location == [10.0, 20.0]

    # Test default belief values
    assert "satisfaction" in poi.belief
    assert "price" in poi.belief


def test_kalman_update():
    """Test Kalman filter update function"""
    # Initial belief and uncertainty
    belief = 0.5
    sigma_b = 0.3

    # Test observation matching belief
    new_belief, new_sigma = kalman_update(belief, sigma_b, 0.5, 0.2)
    assert new_belief == pytest.approx(0.5)
    assert new_sigma < sigma_b  # Uncertainty should decrease

    # Test observation different from belief
    new_belief, new_sigma = kalman_update(belief, sigma_b, 0.8, 0.2)
    assert new_belief > belief  # Should move toward observation
    assert new_belief < 0.8  # But not all the way
    assert new_sigma < sigma_b  # Uncertainty should decrease


def test_memory_record_action():
    """Test recording actions to memory"""
    memory = Memory()
    agent = Agent(agent_id="test_agent")
    poi = POI(poi_id="test_poi", location=[10.0, 20.0])
    time = datetime.now()

    # Record an action
    node_id = memory.record_action(
        agent=agent,
        time=time,
        poi=poi,
        activity_key="train",
        observation={
            "activity": "train",
            "poi_id": "test_poi",
            "reason": "test action",
            "expected_duration": 1,
        },
    )

    assert node_id is not None
    
    # Force flush the buffer to check the DataFrame
    memory._flush_buffer()
    assert len(memory.temporal_memory) == 1

    # Check the recorded data
    record = memory.temporal_memory[0]
    assert record["agent_id"] == "test_agent"
    assert record["poi_id"] == "test_poi"
    assert record["activity_key"] == "train"


def test_memory_buffer():
    """Test memory buffer functionality"""
    memory = Memory()
    agent = Agent(agent_id="test_agent")
    poi = POI(poi_id="test_poi", location=[10.0, 20.0])
    time = datetime.now()
    
    # Record multiple actions without exceeding buffer limit
    for i in range(10):
        memory.record_action(
            agent=agent,
            time=time,
            poi=poi,
            activity_key=f"activity_{i}",
        )
    
    # Buffer should have records, but DataFrame should be empty
    assert len(memory._temporal_memory_buffer) == 10
    assert len(memory.temporal_memory) == 0
    
    # Force flush the buffer
    memory._flush_buffer()
    
    # Buffer should be empty, DataFrame should have records
    assert len(memory._temporal_memory_buffer) == 0
    assert len(memory.temporal_memory) == 10


def test_memory_reflective_summary():
    """Test generating reflective memory summaries"""
    memory = Memory()
    agent = Agent(agent_id="test_agent")
    poi = POI(poi_id="test_poi", location=[10.0, 20.0])

    # Record a few actions on the same day
    date = "2025-01-01"
    for i in range(3):
        time = datetime.fromisoformat(f"{date}T0{i}:00:00")
        memory.record_action(
            agent=agent,
            time=time,
            poi=poi,
            activity_key="train",
            observation={
                "activity": "train",
                "poi_id": "test_poi",
                "reason": "test reflective memory",
                "expected_duration": 1,
            },
        )

    # Generate summary
    summary = memory.generate_daily_summary("test_agent", date)

    assert summary is not None
    assert summary["activity_count"] == 3
    assert "activity_distribution" in summary

    # Check it's stored in reflective memory
    assert "test_agent" in memory.reflective_memory
    assert date in memory.reflective_memory["test_agent"]


def test_simulation_basic_flow():
    """Test basic simulation flow without LLM integration"""
    # Create simulation
    sim = Simulation(days=2, time_steps_per_day=4)

    # Add agents and POIs
    agent = Agent(agent_id="test_agent")
    sim.add_agent(agent)

    # Add POIs of different categories
    training_poi = POI(
        poi_id="training_1",
        name="Training Ground",
        category="training",
        location=[10.0, 20.0],
    )
    food_poi = POI(
        poi_id="food_1", name="Cafeteria", category="food", location=[5.0, 10.0]
    )
    rest_poi = POI(
        poi_id="rest_1", name="Barracks", category="rest", location=[3.0, 5.0]
    )

    sim.add_poi(training_poi)
    sim.add_poi(food_poi)
    sim.add_poi(rest_poi)

    # Run one step
    step_stats = sim.step()

    assert step_stats is not None
    assert "day" in step_stats
    assert "agent_actions" in step_stats
    assert len(step_stats["agent_actions"]) == 1  # One agent
    
    # Force flush memory buffer to check DataFrame
    sim.memory._flush_buffer()
    # Check temporal memory has a record
    assert len(sim.memory.temporal_memory) == 1


def test_agent_poi_interaction():
    """Test agent-POI interaction effects"""
    agent = Agent(energy=0.5, hunger=0.5)
    poi = POI(
        category="food",
        effects={
            "energy": 0.1,
            "hunger": -0.3,
        },
    )

    # Apply POI effects
    initial_energy = agent.energy
    initial_hunger = agent.hunger
    agent.apply_poi_effect(poi.get_effects())

    # Check the agent state was updated
    assert agent.energy > initial_energy
    assert agent.hunger < initial_hunger