import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import yaml

from src.agent_system.agent import ARCHETYPES, Agent
from src.agent_system.poi import POI
from src.agent_system.simulation import Simulation
from src.configs.settings import app_settings
from src.llm_app.client_azure import AzureGPTClient
from src.llm_app.client_gemini import GeminiClient
from src.llm_app.planner import Planner
from src.utils.get_logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_agents(personas_path, count=None, config=None):
    """Load agent personas from JSON file with configurable archetype distribution"""
    with open(personas_path, "r") as f:
        data = json.load(f)

    # Get list of available archetypes
    available_archetypes = list(ARCHETYPES.keys())
    
    # Get archetype distribution from config or use equal distribution
    archetype_weights = None
    if config and "agents" in config and "archetype_distribution" in config["agents"]:
        archetype_dist = config["agents"]["archetype_distribution"]
        archetype_weights = [archetype_dist.get(arch, 1.0) for arch in available_archetypes]
        logger.info(f"Using configured archetype distribution: {dict(zip(available_archetypes, archetype_weights))}")
    else:
        # Default to equal weights if no distribution is configured
        archetype_weights = [1.0] * len(available_archetypes)
        logger.info("Using equal archetype distribution (no configuration found)")

    # If a specific count is requested, handle it
    if count is not None:
        if count <= len(data):
            # If we need fewer agents than available, sample randomly
            data = random.sample(data, count)
        else:
            # If we need more agents than available, clone and modify existing ones
            original_count = len(data)
            for i in range(original_count, count):
                # Clone a random persona
                new_persona = random.choice(data).copy()

                # Modify the cloned persona to make it unique
                new_persona["id"] = f"agent_{i + 1}"

                # Randomize some attributes
                new_persona["age"] = random.randint(19, 40)

                # Assign fixed rank (removing rank selection logic)
                new_persona["rank"] = "private"

                # Assign archetype based on configured weights
                new_persona["archetype"] = random.choices(
                    available_archetypes,
                    weights=archetype_weights,
                    k=1
                )[0]

                # Randomize personality slightly
                for trait in new_persona["personality"]:
                    new_persona["personality"][trait] = max(
                        0.1,
                        min(
                            0.9,
                            new_persona["personality"][trait]
                            + random.uniform(-0.2, 0.2),
                        ),
                    )

                # Randomize initial stats slightly
                for stat in new_persona["initial_stats"]:
                    if stat == "management_skill":
                        # Special handling for management_skill: keep it in 0.0-0.3 range
                        base_value = new_persona["initial_stats"][stat]
                        new_value = base_value + random.uniform(-0.1, 0.1)
                        new_persona["initial_stats"][stat] = max(
                            0.0, min(0.3, new_value)
                        )
                    else:
                        # Normal randomization for other stats
                        new_persona["initial_stats"][stat] = max(
                            0.1,
                            min(
                                0.9,
                                new_persona["initial_stats"][stat]
                                + random.uniform(-0.15, 0.15),
                            ),
                        )

                # Add to data
                data.append(new_persona)

    agents = []
    for persona in data:
        # Assign archetype if not already present, using configured weights
        if "archetype" not in persona:
            persona["archetype"] = random.choices(
                available_archetypes,
                weights=archetype_weights,
                k=1
            )[0]

        agent = Agent(
            agent_id=persona.get("id"),
            age=persona.get("age", 30),
            personality=persona.get("personality", {}),
            rank=persona.get("rank", "private"),
            archetype=persona.get("archetype"),
            energy=persona.get("initial_stats", {}).get("energy", 1.0),
            social=persona.get("initial_stats", {}).get("social", 0.5),
            hunger=persona.get("initial_stats", {}).get("hunger", 0.0),
            weapon_strength=persona.get("initial_stats", {}).get(
                "weapon_strength", 0.5
            ),
            management_skill=persona.get("initial_stats", {}).get(
                "management_skill", 0.3
            ),
            sociability=persona.get("initial_stats", {}).get("sociability", 0.5),
            power=persona.get("initial_stats", {}).get("power", 0.5),
        )
        agents.append(agent)

    return agents


def load_pois(pois_path, count=None):
    """Load POIs from JSON file"""
    with open(pois_path, "r") as f:
        data = json.load(f)

    # If a specific count is requested, handle it
    if count is not None:
        if count <= len(data):
            # If we need fewer POIs than available, sample randomly
            data = random.sample(data, count)
        else:
            # If we need more POIs than available, clone and modify existing ones
            original_count = len(data)
            categories = list(set([poi["category"] for poi in data]))

            for i in range(original_count, count):
                # Clone a random POI
                new_poi = random.choice(data).copy()

                # Modify the cloned POI to make it unique
                new_poi["id"] = f"{random.choice(categories)}_{i + 1}"
                new_poi["name"] = f"{random.choice(categories).title()} Area {i + 1}"

                # Randomize location
                new_poi["location"] = [random.uniform(0, 50), random.uniform(0, 50)]

                # Randomize belief slightly
                for belief in new_poi["belief"]:
                    new_poi["belief"][belief] = max(
                        0.1,
                        min(0.9, new_poi["belief"][belief] + random.uniform(-0.2, 0.2)),
                    )

                # Randomize effects slightly
                for effect in new_poi["effects"]:
                    # Keep the sign of the effect, but randomize magnitude
                    original = new_poi["effects"][effect]
                    sign = 1 if original >= 0 else -1
                    magnitude = abs(original)
                    new_magnitude = magnitude * random.uniform(0.8, 1.2)
                    new_poi["effects"][effect] = sign * new_magnitude

                # Add to data
                data.append(new_poi)

    pois = []
    for poi_data in data:
        poi = POI(
            poi_id=poi_data.get("id"),
            name=poi_data.get("name", ""),
            category=poi_data.get("category", "generic"),
            location=poi_data.get("location", [0.0, 0.0]),
            belief=poi_data.get("belief", {}),
            belief_sigma=poi_data.get("belief_sigma", {}),
            effects=poi_data.get("effects", {}),
        )
        pois.append(poi)

    return pois


def initialize_llm_client(settings, config):
    """Initialize appropriate LLM client based on settings"""
    provider = config["llm"].get("default_provider", "gemini")

    if provider == "azure":
        if (
            not settings.azure_openai_api_key
            or not settings.azure_openai_endpoint
            or not settings.azure_openai_deployment_id
        ):
            logger.warning(
                "Azure credentials not fully configured, falling back to Gemini"
            )
            provider = "gemini"
        else:
            return AzureGPTClient(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                azure_deployment=settings.azure_openai_deployment_id,
                api_version=settings.azure_openai_api_version,
            )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not configured in settings")
        return GeminiClient(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model_name,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def save_results(results, output_dir):
    """Save simulation results to output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # Save logs as JSON
    with open(output_path / "logs.json", "w") as f:
        json.dump(results["logs"], f, indent=2, default=str)

    # Save temporal memory as Parquet
    results["temporal_memory"].write_parquet(output_path / "temporal_memory.parquet")

    # Save reflective memory as JSON
    with open(output_path / "reflective_memory.json", "w") as f:
        json.dump(results["reflective_memory"], f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="City Sim - Military Multi-Agent Simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="data/personas.json",
        help="Path to agent personas file",
    )
    parser.add_argument(
        "--pois", type=str, default="data/pois.json", help="Path to POIs file"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to simulate (overrides config)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of time steps per day (overrides config)",
    )
    parser.add_argument(
        "--use-llm", action="store_true", help="Use LLM for agent planning"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Output directory for results"
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=None,
        help="Number of agents to simulate (default: use all available in JSON)",
    )
    parser.add_argument(
        "--poi-count",
        type=int,
        default=None,
        help="Number of POIs to use in simulation (default: use all available in JSON)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulations",
    )

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")

    # Load settings and config
    config = load_config(args.config)

    # Override settings from command line args
    days = args.days or config["simulation"]["days"]
    steps_per_day = args.steps or config["simulation"]["time_steps_per_day"]

    # Use config settings for agent and POI counts, with command line override
    agent_count = config["agents"]["count"]
    poi_count = config["pois"]["count"]

    logger.info(
        f"Running simulation for {days} days with {steps_per_day} steps per day"
    )
    if agent_count:
        logger.info(f"Using {agent_count} agents")
    if poi_count:
        logger.info(f"Using {poi_count} POIs")

    # Initialize LLM planner if requested
    planner = None
    if args.use_llm:
        try:
            llm_client = initialize_llm_client(app_settings, config)
            planner = Planner(llm_client)
            logger.info(
                f"Using LLM planner with {config['llm']['default_provider']} model"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM planner: {e}")
            logger.info("Continuing with rule-based planning")

    # Create simulation
    simulation = Simulation(
        days=days, time_steps_per_day=steps_per_day, planner=planner
    )

    # Load and add agents
    agents = load_agents(args.agents, count=agent_count, config=config)
    logger.info(f"Loaded {len(agents)} agents")
    for agent in agents:
        simulation.add_agent(agent)

    # Load and add POIs
    pois = load_pois(args.pois, count=poi_count)
    logger.info(f"Loaded {len(pois)} POIs")
    for poi in pois:
        simulation.add_poi(poi)

    # Run simulation
    logger.info("Starting simulation...")
    results = simulation.run()

    # Save results
    output_path = save_results(results, args.output)

    logger.info(f"Simulation completed. {results['days_completed']} days simulated.")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
