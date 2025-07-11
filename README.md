# City Sim - Multi-Agent LLM Military Simulation

A multi-agent simulation system using LLM (Gemini, Azure OpenAI) for military scenario modeling.
Built with Polars for high-performance data processing, Pydantic for data validation, and LangChain for LLM integration.

## Features

- **Agent System**: Simulate up to 100 intelligent agents
- **POI (Point of Interest)**: Locations where agents can interact
- **Memory System**: 
  - Temporal Memory (time-based experiences)
  - Reflective Memory (processed insights)
  - Spatial Memory (location-based information)
- **LLM Integration**: Agent decision-making powered by LLMs
- **Data Processing**: Efficient DataFrame operations with Polars

## Requirements

- Python 3.10+
- Dependencies:
  - polars
  - pydantic
  - pydantic-settings
  - langchain
  - langchain_google_genai or langchain_openai
  - matplotlib
  - plotly
  - PyYAML

## Installation

```bash
# Clone repository
git clone https://github.com/TakumiMukaiyama/city_sim.git
cd city-sim

# Install dependencies
uv add polars pydantic pydantic-settings langchain langchain-gemini matplotlib plotly PyYAML

# Development dependencies
uv add -d pytest black isort ruff
```

## Configuration

Before running, set up your environment:

1. Copy `.env.template` to `.env`
2. Configure API keys

```env
# LLM API Keys
GEMINI_API_KEY=your_gemini_api_key_here
AZURE_API_KEY=your_azure_api_key_here

# Simulation Settings
SIMULATION_DAYS=100
POI_COUNT=20
AGENT_COUNT=100
```

## Usage

### Basic Simulation

```bash
python main.py
```

### Advanced Options

```bash
# Use production config
python main.py --config configs/production.yaml

# Custom simulation parameters
python main.py --days 10 --steps 4

# Enable LLM integration
python main.py --use-llm

# Specify output directory
python main.py --output results/experiment1
```

### Running Tests

```bash
pytest
```

## Project Structure

```
city-sim/
├── configs/                 # Configuration files
│   ├── default.yaml
│   └── production.yaml
├── data/                    # Data files
│   ├── personas.json        # Agent personas
│   └── pois.json            # POI definitions
├── src/                     # Source code
│   ├── agent_system/        # Agent system
│   │   ├── agent.py         # Agent implementation
│   │   ├── poi.py           # POI system
│   │   ├── kalman.py        # Kalman filtering
│   │   ├── memory.py        # Memory system
│   │   └── simulation.py    # Simulation engine
│   ├── llm_app/             # LLM integration
│   │   ├── client_gemini.py # Gemini client
│   │   ├── client_azure.py  # Azure GPT client
│   │   └── planner.py       # Planning system
│   └── utils/               # Utilities
│       └── df_utils.py      # Polars utilities
├── tests/                   # Test files
├── .env                     # Environment variables
├── pyproject.toml           # Project configuration
└── main.py                  # Main entry point
```

## Core Concepts

- **Agent**: Intelligent entities with personas, memories, and decision-making capabilities
- **POI**: Strategic locations where agents can interact and gather information
- **Memory**: Multi-layered memory system for realistic agent behavior
- **Planning**: Goal-oriented decision making using LLMs
- **Simulation**: Time-stepped simulation with configurable parameters
- **LLM Integration**: Natural language processing for agent reasoning

## Output

The simulation generates various output files:

- `logs.json`: Simulation logs
- `temporal_memory.parquet`: Time-based memories (Polars DataFrame)
- `reflective_memory.json`: Processed insights

