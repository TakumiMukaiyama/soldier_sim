# Soldier Sim - LLM-Driven Simulation

LLM（Gemini、Azure OpenAI）を活用した兵士育成シミュレーションシステム。

## 論文について

このプロジェクトは以下の論文を基にして開発されています。

**CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation**  
*Authors: Nicolas Bougie, Narimasa Watanabe*  
*arXiv:2506.21805*  
*URL: https://arxiv.org/abs/2506.21805*

## Requirements

- Python 3.10+
- Dependencies defined in `pyproject.toml`

## Installation

```bash
# Clone repository
git clone [<repository_url>](https://github.com/TakumiMukaiyama/soldier_sim.git)
cd soldier_sim

# Install dependencies using uv
uv sync

# Or install specific extras
uv sync --extra gemini  # For Gemini LLM
uv sync --extra azure   # For Azure OpenAI
uv sync --extra dev     # For development tools
```

## Configuration

### 設定ファイルの構成

シミュレーションの設定は以下のファイルで管理されています。

#### 1. 環境変数設定（`.env`ファイル）

API キーやパスの設定を行います。プロジェクトルートに`.env`ファイルを作成してください。

```env
# Azure OpenAI settings
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_ID=your_deployment_id

# Gemini settings
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-1.5-pro

# Simulation Settings
SIMULATION_DAYS=100
POI_COUNT=100
AGENT_COUNT=100
TIME_STEPS_PER_DAY=8

# Data Paths
CONFIG_PATH=configs/default.yaml
PERSONAS_PATH=data/personas.json
POIS_PATH=data/pois.json
```

#### 2. シミュレーション設定（`configs/default.yaml`）

エージェントやPOIの詳細な動作パラメータを設定します。

```yaml
# Simulation parameters
simulation:
  days: 100
  time_steps_per_day: 8
  start_hour: 6

# Agent parameters  
agents:
  count: 100
  energy_decay_rate: 0.05
  hunger_increase_rate: 0.08
  social_need_base_rate: 0.03

# POI parameters
pois:
  count: 20
  categories: [training, food, rest, office, armory]
  distribution: {...}

# Memory parameters
memory:
  max_temporal_size: 1000000
  reflective_summary_days: 7

# LLM parameters
llm:
  default_provider: "gemini"
  request_timeout: 30
```

#### 3. 本番環境設定（`configs/production.yaml`）

本番環境用の最適化された設定を行います。

#### 4. 設定管理（`src/configs/settings.py`）

Pydanticベースの型安全な設定管理を行います。環境変数の読み込みとバリデーションを担当します。

### 設定の優先順位

1. **コマンドライン引数**（最優先）
2. **環境変数**（`.env`ファイル）
3. **YAMLファイル**（`configs/*.yaml`）
4. **デフォルト値**（`src/configs/settings.py`内）

### 設定変更の方法

- **API キー等の機密情報**: `.env`ファイルで設定
- **シミュレーションパラメータ**: `configs/default.yaml`で設定
- **本番環境**: `configs/production.yaml`を作成・編集
- **一時的な変更**: コマンドライン引数で上書き

```bash
# 例：日数とエージェント数を一時的に変更
python main.py --days 50 --agents 200
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



## Project Structure

```
soldier_sim/
├── configs/                 # Configuration files
│   ├── default.yaml         # Default simulation settings
│   └── production.yaml      # Production configuration
├── data/                    # Data files
│   ├── personas.json        # Agent personas and profiles
│   └── pois.json            # POI definitions and properties
├── docs/                    # Documentation
│   ├── CitySim Urban Modeling.pdf
│   ├── design.md            # System design documentation
│   ├── memory_logic.md      # Memory system logic
│   ├── Planner.md           # Planning system documentation
│   ├── processing_flow.md   # Data processing flow
│   ├── system_overview.md   # System overview
│   └── technical_details.md # Technical implementation details
├── src/                     # Source code
│   ├── agent_system/        # Agent system core
│   │   ├── agent.py         # Agent implementation
│   │   ├── kalman.py        # Kalman filtering for state estimation
│   │   ├── memory.py        # Memory system (temporal/reflective/spatial)
│   │   ├── poi.py           # POI system implementation
│   │   └── simulation.py    # Main simulation engine
│   ├── configs/             # Configuration management
│   │   └── settings.py      # Pydantic settings models
│   ├── llm_app/             # LLM integration
│   │   ├── client_azure.py  # Azure OpenAI client
│   │   ├── client_gemini.py # Gemini client
│   │   ├── planner.py       # LLM-powered planning system
│   │   └── schemas.py       # Data schemas for LLM interaction
│   └── utils/               # Utilities
│       └── df_utils.py      # Polars DataFrame utilities
├── tests/                   # Test files
│   ├── test_agent_improvements.py
│   └── test_simulation.py
├── check.ipynb              # Analysis and visualization notebook
├── main.py                  # Main entry point
├── pyproject.toml           # Project configuration and dependencies
└── uv.lock                  # Dependency lock file
```

## Core Concepts

- **Agent**: 都市環境で行動する知的エンティティ（personas、memories、意思決定機能を持つ）
- **POI**: エージェントが相互作用し情報を収集できる戦略的拠点
- **Memory**: リアルなエージェント行動のための多層メモリシステム
- **Planning**: LLMを使用した目標指向の意思決定
- **Simulation**: 設定可能なパラメータを持つ時間ステップシミュレーション
- **LLM Integration**: エージェントの推論のための自然言語処理
- **Kalman Filtering**: 不確実な環境での状態推定

## Configuration Details

### Simulation Parameters
- **Days**: シミュレーション日数 (デフォルト: 100)
- **Time Steps per Day**: 1日あたりのタイムステップ数 (デフォルト: 8)
- **Start Hour**: シミュレーション開始時間 (デフォルト: 6時)

### Agent Parameters
- **Count**: エージェント数 (デフォルト: 100)
- **Energy Decay Rate**: タイムステップごとのエネルギー減少率
- **Hunger Increase Rate**: 空腹度上昇率
- **Social Need Base Rate**: 社会的欲求の基本上昇率

### POI Categories
- **training**: 訓練施設 (40%)
- **food**: 食事施設 (20%)
- **rest**: 休憩施設 (20%)
- **office**: オフィス (10%)
- **armory**: 武器庫 (10%)

## Output

シミュレーションは以下の出力ファイルを生成します。

- `logs.json`: シミュレーションログ
- `temporal_memory.parquet`: 時間ベースのメモリ (Polars DataFrame)
- `reflective_memory.json`: 処理された洞察とリフレクション
- `spatial_memory.parquet`: 空間ベースのメモリ


