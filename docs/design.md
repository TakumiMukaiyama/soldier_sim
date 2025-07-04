以下は、ご希望の要件を踏まえた**設計書（Markdown）**です。Polars・Pydantic・LLMラッパー・Structured Outputなどの技術を統合したベース構成で、ClaudeCodeにレビューしてもらい、叩き台実装へ移ることを想定しています。

⸻

軍隊中隊規模 Multi-Agent LLM シミュレーション 設計書

目的
	•	LLM（Gemini, Azure GPT‑4.1）を活用して、中隊（100名）規模のエージェントを100日間シミュレーション。
	•	Polarsによる高速な行動履歴管理と集計。
	•	Pydanticによる環境変数・LLM出力の型安全なバリデーション。
	•	ClaudeCodeによる実装レビュー・自動生成に適した構造。

⸻

目录
	1.	ディレクトリ構成
	2.	環境変数設定
	3.	技術スタックまとめ
	4.	モジュール設計概要
	5.	シミュレーションフロー
	6.	Structured Output仕様
	7.	テスト戦略＆開発フロー

⸻

1. ディレクトリ構成

military_sim/
├── configs/
│   ├── default.yaml
│   └── production.yaml
├── data/
│   ├── personas.json
│   └── pois.json
├── src/
│   ├── agent_system/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── poi.py
│   │   ├── kalman.py
│   │   ├── memory.py
│   │   └── simulation.py
│   ├── llm_app/
│   │   ├── __init__.py
│   │   ├── client_gemini.py
│   │   ├── client_azure.py
│   │   └── planner.py
│   └── utils/
│       └── df_utils.py
├── tests/
│   └── test_simulation.py
├── .env
├── pyproject.toml
└── main.py

	•	agent_system/: エージェントやPOI、メモリ、カルマン、シミュレーションのロジック。
	•	llm_app/: 各LLMラッパーとプランナー（Structured Output設計）。
	•	utils/: Polars用DFヘルパー関数。
	•	configs/: Pydantic Settingsに基づく設定管理。
	•	tests/: ユニットテスト。

⸻

2. 環境変数設定

# src/configs/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class AppSettings(BaseSettings):
    gemini_api_key: SecretStr
    azure_api_key: SecretStr
    simulation_days: int = 100
    poi_count: int = 20
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

	•	.envから安全かつ型検証付きで設定値をロード。
	•	Pydantic BaseSettingsによる環境変数管理はLLMアプリにおけるベストプラクティスとされています  ￼ ￼。

⸻

3. 技術スタックまとめ

項目	選択技術	理由
言語	Python 3.10+	標準
データ処理	Polars	高速・並列対応・型安全性  ￼ ￼
環境変数	Pydantic BaseSettings	Prisma
LLM	Gemini‑2.0‑flash, Azure GPT‑4.1	高性能応答モデル
LLMクライアント	独自ラッパー＋Structured Output	Pydantic出力検証による信頼性
アプリ構成	モジュール分割	保守性向上
テスト	pytest	品質担保
可視化	Matplotlib/Plotly（必要に応じpandas変換）	標準


⸻

4. モジュール設計

4.1 Polarsユーティリティ (df_utils.py)
	•	append_records, aggregate_daily_energy, poi_visit_stats など共通処理を提供

4.2 LLMクライアント (client_gemini.py / client_azure.py)
	•	plan_action(prompt: str) -> dictを共通インターフェースで提供
	•	Pydanticモデルによるレスポンス検証 ()

4.3 Planner (planner.py)
	•	エージェント状態 + 行動候補POIをプロンプト化
	•	LLMが返すStructured JSONを planner.schemas にPydanticモデル化

4.4 Agent (agent.py)
	•	属性: age, personality, 階級
	•	状態: energy, social, hunger, weapon_strength, management_skill, needs
	•	メソッド: update_needs(), apply_poi_effect()

4.5 POI (poi.py)
	•	属性: location, category, belief vector + sigma
	•	メソッド: update_belief()（カルマンループ）

4.6 カルマン (kalman.py)
	•	def kalman_update(b, sigma, obs, sigma_o=0.2)

4.7 メモリ (memory.py)
	•	record_action(...)によりTemporal Memoryノードを保存:

- time
- agent_id
- poi_id
- activity_key
- observation
- current_energy
- current_social
- current_weapon_strength
- current_hunger
- current_management_skill


	•	Reflective Memoryは日次で要約生成

4.8 シミュレーション (simulation.py)
	•	日次ループ→時間ステップ→Actions→POI効果→記録
	•	初期はルールベース、その後LLM導入

⸻

5. シミュレーションフロー

for day in range(simulation_days):
    reflective_memory_update()
    daily_plan = planner.plan_action(agent_state, poi_states)

    for step in time_steps_per_day:
        agent.update_needs()
        action = planner.choose_action()
        poi = action.poi
        agent.move_to(poi)
        agent.apply_poi_effect(poi)
        poi.update_belief(obs)
        memory.record_action(...)
        logs.append(...)

	•	Polarsで行動ノードを蓄積し、日次・週次の集計を行う

⸻

6. Structured Output仕様
	•	LLMからの戻り値はJSON形式かつPydanticモデルトレースで検証
	•	JSON例：

{"agent_id": "agent_5", "chosen_poi": "training_ground_3", "activity": "train", "expected_duration": 2}

	•	Pydanticでフィールド型チェック・enum制約などを行い、信頼性を担保  ￼ ￼

⸻

7. テスト戦略
	•	test_simulation.py：
	•	Agent→Needs更新・記録・POI効果適用の動作テスト
	•	kalman_updateの数値検証
	•	plannerによるStructured JSONのバリデーションテスト

フェーズ開発
	1.	ルールベースでPOCまで
	2.	Polars集計・可視化追加
	3.	LLMラッパー+Structured Output統合
	4.	スケールアップ（100日・POI20・Agent100）

⸻
