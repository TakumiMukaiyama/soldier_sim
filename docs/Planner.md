Plannerモジュール（PydanticChain版）詳細設計

目的
	•	CitySimのRecursive Planning、POI選択、行動計画生成を
	•	LLM (Gemini-2.0-Flash, Azure GPT-4.1) + PydanticChainを用いて実行し、
	•	Structured Output（型安全）で結果を受け取り、エージェントの行動に適用する。

⸻

1. 設計方針

1.1 PydanticChainの役割
	•	Plannerの責務を
	1.	エージェントの現在状態・反射記憶・周辺POI情報を元にPromptを作成
	2.	LLMへ送信しStructured JSON形式で出力
	3.	Pydanticで検証済みの型安全な戻り値へ変換
	4.	シミュレーションループで直接使用可能な行動計画辞書に変換
	•	とする。

1.2 LLM使用方針
	•	Gemini-2.0-Flash と Azure GPT-4.1 の両方に対応可能な抽象化された呼び出しクラスを使用。
	•	高負荷時はAzure GPT-4.1、速度優先時はGeminiを使い分け。

1.3 モジュール構成

src/llm_app/
├── planner.py            ← Planner (PydanticChain)
├── client_gemini.py      ← GeminiClient
├── client_azure.py       ← AzureGPTClient
└── schemas.py            ← Pydantic出力スキーマ


⸻

2. Pydanticスキーマ設計（schemas.py）

from pydantic import BaseModel, Field
from typing import Literal

class PlanOutput(BaseModel):
    agent_id: str = Field(..., description="Agentの一意なID")
    chosen_poi: str = Field(..., description="次に訪問するPOIのID")
    activity: Literal["train", "eat", "rest", "manage", "arm"] = Field(..., description="実施するアクティビティ")
    expected_duration: int = Field(..., ge=1, le=8, description="活動予定時間（時間単位）")
    reason: str = Field(..., description="なぜその行動・POIを選んだのかの簡潔な理由")

特徴
	•	Literal型でactivityの入力ミス防止
	•	expected_durationに範囲制約
	•	ClaudeCodeによるコード生成・保守時に自動テスト容易

⸻

3. PydanticChainでのPlanner構築（planner.py）

3.1 インターフェース設計

from langchain.pydantic_v1 import PydanticChain
from .schemas import PlanOutput

class Planner:
    def __init__(self, llm_client):
        self.chain = PydanticChain(
            llm=llm_client,
            output_model=PlanOutput,
            prompt=self._build_prompt()
        )
    
    def _build_prompt(self):
        # return LangChain PromptTemplate or system+user prompt文字列
        ...

    def plan_action(self, agent_state: dict, reflective_memory: dict, poi_list: list[dict]) -> PlanOutput:
        prompt_vars = {
            "agent_state": agent_state,
            "reflective_memory": reflective_memory,
            "poi_list": poi_list
        }
        return self.chain.invoke(prompt_vars)

3.2 Prompt設計例（_build_prompt()）
	•	System: 「あなたは都市シミュレーションエージェントのプランナーです。与えられたエージェント状態、反射記憶、POIリストから次の行動計画をJSON形式で返答してください。」
	•	User:

Agent State:
{agent_state}

Reflective Memory:
{reflective_memory}

POI List:
{poi_list}

必ず次のJSON形式で返答してください:
{
  "agent_id": "",
  "chosen_poi": "",
  "activity": "",
  "expected_duration": ,
  "reason": ""
}



⸻

4. LLMクライアント（client_gemini.py, client_azure.py）

PydanticChainで使用するため、

from langchain_gemini import ChatGemini
from langchain_azure_openai import ChatAzureOpenAI

class GeminiClient(ChatGemini):
    ...

class AzureGPTClient(ChatAzureOpenAI):
    ...

とし、Plannerに差し替え可能な形で使用。

⸻

5. シミュレーション統合例

planner = Planner(llm_client=gemini_client)  # または azure_client

plan_output = planner.plan_action(agent_state, reflective_memory, poi_list)

agent.apply_plan(plan_output)


⸻

6. テスト設計

ユニットテスト（tests/test_planner.py）
	•	PydanticChainによるStructured JSON出力が PlanOutput と一致するか。
	•	activityが不正な場合にValidationErrorが返るか。
	•	LLMの出力変動による安定性を確保するためモック応答でテスト。

⸻

7. 今後の実装ステップ
	1.	schemas.py に PlanOutput を作成
	2.	LLMクライアント (Gemini, Azure) ラッパー準備
	3.	planner.py で PydanticChain組込み、Prompt作成
	4.	simulation.py に統合し、行動計画生成 → POI訪問・記録・能力更新に連動
	5.	テストを作成し、ClaudeCodeで初期実装のレビューを受けて改善

⸻

メリット
	•	Structured Outputによる安全な行動計画取得
	•	LLMモデル・プロンプトの差し替え可能性
	•	POI選択・Recursive Planningの柔軟な適応
	•	ログ・可視化・デバッグ効率の向上

⸻

