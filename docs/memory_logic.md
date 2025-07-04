記憶蓄積ロジック詳細設計

目的

CitySimのLLM駆動Multi-Agentシステムにおける
	•	Temporal Memory（時系列記憶）
	•	Reflective Memory（反射記憶・日次統合記憶）
	•	Spatial Memory（空間記憶）

の設計を詳細化し、LLMによるRecursive Planning・Belief更新・POI選択の適応性を高める。

⸻

1. Temporal Memory（時系列記憶）

概要
	•	各Agentが 行動ごとに1ノードを時系列で追記。
	•	記録後、Polars DataFrame に保持し、時系列分析・可視化・集約に使用。
	•	直近24時間のノードを類似度検索でLLMプロンプトに取り込み可能。

記録タイミング
	•	POI訪問
	•	行動完了（訓練終了・休憩終了など）
	•	状態変化イベント（Needs閾値割れ時の計画変更）

ノード構造

- id: UUID (内部管理)
- time: ISO timestamp
- agent_id: str
- poi_id: str
- location: [float, float]
- activity_key: str (e.g., "train", "eat", "rest")
- observation: dict
- current_energy: float
- current_social: float
- current_weapon_strength: float
- current_hunger: float
- current_management_skill: float

実装方法
	•	memory.py 内 record_action(agent, time, poi, activity_key, observation_dict) 関数で追加。
	•	Polarsで以下のように管理：

import polars as pl

def append_to_temporal_memory(memory_df: pl.DataFrame, record: dict) -> pl.DataFrame:
    return pl.concat([memory_df, pl.DataFrame([record])])


⸻

2. Reflective Memory（反射記憶）

概要
	•	1日終了時点で Temporal Memory を要約し、高次の自己認識（習慣・好み・行動パターン）を形成。
	•	次回以降の行動計画の事前文脈としてLLMへ渡す。
	•	「夕方は食堂より訓練場を選びがち」「週末は休息時間が多い」など抽象化された知識を保持。

要約方法
	•	日次で対象期間の Polars DataFrame をフィルタリング：
	•	期間：00:00 ~ 23:59
	•	agent_id 別に集計
	•	抽出特徴例：
	•	最多訪問POIカテゴリ
	•	Needs別平均値変化
	•	能力値の増減
	•	Pydanticモデルに変換後、LLMプロンプトで使用可能な要約文字列生成。

実装例

def generate_reflective_summary(df: pl.DataFrame, agent_id: str, date: str) -> dict:
    daily_df = df.filter(
        (pl.col("agent_id") == agent_id) & 
        (pl.col("time").str.contains(date))
    )
    summary = {
        "most_visited_poi": daily_df.groupby("poi_id").count().sort("count", descending=True).first()["poi_id"],
        "avg_energy": daily_df["current_energy"].mean(),
        "avg_hunger": daily_df["current_hunger"].mean(),
        ...
    }
    return summary


⸻

3. Spatial Memory（空間記憶）

概要
	•	POIごとに「信念ベクトル」を保持し、訪問時にカルマンフィルタで更新。
	•	信念ベクトル例：

- satisfaction
- price
- convenience
- atmosphere


	•	未訪問POIは類似POIの平均信念値で初期化。

更新方法（カルマンフィルタ）

def kalman_update(belief, sigma_b, observation, sigma_o=0.2):
    K = sigma_b / (sigma_b + sigma_o)
    updated_belief = K * observation + (1 - K) * belief
    updated_sigma = (1 - K) * sigma_b
    return updated_belief, updated_sigma


⸻

4. LLM連携における使用箇所

Recursive Planningで使用
	•	Temporal Memory：
	•	直近24時間の行動ノードの要約・類似度検索結果をPromptに含める
	•	Reflective Memory：
	•	習慣・好みをPromptに含め、計画の個別化
	•	Spatial Memory：
	•	POIの信念値と距離情報を使い、訪問候補POIの価値推定

⸻

5. 保存と可視化
	•	Temporal Memory：Polars DataFrameをParquet/CSVに日次で追記保存。
	•	Reflective Memory：JSON形式で日次保存。
	•	Spatial Memory：POIごとのJSONまたはParquet管理。
	•	可視化：体力/空腹度推移、POI訪問ヒートマップ、能力成長グラフをPlotly/Matplotlibで生成。

⸻

6. 今後の実装フェーズ
	1.	Temporal MemoryのPolars実装 + 記録のE2Eテスト
	2.	Reflective Memory要約スクリプト作成、日次生成テスト
	3.	カルマンフィルタ更新ロジックテスト
	4.	LLMプロンプト統合（Plannerモジュール連携）

