# City Sim - 処理フロー詳細

## 1. システム全体の処理フロー

### 1.1 メインフロー概要

City Simの処理は以下のフェーズで構成されています。

1. **初期化フェーズ**: 設定読み込み、エージェント・POI生成、LLMクライアント初期化
2. **シミュレーション実行フェーズ**: 日次ループによる行動シミュレーション（ランダムイベント含む）
3. **データ出力フェーズ**: 結果の保存と分析用データの生成

### 1.2 詳細処理ステップ

#### 初期化フェーズ（1回のみ）
```python
# 1. 設定読み込み
settings = get_settings()
config = load_config(config_path)

# 2. エージェント初期化
agents = load_agents("data/personas.json", count=config.agent_count)

# 3. POI初期化
pois = load_pois("data/pois.json", count=config.poi_count)

# 4. LLMクライアント初期化
llm_client = initialize_llm_client(settings, config)
planner = Planner(llm_client) if config.use_llm else None
```

#### シミュレーション実行フェーズ（100日間）
```python
for day in range(simulation_days):
    # 日次処理
    simulation._end_day()  # 前日の処理完了
    
    for step in range(time_steps_per_day):  # 通常8ステップ/日
        # 時間ステップ処理
        step_results = simulation.step()
        
        # ランダムイベントのトリガーと更新
        simulation._trigger_random_events()
        simulation._update_active_events()
        
        # 各エージェントの処理
        for agent in agents:
            # 1. 需要更新（時間帯考慮）
            agent.update_needs(time_delta=1.0, current_hour=current_time.hour)
            
            # 2. 行動計画（時間帯・ランダムイベント考慮）
            action = simulation._choose_action(agent)
            
            # 3. 行動実行
            simulation._execute_action(agent, action)
            
            # 4. メモリ記録
            memory.record_action(agent, time, poi, activity, observation)
```

## 2. エージェント行動決定プロセス

### 2.1 時間帯とランダムイベントによる行動優先度

```python
def _choose_action(self, agent):
    # 現在時刻の取得
    current_hour = self.current_time.hour
    
    # アクティブなイベントのチェック（最優先）
    if agent.id in self.active_events:
        event_data = self.active_events[agent.id]
        forced_category = event_data["forced_poi_category"]
        
        # イベントに基づくPOI選択
        forced_pois = [poi for poi in self.pois.values() if poi.category == forced_category]
        if forced_pois:
            chosen_poi = random.choice(forced_pois)
            activity = activity_map.get(forced_category, "idle")
            return {
                "poi_id": chosen_poi.id,
                "activity": activity,
                "reason": f"Event: {event_data['description']}",
            }
    
    # 時間帯に基づく行動優先度
    is_night_time = current_hour >= 22 or current_hour <= 5  # 夜間
    is_meal_time = current_hour in [12, 18]  # 食事時間
    is_early_morning = current_hour in [6, 7, 8]  # 早朝
    
    # 夜間の睡眠優先処理
    if is_night_time and agent.energy < 60.0:
        rest_pois = [poi for poi in self.pois.values() if poi.category in ["rest", "sleep"]]
        if rest_pois:
            sleep_pois = [poi for poi in rest_pois if poi.category == "sleep"]
            chosen_poi = sleep_pois[0] if sleep_pois else rest_pois[0]
            return {
                "poi_id": chosen_poi.id,
                "activity": "sleep",
                "reason": "Night time sleep schedule",
            }
    
    # 食事時間の優先処理
    if is_meal_time or agent.hunger > 60.0:
        food_pois = [poi for poi in self.pois.values() if poi.category == "food"]
        if food_pois:
            # アーキタイプ嗜好を考慮したスコア計算
            food_poi_scores = []
            for poi in food_pois:
                base_score = 1.0
                # 食事時間でスコア上昇
                if is_meal_time:
                    base_score *= 2.0
                # アーキタイプに基づく嗜好適用
                archetype_multiplier = agent.get_poi_preference_multiplier(poi.category)
                score = base_score * archetype_multiplier
                food_poi_scores.append((poi, score))
            
            # スコア順に選択
            food_poi_scores.sort(key=lambda x: x[1], reverse=True)
            chosen_poi = food_poi_scores[0][0]
            return {
                "poi_id": chosen_poi.id,
                "activity": "eat",
                "reason": "Meal time or high hunger",
            }
    
    # LLMプランナー使用（複雑な意思決定用）
    if self.planner and not is_night_time:
        try:
            # コンテキスト情報の集約
            recent_memories = self.memory.get_recent_memories(agent.id, limit=10)
            reflective_memory = self.memory.get_reflective_memory(agent.id)
            poi_list = [poi.to_dict() for poi in self.pois.values()]
            
            # 時間帯情報をエージェント状態に追加
            agent_state = agent.to_dict()
            agent_state["current_hour"] = current_hour
            agent_state["time_context"] = {
                "is_night_time": is_night_time,
                "is_meal_time": is_meal_time,
                "is_early_morning": is_early_morning,
            }
            agent_state["active_event"] = self.active_events.get(agent.id, None)
            
            # LLMからプラン取得
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
            logger.warning(f"LLM planner failed: {e}, falling back to rule-based planning")
    
    # バックアップ: ルールベース行動選択（時間帯と嗜好を考慮）
    # 重要な需要を優先
    if agent.hunger > 70.0:
        food_pois = [poi for poi in self.pois.values() if poi.category == "food"]
        if food_pois:
            chosen_poi = food_pois[0]
            return {
                "poi_id": chosen_poi.id,
                "activity": "eat",
                "reason": "Critical hunger level",
            }
    
    if agent.energy < 30.0:
        rest_pois = [poi for poi in self.pois.values() if poi.category in ["rest", "sleep"]]
        if rest_pois:
            chosen_poi = rest_pois[0]
            activity = "sleep" if is_night_time else "rest"
            return {
                "poi_id": chosen_poi.id,
                "activity": activity,
                "reason": "Critical energy level",
            }
    
    # アーキタイプに基づくPOI選択
    available_pois = list(self.pois.values())
    poi_scores = []
    
    for poi in available_pois:
        # 基本スコア
        base_score = 1.0
        
        # 時間帯に基づく修飾子
        if is_night_time and poi.category not in ["rest", "sleep"]:
            base_score *= 0.1  # 夜間の休息以外に強いペナルティ
        elif is_early_morning and poi.category == "training":
            base_score *= 1.5  # 朝のトレーニングボーナス
        elif poi.category == "recreation" and current_hour in [19, 20, 21]:
            base_score *= 1.3  # 夕方のレクリエーションボーナス
        
        # アーキタイプ嗜好の適用
        archetype_multiplier = agent.get_poi_preference_multiplier(poi.category)
        score = base_score * archetype_multiplier
        poi_scores.append((poi, score))
    
    # スコア順に選択
    poi_scores.sort(key=lambda x: x[1], reverse=True)
    if poi_scores:
        chosen_poi = poi_scores[0][0]
        
        # POIカテゴリに基づくアクティビティ決定
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
    
    # フォールバック（POIが見つからない場合）
    return {
        "poi_id": None,
        "activity": "idle",
        "expected_duration": 1,
        "reason": "No suitable POI found",
    }
```

### 2.2 LLMプランナーによる行動決定

```python
def plan_action(self, agent_state, reflective_memory, poi_list):
    # プロンプト変数の整形
    prompt_vars = {
        "agent_state": self._format_dict(agent_state),
        "reflective_memory": self._format_dict(reflective_memory),
        "poi_list": self._format_list(poi_list),
    }
    
    # チェーン呼び出し
    return self.chain.invoke(prompt_vars)

# プランナープロンプト
"""
You are a planner for a military simulation agent. Your task is to decide the agent's next action based on their current state, memory, and available Points of Interest (POIs).

Agent State:
{agent_state}

Reflective Memory (Agent's past experiences and patterns):
{reflective_memory}

Available POIs:
{poi_list}

Based on this information, determine:
1. Which POI the agent should visit next
2. What activity they should perform there  
3. How long they should spend (in hours, 1-8)
4. Why this is the optimal choice given their current state and needs

IMPORTANT: You must respond with ONLY valid JSON, no additional text or explanation outside the JSON.

Available activities: train, eat, rest, manage, arm, socialize, heal, exercise, study, craft, communicate, maintain, outdoor_train, reflect, organize, idle

JSON response format:
{
  "agent_id": "extract from agent state",
  "chosen_poi": "select POI ID from available list",
  "activity": "select one activity from available activities",
  "expected_duration": 2,
  "reason": "Brief explanation of choice"
}
"""
```

## 3. ランダムイベントシステム

### 3.1 イベント定義

```python
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
    # その他のイベント...
}
```

### 3.2 イベントトリガーと条件チェック

```python
def _trigger_random_events(self) -> None:
    """エージェントのランダムイベントをチェックしトリガーする"""
    for agent_id, agent in self.agents.items():
        # エージェントがすでにアクティブなイベントを持っている場合はスキップ
        if agent_id in self.active_events:
            continue
        
        # 各イベントのチェック
        for event_type, event_data in RANDOM_EVENTS.items():
            # 確率チェック
            if random.random() > event_data["probability"]:
                continue
            
            # 条件チェック
            if not self._check_event_conditions(agent, event_data):
                continue
            
            # イベントトリガー
            self.active_events[agent_id] = {
                "type": event_type,
                "description": event_data["description"],
                "remaining_duration": event_data["duration"],
                "forced_poi_category": event_data["forced_poi_category"],
                "effects": event_data["effects"],
            }
            
            # 即時効果の適用
            for effect, value in event_data["effects"].items():
                if hasattr(agent, effect):
                    current_value = getattr(agent, effect)
                    new_value = max(0.0, min(100.0, current_value + value))
                    setattr(agent, effect, new_value)
            
            logger.info(f"Event triggered for {agent_id}: {event_data['description']}")
            break  # 1ステップにつき1エージェント1イベントのみ

def _check_event_conditions(self, agent: Agent, event_data: Dict) -> bool:
    """エージェントが特定のイベントの条件を満たしているかチェック"""
    conditions = event_data.get("required_conditions", [])
    
    for condition in conditions:
        if condition == "recent_training":
            # 最近のアクティビティでトレーニングをしたかチェック
            if not any(activity in ["train", "exercise", "outdoor_train"] 
                      for activity in agent._daily_activities[-3:]):
                return False
        elif condition == "high_activity":
            # エージェントが非常にアクティブだったかチェック
            if len(agent._daily_activities) < 5:
                return False
        # 必要に応じて他の条件を追加
    
    return True
```

### 3.3 アクティブイベントの更新

```python
def _update_active_events(self) -> None:
    """アクティブイベントの持続時間を更新し、期限切れのものを削除"""
    expired_events = []
    
    for agent_id, event_data in self.active_events.items():
        event_data["remaining_duration"] -= 1
        if event_data["remaining_duration"] <= 0:
            expired_events.append(agent_id)
            logger.info(f"Event ended for {agent_id}: {event_data['description']}")
    
    # 期限切れイベントの削除
    for agent_id in expired_events:
        del self.active_events[agent_id]
```

## 4. エージェントシステム

### 4.1 アーキタイプとスキル定義

```python
ARCHETYPES = {
    "weapon_specialist": {
        "description": "Weapon enthusiast who loves training and armory work",
        "skill_multipliers": {
            "weapon_strength": 1.5,
            "management_skill": 0.8,
            "sociability": 0.9,
            "power": 1.4,
        },
        "poi_preferences": {
            "training": 2.0,
            "armory": 2.5,
            "food": 1.0,
            "rest": 0.8,
            "sleep": 1.0,
            "office": 0.5,
            "recreation": 0.7,
            # その他の嗜好...
        },
    },
    # その他のアーキタイプ定義...
}
```

### 4.2 ニーズ更新と時間帯の影響

```python
def update_needs(self, time_delta: float = 1.0, current_hour: int = 12) -> None:
    """時間経過と時刻に基づくエージェントのニーズ更新"""
    # 時間帯に基づく修飾子
    is_night_time = current_hour >= 22 or current_hour <= 5
    is_meal_time = current_hour in [12, 18]
    
    # エネルギー減少（軽いランダム化あり）
    base_energy_decay = 0.05 * time_delta
    # 夜間に睡眠していない場合のエネルギー減少増加
    if is_night_time:
        base_energy_decay *= 1.5
    energy_decay = base_energy_decay * random.uniform(0.8, 1.2)
    self.energy = max(0.0, self.energy - energy_decay)
    
    # 空腹増加（軽いランダム化あり）
    base_hunger_increase = 0.08 * time_delta
    # 食事時間が近づくと空腹が増加
    if is_meal_time:
        base_hunger_increase *= 1.3
    hunger_increase = base_hunger_increase * random.uniform(0.8, 1.2)
    self.hunger = min(100.0, self.hunger + hunger_increase)
    
    # 性格に基づく社交的ニーズの変化
    social_base_change = 0.03 * time_delta
    # 夜間は社交的ニーズ減少
    if is_night_time:
        social_base_change *= 0.5
    
    # 外向的な性格は一人の時に社交的エネルギーを速く失う
    if self.personality["extroversion"] > 0.5:
        social_change = social_base_change * (self.personality["extroversion"] + 0.5)
    else:
        # 内向的な性格は社交的エネルギーをゆっくり失う
        social_change = social_base_change * self.personality["extroversion"]
    
    self.social = max(0.0, min(100.0, self.social - social_change))
    
    # 最近練習していなければスキルの自然減衰を適用
    if "train" not in self._daily_activities and "exercise" not in self._daily_activities:
        self._apply_skill_decay("weapon_strength", 0.01 * time_delta)
        self._apply_skill_decay("power", 0.01 * time_delta)
    
    if "manage" not in self._daily_activities and "study" not in self._daily_activities:
        self._apply_skill_decay("management_skill", 0.01 * time_delta)
    
    if "socialize" not in self._daily_activities:
        self._apply_skill_decay("sociability", 0.005 * time_delta)
    
    # アクティビティリストが長くなりすぎたらリセット
    if len(self._daily_activities) > 10:
        self._daily_activities = self._daily_activities[-5:]
```

### 4.3 POI効果の適用と逓減効果

```python
def apply_poi_effect(self, poi_effects: Dict[str, float]) -> None:
    """
    POI訪問の効果を適用
    
    効果は値が1.0に近づくにつれて逓減
    アーキタイプがスキル向上の効果に影響
    """
    # POI効果に基づいてアクティビティを追跡
    if "weapon_strength" in poi_effects and poi_effects["weapon_strength"] > 0:
        self._daily_activities.append("train")
        self._training_sessions += 1
    
    if "management_skill" in poi_effects and poi_effects["management_skill"] > 0:
        self._daily_activities.append("manage")
        self._management_sessions += 1
    
    if "sociability" in poi_effects and poi_effects["sociability"] > 0:
        self._daily_activities.append("socialize")
        self._social_sessions += 1
    
    # スキルキャップの更新
    self._update_skill_caps()
    
    # アーキタイプの乗数を取得
    archetype_multipliers = ARCHETYPES.get(self.archetype, {}).get("skill_multipliers", {})
    
    # 逓減効果とアーキタイプボーナスで効果を適用
    self._apply_effect_with_diminishing_returns("energy", poi_effects.get("energy", 0))
    self._apply_effect_with_diminishing_returns("hunger", poi_effects.get("hunger", 0))
    self._apply_effect_with_diminishing_returns("social", poi_effects.get("social", 0), 
                                              self._skill_caps["social"])
    self._apply_effect_with_diminishing_returns("power", poi_effects.get("power", 0), 
                                              self._skill_caps["power"])
    
    # スキル向上にアーキタイプボーナスを適用
    weapon_effect = poi_effects.get("weapon_strength", 0) * archetype_multipliers.get("weapon_strength", 1.0)
    self._apply_effect_with_diminishing_returns("weapon_strength", weapon_effect,
                                              self._skill_caps["weapon_strength"])
    
    management_effect = poi_effects.get("management_skill", 0) * archetype_multipliers.get("management_skill", 1.0)
    self._apply_effect_with_diminishing_returns("management_skill", management_effect,
                                              self._skill_caps["management_skill"])
    
    sociability_effect = poi_effects.get("sociability", 0) * archetype_multipliers.get("sociability", 1.0)
    self._apply_effect_with_diminishing_returns("sociability", sociability_effect,
                                             self._skill_caps["sociability"])
    
    power_effect = poi_effects.get("power", 0) * archetype_multipliers.get("power", 1.0)
    self._apply_effect_with_diminishing_returns("power", power_effect,
                                              self._skill_caps["power"])
```

## 5. メモリシステム

### 5.1 時系列メモリ（Temporal Memory）記録

```python
def record_action(
    self,
    agent: Agent,
    time: datetime,
    poi: Optional[POI] = None,
    activity_key: str = "idle",
    observation: Dict[str, Any] = None,
) -> str:
    """
    エージェントの行動を時系列メモリに記録
    
    Args:
        agent: 行動を実行するエージェント
        time: 行動のタイムスタンプ
        poi: 行動が発生したPOI（またはNone）
        activity_key: アクティビティの種類（"train", "eat", "rest"など）
        observation: この行動からの観察データ
    
    Returns:
        作成されたメモリノードのID
    """
    node_id = str(uuid4())
    time_str = time.isoformat()
    poi_id = poi.id if poi else None
    location = poi.location if poi else agent.location
    
    # 必要なフィールドでデフォルト値を持つ観察を確保
    if not observation:
        observation = {}
    observation = {
        "activity": observation.get("activity", activity_key),
        "poi_id": observation.get("poi_id", poi_id or ""),
        "reason": observation.get("reason", ""),
        "expected_duration": observation.get("expected_duration", 1),
    }
    
    # 新しいレコードの作成
    new_record = {
        "id": node_id,
        "time": time_str,
        "agent_id": agent.id,
        "poi_id": poi_id or "",
        "location": location,
        "activity_key": activity_key,
        "observation": observation,
        "current_energy": agent.energy,
        "current_social": agent.social,
        "current_weapon_strength": agent.weapon_strength,
        "current_hunger": agent.hunger,
        "current_management_skill": agent.management_skill,
        "current_sociability": agent.sociability,
        "current_power": agent.power,
    }
    
    # 時系列メモリバッファに追加
    self._temporal_memory_buffer.append(new_record)
    
    # バッファが大きくなっていればフラッシュ
    if len(self._temporal_memory_buffer) >= 100:
        self._flush_buffer()
    
    return node_id
```

### 5.2 メモリバッファとPolars DataFrameの管理

```python
def _flush_buffer(self) -> None:
    """バッファをtemporal memoryのDataFrameにフラッシュ"""
    if not self._temporal_memory_buffer:
        return
    
    # バッファからDataFrameを作成し、既存のメモリと連結
    buffer_df = pl.DataFrame(self._temporal_memory_buffer)
    self.temporal_memory = pl.concat([self.temporal_memory, buffer_df])
    
    # バッファをクリア
    self._temporal_memory_buffer = []
```

### 5.3 反射メモリ（Reflective Memory）生成

```python
def generate_daily_summary(self, agent_id: str, date: str) -> Dict[str, Any]:
    """
    特定の日のエージェントの反射サマリーを生成
    
    Args:
        agent_id: エージェントのID
        date: YYYY-MM-DD形式の日付文字列
    
    Returns:
        反射メモリのサマリー辞書
    """
    # バッファをフラッシュして全レコードがDataFrameにあることを確保
    self._flush_buffer()
    
    # この日付のこのエージェントのメモリをフィルタリング
    daily_df = self.temporal_memory.filter(
        (pl.col("agent_id") == agent_id) & (pl.col("time").str.contains(date))
    )
    
    if len(daily_df) == 0:
        return {"no_activity": True}
    
    # サマリー統計の生成
    summary = {
        "date": date,
        "activity_count": len(daily_df),
        "activity_distribution": daily_df.group_by("activity_key")
                                .len()
                                .sort("len", descending=True)
                                .to_dict(),
        "most_visited_poi": daily_df.group_by("poi_id")
                           .len()
                           .sort("len", descending=True)
                           .head(1)
                           .to_dict(),
        "energy_stats": {
            "start": daily_df["current_energy"].first(),
            "end": daily_df["current_energy"].last(),
            "avg": daily_df["current_energy"].mean(),
            "min": daily_df["current_energy"].min(),
            "max": daily_df["current_energy"].max(),
        },
        "hunger_stats": {
            "start": daily_df["current_hunger"].first(),
            "end": daily_df["current_hunger"].last(),
            "avg": daily_df["current_hunger"].mean(),
        },
        "weapon_strength_change": daily_df["current_weapon_strength"].last()
                                - daily_df["current_weapon_strength"].first(),
        "management_skill_change": daily_df["current_management_skill"].last()
                                 - daily_df["current_management_skill"].first(),
        "power_change": daily_df["current_power"].last()
                      - daily_df["current_power"].first(),
    }
    
    # 反射メモリに保存
    if agent_id not in self.reflective_memory:
        self.reflective_memory[agent_id] = {}
    
    self.reflective_memory[agent_id][date] = summary
    return summary
```

### 5.4 メモリデータの取得

```python
def get_recent_memories(self, agent_id: str, limit: int = 10) -> pl.DataFrame:
    """エージェントの最近の時系列メモリを取得"""
    # バッファをフラッシュして全レコードがDataFrameにあることを確保
    self._flush_buffer()
    
    return (
        self.temporal_memory.filter(pl.col("agent_id") == agent_id)
        .sort("time", descending=True)
        .head(limit)
    )

def get_reflective_memory(self, agent_id: str, date: Optional[str] = None) -> Dict:
    """エージェントの反射メモリを取得、オプションで特定の日付のものを取得"""
    if agent_id not in self.reflective_memory:
        return {}
    
    if date:
        return self.reflective_memory[agent_id].get(date, {})
    
    return self.reflective_memory[agent_id]
```

## 6. POI信念更新システム

### 6.1 カルマンフィルタによるPOI信念更新

```python
def update_belief(self, observation: Dict[str, float], sigma_o: float = 0.2) -> None:
    """観測に基づいてカルマンフィルタを使用して信念を更新"""
    for key in observation:
        if key in self.belief and key in self.belief_sigma:
            self.belief[key], self.belief_sigma[key] = kalman_update(
                belief=self.belief[key],
                sigma_b=self.belief_sigma[key],
                observation=observation[key],
                sigma_o=sigma_o,
            )
```

## 7. 日次処理と結果生成

### 7.1 日次終了処理

```python
def _end_day(self) -> None:
    """日終了アクティビティの処理"""
    self.current_day += 1
    self.current_step = 0
    
    # --- 修正: サマリー生成用の日付をadvance前に取得 ---
    summary_date_str = self.current_time.strftime("%Y-%m-%d")
    
    # 時間を翌日の午前6時にリセット
    self.current_time = self.current_time.replace(hour=6, minute=0, second=0, microsecond=0)
    self.current_time += timedelta(days=1)
    
    # 全エージェントの反射メモリを生成 (修正: summary_date_strを使用)
    for agent_id in self.agents:
        self.memory.generate_daily_summary(agent_id, summary_date_str)
```

### 7.2 シミュレーション結果の生成

```python
def run(self) -> Dict[str, Any]:
    """指定した日数の完全シミュレーションを実行"""
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
    
    # 結果の準備
    results["logs"] = self.logs
    results["temporal_memory"] = self.memory.temporal_memory
    results["reflective_memory"] = self.memory.reflective_memory
    
    return results
```

## 8. データ処理とパフォーマンス最適化

### 8.1 バッチ処理とバッファ管理

```python
# メモリバッファ管理
BUFFER_SIZE = 100
if len(self._temporal_memory_buffer) >= BUFFER_SIZE:
    self._flush_buffer()

# Polars DataFrameによる高速処理
with pl.Config(streaming_chunk_size=10000):
    daily_stats = temporal_memory.lazy().groupby(["agent_id", "date"]).agg([
        pl.col("current_energy").mean(),
        pl.col("activity_key").count()
    ]).collect()
```

### 8.2 出力データ生成

```python
# 1. エージェント別スキル成長
skill_growth = temporal_memory.groupby("agent_id").agg([
    (pl.col("current_weapon_strength").last() - pl.col("current_weapon_strength").first()).alias("weapon_growth"),
    (pl.col("current_management_skill").last() - pl.col("current_management_skill").first()).alias("mgmt_growth"),
    (pl.col("current_power").last() - pl.col("current_power").first()).alias("power_growth")
])

# 2. POI使用統計
poi_usage = temporal_memory.groupby("poi_id").agg([
    pl.col("agent_id").count().alias("visit_count"),
    pl.col("current_energy").mean().alias("avg_energy_effect")
])

# 3. タイムスタンプ付き結果の保存
def save_results(results, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 実行ログ
    with open(output_path / "logs.json", "w") as f:
        json.dump(results["logs"], f, indent=2, default=str)
    
    # 時系列メモリ（高速アクセス用）
    results["temporal_memory"].write_parquet(output_path / "temporal_memory.parquet")
    
    # 反射メモリ
    with open(output_path / "reflective_memory.json", "w") as f:
        json.dump(results["reflective_memory"], f, indent=2, default=str)
```

この処理フローにより、ランダムイベント、時間帯ベースの行動決定、アーキタイプ嗜好、スキル成長と減衰の高度なロジックを含む大規模なマルチエージェントシミュレーションを効率的かつ安定的に実行できます。各コンポーネントの相互作用とデータフローが明確に定義されており、拡張性と保守性を確保しています。
