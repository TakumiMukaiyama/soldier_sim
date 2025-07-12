# Rule-Based Simulation System Documentation

## 1. Overview

軍事シミュレーションにおけるルールベースの行動選択システムは、LLMが使用できない場合のフォールバックメカニズムとして機能し、現実的な軍事環境での兵士の行動を模倣します。このシステムは時間ベース、需要ベース、性格ベースの優先度ルールを組み合わせています。

## 2. Rule-Based Decision Making Architecture

### 2.1 Decision Priority Hierarchy

エージェントの行動選択は以下の優先度順で決定されます：

1. **Active Events (最優先)**
   - 進行中のランダムイベント（怪我、病気、特別任務など）
   - 通常の行動選択を上書きする

2. **Time-Based Priorities (時間ベース)**
   - 夜間睡眠 (22:00-05:00)
   - 食事時間 (12:00, 18:00)
   - 朝の訓練時間 (06:00-08:00)

3. **Critical Needs (緊急度ベース)**
   - 空腹度 > 70% → 強制的に食事
   - エネルギー < 30% → 強制的に休憩/睡眠

4. **Archetype-Based Selection (性格ベース)**
   - エージェントのアーキタイプに基づくPOI選好度
   - 時間帯による活動ボーナス/ペナルティ

### 2.2 Rule Implementation Details

#### 2.2.1 Active Events Rule
```python
# Priority 1: Handle active events
if agent.id in self.active_events:
    event_data = self.active_events[agent.id]
    forced_category = event_data["forced_poi_category"]
    # Force agent to visit specific POI category
    return forced_action_based_on_event()
```

#### 2.2.2 Time-Based Rules
```python
# Priority 2: Time-based behavior
is_night_time = current_hour >= 22 or current_hour <= 5
is_meal_time = current_hour in [12, 18]
is_early_morning = current_hour in [6, 7, 8]

# Night time sleep enforcement
if is_night_time and agent.energy < 60.0:
    return sleep_action()

# Meal time enforcement
if is_meal_time or agent.hunger > 60.0:
    return meal_action()
```

#### 2.2.3 Critical Needs Rules
```python
# Priority 3: Critical needs override
if agent.hunger > 70.0:
    return emergency_food_action()

if agent.energy < 30.0:
    return emergency_rest_action()
```

#### 2.2.4 Archetype-Based Rules
```python
# Priority 4: Archetype-based selection
poi_scores = []
for poi in available_pois:
    base_score = 1.0
    
    # Time-based modifiers
    if is_night_time and poi.category not in ["rest", "sleep"]:
        base_score *= 0.1  # Strong penalty
    elif is_early_morning and poi.category == "training":
        base_score *= 1.5  # Morning training bonus
    
    # Apply archetype preferences
    archetype_multiplier = agent.get_poi_preference_multiplier(poi.category)
    score = base_score * archetype_multiplier
    poi_scores.append((poi, score))

# Select highest scoring POI
return select_best_poi(poi_scores)
```

## 3. Agent Archetypes and Behavior Rules

### 3.1 Archetype Definitions

システムは6種類のアーキタイプを定義し、それぞれ異なる行動パターンを持ちます：

#### 3.1.1 Weapon Specialist (武器専門家)
- **Description**: Weapon enthusiast who loves training and armory work
- **Skill Multipliers**: 
  - weapon_strength: 1.5x
  - management_skill: 0.8x
  - power: 1.4x
- **Key POI Preferences**:
  - armory: 2.5x (最高優先度)
  - training: 2.0x
  - outdoor: 1.9x
  - workshop: 1.8x

#### 3.1.2 Natural Leader (天然リーダー)
- **Description**: Born leader with management and strategy focus
- **Skill Multipliers**:
  - management_skill: 1.6x
  - sociability: 1.3x
- **Key POI Preferences**:
  - office: 2.0x
  - logistics: 1.9x
  - communications: 1.8x
  - library: 1.6x

#### 3.1.3 Social Butterfly (社交的)
- **Description**: Highly social person who excels in communication
- **Skill Multipliers**:
  - sociability: 1.7x
  - management_skill: 1.2x
- **Key POI Preferences**:
  - recreation: 2.2x
  - communications: 2.1x
  - food: 1.8x (食事は社交の場)
  - spiritual: 1.5x

#### 3.1.4 Scholar (学者)
- **Description**: Intellectual type who prefers study and strategic planning
- **Skill Multipliers**:
  - management_skill: 1.4x
  - weapon_strength: 0.8x
- **Key POI Preferences**:
  - library: 2.5x (最高優先度)
  - office: 2.2x
  - communications: 1.7x
  - logistics: 1.6x

#### 3.1.5 Fitness Enthusiast (フィットネス愛好家)
- **Description**: Physical fitness focused with high energy
- **Skill Multipliers**:
  - fitness: 2.5x
  - power: 1.6x
  - weapon_strength: 1.3x
- **Key POI Preferences**:
  - fitness: 2.5x (最高優先度)
  - outdoor: 2.4x
  - training: 2.3x
  - rest: 0.6x (休憩を好まない)

#### 3.1.6 Introvert (内向的)
- **Description**: Quiet type who prefers individual work and rest
- **Skill Multipliers**:
  - sociability: 0.7x
  - weapon_strength: 1.1x
- **Key POI Preferences**:
  - rest: 2.0x
  - library: 1.9x
  - sleep: 1.8x
  - spiritual: 1.7x
  - recreation: 0.5x (社交を避ける)

### 3.2 Preference Calculation

各エージェントのPOI選択は以下の式で計算されます：

```
final_score = base_score × time_modifier × archetype_multiplier

where:
- base_score = 1.0 (基本スコア)
- time_modifier = 時間帯による修正係数
- archetype_multiplier = アーキタイプ別の選好度係数
```

## 4. Random Events System

### 4.1 Event Categories

システムは6種類のランダムイベントを定義しています：

#### 4.1.1 Injury (怪我)
- **Probability**: 2% per step
- **Duration**: 2 time steps
- **Trigger Condition**: recent_training (最近の訓練後)
- **Forced POI**: medical
- **Effects**: 
  - energy: -15.0
  - weapon_strength: -5.0
  - power: -8.0

#### 4.1.2 Illness (病気)
- **Probability**: 1.5% per step
- **Duration**: 3 time steps
- **Trigger Condition**: None (いつでも発生可能)
- **Forced POI**: medical
- **Effects**:
  - energy: -20.0
  - social: -10.0
  - power: -5.0

#### 4.1.3 Special Training (特別訓練)
- **Probability**: 3% per step
- **Duration**: 1 time step
- **Trigger Condition**: None
- **Forced POI**: outdoor
- **Effects**:
  - weapon_strength: +15.0
  - management_skill: +8.0
  - power: +20.0

#### 4.1.4 Equipment Maintenance (装備メンテナンス)
- **Probability**: 2.5% per step
- **Duration**: 1 time step
- **Trigger Condition**: None
- **Forced POI**: workshop
- **Effects**:
  - weapon_strength: +10.0
  - management_skill: +5.0
  - power: +8.0

#### 4.1.5 Communication Duty (通信当番)
- **Probability**: 2% per step
- **Duration**: 1 time step
- **Trigger Condition**: None
- **Forced POI**: communications
- **Effects**:
  - management_skill: +12.0
  - sociability: +10.0

#### 4.1.6 Stress Fatigue (ストレス疲労)
- **Probability**: 2.5% per step
- **Duration**: 2 time steps
- **Trigger Condition**: high_activity (高活動量)
- **Forced POI**: spiritual
- **Effects**:
  - energy: -10.0
  - social: -5.0
  - power: -3.0

### 4.2 Event Trigger Logic

```python
def _trigger_random_events(self) -> None:
    for agent_id, agent in self.agents.items():
        # Skip if already has active event
        if agent_id in self.active_events:
            continue
            
        for event_type, event_data in RANDOM_EVENTS.items():
            # Check probability
            if random.random() > event_data["probability"]:
                continue
                
            # Check conditions
            if not self._check_event_conditions(agent, event_data):
                continue
                
            # Trigger event
            self.active_events[agent_id] = event_data
            break  # Only one event per agent per step
```

## 5. POI Categories and Activities

### 5.1 POI Categories

システムは16種類のPOIカテゴリを定義しています：

| Category | Activity | Primary Effect | Description |
|----------|----------|----------------|-------------|
| training | train | weapon_strength+ | 基本的な軍事訓練 |
| armory | arm | weapon_strength+ | 武器の管理・メンテナンス |
| office | manage | management_skill+ | 事務作業・管理業務 |
| food | eat | hunger- | 食事・栄養補給 |
| rest | rest | energy+ | 休憩・体力回復 |
| sleep | sleep | energy++ | 睡眠・完全休息 |
| recreation | socialize | social+ | 娯楽・社交活動 |
| medical | heal | health+ | 医療・治療 |
| fitness | exercise | power+ | 体力トレーニング |
| library | study | management_skill+ | 学習・研究 |
| workshop | craft | weapon_strength+ | 工作・修理 |
| communications | communicate | management_skill+ | 通信・情報伝達 |
| maintenance | maintain | weapon_strength+ | 設備保守 |
| outdoor | outdoor_train | weapon_strength+ | 野外訓練 |
| spiritual | reflect | social+ | 精神的ケア |
| logistics | organize | management_skill+ | 補給・物資管理 |

### 5.2 Activity Mapping

各POIカテゴリは特定の活動にマッピングされます：

```python
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
```

## 6. Time-Based Behavior Rules

### 6.1 Time Periods

システムは以下の時間帯を定義しています：

#### 6.1.1 Night Time (夜間)
- **Time Range**: 22:00-05:00
- **Behavior Rules**:
  - 非休憩活動に強いペナルティ (0.1x)
  - エネルギー < 60% で強制睡眠
  - LLMプランナーは使用しない（単純な睡眠判定）

#### 6.1.2 Meal Time (食事時間)
- **Time Range**: 12:00, 18:00
- **Behavior Rules**:
  - 食事POIスコアに2.0xボーナス
  - 空腹度 > 60% で強制食事

#### 6.1.3 Early Morning (早朝)
- **Time Range**: 06:00-08:00
- **Behavior Rules**:
  - 訓練POIスコアに1.5xボーナス
  - 朝の訓練時間として優先

#### 6.1.4 Evening Recreation (夕方娯楽)
- **Time Range**: 19:00-21:00
- **Behavior Rules**:
  - 娯楽POIスコアに1.3xボーナス
  - 仕事後のリラックス時間

### 6.2 Time Modifier Implementation

```python
def apply_time_modifiers(poi_category: str, current_hour: int) -> float:
    base_score = 1.0
    
    if is_night_time(current_hour):
        if poi_category not in ["rest", "sleep"]:
            base_score *= 0.1  # Strong penalty
    elif is_early_morning(current_hour):
        if poi_category == "training":
            base_score *= 1.5  # Morning training bonus
    elif is_evening_recreation(current_hour):
        if poi_category == "recreation":
            base_score *= 1.3  # Evening recreation bonus
    
    return base_score
```

## 7. Agent Needs and Decay System

### 7.1 Need Types

エージェントは以下の需要を持ちます：

#### 7.1.1 Energy (エネルギー)
- **Decay Rate**: 0.05 per time step
- **Night Time Modifier**: 1.5x (夜間は疲労が蓄積しやすい)
- **Randomization**: 0.8-1.2x
- **Recovery**: rest, sleep POIs

#### 7.1.2 Hunger (空腹)
- **Increase Rate**: 0.08 per time step
- **Meal Time Modifier**: 1.3x (食事時間前は急激に増加)
- **Randomization**: 0.8-1.2x
- **Recovery**: food POIs

#### 7.1.3 Social (社交性)
- **Decay Rate**: 0.03 per time step
- **Personality Factor**: extroversion level affects decay rate
- **Night Time Modifier**: 0.5x (夜間は社交需要が減少)
- **Recovery**: recreation, food POIs

### 7.2 Skill Decay System

スキルは使用しないと自然に減衰します：

```python
def apply_skill_decay(self, skill: str, amount: float) -> None:
    if skill not in self._daily_activities:
        decay_resistance = self._decay_resistance[skill]
        actual_decay = amount * (1.0 - decay_resistance)
        current_value = getattr(self, skill)
        new_value = max(0.0, current_value - actual_decay)
        setattr(self, skill, new_value)
```

## 8. Fallback Mechanism

### 8.1 LLM Fallback

ルールベースシステムは、LLMが利用できない場合のフォールバックとして機能します：

```python
def _choose_action(self, agent: Agent) -> Dict[str, Any]:
    # Try LLM planner first
    if self.planner and not is_night_time:
        try:
            return self.planner.plan_action(agent_state, memories, poi_list)
        except Exception as e:
            logger.warning(f"LLM planner failed: {e}, falling back to rule-based")
    
    # Fall back to rule-based planning
    return self._choose_action_rule_based(agent)
```

### 8.2 Rule-Based Reliability

ルールベースシステムは以下の特徴を持ちます：

- **Deterministic**: 同じ条件では同じ結果
- **Fast**: 複雑な計算を必要としない
- **Reliable**: 外部API依存なし
- **Interpretable**: 行動理由が明確

## 9. Configuration and Tuning

### 9.1 Configuration Parameters

主要なパラメータは設定ファイルで調整可能です：

```yaml
# configs/default.yaml
agents:
  energy_decay_rate: 0.05
  hunger_increase_rate: 0.08
  social_need_base_rate: 0.03
  
pois:
  distribution:
    training: 0.25
    food: 0.15
    rest: 0.10
    # ... other categories
```

### 9.2 Tuning Guidelines

- **Energy Decay**: 高い値は頻繁な休憩を促進
- **Hunger Rate**: 高い値は食事POIの利用を増加
- **Archetype Multipliers**: 1.0未満で抑制、1.0超で促進
- **Time Modifiers**: 特定時間帯の行動を強化/抑制

## 10. Performance Characteristics

### 10.1 Computational Complexity

- **Time Complexity**: O(P) where P = number of POIs
- **Space Complexity**: O(A + P) where A = number of agents
- **Scalability**: 100+ agents with minimal performance impact

### 10.2 Decision Quality

- **Realistic Behavior**: 軍事環境の日常的な行動パターンを再現
- **Individual Variation**: アーキタイプによる個性の表現
- **Temporal Consistency**: 時間的制約を考慮した行動選択

## 11. Future Enhancements

### 11.1 Potential Improvements

1. **Dynamic Rule Adjustment**: 学習に基づくルール重み調整
2. **Hierarchical Rules**: より複雑な条件分岐
3. **Group Behavior**: 集団行動ルールの追加
4. **Seasonal Patterns**: 長期的な行動パターンの変化

### 11.2 Integration with LLM

ルールベースシステムは以下の方法でLLMと統合できます：

1. **Hybrid Decision Making**: LLMが複雑な判断、ルールが基本的な判断
2. **Rule Generation**: LLMがルールを生成・調整
3. **Explanation Generation**: LLMが行動理由を説明

## 12. Conclusion

このルールベースシステムは、現実的で一貫性のある軍事シミュレーションを提供します。LLMとの併用により、高度な行動生成と安定した基本動作を両立することができます。 