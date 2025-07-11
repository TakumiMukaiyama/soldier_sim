# City Sim - 処理フロー詳細

## 1. システム全体の処理フロー

### 1.1 メインフロー概要

City Simの処理は以下のフェーズで構成されています。

1. **初期化フェーズ**: 設定読み込み、エージェント・POI生成、LLMクライアント初期化
2. **シミュレーション実行フェーズ**: 日次ループによる行動シミュレーション
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
        
        # 各エージェントの処理
        for agent in agents:
            # 1. 需要更新
            agent.update_needs(time_delta=1.0)
            
            # 2. 行動計画
            action = choose_action(agent)
            
            # 3. 行動実行
            execute_action(agent, action)
            
            # 4. 効果適用
            apply_poi_effects(agent, action)
            
            # 5. 信念更新
            update_poi_beliefs(agent, action)
            
            # 6. メモリ記録
            record_action(agent, action)
```

## 2. エージェント行動決定プロセス

### 2.1 LLMプランナーによる行動決定

```python
def plan_action(agent_state, reflective_memory, poi_list):
    # 1. プロンプト構築
    prompt = f"""
    エージェント状態: {agent_state}
    反射メモリ: {reflective_memory}
    利用可能POI: {poi_list}
    
    最適な行動を選択してください。
    """
    
    # 2. LLM呼び出し
    response = llm_client.invoke_with_structured_output(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        json_structure=PlanOutput.model_json_schema()
    )
    
    # 3. 構造化検証
    return PlanOutput(**response)
```

### 2.2 ルールベース行動選択（バックアップ）

```python
def choose_action_rule_based(agent):
    # 優先度順に判定
    if agent.hunger > 0.7:
        return find_food_poi()
    elif agent.energy < 0.3:
        return find_rest_poi()
    else:
        return find_training_poi()
```

## 3. メモリシステムの処理フロー

### 3.1 時系列メモリ（Temporal Memory）記録

```python
def record_action(agent, time, poi, activity_key, observation):
    # 1. データ構造化
    record = {
        "id": str(uuid4()),
        "time": time.isoformat(),
        "agent_id": agent.id,
        "poi_id": poi.id if poi else None,
        "location": poi.location if poi else agent.location,
        "activity_key": activity_key,
        "observation": observation,
        "current_energy": agent.energy,
        "current_social": agent.social,
        "current_weapon_strength": agent.weapon_strength,
        "current_hunger": agent.hunger,
        "current_management_skill": agent.management_skill,
    }
    
    # 2. バッファに追加
    temporal_memory_buffer.append(record)
    
    # 3. バッファサイズ制御
    if len(temporal_memory_buffer) >= 100:
        flush_buffer_to_dataframe()
```

### 3.2 反射メモリ（Reflective Memory）生成

```python
def generate_daily_summary(agent_id, date):
    # 1. 日次データ抽出
    daily_data = temporal_memory.filter(
        (pl.col("agent_id") == agent_id) & 
        (pl.col("time").str.contains(date))
    )
    
    # 2. 統計計算
    summary = {
        "date": date,
        "activity_count": len(daily_data),
        "most_visited_poi": daily_data.groupby("poi_id").count().sort("count", descending=True).head(1),
        "energy_stats": {
            "start": daily_data["current_energy"].first(),
            "end": daily_data["current_energy"].last(),
            "avg": daily_data["current_energy"].mean(),
            "min": daily_data["current_energy"].min(),
            "max": daily_data["current_energy"].max(),
        },
        "skill_changes": {
            "weapon_strength": daily_data["current_weapon_strength"].last() - daily_data["current_weapon_strength"].first(),
            "management_skill": daily_data["current_management_skill"].last() - daily_data["current_management_skill"].first(),
        }
    }
    
    # 3. 反射メモリに保存
    reflective_memory[agent_id][date] = summary
    return summary
```

## 4. カルマンフィルタによる信念更新

### 4.1 信念更新プロセス

```python
def update_poi_beliefs(agent, action):
    poi = get_poi(action.poi_id)
    
    # 1. 観測データ生成
    observation = generate_observation(agent, poi, action)
    
    # 2. 各信念次元の更新
    for belief_dim in ["satisfaction", "convenience", "atmosphere", "effectiveness"]:
        current_belief = poi.belief[belief_dim]
        current_sigma = poi.belief_sigma[belief_dim]
        observed_value = observation[belief_dim]
        
        # 3. カルマンフィルタ適用
        updated_belief, updated_sigma = kalman_update(
            belief=current_belief,
            sigma_b=current_sigma,
            observation=observed_value,
            sigma_o=0.2  # 観測ノイズ
        )
        
        # 4. 信念更新
        poi.belief[belief_dim] = updated_belief
        poi.belief_sigma[belief_dim] = updated_sigma
```

### 4.2 観測データ生成

```python
def generate_observation(agent, poi, action):
    # POI効果とエージェント状態から観測値を計算
    effects = poi.get_effects()
    
    observation = {
        "satisfaction": calculate_satisfaction(effects, agent),
        "convenience": calculate_convenience(poi, agent),
        "atmosphere": calculate_atmosphere(poi, agent),
        "effectiveness": calculate_effectiveness(effects, action)
    }
    
    return observation
```

## 5. データ処理パイプライン

### 5.1 リアルタイム処理

```python
# 1. 行動記録（バッファ使用）
memory.record_action(agent, time, poi, activity, observation)

# 2. バッファ管理
if len(memory._temporal_memory_buffer) >= 100:
    memory._flush_buffer()

# 3. 集計処理（日次）
daily_summary = memory.generate_daily_summary(agent.id, current_date)
```

### 5.2 バッチ処理

```python
# 1. データ集約
aggregated_data = temporal_memory.groupby(["agent_id", "date"]).agg([
    pl.col("current_energy").mean().alias("avg_energy"),
    pl.col("current_weapon_strength").last().alias("final_weapon_strength"),
    pl.col("activity_key").count().alias("activity_count")
])

# 2. 出力生成
aggregated_data.write_parquet("output/daily_aggregates.parquet")
```

## 6. 出力データ生成

### 6.1 ログ出力

```python
def save_results(results, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 実行ログ
    with open(output_path / "logs.json", "w") as f:
        json.dump(results["logs"], f, indent=2, default=str)
    
    # 2. 時系列メモリ（高速アクセス用）
    results["temporal_memory"].write_parquet(output_path / "temporal_memory.parquet")
    
    # 3. 反射メモリ
    with open(output_path / "reflective_memory.json", "w") as f:
        json.dump(results["reflective_memory"], f, indent=2, default=str)
```

### 6.2 分析用データ生成

```python
# 1. エージェント別スキル成長
skill_growth = temporal_memory.groupby("agent_id").agg([
    (pl.col("current_weapon_strength").last() - pl.col("current_weapon_strength").first()).alias("weapon_growth"),
    (pl.col("current_management_skill").last() - pl.col("current_management_skill").first()).alias("mgmt_growth")
])

# 2. POI使用統計
poi_usage = temporal_memory.groupby("poi_id").agg([
    pl.col("agent_id").count().alias("visit_count"),
    pl.col("current_energy").mean().alias("avg_energy_effect")
])

# 3. 時系列分析
energy_trends = temporal_memory.groupby_dynamic("time", every="1d").agg([
    pl.col("current_energy").mean().alias("avg_energy"),
    pl.col("current_hunger").mean().alias("avg_hunger")
])
```

## 7. エラーハンドリングと安定性

### 7.1 LLM呼び出しエラー処理

```python
def safe_plan_action(agent_state, reflective_memory, poi_list):
    try:
        # LLMプランナー試行
        return planner.plan_action(agent_state, reflective_memory, poi_list)
    except Exception as e:
        # フォールバック: ルールベース
        logger.warning(f"LLM planning failed: {e}, falling back to rule-based")
        return rule_based_planner.plan_action(agent_state, poi_list)
```

### 7.2 データ整合性チェック

```python
def validate_simulation_state():
    # 1. エージェント状態検証
    for agent in agents:
        assert 0.0 <= agent.energy <= 1.0, f"Invalid energy: {agent.energy}"
        assert 0.0 <= agent.hunger <= 1.0, f"Invalid hunger: {agent.hunger}"
    
    # 2. POI信念検証
    for poi in pois:
        for belief_val in poi.belief.values():
            assert 0.0 <= belief_val <= 1.0, f"Invalid belief: {belief_val}"
    
    # 3. メモリ整合性検証
    assert len(memory.temporal_memory) > 0, "No temporal memory records"
```

## 8. 性能最適化

### 8.1 Polarsによる高速処理

```python
# 1. 効率的な集約
daily_stats = temporal_memory.lazy().groupby(["agent_id", "date"]).agg([
    pl.col("current_energy").mean(),
    pl.col("activity_key").count()
]).collect()

# 2. 並列処理
with pl.Config(streaming_chunk_size=10000):
    result = temporal_memory.lazy().filter(
        pl.col("time") >= start_date
    ).select([
        "agent_id", "current_energy", "activity_key"
    ]).collect()
```

### 8.2 メモリ使用量最適化

```python
# 1. バッファサイズ管理
BUFFER_SIZE = 100
if len(temporal_memory_buffer) >= BUFFER_SIZE:
    flush_buffer()

# 2. 古いデータのアーカイブ
if simulation_day > 30:
    archive_old_data(simulation_day - 30)
```

この処理フローにより、大規模なマルチエージェントシミュレーションを効率的かつ安定的に実行できます。各コンポーネントの相互作用とデータフローが明確に定義されており、拡張性と保守性を確保しています。 