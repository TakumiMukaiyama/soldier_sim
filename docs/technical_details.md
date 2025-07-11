# City Sim - 技術的詳細

## 1. 技術的アーキテクチャ詳細

### 1.1 Polarsによる高性能データ処理

#### 1.1.1 メモリ効率的なデータ構造

```python
# 最適化されたスキーマ定義
temporal_memory_schema = {
    "id": pl.Utf8,
    "time": pl.Utf8,
    "agent_id": pl.Utf8,
    "poi_id": pl.Utf8,
    "location": pl.List(pl.Float64),
    "activity_key": pl.Utf8,
    "observation": pl.Struct([
        pl.Field("activity", pl.Utf8),
        pl.Field("poi_id", pl.Utf8),
        pl.Field("reason", pl.Utf8),
        pl.Field("expected_duration", pl.Int64),
    ]),
    "current_energy": pl.Float64,
    "current_social": pl.Float64,
    "current_weapon_strength": pl.Float64,
    "current_hunger": pl.Float64,
    "current_management_skill": pl.Float64,
}
```

#### 1.1.2 遅延評価（Lazy Evaluation）の活用

```python
# 効率的な集約処理
def analyze_agent_performance(temporal_memory: pl.DataFrame, agent_id: str) -> pl.DataFrame:
    return (
        temporal_memory.lazy()
        .filter(pl.col("agent_id") == agent_id)
        .groupby_dynamic("time", every="1d")
        .agg([
            pl.col("current_energy").mean().alias("avg_energy"),
            pl.col("current_weapon_strength").last().alias("final_weapon_strength"),
            pl.col("activity_key").count().alias("activity_count"),
            pl.col("poi_id").n_unique().alias("unique_pois_visited")
        ])
        .collect()
    )
```

#### 1.1.3 並列処理の最適化

```python
# 大規模データセットの効率的な処理
def process_large_dataset(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.lazy()
        .with_columns([
            pl.col("time").str.to_datetime().alias("datetime"),
            pl.col("current_energy").rolling_mean(window_size=24).alias("energy_24h_avg")
        ])
        .groupby(["agent_id", pl.col("datetime").dt.date()])
        .agg([
            pl.col("current_energy").mean().alias("daily_avg_energy"),
            pl.col("current_energy").std().alias("daily_energy_variance"),
            pl.col("activity_key").value_counts().alias("activity_distribution")
        ])
        .collect()
    )
```

### 1.2 Pydantic型安全システム

#### 1.2.1 設定管理

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False
    )
    
    # LLM設定
    gemini_api_key: SecretStr = Field(..., description="Gemini API Key")
    azure_api_key: SecretStr = Field(..., description="Azure OpenAI API Key")
    azure_endpoint: str = Field(..., description="Azure OpenAI Endpoint")
    azure_deployment: str = Field(..., description="Azure OpenAI Deployment Name")
    azure_api_version: str = Field(default="2024-02-01", description="Azure OpenAI API Version")
    
    # シミュレーション設定
    simulation_days: int = Field(default=100, ge=1, le=365, description="シミュレーション日数")
    time_steps_per_day: int = Field(default=8, ge=1, le=24, description="1日あたりの時間ステップ数")
    agent_count: int = Field(default=100, ge=1, le=1000, description="エージェント数")
    poi_count: int = Field(default=20, ge=1, le=100, description="POI数")
    
    # LLM設定
    gemini_model: str = Field(default="gemini-2.0-flash-exp", description="Gemini model name")
    max_tokens: int = Field(default=1000, ge=100, le=4000, description="最大トークン数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for LLM")
    
    # 出力設定
    output_dir: str = Field(default="output", description="出力ディレクトリ")
    log_level: str = Field(default="INFO", description="ログレベル")
```

#### 1.2.2 構造化出力スキーマ

```python
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from enum import Enum

class ActivityType(str, Enum):
    TRAIN = "train"
    EAT = "eat"
    REST = "rest"
    MANAGE = "manage"
    ARM = "arm"
    SOCIALIZE = "socialize"

class PlanOutput(BaseModel):
    agent_id: str = Field(..., description="エージェントの一意なID")
    chosen_poi: str = Field(..., description="次に訪問するPOIのID")
    activity: ActivityType = Field(..., description="実施するアクティビティ")
    expected_duration: int = Field(..., ge=1, le=8, description="活動予定時間（1-8時間）")
    reason: str = Field(..., min_length=10, max_length=200, description="行動選択の理由")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="行動の優先度")
    
    @validator('reason')
    def validate_reason(cls, v):
        if not v.strip():
            raise ValueError('理由は空白ではいけません')
        return v.strip()
    
    @validator('chosen_poi')
    def validate_poi_format(cls, v):
        if not v.startswith(('training_', 'food_', 'rest_', 'office_', 'armory_', 'recreation_')):
            raise ValueError('POI IDは適切なプレフィックスを持つ必要があります')
        return v

class ObservationData(BaseModel):
    satisfaction: float = Field(..., ge=0.0, le=1.0, description="満足度")
    convenience: float = Field(..., ge=0.0, le=1.0, description="利便性")
    atmosphere: float = Field(..., ge=0.0, le=1.0, description="雰囲気")
    effectiveness: float = Field(..., ge=0.0, le=1.0, description="効果性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "satisfaction": 0.8,
                "convenience": 0.6,
                "atmosphere": 0.7,
                "effectiveness": 0.9
            }
        }
```

### 1.3 カルマンフィルタの数学的実装

#### 1.3.1 拡張カルマンフィルタ

```python
import numpy as np
from typing import Tuple, Dict, Any

class ExtendedKalmanFilter:
    """多次元状態空間でのカルマンフィルタ実装"""
    
    def __init__(self, dim: int, process_noise: float = 0.1, measurement_noise: float = 0.2):
        self.dim = dim
        self.state = np.zeros(dim)  # 状態ベクトル
        self.covariance = np.eye(dim) * 0.5  # 共分散行列
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def predict(self, dt: float = 1.0) -> None:
        """予測ステップ"""
        # 状態遷移行列（単位行列：状態は保持される）
        F = np.eye(self.dim)
        
        # プロセスノイズ共分散行列
        Q = np.eye(self.dim) * self.process_noise * dt
        
        # 予測
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
    
    def update(self, measurement: np.ndarray, measurement_matrix: np.ndarray = None) -> None:
        """更新ステップ"""
        if measurement_matrix is None:
            measurement_matrix = np.eye(self.dim)
        
        H = measurement_matrix
        R = np.eye(measurement.shape[0]) * self.measurement_noise
        
        # イノベーション
        y = measurement - H @ self.state
        S = H @ self.covariance @ H.T + R
        
        # カルマンゲイン
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # 更新
        self.state = self.state + K @ y
        self.covariance = (np.eye(self.dim) - K @ H) @ self.covariance
    
    def get_state(self) -> Dict[str, float]:
        """現在の状態を辞書形式で返す"""
        return {
            "satisfaction": float(self.state[0]),
            "convenience": float(self.state[1]),
            "atmosphere": float(self.state[2]),
            "effectiveness": float(self.state[3])
        }
    
    def get_uncertainty(self) -> Dict[str, float]:
        """不確実性を辞書形式で返す"""
        return {
            "satisfaction": float(np.sqrt(self.covariance[0, 0])),
            "convenience": float(np.sqrt(self.covariance[1, 1])),
            "atmosphere": float(np.sqrt(self.covariance[2, 2])),
            "effectiveness": float(np.sqrt(self.covariance[3, 3]))
        }
```

#### 1.3.2 適応的ノイズ調整

```python
class AdaptiveKalmanFilter(ExtendedKalmanFilter):
    """適応的ノイズ調整機能付きカルマンフィルタ"""
    
    def __init__(self, dim: int, initial_process_noise: float = 0.1, 
                 initial_measurement_noise: float = 0.2):
        super().__init__(dim, initial_process_noise, initial_measurement_noise)
        self.innovation_history = []
        self.adaptation_window = 10
    
    def update(self, measurement: np.ndarray, measurement_matrix: np.ndarray = None) -> None:
        """適応的更新"""
        if measurement_matrix is None:
            measurement_matrix = np.eye(self.dim)
        
        H = measurement_matrix
        
        # イノベーション計算
        y = measurement - H @ self.state
        self.innovation_history.append(np.linalg.norm(y))
        
        # 履歴を制限
        if len(self.innovation_history) > self.adaptation_window:
            self.innovation_history.pop(0)
        
        # 適応的ノイズ調整
        if len(self.innovation_history) >= self.adaptation_window:
            recent_innovation = np.mean(self.innovation_history[-5:])
            historical_innovation = np.mean(self.innovation_history[:-5])
            
            if recent_innovation > historical_innovation * 1.5:
                # 測定ノイズを増加
                self.measurement_noise *= 1.1
            elif recent_innovation < historical_innovation * 0.5:
                # 測定ノイズを減少
                self.measurement_noise *= 0.9
        
        # 通常の更新処理
        super().update(measurement, measurement_matrix)
```

### 1.4 LLMクライアントの実装詳細

#### 1.4.1 エラーハンドリングとリトライ機構

```python
import asyncio
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

class RobustLLMClient:
    """堅牢なLLMクライアント実装"""
    
    def __init__(self, client, max_retries: int = 3, base_delay: float = 1.0):
        self.client = client
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def invoke_with_structured_output(self, system_prompt: str, user_prompt: str, 
                                    json_structure: Dict[str, Any]) -> Dict[str, Any]:
        """構造化出力でのLLM呼び出し"""
        try:
            # プロンプトの最適化
            optimized_prompt = self._optimize_prompt(system_prompt, user_prompt, json_structure)
            
            # LLM呼び出し
            response = self.client.chat.completions.create(
                model=self.client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": optimized_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # レスポンス解析
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # 構造検証
            self._validate_response(result, json_structure)
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析エラー: {e}")
            raise
        except Exception as e:
            self.logger.error(f"LLM呼び出しエラー: {e}")
            raise
    
    def _optimize_prompt(self, system_prompt: str, user_prompt: str, 
                        json_structure: Dict[str, Any]) -> str:
        """プロンプトの最適化"""
        schema_example = self._generate_schema_example(json_structure)
        
        optimized = f"""
{user_prompt}

以下のJSON形式で回答してください。他の形式は受け付けません。

{json.dumps(schema_example, indent=2, ensure_ascii=False)}

重要な注意事項
1. 必ず有効なJSONを返してください
2. 全ての必須フィールドを含めてください
3. 値の型は指定されたものと一致させてください
4. 追加のテキストや説明は含めないでください
"""
        return optimized
    
    def _generate_schema_example(self, json_structure: Dict[str, Any]) -> Dict[str, Any]:
        """JSONスキーマから例を生成"""
        example = {}
        properties = json_structure.get("properties", {})
        
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            
            if field_type == "string":
                if "enum" in field_info:
                    example[field_name] = field_info["enum"][0]
                elif field_name == "agent_id":
                    example[field_name] = "agent_1"
                elif field_name == "chosen_poi":
                    example[field_name] = "training_ground_1"
                elif field_name == "reason":
                    example[field_name] = "適切な理由を記載"
                else:
                    example[field_name] = "example_value"
            elif field_type == "integer":
                example[field_name] = 2
            elif field_type == "number":
                example[field_name] = 0.5
            elif field_type == "boolean":
                example[field_name] = True
        
        return example
    
    def _validate_response(self, response: Dict[str, Any], 
                          json_structure: Dict[str, Any]) -> None:
        """レスポンスの検証"""
        required_fields = json_structure.get("required", [])
        properties = json_structure.get("properties", {})
        
        # 必須フィールドの確認
        for field in required_fields:
            if field not in response:
                raise ValueError(f"必須フィールド '{field}' が見つかりません")
        
        # 型の確認
        for field_name, field_value in response.items():
            if field_name in properties:
                expected_type = properties[field_name].get("type")
                if expected_type == "string" and not isinstance(field_value, str):
                    raise ValueError(f"フィールド '{field_name}' は文字列である必要があります")
                elif expected_type == "integer" and not isinstance(field_value, int):
                    raise ValueError(f"フィールド '{field_name}' は整数である必要があります")
                elif expected_type == "number" and not isinstance(field_value, (int, float)):
                    raise ValueError(f"フィールド '{field_name}' は数値である必要があります")
```

#### 1.4.2 非同期処理による高速化

```python
import asyncio
import aiohttp
from typing import List, Dict, Any

class AsyncLLMClient:
    """非同期LLMクライアント"""
    
    def __init__(self, api_key: str, base_url: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def batch_plan_actions(self, agent_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """バッチでのアクション計画"""
        tasks = []
        
        for request in agent_requests:
            task = self._plan_single_action(request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # エラーハンドリング
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # フォールバック処理
                processed_results.append(self._fallback_plan(agent_requests[i]))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _plan_single_action(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """単一エージェントのアクション計画"""
        async with self.semaphore:
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {"role": "system", "content": "エージェントプランナーとして行動してください"},
                            {"role": "user", "content": self._build_prompt(request)}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    },
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        return json.loads(content)
                    else:
                        raise aiohttp.ClientError(f"HTTP {response.status}")
                        
            except Exception as e:
                logging.error(f"非同期LLM呼び出しエラー: {e}")
                raise
    
    def _fallback_plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """フォールバック計画"""
        agent_state = request["agent_state"]
        
        # 簡単なルールベース判定
        if agent_state.get("hunger", 0) > 0.7:
            activity = "eat"
            poi_type = "food"
        elif agent_state.get("energy", 1) < 0.3:
            activity = "rest"
            poi_type = "rest"
        else:
            activity = "train"
            poi_type = "training"
        
        return {
            "agent_id": agent_state["id"],
            "chosen_poi": f"{poi_type}_1",
            "activity": activity,
            "expected_duration": 2,
            "reason": "フォールバック計画による決定"
        }
```

### 1.5 性能プロファイリングと最適化

#### 1.5.1 プロファイリング機能

```python
import time
import memory_profiler
from functools import wraps
from typing import Callable, Any

class PerformanceProfiler:
    """パフォーマンスプロファイリング"""
    
    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
    
    def profile_time(self, func_name: str = None):
        """実行時間のプロファイリング"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    
                    if name not in self.timing_data:
                        self.timing_data[name] = []
                    self.timing_data[name].append(execution_time)
                    
                    if execution_time > 1.0:  # 1秒以上の場合は警告
                        logging.warning(f"{name} took {execution_time:.2f} seconds")
            
            return wrapper
        return decorator
    
    def profile_memory(self, func_name: str = None):
        """メモリ使用量のプロファイリング"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                
                # メモリ使用量計測
                mem_before = memory_profiler.memory_usage()[0]
                result = func(*args, **kwargs)
                mem_after = memory_profiler.memory_usage()[0]
                
                memory_used = mem_after - mem_before
                
                if name not in self.memory_data:
                    self.memory_data[name] = []
                self.memory_data[name].append(memory_used)
                
                if memory_used > 100:  # 100MB以上の場合は警告
                    logging.warning(f"{name} used {memory_used:.2f} MB")
                
                return result
            
            return wrapper
        return decorator
    
    def get_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""
        report = {
            "timing": {},
            "memory": {}
        }
        
        # 実行時間統計
        for func_name, times in self.timing_data.items():
            report["timing"][func_name] = {
                "calls": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "max_time": max(times),
                "min_time": min(times)
            }
        
        # メモリ使用量統計
        for func_name, memories in self.memory_data.items():
            report["memory"][func_name] = {
                "calls": len(memories),
                "total_memory": sum(memories),
                "avg_memory": sum(memories) / len(memories),
                "max_memory": max(memories),
                "min_memory": min(memories)
            }
        
        return report
```

#### 1.5.2 最適化された実装例

```python
# プロファイラーのインスタンス
profiler = PerformanceProfiler()

@profiler.profile_time("simulation_step")
@profiler.profile_memory("simulation_step")
def optimized_simulation_step(simulation):
    """最適化されたシミュレーションステップ"""
    # バッチ処理でエージェントの需要を更新
    agents = list(simulation.agents.values())
    
    # 並列処理での需要更新
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(agent.update_needs, 1.0) for agent in agents]
        for future in futures:
            future.result()
    
    # 非同期でのLLM呼び出し
    if simulation.planner:
        agent_requests = []
        for agent in agents:
            request = {
                "agent_state": agent.to_dict(),
                "reflective_memory": simulation.memory.get_reflective_memory(agent.id),
                "poi_list": [poi.to_dict() for poi in simulation.pois.values()]
            }
            agent_requests.append(request)
        
        # バッチ処理
        actions = asyncio.run(simulation.planner.batch_plan_actions(agent_requests))
    else:
        actions = [simulation._choose_action(agent) for agent in agents]
    
    # 行動実行
    for agent, action in zip(agents, actions):
        simulation._execute_action(agent, action)
    
    return {
        "processed_agents": len(agents),
        "timestamp": time.time()
    }
```

### 1.6 デバッグとトラブルシューティング

#### 1.6.1 デバッグ用ログ設定

```python
import logging
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """ログ設定の初期化"""
    
    # ログレベル設定
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'無効なログレベル: {log_level}')
    
    # フォーマッタ設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ハンドラー設定
    handlers = []
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # ファイルハンドラー
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # ルートロガー設定
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True
    )
    
    # 外部ライブラリのログレベル調整
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
```

#### 1.6.2 状態検証とアサーション

```python
def validate_simulation_state(simulation):
    """シミュレーション状態の検証"""
    
    # エージェント状態検証
    for agent in simulation.agents.values():
        assert 0.0 <= agent.energy <= 1.0, f"Agent {agent.id}: Invalid energy {agent.energy}"
        assert 0.0 <= agent.hunger <= 1.0, f"Agent {agent.id}: Invalid hunger {agent.hunger}"
        assert 0.0 <= agent.social <= 1.0, f"Agent {agent.id}: Invalid social {agent.social}"
        assert 0.0 <= agent.weapon_strength <= 1.0, f"Agent {agent.id}: Invalid weapon_strength {agent.weapon_strength}"
        assert 0.0 <= agent.management_skill <= 1.0, f"Agent {agent.id}: Invalid management_skill {agent.management_skill}"
    
    # POI状態検証
    for poi in simulation.pois.values():
        for belief_dim, belief_val in poi.belief.items():
            assert 0.0 <= belief_val <= 1.0, f"POI {poi.id}: Invalid belief {belief_dim}={belief_val}"
        
        for sigma_dim, sigma_val in poi.belief_sigma.items():
            assert sigma_val >= 0.0, f"POI {poi.id}: Invalid sigma {sigma_dim}={sigma_val}"
    
    # メモリ整合性検証
    assert len(simulation.memory.temporal_memory) >= 0, "Temporal memory should not be negative"
    
    # 時間整合性検証
    assert simulation.current_day >= 0, f"Invalid current_day: {simulation.current_day}"
    assert 0 <= simulation.current_step < simulation.time_steps_per_day, f"Invalid current_step: {simulation.current_step}"
    
    logging.info("シミュレーション状態の検証が完了しました")
```

## 2. 運用・保守のベストプラクティス

### 2.1 監視とアラート

```python
class SimulationMonitor:
    """シミュレーション監視"""
    
    def __init__(self):
        self.metrics = {
            "simulation_errors": 0,
            "llm_failures": 0,
            "memory_usage_peak": 0,
            "processing_time_avg": 0
        }
    
    def check_health(self, simulation) -> Dict[str, Any]:
        """ヘルスチェック"""
        health_status = {
            "status": "healthy",
            "issues": []
        }
        
        # メモリ使用量チェック
        current_memory = memory_profiler.memory_usage()[0]
        if current_memory > 4000:  # 4GB以上
            health_status["issues"].append("High memory usage")
            health_status["status"] = "warning"
        
        # エラー率チェック
        error_rate = self.metrics["simulation_errors"] / max(1, simulation.current_step)
        if error_rate > 0.1:  # 10%以上
            health_status["issues"].append("High error rate")
            health_status["status"] = "critical"
        
        return health_status
```

### 2.2 設定管理

```python
def load_environment_specific_config(env: str = "development"):
    """環境別設定の読み込み"""
    config_files = {
        "development": "configs/development.yaml",
        "staging": "configs/staging.yaml",
        "production": "configs/production.yaml"
    }
    
    config_file = config_files.get(env, "configs/default.yaml")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 環境変数でのオーバーライド
    for key, value in os.environ.items():
        if key.startswith("CITYSIM_"):
            config_key = key[8:].lower()  # CITYSIM_プレフィックスを除去
            config[config_key] = value
    
    return config
```

この技術文書により、City Simシステムの実装、最適化、運用に関する詳細な技術的知識を提供します。これらの実装により、高性能で信頼性の高いマルチエージェントシミュレーションシステムを構築できます。 