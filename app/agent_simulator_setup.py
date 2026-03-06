# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgentSimulator 설정: 429/503 에러 및 지연 시나리오 모킹."""

from google.adk.tools.agent_simulator.agent_simulator_config import (
    AgentSimulatorConfig, InjectedError, InjectionConfig, MockStrategy,
    ToolSimulationConfig)
from google.adk.tools.agent_simulator.agent_simulator_factory import \
    AgentSimulatorFactory


def create_error_simulator_config(
    *,
    enable_429: bool = True,
    enable_503: bool = True,
    enable_latency: bool = False,
    latency_seconds: float = 5.0,
    simulation_model: str = "gemini-2.5-flash",
) -> AgentSimulatorConfig:
    """429/503 에러 시뮬레이션용 AgentSimulatorConfig 생성.

    테스트 시 특정 query 값으로 에러를 트리거할 수 있습니다:
    - query="trigger_429" → 429 Resource Exhausted 에러
    - query="trigger_503" → 503 Service Unavailable 에러
    - query="trigger_latency" → 지연 시뮬레이션 (enable_latency=True일 때)

    Args:
        enable_429: 429 에러 인젝션 활성화
        enable_503: 503 에러 인젝션 활성화
        enable_latency: 지연 인젝션 활성화
        latency_seconds: 지연 시간(초)
        simulation_model: 시뮬레이터 내부 LLM 모델

    Returns:
        AgentSimulatorConfig 인스턴스
    """
    injection_configs = []

    if enable_429:
        injection_configs.append(
            InjectionConfig(
                injection_probability=1.0,
                match_args={"query": "trigger_429"},
                injected_error=InjectedError(
                    injected_http_error_code=429,
                    error_message="Resource exhausted. Rate limit exceeded. Please retry later.",
                ),
            )
        )

    if enable_503:
        injection_configs.append(
            InjectionConfig(
                injection_probability=1.0,
                match_args={"query": "trigger_503"},
                injected_error=InjectedError(
                    injected_http_error_code=503,
                    error_message="Service unavailable. The server is temporarily overloaded.",
                ),
            )
        )

    if enable_latency:
        # InjectionConfig는 injected_error 또는 injected_response 중 하나 필수
        injection_configs.append(
            InjectionConfig(
                injection_probability=1.0,
                match_args={"query": "trigger_latency"},
                injected_latency_seconds=latency_seconds,
                injected_response={
                    "status": "delayed",
                    "message": f"Simulated {latency_seconds}s latency",
                },
            )
        )

    tool_configs = [
        ToolSimulationConfig(
            tool_name="get_weather",
            injection_configs=injection_configs,
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        ToolSimulationConfig(
            tool_name="get_current_time",
            injection_configs=injection_configs,
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ]

    return AgentSimulatorConfig(
        tool_simulation_configs=tool_configs,
        simulation_model=simulation_model,
    )


def create_simulator_callback(config: AgentSimulatorConfig):
    """AgentSimulator 콜백 생성 (before_tool_callback용)."""
    return AgentSimulatorFactory.create_callback(config)


def create_simulator_plugin(config: AgentSimulatorConfig):
    """AgentSimulator 플러그인 생성 (plugins용)."""
    return AgentSimulatorFactory.create_plugin(config)
