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

import datetime
import os
from zoneinfo import ZoneInfo

import google.auth
from google.adk.agents import Agent
from google.adk.apps.app import App

from .agent_simulator_setup import (
    create_error_simulator_config,
    create_simulator_callback,
)

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


# 429/503 에러 처리 가이드라인 (LLM instruction)
ERROR_HANDLING_INSTRUCTION = """

When a tool returns an error response with error_code and error_message:
- **429 (Resource Exhausted)**: Inform the user that the service is rate-limited. Suggest waiting a moment and retrying. Be polite and helpful.
- **503 (Service Unavailable)**: Inform the user that the service is temporarily overloaded. Suggest trying again in a few minutes.
- For both cases: Do not panic, provide a clear explanation, and offer actionable next steps.
"""


def _build_root_agent():
    """root_agent 생성. ENABLE_ERROR_SIMULATOR=true 시 429/503 모킹 활성화."""
    use_simulator = os.environ.get("ENABLE_ERROR_SIMULATOR", "").lower() in (
        "1",
        "true",
        "yes",
    )

    agent_kwargs = dict(
        name="root_agent",
        model="gemini-2.5-flash",
        instruction=(
            "학생이 어려워하는 영어질문에 대해서 친절히 알려주는 영어튜터 선생님. 학생이 어려워하는 부분을 정확히 파악하고 2~3문장 수준으로 가이드 해주세요"
            + ERROR_HANDLING_INSTRUCTION
        ),
    )

    if use_simulator:
        config = create_error_simulator_config(
            enable_429=True,
            enable_503=True,
            enable_latency=False,
        )
        agent_kwargs["before_tool_callback"] = create_simulator_callback(config)

    return Agent(**agent_kwargs)


root_agent = _build_root_agent()
app = App(root_agent=root_agent, name="app")
