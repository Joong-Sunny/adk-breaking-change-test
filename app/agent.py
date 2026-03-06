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

import os

import google.auth
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.planners import BuiltInPlanner
from google.genai import types

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


# 단일 LLM 에이전트 (툴 없음). 429/503은 Gemini API 호출 시 발생하며,
# ADK/Vertex AI SDK의 재시도 로직으로 처리됩니다.
root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    instruction="학생이 어려워하는 영어질문에 대해서 친절히 알려주는 영어튜터 선생님. 학생이 어려워하는 부분을 정확히 파악하고 2~3문장 수준으로 가이드 해주세요",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=0,
        )
    ),
)

app = App(root_agent=root_agent, name="app")
