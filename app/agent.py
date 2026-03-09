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
from typing import Optional

import google.auth
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App
from google.adk.models import LlmRequest, LlmResponse
from google.adk.planners import BuiltInPlanner
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .error_mocking_model import MockErrorModel

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


async def log_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    print("📍 [BEFORE_MODEL] 모델 호출 직전 — before_model_callback 진입")
    return None


def log_after_model(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    print("📍 [AFTER_MODEL] 모델 응답 수신 — after_model_callback 진입")
    return llm_response


class DefenseGuardPlugin(BasePlugin):
    """API 에러 방어 및 Fallback을 처리하는 커스텀 플러그인"""

    def __init__(self) -> None:
        super().__init__(name="defense_guard")

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:

        error_msg = str(error)

        # 503 에러 발생 시: 특별한 로깅 후 예외를 그대로 발생시킴
        if "503" in error_msg or "Internal Server Error" in error_msg:
            print(f"🚨 [CRITICAL] Gemini API 503 서버 에러 발생! 상세내용: {error_msg}")
            return None

        # 429 에러 발생 시: OpenAI로 Fallback 처리
        if "429" in error_msg or "Resource Exhausted" in error_msg:
            print("⚠️ [WARNING] 429 에러 발생! OpenAI 모델로 Fallback을 시도합니다.")
            try:
                # 가상의 OpenAI 호출 로직 (실제 프로젝트에 맞게 수정 필요)
                # openai_text = await call_openai_api(llm_request.prompt)
                openai_text = "이것은 OpenAI를 통해 받아온 Fallback 응답입니다."

                # ADK가 이해할 수 있는 형태로 포장하여 반환 (예외 억제)
                return LlmResponse(
                    content=types.Content(
                        role="model", parts=[types.Part.from_text(text=openai_text)]
                    )
                )
            except Exception as fallback_error:
                print(f"❌ OpenAI Fallback 실패: {fallback_error}")
                return None

        return None


# 단일 LLM 에이전트 (툴 없음). 429/503은 Gemini API 호출 시 발생하며,
# ADK/Vertex AI SDK의 재시도 로직으로 처리됩니다.
root_agent = Agent(
    name="root_agent",
    model=MockErrorModel(target_error_code=429),
    instruction="학생이 어려워하는 영어질문에 대해서 친절히 알려주는 영어튜터 선생님. 학생이 어려워하는 부분을 정확히 파악하고 2~3문장 수준으로 가이드 해주세요",
    before_model_callback=log_before_model,
    after_model_callback=log_after_model,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=0,
        )
    ),
)

app = App(root_agent=root_agent, name="app", plugins=[DefenseGuardPlugin()])
