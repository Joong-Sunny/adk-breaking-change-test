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

import logging
import os
from typing import Optional

import google.auth
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App
from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.planners import BuiltInPlanner
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .error_mocking_model import MockErrorModel

try:
    import google.cloud.logging
    google.cloud.logging.Client().setup_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

logger.info("===========================")
logger.info("GOOGLE_CLOUD_PROJECT: %s", os.getenv("GOOGLE_CLOUD_PROJECT"))
logger.info("GOOGLE_CLOUD_LOCATION: %s", os.getenv("GOOGLE_CLOUD_LOCATION"))
logger.info("GOOGLE_GENAI_USE_VERTEXAI: %s", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))
logger.info("===========================")
logger.info("OPENAI_API_KEY: %s", os.getenv("OPENAI_API_KEY"))
logger.info("===========================")

async def log_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    logger.info("[BEFORE_MODEL] 모델 호출 직전 — before_model_callback 진입")
    return None


def log_after_model(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    logger.info("[AFTER_MODEL] 모델 응답 수신 — after_model_callback 진입")
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
            logger.error("[CRITICAL] Gemini API 503 서버 에러 발생! 상세내용: %s", error_msg)
            return None

        # 429 에러 발생 시: OpenAI로 Fallback 처리
        if "429" in error_msg or "Resource Exhausted" in error_msg:
            logger.warning("[WARNING] 429 에러 발생! OpenAI 모델로 Fallback을 시도합니다.")
            try:
                # gemini-3-flash-preview
                # claude-haiku-4-5
                # openai/gpt-4o
                os.environ["VERTEXAI_PROJECT"] = "ai-lamp-dev-479401"
                os.environ["VERTEXAI_LOCATION"] = "global"

                fallback_model = "vertex_ai/gemini-3-flash-preview"
                openai_model = LiteLlm(model=fallback_model)
                llm_request.model = fallback_model

                response = None
                async for chunk in openai_model.generate_content_async(
                    llm_request=llm_request, stream=False
                ):
                    response = chunk

                logger.info("OpenAI에 최종적으로 들어간 내용: %s", llm_request)
                logger.info("OpenAI Fallback 응답: %s", response)

                return response
            except Exception as fallback_error:
                logger.error("OpenAI Fallback 실패: %s", fallback_error)
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
