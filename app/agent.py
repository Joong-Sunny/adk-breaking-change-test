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
import sys
import time
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
    # StructuredLogHandler: stdout에 JSON 출력 → Agent Engine이 Cloud Logging으로 라우팅
    # CloudLoggingHandler(setup_logging) 대신 사용 — 쓰레드 문제 없음
    from google.cloud.logging_v2.handlers import StructuredLogHandler

    _handler = StructuredLogHandler()
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(_handler)
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
logger.info("GEMINI_API_KEY: %s", os.getenv("GEMINI_API_KEY"))
logger.info("===========================")

async def log_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    print("[BEFORE_MODEL] 모델 호출 직전 — before_model_callback 진입", file=sys.stdout)
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
            print("[error] 503 에러발생, 기존 에러 그대로 전파", error_msg)
            return None

        # 429 에러 발생 시: Fallback 처리
        # 순서: AI Studio gemini-2.5-flash → Vertex AI gemini-3-flash-preview → OpenAI gpt-4o
        if "429" in error_msg or "Resource Exhausted" in error_msg:
            logger.warning("[WARNING] 429 에러 발생! 다른 모델로 시도합니다.")
            print("[warning] 429 에러발생, fallback 시도")

            # fallback 1: Google AI Studio gemini-2.5-flash (Vertex AI와 완전히 독립된 인프라)
            try:
                fallback1_model = "gemini/gemini-2.5-flash"
                fallback1 = LiteLlm(model=fallback1_model)
                llm_request.model = fallback1_model

                response = None
                _t1 = time.time()
                print("[warning] fallback 1차시도: AI Studio gemini-2.5-flash")
                async for chunk in fallback1.generate_content_async(
                    llm_request=llm_request, stream=False
                ):
                    response = chunk
                print(
                    f"[warning] fallback 1차 성공: AI Studio gemini-2.5-flash ({time.time() - _t1:.2f}s)"
                )
                return response
            except Exception as fallback1_error:
                print(
                    f"[error] AI Studio gemini-2.5-flash 시도 실패 ({time.time() - _t1:.2f}s): {fallback1_error}"
                )
                logger.error(
                    "[error] AI Studio gemini-2.5-flash 시도 실패: %s", fallback1_error
                )

            # fallback 2: Vertex AI gemini-3-flash-preview (다른 모델, 별도 수요 풀)
            try:
                fallback2_model = "vertex_ai/gemini-3-flash-preview"
                fallback2 = LiteLlm(model=fallback2_model)
                llm_request.model = fallback2_model

                response = None
                _t2 = time.time()
                print("[warning] fallback 2차 시도: Vertex AI gemini-3-flash-preview")
                async for chunk in fallback2.generate_content_async(
                    llm_request=llm_request, stream=False
                ):
                    response = chunk
                print(
                    f"[warning] fallback 2차 성공: Vertex AI gemini-3-flash-preview ({time.time() - _t2:.2f}s)"
                )
                return response
            except Exception as fallback2_error:
                print(
                    f"[error] Vertex AI gemini-3-flash-preview 시도 실패 ({time.time() - _t2:.2f}s): {fallback2_error}"
                )
                logger.error(
                    "[error] Vertex AI gemini-3-flash-preview 시도 실패: %s",
                    fallback2_error,
                )

            # fallback 3: OpenAI gpt-4o (완전히 다른 제공자)
            try:
                fallback3_model = "openai/gpt-4o"
                fallback3 = LiteLlm(model=fallback3_model)
                llm_request.model = fallback3_model

                response = None
                _t3 = time.time()
                print("[warning] fallback 3 시도: OpenAI gpt-4o")
                async for chunk in fallback3.generate_content_async(
                    llm_request=llm_request, stream=False
                ):
                    response = chunk
                print(
                    f"[warning] fallback 3 성공: OpenAI gpt-4o ({time.time() - _t3:.2f}s)"
                )
                return response
            except Exception as fallback3_error:
                print(
                    f"[error] OpenAI gpt-4o 시도 실패 ({time.time() - _t3:.2f}s): {fallback3_error}"
                )
                logger.error("[error] OpenAI gpt-4o 시도 실패: %s", fallback3_error)
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
