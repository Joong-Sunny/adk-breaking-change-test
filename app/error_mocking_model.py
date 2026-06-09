import asyncio
import logging
from typing import AsyncGenerator, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.api_core import exceptions as google_exceptions
from google.genai import types

logging.basicConfig(level=logging.INFO)


# ==========================================
# 1. 원하는 API 예외를 던질 수 있는 통합 Mock Model
# ==========================================
class MockErrorModel(BaseLlm):
    """
    설정에 따라 429, 503 등 구체적인 Google API 예외를 발생시키는 Mock 모델.
    delay_seconds: 에러를 발생시키기 전에 대기할 시간 (초). 느린 LLM 시뮬레이션.
    """
    model: str = "mock-error-model"
    target_error_code: int = 429
    delay_seconds: int = 0  # 느린 LLM 시뮬레이션용 지연 시간

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:

        if self.delay_seconds > 0:
            logging.info(
                f"😴 [MockErrorModel] {self.delay_seconds}초 대기 후 "
                f"HTTP {self.target_error_code} 에러를 발생시킵니다..."
            )
            await asyncio.sleep(self.delay_seconds)

        logging.info(f"😈 [MockErrorModel] HTTP {self.target_error_code} 에러 발생!")

        if self.target_error_code == 429:
            raise google_exceptions.ResourceExhausted(
                "Quota exceeded for quota metric 'Generate requests'."
            )
        elif self.target_error_code == 503:
            raise google_exceptions.ServiceUnavailable("The service is currently unavailable.")
        elif self.target_error_code == 500:
            raise google_exceptions.InternalServerError("Internal error encountered.")
        elif self.target_error_code == 504:
            raise google_exceptions.DeadlineExceeded("The request timed out.")
        else:
            raise Exception("알 수 없는 에러")

        yield  # async generator 문법 충족용


# ==========================================
# 2. asyncio 기반 타임아웃 래퍼 모델
#
# 실제 Gemini 모델 사용 시엔 http_options.timeout 으로 HTTP 레벨 타임아웃이 가능하지만,
# MockErrorModel처럼 HTTP를 쓰지 않는 커스텀 BaseLlm에는 http_options가 적용되지 않음.
# 이 클래스는 asyncio.timeout()으로 임의의 BaseLlm 호출에 타임아웃을 적용하는 범용 래퍼.
# ==========================================
class TimeoutModel(BaseLlm):
    """
    임의의 BaseLlm 호출에 asyncio 타임아웃을 적용하는 래퍼.
    timeout_seconds 초과 시 DeadlineExceeded를 발생시켜 on_model_error_callback을 트리거.
    """
    model: str = "timeout-wrapper"
    wrapped: BaseLlm
    timeout_seconds: float = 15.0

    model_config = {"arbitrary_types_allowed": True}

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async for chunk in self.wrapped.generate_content_async(llm_request, stream):
                    yield chunk
        except asyncio.TimeoutError:
            logging.warning(
                f"⏰ [TimeoutModel] {self.timeout_seconds}초 타임아웃 초과! "
                "DeadlineExceeded를 발생시킵니다 → on_model_error_callback 트리거"
            )
            raise google_exceptions.DeadlineExceeded(
                f"LLM 응답이 {self.timeout_seconds}초를 초과했습니다."
            )
