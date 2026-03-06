import logging
from typing import AsyncGenerator, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
# ✅ [핵심 추가] Google API 코어 예외 클래스들 임포트
from google.api_core import exceptions as google_exceptions
from google.genai import types

logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. 원하는 API 예외를 던질 수 있는 통합 Mock Model
# ==========================================
class MockErrorModel(BaseLlm):
    """
    설정에 따라 429, 503 등 구체적인 Google API 예외를 발생시키는 Mock 모델
    """
    model: str = "mock-error-model"
    
    # 어떤 에러를 던질지 결정하는 필드 (Pydantic 모델이므로 타입 힌트 필수)
    target_error_code: int = 429 

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        
        logging.info(f"😈 [Mock Model] 강제로 HTTP {self.target_error_code} 에러를 발생시킵니다!")

        # target_error_code에 따라 실제 Google API 코어 예외 객체를 던짐
        if self.target_error_code == 429:
            raise google_exceptions.ResourceExhausted("Quota exceeded for quota metric 'Generate requests'.")
        
        elif self.target_error_code == 503:
            raise google_exceptions.ServiceUnavailable("The service is currently unavailable.")
        
        elif self.target_error_code == 500:
            raise google_exceptions.InternalServerError("Internal error encountered.")
            
        elif self.target_error_code == 504:
            raise google_exceptions.DeadlineExceeded("The request timed out.")
            
        else:
            raise Exception("알 수 없는 에러")
            
        yield # 문법 충족용 더미 코드


# ==========================================
# 2. 예외 '타입(Type)'을 검사하는 견고한 플러그인
# ==========================================
class DefenseGuardPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="defense_guard")

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        
        # 1) 503 / 500 (서버/네트워크 레벨 에러) 처리
        # isinstance를 사용하여 다수의 예외 클래스를 한 번에 우아하게 검사합니다.
        if isinstance(error, (google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError)):
            logging.critical(f"🚨 [CRITICAL] Gemini API 50X 서버 에러! 로깅 시스템에 기록합니다. 내용: {error.message}")
            return None # 기존 에러 그대로 전파

        # 2) 429 (자원 고갈 / 할당량 초과) 처리
        if isinstance(error, google_exceptions.ResourceExhausted):
            logging.warning("⚠️ [WARNING] 429 Resource Exhausted 감지! OpenAI Fallback 개입.")
            
            # 가상의 Fallback 처리
            return LlmResponse(
                content=types.Content(
                    role='model',
                    parts=[types.Part.from_text(text="429 에러를 방어하고 대체 모델이 응답했습니다.")]
                )
            )

        # 3) 504 (타임아웃) 처리
        if isinstance(error, google_exceptions.DeadlineExceeded):
            logging.error("⏳ [ERROR] 요청 시간 초과(Timeout) 발생!")
            return None

        # 그 외 예외
        return None