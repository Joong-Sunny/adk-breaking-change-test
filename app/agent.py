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

# ============================================================
# Region 레이턴시 측정용 단순 인사 에이전트
#
# 목적:
#   Vertex AI 의 global region vs. korea(asia-northeast3) region 의
#   응답 속도/에러율 차이를 측정한다.
#
#   - korea region   : 피크타임에 429/503 이 직접 노출되도록 fallback 없음
#   - global region  : Google 측에서 자동 라우팅 → 에러/지연 완화 기대
#
#   region 은 GOOGLE_CLOUD_LOCATION 환경변수로 제어한다.
#   (배포 시 global / asia-northeast3 를 바꿔가며 동일 코드로 측정)
#
# 측정을 오염시키지 않기 위해 의도적으로 제외한 것:
#   - 인위적 timeout (1초 컷) — 실제 region 지연을 봐야 하므로 제거
#   - fallback 플러그인     — 피크타임 429/503 을 가리므로 제거
#   - 에러 목킹             — 실제 트래픽만 측정
# ============================================================

import logging
import os
import time
from typing import Optional

import google.auth
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App
from google.adk.models import LlmRequest, LlmResponse
from google.adk.planners import BuiltInPlanner
from google.genai import types

try:
    # StructuredLogHandler: stdout에 JSON 출력 → Agent Engine이 Cloud Logging으로 라우팅
    from google.cloud.logging_v2.handlers import StructuredLogHandler

    _handler = StructuredLogHandler()
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(_handler)
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
# 실험 대상 region. 배포 환경에서 "global" / "asia-northeast3" 로 바꿔가며 측정.
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

logger.info("===========================")
logger.info("GOOGLE_CLOUD_PROJECT: %s", os.getenv("GOOGLE_CLOUD_PROJECT"))
logger.info("GOOGLE_CLOUD_LOCATION: %s", os.getenv("GOOGLE_CLOUD_LOCATION"))
logger.info("GOOGLE_GENAI_USE_VERTEXAI: %s", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))
logger.info("===========================")


# ── 레이턴시 측정용 콜백 ────────────────────────────────────
# before/after_model 사이 경과 시간을 로그로 남겨 region 별 모델 호출
# 지연(network + inference)을 Cloud Logging 에서 바로 집계할 수 있게 한다.
_MODEL_CALL_START: dict[str, float] = {}


def mark_model_start(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    session_id = callback_context.session.id
    _MODEL_CALL_START[session_id] = time.time()
    logger.info(
        "[LATENCY] model call start | region=%s | session_id=%s",
        os.getenv("GOOGLE_CLOUD_LOCATION"),
        session_id,
    )
    return None


def mark_model_end(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    session_id = callback_context.session.id
    start = _MODEL_CALL_START.pop(session_id, None)
    elapsed_ms = (time.time() - start) * 1000 if start is not None else -1
    logger.info(
        "[LATENCY] model call end | region=%s | elapsed_ms=%.1f | session_id=%s",
        os.getenv("GOOGLE_CLOUD_LOCATION"),
        elapsed_ms,
        session_id,
    )
    return llm_response


root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    instruction=(
        "당신은 인사 챗봇입니다. 사용자가 어떤 언어로든 인사하면, "
        "매번 무작위로 다른 나라 언어의 짧은 인사말 하나로만 응답하세요. "
        "설명·이모지·부연 없이 인사말 단어 하나만 출력합니다.\n"
        "예) 사용자: 안녕  → Hello\n"
        "예) 사용자: 你好  → Hola\n"
        "예) 사용자: Hi   → Bonjour"
    ),
    generate_content_config=types.GenerateContentConfig(
        temperature=1.0,  # 매번 다른 언어가 나오도록 다양성 확보
    ),
    before_model_callback=mark_model_start,
    after_model_callback=mark_model_end,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=0,  # thinking 비활성화 → 순수 응답 지연만 측정
        )
    ),
)

app = App(root_agent=root_agent, name="app")
