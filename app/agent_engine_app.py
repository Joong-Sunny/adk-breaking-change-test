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

# mypy: disable-error-code="attr-defined,arg-type"
import os
from typing import Any

import google.cloud.logging
import vertexai
from app.agent import app as adk_app
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.cloud import logging as google_cloud_logging
from vertexai.agent_engines.templates.adk import AdkApp


class AgentEngineApp(AdkApp):
    def set_up(self) -> None:
        """Initialize the agent engine app with logging and telemetry."""
        vertexai.init()
        setup_telemetry()
        super().set_up()

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "ai-lamp-dev-479401")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-northeast3")
        # GOOGLE_CLOUD_AGENT_ENGINE_ID 는 Agent Engine이 런타임에 자동 주입하는 예약 변수
        # 로컬 실행 시엔 없을 수 있으므로 빈 문자열로 fallback
        agent_engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "")

        logging_client = google_cloud_logging.Client(project=project_id)
        # resource를 지정해야 Agent Engine 로그 탐색기에서 올바르게 조회됨
        self.logger = logging_client.logger(
            name="agent_engine_app",
            resource=google.cloud.logging.Resource(
                type="aiplatform.googleapis.com/ReasoningEngine",
                labels={
                    "location": location,
                    "resource_container": project_id,
                    "reasoning_engine_id": agent_engine_id,
                },
            ),
        )
        if gemini_location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = gemini_location

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback."""
        feedback_obj = Feedback.model_validate(feedback)
        self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent."""
        operations = super().register_operations()
        operations[""] = operations.get("", []) + ["register_feedback"]
        return operations


gemini_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")
agent_engine = AgentEngineApp(
    app=adk_app,
    artifact_service_builder=lambda: GcsArtifactService(bucket_name=logs_bucket_name)
    if logs_bucket_name
    else InMemoryArtifactService(),
)
