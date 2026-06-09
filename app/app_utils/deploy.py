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

import asyncio
import datetime
import importlib
import inspect
import json
import logging
import subprocess
import warnings
from typing import Any

import click
import google.auth
import vertexai
from vertexai._genai import _agent_engines_utils
from vertexai._genai.types import AgentEngine, AgentEngineConfig

# Suppress google-cloud-storage version compatibility warning
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.cloud.aiplatform"
)


def generate_class_methods_from_agent(agent_instance: Any) -> list[dict[str, Any]]:
    """Generate method specifications with schemas from agent's register_operations().

    See: https://docs.cloud.google.com/agent-builder/agent-engine/use/custom#supported-operations
    """
    registered_operations = _agent_engines_utils._get_registered_operations(
        agent=agent_instance
    )
    class_methods_spec = _agent_engines_utils._generate_class_methods_spec_or_raise(
        agent=agent_instance,
        operations=registered_operations,
    )
    class_methods_list = [
        _agent_engines_utils._to_dict(method_spec) for method_spec in class_methods_spec
    ]
    return class_methods_list


def parse_key_value_pairs(kv_string: str | None) -> dict[str, str]:
    """Parse key-value pairs from a comma-separated KEY=VALUE string."""
    result = {}
    if kv_string:
        for pair in kv_string.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key.strip()] = value.strip()
            else:
                logging.warning(f"Skipping malformed key-value pair: {pair}")
    return result


def write_deployment_metadata(
    remote_agent: Any,
    metadata_file: str = "deployment_metadata.json",
) -> None:
    """Write deployment metadata to file."""
    metadata = {
        "remote_agent_engine_id": remote_agent.api_resource.name,
        "deployment_target": "agent_engine",
        "is_a2a": False,
        "deployment_timestamp": datetime.datetime.now().isoformat(),
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Agent Engine ID written to {metadata_file}")


def ensure_logging_iam_binding(project: str, project_number: str) -> None:
    """Reasoning Engine 서비스 에이전트에 Cloud Logging/Trace 쓰기 권한을 부여합니다.

    Agent Engine의 실제 런타임 SA:
      service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com

    이 SA는 deploy SA(deploy-agent-engine-lamp-dev@...)와 다르며,
    GCP가 첫 Agent Engine 배포 시 자동으로 생성합니다.
    기본적으로 logging.logWriter 가 없어 StructuredLogHandler 외의
    Cloud Logging 직접 API 호출 로그가 기록되지 않습니다.

    NOTE: gcloud add-iam-policy-binding 은 idempotent 명령어입니다.
    이미 바인딩이 존재해도 returncode=0 을 반환하므로, returncode != 0 은
    '이미 설정됨'이 아니라 진짜 실패(권한 부족 등)를 의미합니다.
    """
    # gcp-sa-aiplatform-re: Reasoning Engine(Agent Engine) 전용 서비스 에이전트
    # gcp-sa-aiplatform (일반 Vertex AI SA)과 다른 별개의 SA임
    runtime_sa = f"service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"

    roles_to_grant = [
        "roles/logging.logWriter",   # Cloud Logging 직접 API 호출 (self.logger.log_struct 등)
        "roles/cloudtrace.agent",    # OpenTelemetry 트레이스 전송
    ]

    for role in roles_to_grant:
        result = subprocess.run(
            [
                "gcloud", "projects", "add-iam-policy-binding", project,
                f"--member=serviceAccount:{runtime_sa}",
                f"--role={role}",
                "--condition=None",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"✅ {role} 부여 완료 → {runtime_sa}")
        else:
            logging.error(
                "❌ %s 바인딩 실패! "
                "배포된 에이전트에서 Cloud Logging/Trace가 동작하지 않을 수 있습니다. "
                "배포 실행 계정에 resourcemanager.projects.setIamPolicy 권한이 있는지 확인하세요. "
                "오류 내용: %s",
                role,
                result.stderr.strip(),
            )


def print_deployment_success(
    remote_agent: Any,
    location: str,
    project: str,
) -> None:
    """Print deployment success message with console URL."""
    # Extract agent engine ID and project number for console URL
    resource_name_parts = remote_agent.api_resource.name.split("/")
    agent_engine_id = resource_name_parts[-1]
    project_number = resource_name_parts[1]
    print("\n✅ Deployment successful!")
    service_account = remote_agent.api_resource.spec.service_account
    if service_account:
        print(f"Service Account: {service_account}")
    else:
        default_sa = (
            f"service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
        )
        print(f"Service Account: {default_sa}")
    playground_url = f"https://console.cloud.google.com/vertex-ai/agents/locations/{location}/agent-engines/{agent_engine_id}/playground?project={project}"
    print(f"\n📊 Open Console Playground: {playground_url}\n")


@click.command()
@click.option(
    "--project",
    default=None,
    help="GCP project ID (defaults to application default credentials)",
)
@click.option(
    "--location",
    default="asia-northeast3",
    help="GCP region (defaults to asia-northeast3)",
)
@click.option(
    "--display-name",
    default="sunny-good-agent",
    help="Display name for the agent engine",
)
@click.option(
    "--description",
    default="A base ReAct agent built with Google's Agent Development Kit (ADK)",
    help="Description of the agent",
)
@click.option(
    "--source-packages",
    multiple=True,
    default=["./app"],
    help="Source packages to deploy. Can be specified multiple times (e.g., --source-packages=./app --source-packages=./lib)",
)
@click.option(
    "--entrypoint-module",
    default="app.agent_engine_app",
    help="Python module path for the agent entrypoint (required)",
)
@click.option(
    "--entrypoint-object",
    default="agent_engine",
    help="Name of the agent instance at module level (required)",
)
@click.option(
    "--requirements-file",
    default="app/app_utils/.requirements.txt",
    help="Path to requirements.txt file",
)
@click.option(
    "--set-env-vars",
    default=None,
    help="Comma-separated list of environment variables in KEY=VALUE format",
)
@click.option(
    "--labels",
    default=None,
    help="Comma-separated list of labels in KEY=VALUE format",
)
@click.option(
    "--service-account",
    default=None,
    help="Service account email to use for the agent engine",
)
@click.option(
    "--min-instances",
    type=int,
    default=1,
    help="Minimum number of instances (default: 1)",
)
@click.option(
    "--max-instances",
    type=int,
    default=10,
    help="Maximum number of instances (default: 10)",
)
@click.option(
    "--cpu",
    default="4",
    help="CPU limit (default: 4)",
)
@click.option(
    "--memory",
    default="8Gi",
    help="Memory limit (default: 8Gi)",
)
@click.option(
    "--container-concurrency",
    type=int,
    default=9,
    help="Container concurrency (default: 9)",
)
@click.option(
    "--num-workers",
    type=int,
    default=1,
    help="Number of worker processes (default: 1)",
)
def deploy_agent_engine_app(
    project: str | None,
    location: str,
    display_name: str,
    description: str,
    source_packages: tuple[str, ...],
    entrypoint_module: str,
    entrypoint_object: str,
    requirements_file: str,
    set_env_vars: str | None,
    labels: str | None,
    service_account: str | None,
    min_instances: int,
    max_instances: int,
    cpu: str,
    memory: str,
    container_concurrency: int,
    num_workers: int,
) -> AgentEngine:
    """Deploy the agent engine app to Vertex AI."""

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Parse environment variables and labels if provided
    env_vars = parse_key_value_pairs(set_env_vars)
    labels_dict = parse_key_value_pairs(labels)

    # Set GOOGLE_CLOUD_REGION to match deployment location
    env_vars["GOOGLE_CLOUD_REGION"] = location

    # Add NUM_WORKERS from CLI argument (can be overridden via --set-env-vars)
    if "NUM_WORKERS" not in env_vars:
        env_vars["NUM_WORKERS"] = str(num_workers)

    # Enable telemetry by default for Agent Engine
    if "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY" not in env_vars:
        env_vars["GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"] = "true"
    if "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT" not in env_vars:
        env_vars["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

    if not project:
        _, project = google.auth.default()

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤖 DEPLOYING AGENT TO VERTEX AI AGENT ENGINE 🤖         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Log deployment parameters
    click.echo("\n📋 Deployment Parameters:")
    click.echo(f"  Project: {project}")
    click.echo(f"  Location: {location}")
    click.echo(f"  Display Name: {display_name}")
    click.echo(f"  Min Instances: {min_instances}")
    click.echo(f"  Max Instances: {max_instances}")
    click.echo(f"  CPU: {cpu}")
    click.echo(f"  Memory: {memory}")
    click.echo(f"  Container Concurrency: {container_concurrency}")
    if service_account:
        click.echo(f"  Service Account: {service_account}")
    if env_vars:
        click.echo("\n🌍 Environment Variables:")
        for key, value in sorted(env_vars.items()):
            click.echo(f"  {key}: {value}")

    source_packages_list = list(source_packages)

    # Initialize vertexai client
    client = vertexai.Client(
        project=project,
        location=location,
    )
    vertexai.init(project=project, location=location)

    # Add agent garden labels if configured

    # Dynamically import the agent instance to generate class_methods
    logging.info(f"Importing {entrypoint_module}.{entrypoint_object}")
    module = importlib.import_module(entrypoint_module)
    agent_instance = getattr(module, entrypoint_object)

    # If the agent_instance is a coroutine, await it to get the actual instance
    if inspect.iscoroutine(agent_instance):
        logging.info(f"Detected coroutine, awaiting {entrypoint_object}...")
        agent_instance = asyncio.run(agent_instance)
    # Generate class methods spec from register_operations
    class_methods_list = generate_class_methods_from_agent(agent_instance)

    config = AgentEngineConfig(
        display_name=display_name,
        description=description,
        source_packages=source_packages_list,
        entrypoint_module=entrypoint_module,
        entrypoint_object=entrypoint_object,
        class_methods=class_methods_list,
        env_vars=env_vars,
        service_account=service_account,
        requirements_file=requirements_file,
        labels=labels_dict,
        min_instances=min_instances,
        max_instances=max_instances,
        resource_limits={"cpu": cpu, "memory": memory},
        container_concurrency=container_concurrency,
        agent_framework="google-adk",
    )

    # Check if an agent with this name already exists
    existing_agents = list(client.agent_engines.list())
    matching_agents = [
        agent
        for agent in existing_agents
        if agent.api_resource.display_name == display_name
    ]

    # Deploy the agent (create or update)
    if matching_agents:
        click.echo(f"\n📝 Updating existing agent: {display_name}")
    else:
        click.echo(f"\n🚀 Creating new agent: {display_name}")

    click.echo("🚀 Deploying to Vertex AI Agent Engine (this can take 3-5 minutes)...")
    if matching_agents:
        remote_agent = client.agent_engines.update(
            name=matching_agents[0].api_resource.name, config=config
        )
    else:
        remote_agent = client.agent_engines.create(config=config)

    write_deployment_metadata(remote_agent)
    print_deployment_success(remote_agent, location, project)

    # 런타임 SA에 Cloud Logging 권한 자동 부여
    # GOOGLE_CLOUD_AGENT_ENGINE_ID 는 예약 prefix 라 직접 설정 불가 — Agent Engine이 런타임에 자동 주입
    resource_name_parts = remote_agent.api_resource.name.split("/")
    project_number = resource_name_parts[1]
    ensure_logging_iam_binding(project, project_number)

    return remote_agent


if __name__ == "__main__":
    deploy_agent_engine_app()
