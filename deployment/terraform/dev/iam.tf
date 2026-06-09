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

locals {
  project_ids = {
    dev = var.dev_project_id
  }
}


# Get the project number for the dev project
data "google_project" "dev_project" {
  project_id = var.dev_project_id
}

# Grant Storage Object Creator role to default compute service account
resource "google_project_iam_member" "default_compute_sa_storage_object_creator" {
  project    = var.dev_project_id
  role       = "roles/cloudbuild.builds.builder"
  member     = "serviceAccount:${data.google_project.dev_project.number}-compute@developer.gserviceaccount.com"
  depends_on = [resource.google_project_service.services]
}

# Agent service account
resource "google_service_account" "app_sa" {
  account_id   = "${var.project_name}-app"
  display_name = "${var.project_name} Agent Service Account"
  project      = var.dev_project_id
  depends_on   = [resource.google_project_service.services]
}

# Grant application SA the required permissions to run the application
resource "google_project_iam_member" "app_sa_roles" {
  for_each = {
    for pair in setproduct(keys(local.project_ids), var.app_sa_roles) :
    join(",", pair) => {
      project = local.project_ids[pair[0]]
      role    = pair[1]
    }
  }

  project    = each.value.project
  role       = each.value.role
  member     = "serviceAccount:${google_service_account.app_sa.email}"
  depends_on = [resource.google_project_service.services]
}


# Grant required permissions to Vertex AI service account (gcp-sa-aiplatform)
# NOTE: 이 SA는 일반 Vertex AI 서비스 에이전트이며 Agent Engine 런타임 SA와 다릅니다.
resource "google_project_iam_member" "vertex_ai_sa_permissions" {
  for_each = {
    for pair in setproduct(keys(local.project_ids), var.app_sa_roles) :
    join(",", pair) => pair[1]
  }

  project = var.dev_project_id
  role    = each.value
  member  = google_project_service_identity.vertex_sa.member
  depends_on = [resource.google_project_service.services]
}


# Grant required permissions to Reasoning Engine (Agent Engine) service agent
# (gcp-sa-aiplatform-re) — deploy SA와 다른 GCP 관리 런타임 SA
#
# 이 SA는 Agent Engine 첫 배포 시 GCP가 자동 생성합니다.
# 만약 terraform apply 시 "member not found" 에러가 나면:
#   1. 먼저 make deploy 로 에이전트를 한 번 배포하여 SA 생성
#   2. 이후 make setup-dev-env 로 terraform apply 재실행
resource "google_project_iam_member" "reasoning_engine_sa_roles" {
  for_each = toset([
    "roles/logging.logWriter",  # Cloud Logging 직접 API 호출 (StructuredLogHandler stdout 외)
    "roles/cloudtrace.agent",   # OpenTelemetry 트레이스 전송
  ])

  project    = var.dev_project_id
  role       = each.value
  member     = "serviceAccount:service-${data.google_project.dev_project.number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
  depends_on = [resource.google_project_service.services]
}


