//! HTTP routes for storing and reading generated wiki trajectories.
//!
//! The request and response bodies intentionally mirror
//! [`crate::trajectories`] types. Axum concerns stay here: Foundation auth,
//! scorecard metering, caller-token extraction, collection resolution, and
//! cache invalidation for stale collection ids. The transaction boundaries and
//! trajectory invariants live next to the trajectory model.

use std::future::Future;

use axum::{
    extract::{Path, Query, State},
    http::HeaderMap,
    Json,
};
use chroma_error::{ChromaError, ErrorCodes};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::{
    auth::AuthzAction,
    errors::ServerError,
    foundation_chroma::{FoundationChromaClient, FoundationChromaClientError},
    routes::{caller_token, whoami::whoami_and_authorize},
    server::FoundationApiServer,
    trajectories::{
        append_open_generate_trajectory, create_open_generate_trajectory,
        finalize_open_generate_trajectory, load_generate_trajectory, save_generate_trajectory,
        Action, AppendTrajectoryEntriesRequest, GenerateTrajectoryFile, Observation,
        ToolCallMetadata, TrajectoryEntry, TrajectoryError, TrajectoryWriteResponse,
    },
};

const WIKI_WRITE_TOOLS: [&str; 2] = ["wiki_apply_patch", "wiki_upsert_file"];

/// Query parameters for `GET /api/trajectories/{id}`.
#[derive(Debug, Default, PartialEq, Eq, Deserialize)]
pub struct ReadTrajectoryQuery {
    /// Reject open trajectories when true. Defaults false so callers can
    /// inspect a partial trajectory while it executes.
    #[serde(default)]
    pub require_finalized: bool,
}

/// Query parameters for `GET /api/trajectories/{id}/reasoning`.
#[derive(Debug, Default, PartialEq, Eq, Deserialize)]
pub struct ReadTrajectoryReasoningQuery {
    /// Wiki page slug to find in trajectory write calls. Present-but-empty is
    /// valid and targets the wiki root page.
    pub slug: Option<String>,
    /// Reject open trajectories when true. Defaults false so callers can
    /// inspect a partial trajectory while it executes.
    #[serde(default)]
    pub require_finalized: bool,
}

/// Reasoning traces associated with one page write.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct TrajectoryReasoningResponse {
    /// The requested page slug.
    pub slug: String,
    /// Trimmed non-empty reasoning traces through the final write action.
    pub reasoning: Vec<String>,
    /// Other slugs written by the same final action, excluding `slug`.
    pub other_slugs: Vec<String>,
}

/// Errors raised while running trajectory routes after request extraction.
#[derive(Debug, thiserror::Error)]
pub enum TrajectoryRouteError {
    /// `frontend_ingress_url` is unset, so the proxying client was never built.
    #[error("trajectory record I/O is not configured")]
    RouteDisabled,
    /// The caller's request carried no usable `x-chroma-token`.
    #[error("missing or invalid x-chroma-token header")]
    MissingToken,
    /// Resolving the trajectory collection through the proxy failed.
    #[error(transparent)]
    Resolve(#[from] FoundationChromaClientError),
    /// The trajectory operation failed.
    #[error(transparent)]
    Trajectory(#[from] TrajectoryError),
    /// The reasoning route requires a slug query parameter.
    #[error("missing slug query parameter")]
    MissingSlug,
}

impl ChromaError for TrajectoryRouteError {
    fn code(&self) -> ErrorCodes {
        match self {
            TrajectoryRouteError::RouteDisabled => ErrorCodes::Internal,
            TrajectoryRouteError::MissingToken | TrajectoryRouteError::MissingSlug => {
                ErrorCodes::InvalidArgument
            }
            TrajectoryRouteError::Resolve(err) if err.is_not_found() => ErrorCodes::NotFound,
            TrajectoryRouteError::Resolve(err) => err.code(),
            TrajectoryRouteError::Trajectory(err) => err.code(),
        }
    }
}

/// `POST /api/trajectories/save` writes a complete finalized trajectory.
pub async fn foundation_save_trajectory(
    headers: HeaderMap,
    State(server): State<FoundationApiServer>,
    Json(file): Json<GenerateTrajectoryFile>,
) -> Result<Json<TrajectoryWriteResponse>, ServerError> {
    let identity =
        whoami_and_authorize(&*server.auth, &headers, AuthzAction::UpsertFoundation).await?;
    let tenant = identity.tenant;
    let _guard = server
        .scorecard_request(&["op:foundation_save_trajectory", &format!("tenant:{tenant}")])?;

    let (client, collection) = trajectory_collection(&server, &headers, &tenant).await?;
    let response = trajectory_op(
        client,
        &tenant,
        save_generate_trajectory(&collection, &file),
    )
    .await?;
    Ok(Json(response))
}

/// `POST /api/trajectories/open` creates an open trajectory with zero entries.
pub async fn foundation_open_trajectory(
    headers: HeaderMap,
    State(server): State<FoundationApiServer>,
    Json(file): Json<GenerateTrajectoryFile>,
) -> Result<Json<TrajectoryWriteResponse>, ServerError> {
    let identity =
        whoami_and_authorize(&*server.auth, &headers, AuthzAction::UpsertFoundation).await?;
    let tenant = identity.tenant;
    let _guard = server
        .scorecard_request(&["op:foundation_open_trajectory", &format!("tenant:{tenant}")])?;

    let (client, collection) = trajectory_collection(&server, &headers, &tenant).await?;
    let response = trajectory_op(
        client,
        &tenant,
        create_open_generate_trajectory(&collection, &file),
    )
    .await?;
    Ok(Json(response))
}

/// `POST /api/trajectories/{id}/entries` appends complete entries.
pub async fn foundation_append_trajectory_entries(
    headers: HeaderMap,
    State(server): State<FoundationApiServer>,
    Path(id): Path<Uuid>,
    Json(request): Json<AppendTrajectoryEntriesRequest>,
) -> Result<Json<TrajectoryWriteResponse>, ServerError> {
    let identity =
        whoami_and_authorize(&*server.auth, &headers, AuthzAction::UpsertFoundation).await?;
    let tenant = identity.tenant;
    let _guard = server.scorecard_request(&[
        "op:foundation_append_trajectory_entries",
        &format!("tenant:{tenant}"),
    ])?;

    let (client, collection) = trajectory_collection(&server, &headers, &tenant).await?;
    let response = trajectory_op(
        client,
        &tenant,
        append_open_generate_trajectory(&collection, id, &request),
    )
    .await?;
    Ok(Json(response))
}

/// `POST /api/trajectories/{id}/finalize` finalizes an open trajectory.
pub async fn foundation_finalize_trajectory(
    headers: HeaderMap,
    State(server): State<FoundationApiServer>,
    Path(id): Path<Uuid>,
    Json(file): Json<GenerateTrajectoryFile>,
) -> Result<Json<TrajectoryWriteResponse>, ServerError> {
    let identity =
        whoami_and_authorize(&*server.auth, &headers, AuthzAction::UpsertFoundation).await?;
    let tenant = identity.tenant;
    let _guard = server.scorecard_request(&[
        "op:foundation_finalize_trajectory",
        &format!("tenant:{tenant}"),
    ])?;

    let (client, collection) = trajectory_collection(&server, &headers, &tenant).await?;
    let response = trajectory_op(
        client,
        &tenant,
        finalize_open_generate_trajectory(&collection, id, &file),
    )
    .await?;
    Ok(Json(response))
}

/// `GET /api/trajectories/{id}` returns a full or partial trajectory.
pub async fn foundation_get_trajectory(
    headers: HeaderMap,
    State(server): State<FoundationApiServer>,
    Path(id): Path<Uuid>,
    Query(query): Query<ReadTrajectoryQuery>,
) -> Result<Json<GenerateTrajectoryFile>, ServerError> {
    let identity =
        whoami_and_authorize(&*server.auth, &headers, AuthzAction::ViewFoundation).await?;
    let tenant = identity.tenant;
    let _guard =
        server.scorecard_request(&["op:foundation_get_trajectory", &format!("tenant:{tenant}")])?;

    let (client, collection) = trajectory_collection(&server, &headers, &tenant).await?;
    let response = trajectory_op(
        client,
        &tenant,
        load_generate_trajectory(&collection, id, query.require_finalized),
    )
    .await?;
    Ok(Json(response))
}

/// `GET /api/trajectories/{id}/reasoning` returns reasoning for one page write.
pub async fn foundation_get_trajectory_reasoning(
    headers: HeaderMap,
    State(server): State<FoundationApiServer>,
    Path(id): Path<Uuid>,
    Query(query): Query<ReadTrajectoryReasoningQuery>,
) -> Result<Json<Vec<TrajectoryReasoningResponse>>, ServerError> {
    let identity =
        whoami_and_authorize(&*server.auth, &headers, AuthzAction::ViewFoundation).await?;
    let tenant = identity.tenant;
    let _guard = server.scorecard_request(&[
        "op:foundation_get_trajectory_reasoning",
        &format!("tenant:{tenant}"),
    ])?;
    let slug = query.slug.ok_or(TrajectoryRouteError::MissingSlug)?;

    let (client, collection) = trajectory_collection(&server, &headers, &tenant).await?;
    let file = trajectory_op(
        client,
        &tenant,
        load_generate_trajectory(&collection, id, query.require_finalized),
    )
    .await?;
    Ok(Json(reasoning_for_slug(&file, &slug)))
}

fn reasoning_for_slug(
    file: &GenerateTrajectoryFile,
    slug: &str,
) -> Vec<TrajectoryReasoningResponse> {
    let entries = &file.trajectory.actions_and_observations;
    let mut reasoning_prefix = Vec::new();
    let mut last_match = None;

    for (index, entry) in entries.iter().enumerate() {
        let TrajectoryEntry::Action(action) = entry else {
            continue;
        };

        if let Some(reasoning) = action.reasoning.as_ref() {
            let trimmed = reasoning.trim();
            if !trimmed.is_empty() {
                reasoning_prefix.push(trimmed.to_string());
            }
        }
        let reasoning_len_after_action = reasoning_prefix.len();

        let observation = next_observation(entries, index);
        if action_writes_slug(action, observation, slug) {
            last_match = Some((index, reasoning_len_after_action));
        }
    }

    let Some((action_index, reasoning_len)) = last_match else {
        return Vec::new();
    };
    let other_slugs = match &entries[action_index] {
        TrajectoryEntry::Action(action) => {
            other_written_slugs(action, next_observation(entries, action_index), slug)
        }
        TrajectoryEntry::Observation(_) => Vec::new(),
    };

    vec![TrajectoryReasoningResponse {
        slug: slug.to_string(),
        reasoning: reasoning_prefix.into_iter().take(reasoning_len).collect(),
        other_slugs,
    }]
}

fn next_observation(entries: &[TrajectoryEntry], action_index: usize) -> Option<&Observation> {
    entries.get(action_index + 1).and_then(|entry| match entry {
        TrajectoryEntry::Observation(observation) => Some(observation),
        TrajectoryEntry::Action(_) => None,
    })
}

fn action_writes_slug(action: &Action, observation: Option<&Observation>, slug: &str) -> bool {
    (0..action.tools.len()).any(|call| {
        call_write_slug(action, observation, call)
            .as_deref()
            .is_some_and(|call_slug| call_slug == slug)
    })
}

fn other_written_slugs(
    action: &Action,
    observation: Option<&Observation>,
    slug: &str,
) -> Vec<String> {
    let mut slugs = Vec::new();
    for call in 0..action.tools.len() {
        let Some(call_slug) = call_write_slug(action, observation, call) else {
            continue;
        };
        if call_slug != slug && !slugs.contains(&call_slug) {
            slugs.push(call_slug);
        }
    }
    slugs
}

fn call_write_slug(
    action: &Action,
    observation: Option<&Observation>,
    call: usize,
) -> Option<String> {
    let tool = action.tools.get(call)?;
    if !WIKI_WRITE_TOOLS.contains(&tool.tool_schema.name.as_str()) {
        return None;
    }

    let metadata = observation_metadata(observation, call);
    if metadata
        .and_then(|metadata| metadata.skipped_due_to_handoff)
        .unwrap_or(false)
    {
        return None;
    }

    param_slug(action.params.get(call))
        .or_else(|| metadata.and_then(|metadata| metadata.slug.clone()))
}

fn param_slug(params: Option<&Value>) -> Option<String> {
    params?
        .get("slug")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn observation_metadata(
    observation: Option<&Observation>,
    call: usize,
) -> Option<&ToolCallMetadata> {
    observation?
        .tool_metadata
        .get(call)
        .and_then(Option::as_ref)
}

async fn trajectory_collection<'a>(
    server: &'a FoundationApiServer,
    headers: &HeaderMap,
    tenant: &str,
) -> Result<(&'a FoundationChromaClient, chroma::ChromaCollection), TrajectoryRouteError> {
    let client = server
        .foundation_chroma_client
        .as_ref()
        .ok_or(TrajectoryRouteError::RouteDisabled)?;
    let token = caller_token(headers).ok_or(TrajectoryRouteError::MissingToken)?;
    let collection = client.trajectories_collection(tenant, token).await?;
    Ok((client, collection))
}

async fn trajectory_op<T, F>(
    client: &FoundationChromaClient,
    tenant: &str,
    fut: F,
) -> Result<T, TrajectoryRouteError>
where
    F: Future<Output = Result<T, TrajectoryError>>,
{
    fut.await.map_err(|err| {
        if err.is_chroma_not_found() {
            client.invalidate_trajectories(tenant);
        }
        TrajectoryRouteError::Trajectory(err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectories::{Source, Tool, ToolSchema, Trajectory};
    use chroma::client::ChromaHttpClientError;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn minimal_file(entries: Vec<TrajectoryEntry>) -> GenerateTrajectoryFile {
        GenerateTrajectoryFile {
            batch_index: None,
            batch_offset: None,
            worker_id: None,
            span: None,
            attempt_id: None,
            deadlock_retries: None,
            attempt_paths: None,
            started_at: None,
            duration_seconds: None,
            status: None,
            error: None,
            usage: None,
            citations: None,
            final_todos: None,
            trajectory: Trajectory {
                id: Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
                actions_and_observations: entries,
            },
            extra: BTreeMap::new(),
        }
    }

    fn tool(name: &str) -> Tool {
        Tool {
            tool_schema: ToolSchema {
                name: name.to_string(),
                description: String::new(),
                parameters: json!({"type": "object"}),
                required: Vec::new(),
                extra: BTreeMap::new(),
            },
            extra: BTreeMap::new(),
        }
    }

    fn action(reasoning: Option<&str>, calls: Vec<(&str, Value)>) -> TrajectoryEntry {
        let call_count = calls.len();
        TrajectoryEntry::Action(Action {
            tools: calls.iter().map(|(name, _)| tool(name)).collect(),
            params: calls.iter().map(|(_, params)| params.clone()).collect(),
            sources: (0..call_count).map(|_| Source::new("agent")).collect(),
            reasoning: reasoning.map(str::to_string),
            reasoning_signature: None,
        })
    }

    fn metadata(slug: Option<&str>, skipped: bool) -> ToolCallMetadata {
        ToolCallMetadata {
            lock_handoff: None,
            lock_waits: None,
            skipped_due_to_handoff: Some(skipped),
            surfaced_page_ids: None,
            read_page_id: None,
            page_id: None,
            record_ids: None,
            todos: None,
            op: None,
            slug: slug.map(str::to_string),
            source_ids: None,
            categories: None,
            latest_raw_source_date: None,
            extra: BTreeMap::new(),
        }
    }

    fn observation(metadata: Vec<Option<ToolCallMetadata>>) -> TrajectoryEntry {
        let call_count = metadata.len();
        TrajectoryEntry::Observation(Observation {
            observations: (0..call_count).map(|_| "ok".to_string()).collect(),
            sources: (0..call_count).map(|_| Source::new("wiki")).collect(),
            tool_metadata: metadata,
        })
    }

    #[test]
    fn route_errors_map_complete_contract_codes() {
        let id = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        assert_eq!(
            vec![
                TrajectoryRouteError::RouteDisabled.code(),
                TrajectoryRouteError::MissingToken.code(),
                TrajectoryRouteError::MissingSlug.code(),
                TrajectoryRouteError::Resolve(FoundationChromaClientError::InvalidToken(
                    "bad token".to_string(),
                ))
                .code(),
                TrajectoryRouteError::Resolve(FoundationChromaClientError::Client(
                    ChromaHttpClientError::ApiError(
                        "missing".to_string(),
                        reqwest::StatusCode::NOT_FOUND,
                    ),
                ))
                .code(),
                TrajectoryRouteError::Trajectory(TrajectoryError::NotFound { tid: id }).code(),
                TrajectoryRouteError::Trajectory(TrajectoryError::AlreadyExists { tid: id }).code(),
                TrajectoryRouteError::Trajectory(TrajectoryError::EmptyAppend { tid: id }).code(),
                TrajectoryRouteError::Trajectory(TrajectoryError::IdMismatch {
                    path: id,
                    body: Uuid::nil(),
                })
                .code(),
                TrajectoryRouteError::Trajectory(TrajectoryError::FinalizedRequired { tid: id })
                    .code(),
                TrajectoryRouteError::Trajectory(TrajectoryError::EntryCountMismatch {
                    tid: id,
                    expected: 1,
                    actual: 0,
                })
                .code(),
                TrajectoryRouteError::Trajectory(TrajectoryError::NotOpen {
                    tid: id,
                    write_state: crate::trajectories::WriteState::Finalized,
                })
                .code(),
            ],
            vec![
                ErrorCodes::Internal,
                ErrorCodes::InvalidArgument,
                ErrorCodes::InvalidArgument,
                ErrorCodes::InvalidArgument,
                ErrorCodes::NotFound,
                ErrorCodes::NotFound,
                ErrorCodes::AlreadyExists,
                ErrorCodes::InvalidArgument,
                ErrorCodes::InvalidArgument,
                ErrorCodes::FailedPrecondition,
                ErrorCodes::FailedPrecondition,
                ErrorCodes::FailedPrecondition,
            ]
        );
    }

    #[test]
    fn read_query_defaults_to_partial_reads() {
        assert_eq!(
            ReadTrajectoryQuery::default(),
            ReadTrajectoryQuery {
                require_finalized: false,
            }
        );
        assert_eq!(
            serde_json::from_value::<ReadTrajectoryQuery>(json!({})).unwrap(),
            ReadTrajectoryQuery {
                require_finalized: false,
            }
        );
    }

    #[test]
    fn read_query_deserializes_finalized_requirement_opt_in() {
        assert_eq!(
            serde_json::from_value::<ReadTrajectoryQuery>(json!({
                "require_finalized": true
            }))
            .unwrap(),
            ReadTrajectoryQuery {
                require_finalized: true,
            }
        );
    }

    #[test]
    fn reasoning_query_requires_slug_but_allows_root_slug() {
        assert_eq!(
            serde_json::from_value::<ReadTrajectoryReasoningQuery>(json!({
                "require_finalized": true
            }))
            .unwrap(),
            ReadTrajectoryReasoningQuery {
                slug: None,
                require_finalized: true,
            }
        );
        assert_eq!(
            serde_json::from_value::<ReadTrajectoryReasoningQuery>(json!({
                "slug": "",
            }))
            .unwrap(),
            ReadTrajectoryReasoningQuery {
                slug: Some(String::new()),
                require_finalized: false,
            }
        );
    }

    #[test]
    fn reasoning_for_slug_returns_empty_when_slug_was_not_written() {
        let file = minimal_file(vec![action(
            Some("thinking"),
            vec![("wiki_upsert_file", json!({"slug": "other"}))],
        )]);

        assert_eq!(reasoning_for_slug(&file, "target"), Vec::new());
    }

    #[test]
    fn reasoning_for_slug_uses_latest_write_and_prefix_reasoning() {
        let file = minimal_file(vec![
            action(
                Some("  first target  "),
                vec![("wiki_upsert_file", json!({"slug": "target"}))],
            ),
            action(
                Some("intermediate"),
                vec![("wiki_apply_patch", json!({"slug": "other"}))],
            ),
            action(Some("  \n  "), vec![("search", json!({"slug": "ignored"}))]),
            action(
                Some("final target"),
                vec![
                    ("wiki_apply_patch", json!({"slug": "target"})),
                    ("wiki_upsert_file", json!({"slug": "sibling"})),
                    ("wiki_apply_patch", json!({"slug": "sibling"})),
                    ("search", json!({"slug": "not-a-write"})),
                ],
            ),
            action(
                Some("after final target"),
                vec![("wiki_upsert_file", json!({"slug": "later"}))],
            ),
        ]);

        assert_eq!(
            reasoning_for_slug(&file, "target"),
            vec![TrajectoryReasoningResponse {
                slug: "target".to_string(),
                reasoning: vec![
                    "first target".to_string(),
                    "intermediate".to_string(),
                    "final target".to_string(),
                ],
                other_slugs: vec!["sibling".to_string()],
            }]
        );
    }

    #[test]
    fn reasoning_for_slug_returns_written_slug_without_reasoning() {
        let file = minimal_file(vec![
            action(
                Some("before"),
                vec![("wiki_upsert_file", json!({"slug": "other"}))],
            ),
            action(None, vec![("wiki_upsert_file", json!({"slug": "target"}))]),
            action(
                Some("after target"),
                vec![("wiki_upsert_file", json!({"slug": "other"}))],
            ),
        ]);

        assert_eq!(
            reasoning_for_slug(&file, "target"),
            vec![TrajectoryReasoningResponse {
                slug: "target".to_string(),
                reasoning: vec!["before".to_string()],
                other_slugs: Vec::new(),
            }]
        );
    }

    #[test]
    fn reasoning_for_slug_uses_metadata_slug_fallback_and_skips_handoffs() {
        let file = minimal_file(vec![
            action(
                Some("skipped"),
                vec![("wiki_upsert_file", json!({"slug": "target"}))],
            ),
            observation(vec![Some(metadata(Some("target"), true))]),
            action(Some("fallback"), vec![("wiki_apply_patch", json!({}))]),
            observation(vec![Some(metadata(Some("target"), false))]),
        ]);

        assert_eq!(
            reasoning_for_slug(&file, "target"),
            vec![TrajectoryReasoningResponse {
                slug: "target".to_string(),
                reasoning: vec!["skipped".to_string(), "fallback".to_string()],
                other_slugs: Vec::new(),
            }]
        );
    }
}
