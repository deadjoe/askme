#![allow(dead_code)]
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Custom datetime deserializer that handles timestamps without timezone
mod datetime_format {
    use chrono::{DateTime, Utc};
    use serde::{self, Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<DateTime<Utc>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Use a visitor to handle missing fields gracefully
        struct DateTimeVisitor;

        impl<'de> serde::de::Visitor<'de> for DateTimeVisitor {
            type Value = Option<DateTime<Utc>>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an optional datetime string")
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(None)
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                let s: Option<String> = Option::deserialize(deserializer)?;
                match s {
                    Some(s) => {
                        // Try parsing with Z suffix first (standard RFC3339)
                        if let Ok(dt) = DateTime::parse_from_rfc3339(&s) {
                            return Ok(Some(dt.with_timezone(&Utc)));
                        }

                        // If that fails, try adding Z suffix (assume UTC)
                        if let Ok(dt) = DateTime::parse_from_rfc3339(&format!("{}Z", s)) {
                            return Ok(Some(dt.with_timezone(&Utc)));
                        }

                        Err(serde::de::Error::custom(format!("Invalid datetime format: {}", s)))
                    }
                    None => Ok(None),
                }
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(None)
            }
        }

        deserializer.deserialize_option(DateTimeVisitor)
    }
}

/// Ingest request payload
#[derive(Debug, Clone, Serialize)]
pub struct IngestRequest {
    pub source: String,
    pub path: String,
    pub tags: Option<Vec<String>>,
    pub overwrite: bool,
}

/// Ingest response (flexible format to handle different API versions)
#[derive(Debug, Clone, Deserialize)]
pub struct IngestResponse {
    pub task_id: String,
    pub status: String,

    // Old format fields (with datetime and detailed progress)
    #[serde(default)]
    pub progress: Option<f64>,
    #[serde(default)]
    pub documents_processed: Option<u32>,
    #[serde(default)]
    pub total_documents: Option<u32>,
    #[serde(default)]
    pub error_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "datetime_format::deserialize")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "datetime_format::deserialize")]
    pub completed_at: Option<DateTime<Utc>>,

    // New format fields
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub document_count: Option<u32>,
}

/// Task status response
#[derive(Debug, Clone, Deserialize)]
pub struct TaskStatus {
    pub task_id: String,
    pub status: String, // "queued", "processing", "completed", "failed"
    pub progress: Option<f64>,
    pub documents_processed: Option<u32>,
    pub total_documents: Option<u32>,
    pub error_message: Option<String>,
    #[serde(deserialize_with = "datetime_format::deserialize")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(deserialize_with = "datetime_format::deserialize")]
    pub completed_at: Option<DateTime<Utc>>,
}

/// Query request payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub q: String,
    pub topk: u32,
    pub alpha: f64,
    pub use_rrf: bool,
    pub rrf_k: u32,
    pub use_hyde: bool,
    pub use_rag_fusion: bool,
    pub reranker: String,
    pub max_passages: u32,
    pub include_debug: bool,
}

impl Default for QueryRequest {
    fn default() -> Self {
        Self {
            q: String::new(),
            topk: 50,
            alpha: 0.5,
            use_rrf: true,
            rrf_k: 60,
            use_hyde: false,
            use_rag_fusion: false,
            reranker: "bge_local".to_string(),
            max_passages: 8,
            include_debug: false,
        }
    }
}

/// Citation information
#[derive(Debug, Clone, Deserialize)]
pub struct Citation {
    pub doc_id: String,
    pub title: String,
    pub content: Option<String>,
    pub score: f64,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Debug information for retrieval
#[derive(Debug, Clone, Deserialize)]
pub struct RetrievalDebug {
    pub bm25_hits: u32,
    pub dense_hits: u32,
    pub fusion_method: String,
    pub latency_ms: f64,
    pub embedding_latency_ms: Option<f64>,
    pub search_latency_ms: Option<f64>,
    pub rerank_latency_ms: Option<f64>,
    pub error: Option<String>,
}

/// Query response
#[derive(Debug, Clone, Deserialize)]
pub struct QueryResponse {
    pub query_id: String,
    pub answer: String,
    pub citations: Vec<Citation>,
    pub timestamp: DateTime<Utc>,
    pub retrieval_debug: Option<RetrievalDebug>,
}

/// API error response
#[derive(Debug, Clone, Deserialize)]
pub struct ApiError {
    pub detail: String,
    pub error_type: Option<String>,
}

/// Health check response
#[derive(Debug, Clone, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub components: Option<serde_json::Value>,
}

/// Ingest stats response (subset used by TUI)
#[derive(Debug, Clone, Deserialize)]
pub struct IngestStatsResponse {
    pub total_documents: Option<u64>,
    pub total_chunks: Option<u64>,
    pub total_size_bytes: Option<u64>,
}

/// Source type for ingestion
#[derive(Debug, Clone, PartialEq)]
pub enum SourceType {
    File,
    Directory,
    Url,
}

impl SourceType {
    pub fn detect(path: &str) -> Self {
        if path.starts_with("http://") || path.starts_with("https://") {
            Self::Url
        } else if std::path::Path::new(path).is_dir() {
            Self::Directory
        } else {
            Self::File
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Directory => "dir",
            Self::Url => "url",
        }
    }
}

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Text,
    Json,
    Markdown,
}

impl OutputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
            Self::Markdown => "markdown",
        }
    }

    pub fn variants() -> &'static [Self] {
        &[Self::Text, Self::Json, Self::Markdown]
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Reranker model options
#[derive(Debug, Clone, PartialEq)]
pub enum Reranker {
    BgeLocal,
    Cohere,
}

impl Reranker {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BgeLocal => "bge_local",
            Self::Cohere => "cohere",
        }
    }

    pub fn variants() -> &'static [Self] {
        &[Self::BgeLocal, Self::Cohere]
    }
}

impl std::fmt::Display for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Reranker {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bge_local" => Ok(Self::BgeLocal),
            "cohere" => Ok(Self::Cohere),
            _ => Err(format!("Invalid reranker: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_type_detection() {
        assert_eq!(SourceType::detect("https://example.com"), SourceType::Url);
        assert_eq!(SourceType::detect("http://example.com"), SourceType::Url);
        assert_eq!(SourceType::detect("/some/file.pdf"), SourceType::File);
    }

    #[test]
    fn test_query_request_default() {
        let req = QueryRequest::default();
        assert_eq!(req.topk, 50);
        assert_eq!(req.alpha, 0.5);
        assert!(req.use_rrf);
        assert_eq!(req.reranker, "bge_local");
    }

    #[test]
    fn test_reranker_parsing() {
        assert_eq!("bge_local".parse::<Reranker>().unwrap(), Reranker::BgeLocal);
        assert_eq!("cohere".parse::<Reranker>().unwrap(), Reranker::Cohere);
        assert!("invalid".parse::<Reranker>().is_err());
    }

    #[test]
    fn test_ingest_response_parsing() {
        let test_json = r#"{"task_id":"a956c280-1977-4bdc-bfd4-5d47a44902d3","status":"processing","progress":0.0,"documents_processed":0,"total_documents":1,"error_message":null,"started_at":"2025-09-23T17:17:19.096087","completed_at":null}"#;

        println!("Testing JSON: {}", test_json);
        println!("JSON length: {}", test_json.len());

        // Test individual components first
        let timestamp_str = "2025-09-23T17:17:19.096087";
        match chrono::DateTime::parse_from_rfc3339(&format!("{}Z", timestamp_str)) {
            Ok(dt) => println!("Timestamp parsed successfully: {}", dt),
            Err(e) => println!("Timestamp parsing failed: {}", e),
        }

        let result = serde_json::from_str::<IngestResponse>(test_json);
        match result {
            Ok(response) => {
                assert_eq!(response.task_id, "a956c280-1977-4bdc-bfd4-5d47a44902d3");
                assert_eq!(response.status, "processing");
                assert_eq!(response.progress, Some(0.0));
                assert_eq!(response.documents_processed, Some(0));
                assert_eq!(response.total_documents, Some(1));
                assert!(response.error_message.is_none());
                assert!(response.started_at.is_some());
                assert!(response.completed_at.is_none());
            }
            Err(e) => {
                println!("JSON parsing failed: {}", e);
                println!("Error details: {:?}", e);
                panic!("Failed to parse IngestResponse: {}", e);
            }
        }
    }

    #[test]
    fn test_task_status_parsing() {
        let test_json = r#"{"task_id":"a956c280-1977-4bdc-bfd4-5d47a44902d3","status":"processing","progress":0.0,"documents_processed":0,"total_documents":1,"error_message":null,"started_at":"2025-09-23T17:17:19.096087","completed_at":null}"#;

        let result = serde_json::from_str::<TaskStatus>(test_json);
        match result {
            Ok(task) => {
                assert_eq!(task.task_id, "a956c280-1977-4bdc-bfd4-5d47a44902d3");
                assert_eq!(task.status, "processing");
                assert_eq!(task.progress, Some(0.0));
                assert_eq!(task.documents_processed, Some(0));
                assert_eq!(task.total_documents, Some(1));
                assert!(task.error_message.is_none());
                assert!(task.started_at.is_some());
                assert!(task.completed_at.is_none());
            }
            Err(e) => {
                panic!("Failed to parse TaskStatus: {}", e);
            }
        }
    }

    #[test]
    fn test_new_ingest_response_parsing() {
        let test_json = r#"{"task_id":"0729fde2-3392-4a0d-813a-e40589d5562d","status":"queued","message":null,"document_count":null}"#;

        let result = serde_json::from_str::<IngestResponse>(test_json);
        match result {
            Ok(response) => {
                assert_eq!(response.task_id, "0729fde2-3392-4a0d-813a-e40589d5562d");
                assert_eq!(response.status, "queued");
                assert!(response.message.is_none());
                assert!(response.document_count.is_none());
                // Old format fields should be None for new format
                assert!(response.progress.is_none());
                assert!(response.documents_processed.is_none());
                assert!(response.total_documents.is_none());
                assert!(response.error_message.is_none());
                assert!(response.started_at.is_none());
                assert!(response.completed_at.is_none());
            }
            Err(e) => {
                panic!("Failed to parse new IngestResponse: {}", e);
            }
        }
    }
}
