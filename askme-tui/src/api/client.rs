use crate::api::types::*;
use anyhow::{Context, Result};
use reqwest::{Client, Response};
use std::time::Duration;
use tracing::{debug, error, info};

/// API client for askme service
#[derive(Debug, Clone)]
pub struct ApiClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl ApiClient {
    /// Create a new API client
    pub fn new(base_url: String, api_key: Option<String>) -> Result<Self> {
        // Allow overriding timeout via env var; default to 180s for long generations
        let timeout_secs: u64 = std::env::var("ASKME_TUI_HTTP_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(180);

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        })
    }

    /// Check if the API is healthy
    pub async fn health_check(&self) -> Result<HealthResponse> {
        let url = format!("{}/health", self.base_url);
        debug!("Health check: {}", url);

        let response = self.client.get(&url).send().await
            .context("Failed to send health check request")?;

        if response.status().is_success() {
            let health: HealthResponse = response.json().await
                .context("Failed to parse health response")?;
            info!("Health check successful: {}", health.status);
            Ok(health)
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Health check failed: {} - {}", status, text);
        }
    }

    /// Get ingestion/global stats (for collection size, etc.)
    pub async fn ingest_stats(&self) -> Result<IngestStatsResponse> {
        let url = format!("{}/ingest/stats", self.base_url);
        debug!("Fetching ingest stats: {}", url);

        let response = self.client.get(&url).send().await
            .context("Failed to get ingest stats")?;

        self.handle_response(response).await
    }

    /// Submit an ingest request
    pub async fn ingest(&self, request: IngestRequest) -> Result<IngestResponse> {
        let url = format!("{}/ingest/", self.base_url);
        debug!("Ingest request: {} -> {}", request.source, request.path);

        let mut req_builder = self.client.post(&url).json(&request);

        if let Some(ref api_key) = self.api_key {
            req_builder = req_builder.header("X-API-Key", api_key);
        }

        let response = req_builder.send().await
            .context("Failed to send ingest request")?;

        self.handle_response(response).await
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: &str) -> Result<TaskStatus> {
        let url = format!("{}/ingest/status/{}", self.base_url, task_id);
        debug!("Getting task status: {}", task_id);

        let response = self.client.get(&url).send().await
            .context("Failed to get task status")?;

        self.handle_response(response).await
    }

    /// Submit a query request
    pub async fn query(&self, request: QueryRequest) -> Result<QueryResponse> {
        let url = format!("{}/query/", self.base_url);

        // Log the full request for debugging
        let request_json = serde_json::to_string_pretty(&request)
            .context("Failed to serialize query request")?;

        debug!("Query request to {}: {}", url, request_json);

        let mut req_builder = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request);

        if let Some(ref api_key) = self.api_key {
            req_builder = req_builder.header("X-API-Key", api_key);
        }

        let response = req_builder.send().await.map_err(|e| {
            if e.is_timeout() {
                anyhow::anyhow!(
                    "Request timed out. Increase ASKME_TUI_HTTP_TIMEOUT (current default: 180s)."
                )
            } else {
                anyhow::anyhow!("Failed to send query request: {}", e)
            }
        })?;

        self.handle_response(response).await
    }

    /// Poll task status until completion
    pub async fn poll_task_status<F>(&self, task_id: &str, mut callback: F) -> Result<TaskStatus>
    where
        F: FnMut(&TaskStatus),
    {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(3600); // 1 hour timeout

        loop {
            interval.tick().await;

            if start.elapsed() > timeout {
                anyhow::bail!("Timeout waiting for task completion");
            }

            let status = self.get_task_status(task_id).await?;
            callback(&status);

            match status.status.as_str() {
                "completed" => {
                    info!("Task {} completed successfully", task_id);
                    return Ok(status);
                }
                "failed" => {
                    let error_msg = status.error_message
                        .unwrap_or_else(|| "Unknown error".to_string());
                    anyhow::bail!("Task {} failed: {}", task_id, error_msg);
                }
                "processing" | "queued" => {
                    // Continue polling
                }
                _ => {
                    debug!("Unknown task status: {}", status.status);
                }
            }
        }
    }

    /// Poll task status with progress updates via channel
    pub async fn poll_task_status_with_updates(&self, task_id: &str) -> Result<(tokio::sync::mpsc::UnboundedReceiver<TaskStatus>, tokio::task::JoinHandle<Result<TaskStatus>>)> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let api_client = self.clone();
        let task_id = task_id.to_string();

        let handle = tokio::spawn(async move {
            api_client.poll_task_status(&task_id, move |status| {
                let _ = tx.send(status.clone());
            }).await
        });

        Ok((rx, handle))
    }

    /// Handle HTTP response and parse JSON or error
    async fn handle_response<T>(&self, response: Response) -> Result<T>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let status = response.status();
        let url = response.url().clone();

        if status.is_success() {
            let response_text = response.text().await
                .context("Failed to read response body")?;

            debug!("API Response from {}: {}", url, &response_text);

            serde_json::from_str(&response_text)
                .with_context(|| format!("Failed to parse JSON response: {}", response_text))
        } else {
            // Get response body for error details
            let error_text = response.text().await
                .unwrap_or_else(|_| "Unknown error".to_string());

            error!("API Error from {}: {} - {}", url, status, error_text);

            // Try to parse as API error first
            if let Ok(api_error) = serde_json::from_str::<ApiError>(&error_text) {
                anyhow::bail!("API Error: {}", api_error.detail);
            } else {
                anyhow::bail!("HTTP {} from {}: {}", status, url, error_text);
            }
        }
    }

    /// Update base URL
    #[allow(dead_code)]
    pub fn set_base_url(&mut self, base_url: String) {
        self.base_url = base_url.trim_end_matches('/').to_string();
    }

    /// Update API key
    #[allow(dead_code)]
    pub fn set_api_key(&mut self, api_key: Option<String>) {
        self.api_key = api_key;
    }

    /// Get current base URL
    #[allow(dead_code)]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Check if API key is set
    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = ApiClient::new(
            "http://localhost:8080".to_string(),
            None,
        ).unwrap();

        assert_eq!(client.base_url(), "http://localhost:8080");
        assert!(!client.has_api_key());
    }

    #[tokio::test]
    async fn test_url_normalization() {
        let client = ApiClient::new(
            "http://localhost:8080/".to_string(),
            None,
        ).unwrap();

        assert_eq!(client.base_url(), "http://localhost:8080");
    }

    #[test]
    fn test_query_request_serialization() {
        let request = QueryRequest {
            q: "test question".to_string(),
            topk: 10,
            alpha: 0.7,
            use_rrf: false,
            ..Default::default()
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("test question"));
        assert!(json.contains("\"topk\":10"));
        assert!(json.contains("\"alpha\":0.7"));
    }

    #[test]
    fn test_ingest_request_serialization() {
        let request = IngestRequest {
            source: "file".to_string(),
            path: "/path/to/file.pdf".to_string(),
            tags: Some(vec!["test".to_string(), "doc".to_string()]),
            overwrite: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("file"));
        assert!(json.contains("/path/to/file.pdf"));
        assert!(json.contains("test"));
        assert!(json.contains("\"overwrite\":true"));
    }
}
