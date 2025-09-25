use crate::api::{ApiClient, QueryRequest, QueryResponse};
use crate::tabs::{QueryTab, IngestTab, SettingsTab};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{debug, error, info};
use tokio::task::JoinHandle;
use throbber_widgets_tui::ThrobberState;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub api_url: String,
    pub api_key: Option<String>,
    pub default_query: QueryRequest,
    pub output_format: String,
    pub auto_save: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_url: "http://localhost:8080".to_string(),
            api_key: std::env::var("ASKME_API_KEY").ok(),
            default_query: QueryRequest::default(),
            output_format: "text".to_string(),
            auto_save: true,
        }
    }
}

/// Active tab in the UI
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tab {
    Query,
    Ingest,
    Settings,
    Help,
}

impl Tab {
    pub fn titles() -> Vec<&'static str> {
        vec!["Query", "Ingest", "Settings", "Help"]
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Query,
            1 => Self::Ingest,
            2 => Self::Settings,
            3 => Self::Help,
            _ => Self::Query,
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::Query => 0,
            Self::Ingest => 1,
            Self::Settings => 2,
            Self::Help => 3,
        }
    }
}

/// Application state
pub struct App {
    pub config: Config,
    pub api_client: ApiClient,
    pub current_tab: Tab,
    pub should_quit: bool,
    pub status_message: String,
    pub error_message: Option<String>,

    // Tab states
    pub query_tab: QueryTab,
    pub ingest_tab: IngestTab,
    pub settings_tab: SettingsTab,

    // Config file path
    config_path: PathBuf,

    // Backend info (for Status bar augmentation)
    pub vector_backend: Option<String>,
    pub collection_name: Option<String>,
    pub collection_chunks: Option<u64>,

    // Background query task
    pub query_task: Option<JoinHandle<anyhow::Result<QueryResponse>>>,
    pub throbber: ThrobberState,

    // Background ingest task (submits request)
    pub ingest_task: Option<JoinHandle<anyhow::Result<crate::api::types::IngestResponse>>>,

    // Current ingest task ID for status polling
    pub current_ingest_task_id: Option<String>,

    // Channel for receiving ingest progress updates
    pub ingest_progress_rx: Option<tokio::sync::mpsc::UnboundedReceiver<crate::api::types::TaskStatus>>,

    // Background ingest polling task handle
    pub ingest_polling_task: Option<JoinHandle<anyhow::Result<crate::api::types::TaskStatus>>>,
}

impl App {
    /// Create a new application instance
    pub fn new() -> Result<Self> {
        let config_path = Self::config_path()?;
        let config = Self::load_config(&config_path)?;

        let api_client = ApiClient::new(
            config.api_url.clone(),
            config.api_key.clone(),
        )?;

        Ok(Self {
            api_client,
            current_tab: Tab::Query,
            should_quit: false,
            status_message: "Ready".to_string(),
            error_message: None,
            query_tab: QueryTab::new(config.default_query.clone()),
            ingest_tab: IngestTab::new(),
            settings_tab: SettingsTab::new(config.clone()),
            config,
            config_path,
            vector_backend: None,
            collection_name: None,
            collection_chunks: None,
            query_task: None,
            throbber: ThrobberState::default(),
            ingest_task: None,
            current_ingest_task_id: None,
            ingest_progress_rx: None,
            ingest_polling_task: None,
        })
    }

    /// Get configuration file path
    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .context("Failed to get config directory")?;

        let app_config_dir = config_dir.join("askme-tui");
        std::fs::create_dir_all(&app_config_dir)
            .context("Failed to create config directory")?;

        Ok(app_config_dir.join("config.toml"))
    }

    /// Load configuration from file
    fn load_config(path: &PathBuf) -> Result<Config> {
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .context("Failed to read config file")?;

            let config: Config = toml::from_str(&content)
                .context("Failed to parse config file")?;

            info!("Config loaded from: {}", path.display());
            Ok(config)
        } else {
            info!("Config file not found, using defaults");
            Ok(Config::default())
        }
    }

    /// Save configuration to file
    pub fn save_config(&self) -> Result<()> {
        let content = toml::to_string_pretty(&self.config)
            .context("Failed to serialize config")?;

        std::fs::write(&self.config_path, content)
            .context("Failed to write config file")?;

        debug!("Config saved to: {}", self.config_path.display());
        Ok(())
    }

    /// Update API client with current config
    pub fn update_api_client(&mut self) -> Result<()> {
        self.api_client = ApiClient::new(
            self.config.api_url.clone(),
            self.config.api_key.clone(),
        )?;

        self.set_status("API client updated");
        Ok(())
    }

    /// Check API health
    pub async fn check_api_health(&mut self) {
        match self.api_client.health_check().await {
            Ok(health) => {
                // Extract vector backend if provided
                if let Some(comps) = &health.components {
                    if let Some(vb) = comps.get("vector_backend").and_then(|v| v.as_str()) {
                        self.vector_backend = Some(vb.to_string());
                    }
                    if let Some(name) = comps.get("collection_name").and_then(|v| v.as_str()) {
                        self.collection_name = Some(name.to_string());
                    }
                }
                // Try to fetch ingest stats for chunk counts (non-fatal)
                if let Ok(stats) = self.api_client.ingest_stats().await {
                    self.collection_chunks = stats.total_chunks;
                }

                self.set_status(&format!("API healthy: {}", health.status));
                self.clear_error();
            }
            Err(e) => {
                self.set_error(&format!("API health check failed: {}", e));
            }
        }
    }

    // Removed unused next_tab/previous_tab to keep code clean

    /// Set current tab by index
    pub fn set_tab(&mut self, index: usize) {
        self.current_tab = Tab::from_index(index);
    }

    /// Set status message
    pub fn set_status(&mut self, message: &str) {
        self.status_message = message.to_string();
        debug!("Status: {}", message);
    }

    /// Set error message
    pub fn set_error(&mut self, message: &str) {
        self.error_message = Some(message.to_string());
        error!("Error: {}", message);
    }

    /// Clear error message
    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    /// Quit the application
    pub fn quit(&mut self) {
        self.should_quit = true;
        if self.config.auto_save {
            if let Err(e) = self.save_config() {
                error!("Failed to save config on quit: {}", e);
            }
        }
    }

    /// Get current status for display
    pub fn display_status(&self) -> String {
        let api_status = if self.api_client.has_api_key() {
            "🔐"
        } else {
            "🔓"
        };

        let error_indicator = if self.error_message.is_some() {
            " ❌"
        } else {
            ""
        };

        let mut extra = String::new();
        if let Some(ref vb) = self.vector_backend {
            extra.push_str(&format!(" | DB: {}", vb));
        }
        if let Some(ref name) = self.collection_name {
            extra.push_str(&format!("/{}", name));
        }
        if let Some(chunks) = self.collection_chunks {
            extra.push_str(&format!(" (chunks: {})", chunks));
        }

        format!(
            "{} | API: {} {}{}{}",
            self.status_message,
            self.api_client.base_url(),
            api_status,
            error_indicator,
            extra
        )
    }

    /// Get current error message for display
    pub fn display_error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    // Removed unused process_query; query now runs in a background task

    // Removed legacy ingest methods; ingestion now runs as a background task
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tab_navigation() {
        assert_eq!(Tab::from_index(0), Tab::Query);
        assert_eq!(Tab::from_index(1), Tab::Ingest);
        assert_eq!(Tab::from_index(2), Tab::Settings);
        assert_eq!(Tab::from_index(3), Tab::Help);
        assert_eq!(Tab::from_index(99), Tab::Query); // Out of bounds defaults to Query
    }

    #[test]
    fn test_tab_titles() {
        let titles = Tab::titles();
        assert_eq!(titles.len(), 4);
        assert_eq!(titles[0], "Query");
        assert_eq!(titles[1], "Ingest");
    }

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.api_url, "http://localhost:8080");
        assert_eq!(config.default_query.topk, 50);
        assert_eq!(config.default_query.alpha, 0.5);
    }
}
