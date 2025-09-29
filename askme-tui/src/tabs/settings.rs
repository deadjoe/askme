use crate::api::QueryRequest;
use crate::app::Config;
use tui_textarea::{TextArea, Input};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

/// Settings tab state and functionality
pub struct SettingsTab {
    /// API URL input
    pub api_url_input: TextArea<'static>,

    /// API key input
    pub api_key_input: TextArea<'static>,

    /// Default query parameters
    pub default_topk: u32,
    pub default_alpha: f64,
    pub default_use_rrf: bool,
    pub default_rrf_k: u32,
    pub default_max_passages: u32,
    pub auto_save: bool,

    /// UI state
    pub focused_field: SettingsField,
    pub config_modified: bool,
    pub show_api_key: bool,
}

/// Fields that can be focused in the settings tab
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SettingsField {
    ApiUrl,
    ApiKey,
    DefaultTopK,
    DefaultAlpha,
    DefaultRrfK,
    DefaultMaxPassages,
    AutoSave,
    ShowApiKey,
    Save,
    Reset,
}

impl SettingsField {
    pub fn all() -> Vec<Self> {
        vec![
            Self::ApiUrl,
            Self::ApiKey,
            Self::DefaultTopK,
            Self::DefaultAlpha,
            Self::DefaultRrfK,
            Self::DefaultMaxPassages,
            Self::AutoSave,
            Self::ShowApiKey,
            Self::Save,
            Self::Reset,
        ]
    }
}

impl SettingsTab {
    /// Create a new settings tab
    pub fn new(config: Config) -> Self {
        let mut api_url_input = TextArea::default();
        api_url_input.set_block(
            Block::default()
                .title("API URL")
                .borders(Borders::ALL)
        );
        api_url_input.insert_str(&config.api_url);

        let mut api_key_input = TextArea::default();
        api_key_input.set_block(
            Block::default()
                .title("API Key")
                .borders(Borders::ALL)
        );
        if let Some(ref api_key) = config.api_key {
            api_key_input.insert_str(api_key);
        }

        Self {
            api_url_input,
            api_key_input,
            default_topk: config.default_query.topk,
            default_alpha: config.default_query.alpha,
            default_use_rrf: config.default_query.use_rrf,
            default_rrf_k: config.default_query.rrf_k,
            default_max_passages: config.default_query.max_passages,
            auto_save: config.auto_save,
            focused_field: SettingsField::ApiUrl,
            config_modified: false,
            show_api_key: false,
        }
    }

    /// Get the current configuration
    pub fn get_config(&self) -> Config {
        let api_url = self.api_url_input.lines().join("");
        let api_key_text = self.api_key_input.lines().join("");
        let api_key = if api_key_text.trim().is_empty() {
            None
        } else {
            Some(api_key_text)
        };

        let default_query = QueryRequest {
            q: String::new(),
            topk: self.default_topk,
            alpha: self.default_alpha,
            use_rrf: self.default_use_rrf,
            rrf_k: self.default_rrf_k,
            use_hyde: false,
            use_rag_fusion: false,
            reranker: "qwen_local".to_string(),
            max_passages: self.default_max_passages,
            include_debug: false,
        };

        Config {
            api_url,
            api_key,
            default_query,
            output_format: "text".to_string(),
            auto_save: self.auto_save,
        }
    }

    /// Update from configuration
    pub fn update_from_config(&mut self, config: &Config) {
        // Clear and update API URL
        self.api_url_input.select_all();
        self.api_url_input.cut();
        self.api_url_input.insert_str(&config.api_url);

        // Clear and update API key
        self.api_key_input.select_all();
        self.api_key_input.cut();
        if let Some(ref api_key) = config.api_key {
            self.api_key_input.insert_str(api_key);
        }

        // Update default parameters
        self.default_topk = config.default_query.topk;
        self.default_alpha = config.default_query.alpha;
        self.default_use_rrf = config.default_query.use_rrf;
        self.default_rrf_k = config.default_query.rrf_k;
        self.default_max_passages = config.default_query.max_passages;
        self.auto_save = config.auto_save;

        self.config_modified = false;
    }

    /// Reset to defaults
    pub fn reset_to_defaults(&mut self) {
        let default_config = Config::default();
        self.update_from_config(&default_config);
        self.config_modified = true;
    }

    /// Mark configuration as modified
    pub fn mark_modified(&mut self) {
        self.config_modified = true;
    }

    /// Check if configuration is modified
    #[allow(dead_code)]
    pub fn is_modified(&self) -> bool {
        self.config_modified
    }

    /// Clear modified flag
    pub fn clear_modified(&mut self) {
        self.config_modified = false;
    }

    /// Handle keyboard input
    pub fn handle_input(&mut self, key: ratatui::crossterm::event::KeyEvent) -> bool {
        match self.focused_field {
            SettingsField::ApiUrl => {
                match key.code {
                    ratatui::crossterm::event::KeyCode::Tab => {
                        self.next_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::BackTab => {
                        self.previous_field();
                        true
                    }
                    _ => {
                        self.api_url_input.input(Input::from(key));
                        self.mark_modified();
                        true
                    }
                }
            }
            SettingsField::ApiKey => {
                match key.code {
                    ratatui::crossterm::event::KeyCode::Tab => {
                        self.next_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::BackTab => {
                        self.previous_field();
                        true
                    }
                    _ => {
                        self.api_key_input.input(Input::from(key));
                        self.mark_modified();
                        true
                    }
                }
            }
            _ => {
                match key.code {
                    ratatui::crossterm::event::KeyCode::Tab => {
                        self.next_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::BackTab => {
                        self.previous_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Enter => {
                        match self.focused_field {
                            SettingsField::Save | SettingsField::Reset => {
                                false // Let app handle these actions
                            }
                            _ => {
                                self.next_field();
                                true
                            }
                        }
                    }
                    ratatui::crossterm::event::KeyCode::Char(' ') => {
                        self.handle_space();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Left | ratatui::crossterm::event::KeyCode::Right => {
                        self.handle_arrow_keys(key);
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Char(c) if c.is_ascii_digit() => {
                        self.handle_digit_input(c);
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Char('.') => {
                        self.handle_decimal_input();
                        true
                    }
                    _ => true
                }
            }
        }
    }

    /// Handle space key for toggleable fields
    fn handle_space(&mut self) {
        match self.focused_field {
            SettingsField::AutoSave => {
                self.auto_save = !self.auto_save;
                self.mark_modified();
            }
            SettingsField::ShowApiKey => {
                self.show_api_key = !self.show_api_key;
            }
            _ => {}
        }
    }

    /// Handle arrow keys for numeric fields
    fn handle_arrow_keys(&mut self, key: ratatui::crossterm::event::KeyEvent) {
        let increment = key.code == ratatui::crossterm::event::KeyCode::Right;

        match self.focused_field {
            SettingsField::DefaultTopK => {
                if increment {
                    self.default_topk = (self.default_topk + 1).min(100);
                } else {
                    self.default_topk = self.default_topk.saturating_sub(1).max(1);
                }
                self.mark_modified();
            }
            SettingsField::DefaultAlpha => {
                if increment {
                    self.default_alpha = (self.default_alpha + 0.1).min(1.0);
                } else {
                    self.default_alpha = (self.default_alpha - 0.1).max(0.0);
                }
                self.mark_modified();
            }
            SettingsField::DefaultRrfK => {
                if increment {
                    self.default_rrf_k = self.default_rrf_k + 1;
                } else {
                    self.default_rrf_k = self.default_rrf_k.saturating_sub(1);
                }
                self.mark_modified();
            }
            SettingsField::DefaultMaxPassages => {
                if increment {
                    self.default_max_passages = (self.default_max_passages + 1).min(20);
                } else {
                    self.default_max_passages = self.default_max_passages.saturating_sub(1).max(1);
                }
                self.mark_modified();
            }
            _ => {}
        }
    }

    /// Handle digit input for numeric fields
    fn handle_digit_input(&mut self, digit: char) {
        let digit_val = digit.to_digit(10).unwrap() as u32;

        match self.focused_field {
            SettingsField::DefaultTopK => {
                let new_val = self.default_topk * 10 + digit_val;
                if new_val <= 100 {
                    self.default_topk = new_val;
                    self.mark_modified();
                }
            }
            SettingsField::DefaultRrfK => {
                let new_val = self.default_rrf_k * 10 + digit_val;
                self.default_rrf_k = new_val;
                self.mark_modified();
            }
            SettingsField::DefaultMaxPassages => {
                let new_val = self.default_max_passages * 10 + digit_val;
                if new_val <= 20 {
                    self.default_max_passages = new_val;
                    self.mark_modified();
                }
            }
            _ => {}
        }
    }

    /// Handle decimal input for alpha
    fn handle_decimal_input(&mut self) {
        if self.focused_field == SettingsField::DefaultAlpha {
            // Reset alpha to allow new decimal input
            self.default_alpha = 0.0;
            self.mark_modified();
        }
    }

    /// Move to next field
    fn next_field(&mut self) {
        let fields = SettingsField::all();
        if let Some(current_idx) = fields.iter().position(|&f| f == self.focused_field) {
            let next_idx = (current_idx + 1) % fields.len();
            self.focused_field = fields[next_idx];
        }
    }

    /// Move to previous field
    fn previous_field(&mut self) {
        let fields = SettingsField::all();
        if let Some(current_idx) = fields.iter().position(|&f| f == self.focused_field) {
            let prev_idx = if current_idx == 0 {
                fields.len() - 1
            } else {
                current_idx - 1
            };
            self.focused_field = fields[prev_idx];
        }
    }

    /// Render the settings tab
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let main_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);

        // Left panel: Configuration form
        self.render_config_form(frame, main_layout[0]);

        // Right panel: Help and actions
        self.render_help_and_actions(frame, main_layout[1]);
    }

    /// Render configuration form
    fn render_config_form(&mut self, frame: &mut Frame, area: Rect) {
        let form_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),  // API URL
                Constraint::Length(4),  // API Key
                Constraint::Min(8),     // Default parameters
            ])
            .split(area);

        // API URL input
        let api_url_focused = self.focused_field == SettingsField::ApiUrl;
        if api_url_focused {
            self.api_url_input.set_style(Style::default().fg(Color::Yellow));
        } else {
            self.api_url_input.set_style(Style::default());
        }
        frame.render_widget(&self.api_url_input, form_layout[0]);

        // API Key input
        let api_key_focused = self.focused_field == SettingsField::ApiKey;
        if api_key_focused {
            self.api_key_input.set_style(Style::default().fg(Color::Yellow));
        } else {
            self.api_key_input.set_style(Style::default());
        }

        // Handle API key display (masked or not)
        if !self.show_api_key && !self.api_key_input.lines().join("").is_empty() {
            // Create a masked version for display
            let masked_text = "*".repeat(self.api_key_input.lines().join("").len());
            let mut masked_textarea = TextArea::from(vec![masked_text]);
            masked_textarea.set_block(self.api_key_input.block().cloned().unwrap_or_default());
            if api_key_focused {
                masked_textarea.set_style(Style::default().fg(Color::Yellow));
            } else {
                masked_textarea.set_style(Style::default());
            }
            frame.render_widget(&masked_textarea, form_layout[1]);
        } else {
            frame.render_widget(&self.api_key_input, form_layout[1]);
        }

        // Default parameters
        self.render_default_parameters(frame, form_layout[2]);
    }

    /// Render default parameters section
    fn render_default_parameters(&self, frame: &mut Frame, area: Rect) {
        let params_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),  // TopK
                Constraint::Length(1),  // Alpha
                Constraint::Length(1),  // RRF K
                Constraint::Length(1),  // Max Passages
                Constraint::Length(1),  // Auto Save
                Constraint::Length(1),  // Show API Key
                Constraint::Min(2),     // Buttons
            ])
            .split(area);

        let mut param_idx = 0;

        // Default TopK
        let topk_style = if self.focused_field == SettingsField::DefaultTopK {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        frame.render_widget(
            Paragraph::new(format!("Default TopK: {} (1-100)", self.default_topk))
                .style(topk_style),
            params_layout[param_idx]
        );
        param_idx += 1;

        // Default Alpha
        let alpha_style = if self.focused_field == SettingsField::DefaultAlpha {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        frame.render_widget(
            Paragraph::new(format!("Default Alpha: {:.2} (0.0-1.0)", self.default_alpha))
                .style(alpha_style),
            params_layout[param_idx]
        );
        param_idx += 1;

        // Default RRF K
        let rrf_k_style = if self.focused_field == SettingsField::DefaultRrfK {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        frame.render_widget(
            Paragraph::new(format!("Default RRF K: {}", self.default_rrf_k))
                .style(rrf_k_style),
            params_layout[param_idx]
        );
        param_idx += 1;

        // Default Max Passages
        let max_passages_style = if self.focused_field == SettingsField::DefaultMaxPassages {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        frame.render_widget(
            Paragraph::new(format!("Default Max Passages: {} (1-20)", self.default_max_passages))
                .style(max_passages_style),
            params_layout[param_idx]
        );
        param_idx += 1;

        // Auto Save
        let auto_save_style = if self.focused_field == SettingsField::AutoSave {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        frame.render_widget(
            Paragraph::new(format!("Auto Save: {} (Space to toggle)",
                if self.auto_save { "✓" } else { "✗" }))
                .style(auto_save_style),
            params_layout[param_idx]
        );
        param_idx += 1;

        // Show API Key
        let show_api_key_style = if self.focused_field == SettingsField::ShowApiKey {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        frame.render_widget(
            Paragraph::new(format!("Show API Key: {} (Space to toggle)",
                if self.show_api_key { "✓" } else { "✗" }))
                .style(show_api_key_style),
            params_layout[param_idx]
        );
        param_idx += 1;

        // Action buttons
        self.render_action_buttons(frame, params_layout[param_idx]);
    }

    /// Render action buttons
    fn render_action_buttons(&self, frame: &mut Frame, area: Rect) {
        let button_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Save button
        let save_style = if self.focused_field == SettingsField::Save {
            Style::default().bg(Color::Green).fg(Color::Black)
        } else if self.config_modified {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Gray)
        };

        let save_text = if self.config_modified {
            "Save Config *"
        } else {
            "Save Config"
        };

        let save_button = Paragraph::new(save_text)
            .block(Block::default().borders(Borders::ALL))
            .style(save_style);

        frame.render_widget(save_button, button_layout[0]);

        // Reset button
        let reset_style = if self.focused_field == SettingsField::Reset {
            Style::default().bg(Color::Red).fg(Color::Black)
        } else {
            Style::default().fg(Color::Red)
        };

        let reset_button = Paragraph::new("Reset to Defaults")
            .block(Block::default().borders(Borders::ALL))
            .style(reset_style);

        frame.render_widget(reset_button, button_layout[1]);
    }

    /// Render help and actions section
    fn render_help_and_actions(&self, frame: &mut Frame, area: Rect) {
        let help_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),    // Help text
                Constraint::Length(5),  // Status
            ])
            .split(area);

        // Help text
        let help_text = vec![
            "Settings Help:",
            "",
            "Tab/Shift+Tab: Navigate fields",
            "Enter: Activate/Next field",
            "Space: Toggle boolean options",
            "←/→: Adjust numeric values",
            "Digits: Direct numeric input",
            "",
            "API URL: askme server address",
            "API Key: Authentication key",
            "Default values apply to new queries",
            "",
            "* indicates unsaved changes",
        ];

        let help_widget = Paragraph::new(help_text.join("\n"))
            .block(Block::default().borders(Borders::ALL).title("Help"))
            .wrap(Wrap { trim: true });

        frame.render_widget(help_widget, help_layout[0]);

        // Status
        let status_text = if self.config_modified {
            "Status: Configuration modified\nUse Save to apply changes"
        } else {
            "Status: Configuration saved\nAll changes applied"
        };

        let status_style = if self.config_modified {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Green)
        };

        let status_widget = Paragraph::new(status_text)
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .style(status_style)
            .wrap(Wrap { trim: true });

        frame.render_widget(status_widget, help_layout[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_tab_creation() {
        let config = Config::default();
        let tab = SettingsTab::new(config.clone());

        assert_eq!(tab.focused_field, SettingsField::ApiUrl);
        assert!(!tab.config_modified);
        assert!(!tab.show_api_key);
        assert_eq!(tab.default_topk, config.default_query.topk);
    }

    #[test]
    fn test_config_generation() {
        let original_config = Config::default();
        let mut tab = SettingsTab::new(original_config);

        // Modify some values
        tab.default_topk = 75;
        tab.default_alpha = 0.8;
        tab.auto_save = false;

        let new_config = tab.get_config();
        assert_eq!(new_config.default_query.topk, 75);
        assert_eq!(new_config.default_query.alpha, 0.8);
        assert!(!new_config.auto_save);
    }

    #[test]
    fn test_field_navigation() {
        let config = Config::default();
        let mut tab = SettingsTab::new(config);

        assert_eq!(tab.focused_field, SettingsField::ApiUrl);
        tab.next_field();
        assert_eq!(tab.focused_field, SettingsField::ApiKey);

        // Test wrap around
        tab.focused_field = SettingsField::Reset;
        tab.next_field();
        assert_eq!(tab.focused_field, SettingsField::ApiUrl);
    }

    #[test]
    fn test_modification_tracking() {
        let config = Config::default();
        let mut tab = SettingsTab::new(config);

        assert!(!tab.is_modified());

        tab.mark_modified();
        assert!(tab.is_modified());

        tab.clear_modified();
        assert!(!tab.is_modified());
    }

    #[test]
    fn test_numeric_input_bounds() {
        let config = Config::default();
        let mut tab = SettingsTab::new(config);

        // Test topk bounds
        tab.focused_field = SettingsField::DefaultTopK;
        tab.default_topk = 95;
        tab.handle_digit_input('9');
        assert_eq!(tab.default_topk, 95); // Should not exceed 100

        // Test max passages bounds
        tab.focused_field = SettingsField::DefaultMaxPassages;
        tab.default_max_passages = 19;
        tab.handle_digit_input('5');
        assert_eq!(tab.default_max_passages, 19); // Should not exceed 20
    }
}
