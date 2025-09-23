use crate::api::{IngestRequest, TaskStatus, SourceType};
use tui_textarea::{TextArea, Input};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

/// Ingest tab state and functionality
pub struct IngestTab {
    /// Path input field
    pub path_input: TextArea<'static>,

    /// Tags input field
    pub tags_input: TextArea<'static>,

    /// Ingest options
    pub overwrite: bool,

    /// UI state
    pub focused_field: IngestField,
    pub source_type: Option<SourceType>,
    pub current_task: Option<TaskStatus>,
    pub task_history: Vec<TaskStatus>,
    pub processing: bool,
}

/// Fields that can be focused in the ingest tab
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IngestField {
    Path,
    Tags,
    Options,
    Submit,
}

impl IngestField {
    pub fn all() -> Vec<Self> {
        vec![Self::Path, Self::Tags, Self::Options, Self::Submit]
    }
}

impl IngestTab {
    /// Create a new ingest tab
    pub fn new() -> Self {
        let mut path_input = TextArea::default();
        path_input.set_block(
            Block::default()
                .title("Source Path/URL")
                .borders(Borders::ALL)
        );
        path_input.set_placeholder_text("Enter file path, directory, or URL...");

        let mut tags_input = TextArea::default();
        tags_input.set_block(
            Block::default()
                .title("Tags (comma-separated)")
                .borders(Borders::ALL)
        );
        tags_input.set_placeholder_text("project,team,v1.0");

        Self {
            path_input,
            tags_input,
            overwrite: false,
            focused_field: IngestField::Path,
            source_type: None,
            current_task: None,
            task_history: Vec::new(),
            processing: false,
        }
    }

    /// Get the current ingest request
    pub fn get_ingest_request(&self) -> IngestRequest {
        let path_text = self.path_input.lines().join("");
        let tags_text = self.tags_input.lines().join("");

        let detected_source_type = SourceType::detect(&path_text);
        let source_type = self.source_type
            .as_ref()
            .unwrap_or(&detected_source_type);

        let tags = if tags_text.trim().is_empty() {
            None
        } else {
            Some(
                tags_text
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            )
        };

        IngestRequest {
            source: source_type.as_str().to_string(),
            path: path_text,
            tags,
            overwrite: self.overwrite,
        }
    }

    /// Update the source type when path changes
    pub fn update_source_type(&mut self) {
        let path_text = self.path_input.lines().join("");
        if !path_text.is_empty() {
            self.source_type = Some(SourceType::detect(&path_text));
        } else {
            self.source_type = None;
        }
    }

    /// Update task status (used in tests)
    #[cfg(test)]
    pub fn update_task_status(&mut self, status: TaskStatus) {
        // Update current task
        self.current_task = Some(status.clone());

        // Update or add to history
        if let Some(existing_idx) = self.task_history
            .iter()
            .position(|t| t.task_id == status.task_id)
        {
            self.task_history[existing_idx] = status;
        } else {
            self.task_history.push(status);
        }

        // Keep only last 10 tasks in history
        if self.task_history.len() > 10 {
            self.task_history.remove(0);
        }

        // Check if processing is complete
        if let Some(ref task) = self.current_task {
            self.processing = !matches!(task.status.as_str(), "completed" | "failed");
        }
    }

    /// Start processing
    pub fn start_processing(&mut self) {
        self.processing = true;
        self.current_task = None;
    }

    /// Clear current task
    #[allow(dead_code)]
    pub fn clear_current_task(&mut self) {
        self.current_task = None;
        self.processing = false;
    }

    /// Handle keyboard input
    pub fn handle_input(&mut self, key: ratatui::crossterm::event::KeyEvent) -> bool {
        // Global Up/Down to move focus like Query tab
        match key.code {
            ratatui::crossterm::event::KeyCode::Up => {
                self.previous_field();
                return true;
            }
            ratatui::crossterm::event::KeyCode::Down => {
                self.next_field();
                return true;
            }
            _ => {}
        }
        match self.focused_field {
            IngestField::Path => {
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
                        self.path_input.input(Input::from(key));
                        self.update_source_type();
                        true
                    }
                }
            }
            IngestField::Tags => {
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
                        self.tags_input.input(Input::from(key));
                        true
                    }
                }
            }
            IngestField::Options => {
                match key.code {
                    ratatui::crossterm::event::KeyCode::Tab => {
                        self.next_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::BackTab => {
                        self.previous_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Char(' ') => {
                        self.overwrite = !self.overwrite;
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Enter => {
                        self.next_field();
                        true
                    }
                    _ => true
                }
            }
            IngestField::Submit => {
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
                        false // Let app handle submission
                    }
                    _ => true
                }
            }
        }
    }

    /// Move to next field
    fn next_field(&mut self) {
        let fields = IngestField::all();
        if let Some(current_idx) = fields.iter().position(|&f| f == self.focused_field) {
            let next_idx = (current_idx + 1) % fields.len();
            self.focused_field = fields[next_idx];
        }
    }

    /// Move to previous field
    fn previous_field(&mut self) {
        let fields = IngestField::all();
        if let Some(current_idx) = fields.iter().position(|&f| f == self.focused_field) {
            let prev_idx = if current_idx == 0 {
                fields.len() - 1
            } else {
                current_idx - 1
            };
            self.focused_field = fields[prev_idx];
        }
    }

    /// Render the ingest tab
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let main_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Left panel: Input form
        self.render_input_form(frame, main_layout[0]);

        // Right panel: Status and history
        self.render_status(frame, main_layout[1]);
    }

    /// Render input form
    fn render_input_form(&mut self, frame: &mut Frame, area: Rect) {
        let form_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),  // Path input
                Constraint::Length(4),  // Tags input
                Constraint::Length(4),  // Options
                Constraint::Length(3),  // Submit button
                Constraint::Min(2),     // Info
            ])
            .split(area);

        // Path input with dynamic border + placeholder color (same as Query.Question)
        let path_focused = self.focused_field == IngestField::Path;
        let mut path_block = Block::default().title("Source Path/URL").borders(Borders::ALL);
        if path_focused {
            path_block = path_block.border_style(Style::default().fg(Color::Yellow));
            self.path_input.set_style(Style::default().fg(Color::Yellow));
            self.path_input.set_placeholder_style(Style::default().fg(Color::Yellow));
        } else {
            self.path_input.set_style(Style::default());
            self.path_input.set_placeholder_style(Style::default().fg(Color::Cyan));
        }
        self.path_input.set_block(path_block);
        frame.render_widget(&self.path_input, form_layout[0]);

        // Tags input with dynamic border + placeholder color
        let tags_focused = self.focused_field == IngestField::Tags;
        let mut tags_block = Block::default().title("Tags (comma-separated)").borders(Borders::ALL);
        if tags_focused {
            tags_block = tags_block.border_style(Style::default().fg(Color::Yellow));
            self.tags_input.set_style(Style::default().fg(Color::Yellow));
            self.tags_input.set_placeholder_style(Style::default().fg(Color::Yellow));
        } else {
            self.tags_input.set_style(Style::default());
            self.tags_input.set_placeholder_style(Style::default().fg(Color::Cyan));
        }
        self.tags_input.set_block(tags_block);
        frame.render_widget(&self.tags_input, form_layout[1]);

        // Options
        let options_style = if self.focused_field == IngestField::Options {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        let options_text = format!(
            "Overwrite existing: {} (Space to toggle)",
            if self.overwrite { "✓" } else { "✗" }
        );

        let options_widget = Paragraph::new(options_text)
            .block(Block::default().borders(Borders::ALL).title("Options"))
            .style(options_style)
            .wrap(Wrap { trim: true });

        frame.render_widget(options_widget, form_layout[2]);

        // Submit button
        let submit_style = if self.focused_field == IngestField::Submit {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        let submit_text = if self.processing {
            "Processing..."
        } else {
            "Start Ingestion (Enter)"
        };

        let submit_button = Paragraph::new(submit_text)
            .block(Block::default().borders(Borders::ALL).title("Submit"))
            .style(submit_style)
            .wrap(Wrap { trim: true });

        frame.render_widget(submit_button, form_layout[3]);

        // Info section
        let info_text = if let Some(ref source_type) = self.source_type {
            format!(
                "Detected source type: {}\n\nTab: Next field\nEnter: Submit",
                match source_type {
                    SourceType::File => "📄 File",
                    SourceType::Directory => "📁 Directory",
                    SourceType::Url => "🌐 URL",
                }
            )
        } else {
            "Enter a path to detect source type\n\nSupported:\n• File paths\n• Directory paths\n• HTTP/HTTPS URLs".to_string()
        };

        let info_widget = Paragraph::new(info_text)
            .block(Block::default().borders(Borders::ALL).title("Info"))
            .wrap(Wrap { trim: true });

        frame.render_widget(info_widget, form_layout[4]);
    }

    /// Render status section
    fn render_status(&self, frame: &mut Frame, area: Rect) {
        let status_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),  // Current task
                Constraint::Min(5),     // Task history
            ])
            .split(area);

        // Current task status
        self.render_current_task(frame, status_layout[0]);

        // Task history
        self.render_task_history(frame, status_layout[1]);
    }

    /// Render current task status
    fn render_current_task(&self, frame: &mut Frame, area: Rect) {
        if let Some(ref task) = self.current_task {
            let task_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),  // Basic info
                    Constraint::Length(2),  // Progress
                    Constraint::Min(2),     // Details
                ])
                .split(area);

            // Basic info
            let basic_info = format!(
                "Task ID: {}\nStatus: {}",
                &task.task_id[..8],
                task.status
            );

            let basic_widget = Paragraph::new(basic_info)
                .block(Block::default().borders(Borders::ALL).title("Current Task"));

            frame.render_widget(basic_widget, task_layout[0]);

            // Progress bar with better colors and animation
            if let Some(progress) = task.progress {
                let progress_percent = (progress as u16).min(100);
                let gauge_color = match task.status.as_str() {
                    "processing" => Color::Cyan,
                    "completed" => Color::Green,
                    "failed" => Color::Red,
                    _ => Color::Yellow,
                };

                let label = if progress_percent == 100 {
                    "✅ Completed".to_string()
                } else {
                    format!("⚙️ Processing: {:.1}%", progress)
                };

                let gauge = Gauge::default()
                    .block(Block::default().borders(Borders::ALL).title("Progress"))
                    .gauge_style(Style::default().fg(gauge_color))
                    .percent(progress_percent)
                    .label(label);

                frame.render_widget(gauge, task_layout[1]);
            } else {
                let placeholder = Paragraph::new("No progress data")
                    .block(Block::default().borders(Borders::ALL).title("Progress"));
                frame.render_widget(placeholder, task_layout[1]);
            }

            // Details
            let details_text = vec![
                format!("Documents processed: {}", task.documents_processed.unwrap_or(0)),
                if let Some(ref error) = task.error_message {
                    format!("Error: {}", error)
                } else {
                    "No errors".to_string()
                },
                if let Some(ref started) = task.started_at {
                    format!("Started: {}", started.format("%H:%M:%S"))
                } else {
                    "Start time: Unknown".to_string()
                },
            ];

            let details_widget = Paragraph::new(details_text.join("\n"))
                .block(Block::default().borders(Borders::ALL).title("Details"))
                .wrap(Wrap { trim: true });

            frame.render_widget(details_widget, task_layout[2]);
        } else if self.processing {
            let processing_widget = Paragraph::new("Submitting ingestion request...\n\nPlease wait...")
                .block(Block::default().borders(Borders::ALL).title("Current Task"))
                .style(Style::default().fg(Color::Yellow))
                .wrap(Wrap { trim: true });

            frame.render_widget(processing_widget, area);
        } else {
            let placeholder_widget = Paragraph::new("No active ingestion task.\n\nFill out the form and submit to start.")
                .block(Block::default().borders(Borders::ALL).title("Current Task"))
                .style(Style::default().fg(Color::Gray))
                .wrap(Wrap { trim: true });

            frame.render_widget(placeholder_widget, area);
        }
    }

    /// Render task history
    fn render_task_history(&self, frame: &mut Frame, area: Rect) {
        if self.task_history.is_empty() {
            let placeholder = Paragraph::new("No task history yet.")
                .block(Block::default().borders(Borders::ALL).title("Recent Tasks"))
                .style(Style::default().fg(Color::Gray));

            frame.render_widget(placeholder, area);
        } else {
            let history_items: Vec<ListItem> = self.task_history
                .iter()
                .rev() // Show most recent first
                .take(10)
                .map(|task| {
                    let status_icon = match task.status.as_str() {
                        "completed" => "✅",
                        "failed" => "❌",
                        "processing" => "⚙️",
                        "queued" => "⏳",
                        _ => "❓",
                    };

                    let docs_processed = task.documents_processed.unwrap_or(0);
                    let display_text = format!(
                        "{} {} | {} docs | {}",
                        status_icon,
                        &task.task_id[..8],
                        docs_processed,
                        task.status
                    );

                    let style = match task.status.as_str() {
                        "completed" => Style::default().fg(Color::Green),
                        "failed" => Style::default().fg(Color::Red),
                        "processing" => Style::default().fg(Color::Yellow),
                        _ => Style::default(),
                    };

                    ListItem::new(display_text).style(style)
                })
                .collect();

            let history_widget = List::new(history_items)
                .block(Block::default().borders(Borders::ALL).title("Recent Tasks"));

            frame.render_widget(history_widget, area);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_ingest_tab_creation() {
        let tab = IngestTab::new();
        assert_eq!(tab.focused_field, IngestField::Path);
        assert!(!tab.overwrite);
        assert!(!tab.processing);
        assert!(tab.current_task.is_none());
    }

    #[test]
    fn test_source_type_detection() {
        let mut tab = IngestTab::new();

        // Test URL detection
        tab.path_input.insert_str("https://example.com/doc.pdf");
        tab.update_source_type();
        assert_eq!(tab.source_type, Some(SourceType::Url));

        // Test file detection
        tab.path_input.select_all();
        tab.path_input.cut();
        tab.path_input.insert_str("/path/to/file.pdf");
        tab.update_source_type();
        assert_eq!(tab.source_type, Some(SourceType::File));
    }

    #[test]
    fn test_ingest_request_generation() {
        let mut tab = IngestTab::new();

        tab.path_input.insert_str("/test/path");
        tab.tags_input.insert_str("tag1,tag2,tag3");
        tab.overwrite = true;
        tab.update_source_type();

        let request = tab.get_ingest_request();

        assert_eq!(request.path, "/test/path");
        assert_eq!(request.source, "file");
        assert_eq!(request.tags, Some(vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()]));
        assert!(request.overwrite);
    }

    #[test]
    fn test_task_status_updates() {
        let mut tab = IngestTab::new();

        let task1 = TaskStatus {
            task_id: "task1".to_string(),
            status: "processing".to_string(),
            progress: Some(50.0),
            documents_processed: Some(10),
            error_message: None,
            started_at: Some(Utc::now()),
            completed_at: None,
        };

        tab.update_task_status(task1.clone());

        assert_eq!(tab.current_task.as_ref().unwrap().task_id, "task1");
        assert_eq!(tab.task_history.len(), 1);
        assert!(tab.processing);

        // Update same task
        let mut task1_updated = task1;
        task1_updated.status = "completed".to_string();
        task1_updated.progress = Some(100.0);

        tab.update_task_status(task1_updated);

        assert_eq!(tab.task_history.len(), 1); // Should update, not add
        assert!(!tab.processing); // Should stop processing
    }

    #[test]
    fn test_field_navigation() {
        let mut tab = IngestTab::new();

        assert_eq!(tab.focused_field, IngestField::Path);
        tab.next_field();
        assert_eq!(tab.focused_field, IngestField::Tags);
        tab.next_field();
        assert_eq!(tab.focused_field, IngestField::Options);
        tab.next_field();
        assert_eq!(tab.focused_field, IngestField::Submit);
        tab.next_field();
        assert_eq!(tab.focused_field, IngestField::Path); // Wrap around
    }
}
