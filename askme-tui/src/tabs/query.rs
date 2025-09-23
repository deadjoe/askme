use crate::api::{QueryRequest, QueryResponse, OutputFormat, Reranker};
use tui_textarea::{TextArea, Input};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

/// Query tab state and functionality
pub struct QueryTab {
    /// Text area for question input
    pub question_input: TextArea<'static>,

    /// Query parameters
    pub topk: u32,
    pub alpha: f64,
    pub use_rrf: bool,
    pub rrf_k: u32,
    pub use_hyde: bool,
    pub use_rag_fusion: bool,
    pub reranker: Reranker,
    pub max_passages: u32,
    pub include_debug: bool,
    pub output_format: OutputFormat,

    /// Current query response
    pub response: Option<QueryResponse>,

    /// UI state
    pub focused_field: QueryField,
    pub editing_field: Option<QueryField>,
    pub response_scroll: usize,
    pub processing: bool,
    /// Buffer for numeric editing while in edit mode
    pub edit_buffer: String,
}

/// Fields that can be focused in the query tab
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryField {
    Question,
    TopK,
    Alpha,
    RrfK,
    MaxPassages,
    Reranker,
    OutputFormat,
    UseRrf,
    UseHyde,
    UseRagFusion,
    IncludeDebug,
    Submit,
}

impl QueryField {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Question,
            Self::TopK,
            Self::Alpha,
            Self::RrfK,
            Self::MaxPassages,
            Self::Reranker,
            Self::OutputFormat,
            Self::UseRrf,
            Self::UseHyde,
            Self::UseRagFusion,
            Self::IncludeDebug,
            Self::Submit,
        ]
    }
}

impl QueryTab {
    /// Create a new query tab
    pub fn new(default_request: QueryRequest) -> Self {
        let mut question_input = TextArea::default();
        let block = Block::default().title("Question").borders(Borders::ALL);
        question_input.set_block(block.clone());
        question_input.set_placeholder_text("Enter your question here...");
        // Set a bright cyan placeholder by default
        question_input.set_placeholder_style(Style::default().fg(Color::Cyan));

        Self {
            question_input,
            topk: default_request.topk,
            alpha: default_request.alpha,
            use_rrf: default_request.use_rrf,
            rrf_k: default_request.rrf_k,
            use_hyde: default_request.use_hyde,
            use_rag_fusion: default_request.use_rag_fusion,
            reranker: default_request.reranker.parse().unwrap_or(Reranker::BgeLocal),
            max_passages: default_request.max_passages,
            include_debug: default_request.include_debug,
            output_format: OutputFormat::Text,
            response: None,
            focused_field: QueryField::Question,
            editing_field: None,
            response_scroll: 0,
            processing: false,
            edit_buffer: String::new(),
        }
    }

    /// Get the current query request
    pub fn get_query_request(&self) -> QueryRequest {
        let question_text = self.question_input.lines().join("\n");

        QueryRequest {
            q: question_text,
            topk: self.topk,
            alpha: self.alpha,
            use_rrf: self.use_rrf,
            rrf_k: self.rrf_k,
            use_hyde: self.use_hyde,
            use_rag_fusion: self.use_rag_fusion,
            reranker: self.reranker.as_str().to_string(),
            max_passages: self.max_passages,
            include_debug: self.include_debug,
        }
    }

    /// Set query response
    pub fn set_response(&mut self, response: QueryResponse) {
        self.response = Some(response);
        self.response_scroll = 0;
        self.processing = false;
    }

    /// Clear query response
    pub fn clear_response(&mut self) {
        self.response = None;
        self.response_scroll = 0;
    }

    /// Start processing
    pub fn start_processing(&mut self) {
        self.processing = true;
        self.clear_response();
    }

    /// Handle keyboard input
    pub fn handle_input(&mut self, key: ratatui::crossterm::event::KeyEvent) -> bool {
        // Handle global navigation first
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
            QueryField::Question => {
                match key.code {
                    ratatui::crossterm::event::KeyCode::Tab => {
                        self.next_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::BackTab => {
                        self.previous_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Enter if key.modifiers.contains(ratatui::crossterm::event::KeyModifiers::CONTROL) => {
                        // Ctrl+Enter to submit
                        false // Let app handle submission
                    }
                    _ => {
                        self.question_input.input(Input::from(key));
                        true
                    }
                }
            }
            _ => {
                match key.code {
                    ratatui::crossterm::event::KeyCode::Tab => {
                        // Leaving edit mode if editing
                        self.editing_field = None;
                        self.next_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::BackTab => {
                        self.editing_field = None;
                        self.previous_field();
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Enter => {
                        if self.focused_field == QueryField::Submit {
                            // Submit handled by app
                            false
                        } else {
                            // Toggle edit mode for the focused parameter
                            if self.editing_field == Some(self.focused_field) {
                                self.editing_field = None; // Finish editing
                            } else {
                                self.editing_field = Some(self.focused_field);
                                self.edit_buffer.clear();
                            }
                            true
                        }
                    }
                    ratatui::crossterm::event::KeyCode::Left | ratatui::crossterm::event::KeyCode::Right => {
                        // Only allow adjustments while editing
                        if self.editing_field == Some(self.focused_field) {
                            self.handle_field_input(key);
                        }
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Char(' ') => {
                        // Optional: allow toggles only in edit mode for fields that use space
                        if self.editing_field == Some(self.focused_field) {
                            match self.focused_field {
                                QueryField::Reranker => {
                                    self.reranker = match self.reranker {
                                        Reranker::BgeLocal => Reranker::Cohere,
                                        Reranker::Cohere => Reranker::BgeLocal,
                                    };
                                }
                                QueryField::OutputFormat => {
                                    self.output_format = match self.output_format {
                                        OutputFormat::Text => OutputFormat::Json,
                                        OutputFormat::Json => OutputFormat::Markdown,
                                        OutputFormat::Markdown => OutputFormat::Text,
                                    };
                                }
                                QueryField::UseRrf => { self.use_rrf = !self.use_rrf; }
                                QueryField::UseHyde => { self.use_hyde = !self.use_hyde; }
                                QueryField::UseRagFusion => { self.use_rag_fusion = !self.use_rag_fusion; }
                                QueryField::IncludeDebug => { self.include_debug = !self.include_debug; }
                                _ => {}
                            }
                        }
                        true
                    }
                    ratatui::crossterm::event::KeyCode::Char(c) => {
                        // Only accept chars while editing
                        if self.editing_field == Some(self.focused_field) {
                            self.handle_char_input(c);
                        }
                        true
                    }
                    _ => true
                }
            }
        }
    }

    /// Move to next field
    fn next_field(&mut self) {
        let fields = QueryField::all();

        if let Some(current_idx) = fields.iter().position(|&f| f == self.focused_field) {
            let next_idx = (current_idx + 1) % fields.len();
            self.focused_field = fields[next_idx];
        }
    }

    /// Move to previous field
    fn previous_field(&mut self) {
        let fields = QueryField::all();

        if let Some(current_idx) = fields.iter().position(|&f| f == self.focused_field) {
            let prev_idx = if current_idx == 0 {
                fields.len() - 1
            } else {
                current_idx - 1
            };
            self.focused_field = fields[prev_idx];
        }
    }

    /// Handle character input for focused field
    fn handle_char_input(&mut self, c: char) {
        match self.focused_field {
            QueryField::TopK => {
                if c.is_ascii_digit() {
                    let candidate = if self.edit_buffer.is_empty() {
                        c.to_string()
                    } else {
                        format!("{}{}", self.edit_buffer, c)
                    };
                    if let Ok(val) = candidate.parse::<u32>() {
                        if (1..=100).contains(&val) {
                            self.topk = val;
                            self.edit_buffer = candidate;
                        }
                    }
                } else if c == 'r' || c == 'R' {
                    // Reset to default
                    self.topk = 50;
                    self.edit_buffer.clear();
                }
            }
            QueryField::Alpha => {
                if c.is_ascii_digit() {
                    let digit = c.to_digit(10).unwrap() as f64 / 10.0;
                    if digit <= 1.0 {
                        self.alpha = digit;
                        self.edit_buffer = c.to_string();
                    }
                } else if c == 'r' || c == 'R' {
                    // Reset to default
                    self.alpha = 0.5;
                    self.edit_buffer.clear();
                }
            }
            QueryField::RrfK => {
                if c.is_ascii_digit() {
                    let candidate = if self.edit_buffer.is_empty() {
                        c.to_string()
                    } else {
                        format!("{}{}", self.edit_buffer, c)
                    };
                    if let Ok(val) = candidate.parse::<u32>() {
                        self.rrf_k = val;
                        self.edit_buffer = candidate;
                    }
                } else if c == 'r' || c == 'R' {
                    self.rrf_k = 60;
                    self.edit_buffer.clear();
                }
            }
            QueryField::MaxPassages => {
                if c.is_ascii_digit() {
                    let candidate = if self.edit_buffer.is_empty() {
                        c.to_string()
                    } else {
                        format!("{}{}", self.edit_buffer, c)
                    };
                    if let Ok(val) = candidate.parse::<u32>() {
                        if (1..=20).contains(&val) {
                            self.max_passages = val;
                            self.edit_buffer = candidate;
                        }
                    }
                } else if c == 'r' || c == 'R' {
                    self.max_passages = 8;
                    self.edit_buffer.clear();
                }
            }
            QueryField::Reranker => {
                // Toggle between reranker options
                if c == ' ' {
                    self.reranker = match self.reranker {
                        Reranker::BgeLocal => Reranker::Cohere,
                        Reranker::Cohere => Reranker::BgeLocal,
                    };
                } else if c == 'r' || c == 'R' {
                    self.reranker = Reranker::BgeLocal;
                }
            }
            QueryField::OutputFormat => {
                // Cycle through output formats
                if c == ' ' {
                    self.output_format = match self.output_format {
                        OutputFormat::Text => OutputFormat::Json,
                        OutputFormat::Json => OutputFormat::Markdown,
                        OutputFormat::Markdown => OutputFormat::Text,
                    };
                } else if c == 'r' || c == 'R' {
                    self.output_format = OutputFormat::Text;
                }
            }
            _ => {}
        }
    }

    /// Handle field-specific input
    fn handle_field_input(&mut self, key: ratatui::crossterm::event::KeyEvent) {
        match key.code {
            ratatui::crossterm::event::KeyCode::Left => {
                match self.focused_field {
                    QueryField::Alpha => {
                        self.alpha = ((self.alpha * 10.0 - 1.0) / 10.0).max(0.0);
                    }
                    QueryField::TopK => {
                        self.topk = self.topk.saturating_sub(5).max(1);
                    }
                    QueryField::RrfK => {
                        self.rrf_k = self.rrf_k.saturating_sub(10).max(1);
                    }
                    QueryField::MaxPassages => {
                        self.max_passages = self.max_passages.saturating_sub(1).max(1);
                    }
                    _ => {}
                }
            }
            ratatui::crossterm::event::KeyCode::Right => {
                match self.focused_field {
                    QueryField::Alpha => {
                        self.alpha = ((self.alpha * 10.0 + 1.0) / 10.0).min(1.0);
                    }
                    QueryField::TopK => {
                        self.topk = (self.topk + 5).min(100);
                    }
                    QueryField::RrfK => {
                        self.rrf_k = self.rrf_k + 10;
                    }
                    QueryField::MaxPassages => {
                        self.max_passages = (self.max_passages + 1).min(20);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }


    // Removed unused helpers; scrolling can be added when needed

    /// Render the query tab
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let main_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Left panel: Input form
        self.render_input_form(frame, main_layout[0]);

        // Right panel: Response
        self.render_response(frame, main_layout[1]);
    }

    /// Render input form
    fn render_input_form(&mut self, frame: &mut Frame, area: Rect) {
        let form_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(9),  // Question input (more vertical space)
                Constraint::Min(6),     // Parameters (slightly less space)
                Constraint::Length(3),  // Submit button
            ])
            .split(area);

        // Question input
        let question_focused = self.focused_field == QueryField::Question;
        // Dynamic border highlight + subtle text color when focused
        let mut block = Block::default().title("Question").borders(Borders::ALL);
        if question_focused {
            block = block.border_style(Style::default().fg(Color::Yellow));
            self.question_input.set_style(Style::default().fg(Color::Yellow));
            self.question_input
                .set_placeholder_style(Style::default().fg(Color::Yellow));
        } else {
            self.question_input.set_style(Style::default());
            self.question_input
                .set_placeholder_style(Style::default().fg(Color::Cyan));
        }
        self.question_input.set_block(block);
        frame.render_widget(&self.question_input, form_layout[0]);

        // Parameters
        self.render_parameters(frame, form_layout[1]);

        // Submit button
        let submit_style = if self.focused_field == QueryField::Submit {
            Style::default().bg(Color::Yellow).fg(Color::Black)
        } else {
            Style::default()
        };

        let submit_text = if self.processing {
            "Processing..."
        } else {
            "Submit Query (Enter/Ctrl+Enter)"
        };

        let submit_button = Paragraph::new(submit_text)
            .block(Block::default().borders(Borders::ALL).title("Submit"))
            .style(submit_style)
            .wrap(Wrap { trim: true });

        frame.render_widget(submit_button, form_layout[2]);
    }

    /// Render parameters section
    fn render_parameters(&self, frame: &mut Frame, area: Rect) {
        // Helper to style a line based on focus/editing
        let style_for = |field: QueryField| {
            if self.editing_field == Some(field) {
                Style::default().bg(Color::Green).fg(Color::Black)
            } else if self.focused_field == field {
                Style::default().bg(Color::Yellow).fg(Color::Black)
            } else {
                Style::default()
            }
        };

        let mut lines: Vec<ratatui::text::Line> = Vec::new();

        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("TopK: {} (1-100)", self.topk),
                style_for(QueryField::TopK),
            ),
        ));

        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("Alpha: {:.2} (0.0-1.0)", self.alpha),
                style_for(QueryField::Alpha),
            ),
        ));

        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("RRF K: {}", self.rrf_k),
                style_for(QueryField::RrfK),
            ),
        ));

        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("Max Passages: {} (1-20)", self.max_passages),
                style_for(QueryField::MaxPassages),
            ),
        ));

        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("Reranker: {}", self.reranker),
                style_for(QueryField::Reranker),
            ),
        ));

        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("Output: {}", self.output_format),
                style_for(QueryField::OutputFormat),
            ),
        ));

        lines.push(ratatui::text::Line::from(""));

        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("RRF: {}", if self.use_rrf { "✓" } else { "✗" }),
                style_for(QueryField::UseRrf),
            ),
        ));
        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("HyDE: {}", if self.use_hyde { "✓" } else { "✗" }),
                style_for(QueryField::UseHyde),
            ),
        ));
        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("RAG-Fusion: {}", if self.use_rag_fusion { "✓" } else { "✗" }),
                style_for(QueryField::UseRagFusion),
            ),
        ));
        lines.push(ratatui::text::Line::from(
            ratatui::text::Span::styled(
                format!("Debug: {}", if self.include_debug { "✓" } else { "✗" }),
                style_for(QueryField::IncludeDebug),
            ),
        ));

        // Controls help
        lines.push(ratatui::text::Line::from(""));
        lines.push("Controls:".into());
        lines.push("Enter: Edit/Finish current parameter".into());
        lines.push("↑↓: Navigate fields".into());
        lines.push("←→: Adjust values (in edit mode)".into());
        lines.push("Numbers/Space: Change value (in edit mode)".into());
        lines.push("R: Reset field to default".into());

        let params_widget = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Parameters"))
            .wrap(Wrap { trim: true });

        frame.render_widget(params_widget, area);
    }

    /// Render response section
    fn render_response(&self, frame: &mut Frame, area: Rect) {
        if let Some(ref response) = self.response {
            let response_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(70), // Answer
                    Constraint::Percentage(30), // Citations
                ])
                .split(area);

            // Answer
            let answer_text = match self.output_format {
                OutputFormat::Text | OutputFormat::Markdown => &response.answer,
                OutputFormat::Json => {
                    // For JSON, show pretty-printed JSON
                    &response.answer
                }
            };

            let answer_widget = Paragraph::new(answer_text.clone())
                .block(Block::default().borders(Borders::ALL).title("Answer"))
                .wrap(Wrap { trim: true })
                .scroll((self.response_scroll as u16, 0));

            frame.render_widget(answer_widget, response_layout[0]);

            // Citations
            let citations: Vec<ListItem> = response.citations
                .iter()
                .enumerate()
                .map(|(i, citation)| {
                    let content = format!(
                        "[{}] {} (Score: {:.3})",
                        i + 1,
                        citation.title,
                        citation.score
                    );
                    ListItem::new(content)
                })
                .collect();

            let citations_widget = List::new(citations)
                .block(Block::default().borders(Borders::ALL).title("Citations"));

            frame.render_widget(citations_widget, response_layout[1]);
        } else {
            let placeholder = if self.processing {
                // While processing, keep Response area simple (popup handles visuals)
                Paragraph::new("")
                    .style(Style::default())
            } else {
                Paragraph::new("No response yet.\n\nEnter a question and submit to see results here.")
                    .style(Style::default().fg(Color::Gray))
            };

            let placeholder_widget = placeholder
                .block(Block::default().borders(Borders::ALL).title("Response"))
                .wrap(Wrap { trim: true });

            frame.render_widget(placeholder_widget, area);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_tab_creation() {
        let default_request = QueryRequest::default();
        let tab = QueryTab::new(default_request.clone());

        assert_eq!(tab.topk, default_request.topk);
        assert_eq!(tab.alpha, default_request.alpha);
        assert_eq!(tab.focused_field, QueryField::Question);
        // All parameters are always visible now
    }

    #[test]
    fn test_field_navigation() {
        let mut tab = QueryTab::new(QueryRequest::default());

        // Test basic field navigation
        assert_eq!(tab.focused_field, QueryField::Question);
        tab.next_field();
        assert_eq!(tab.focused_field, QueryField::TopK);
        tab.next_field();
        assert_eq!(tab.focused_field, QueryField::Alpha);

        // Test wrap around
        tab.focused_field = QueryField::Submit;
        tab.next_field();
        assert_eq!(tab.focused_field, QueryField::Question);
    }

    #[test]
    fn test_parameter_bounds() {
        let mut tab = QueryTab::new(QueryRequest::default());

        // Test topk bounds
        tab.topk = 99;
        tab.handle_char_input('9');
        assert_eq!(tab.topk, 99); // Should not exceed 100

        // Test alpha bounds
        tab.focused_field = QueryField::Alpha;
        tab.alpha = 0.95;
        tab.handle_field_input(ratatui::crossterm::event::KeyEvent::new(
            ratatui::crossterm::event::KeyCode::Right,
            ratatui::crossterm::event::KeyModifiers::NONE
        ));
        assert_eq!(tab.alpha, 1.0); // Should not exceed 1.0
    }
}
