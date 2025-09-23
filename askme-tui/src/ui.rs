use crate::app::{App, Tab};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect, Margin},
    style::{Color, Modifier, Style},
    widgets::{
        Block, Borders, Clear, Paragraph, Tabs, Wrap,
    },
    Frame,
};
use throbber_widgets_tui::Throbber;

/// Render the entire application UI
pub fn render(app: &mut App, frame: &mut Frame) {
    let size = frame.area();

    // Main layout: header, body, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header (tabs)
            Constraint::Min(0),    // Body (content)
            Constraint::Length(3), // Footer (status)
        ])
        .split(size);

    // Render header (tabs)
    render_header(app, frame, chunks[0]);

    // Render main content based on current tab
    render_tab_content(app, frame, chunks[1]);

    // Render footer (status bar)
    render_footer(app, frame, chunks[2]);

    // Render error popup if there's an error
    if app.error_message.is_some() {
        render_error_popup(app, frame, size);
    }

    // Render processing overlay while query is running
    if app.query_tab.processing {
        render_processing_popup(app, frame, size);
    }
}

/// Render the header with tab navigation
fn render_header(app: &App, frame: &mut Frame, area: Rect) {
    let tab_titles = Tab::titles();
    let tabs = Tabs::new(tab_titles)
        .block(Block::default().borders(Borders::ALL).title("askme TUI"))
        .style(Style::default().fg(Color::Gray))
        .highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .bg(Color::Yellow)
                .fg(Color::Black),
        )
        .select(app.current_tab.index());

    frame.render_widget(tabs, area);
}

/// Render the content for the current tab
fn render_tab_content(app: &mut App, frame: &mut Frame, area: Rect) {
    match app.current_tab {
        Tab::Query => {
            app.query_tab.render(frame, area);
        }
        Tab::Ingest => {
            app.ingest_tab.render(frame, area);
        }
        Tab::Settings => {
            app.settings_tab.render(frame, area);
        }
        Tab::Help => {
            render_help_tab(frame, area);
        }
    }
}

/// Render the help tab
fn render_help_tab(frame: &mut Frame, area: Rect) {
    let help_text = vec![
        "askme TUI - Terminal User Interface for askme RAG System",
        "",
        "GLOBAL CONTROLS:",
        "  Ctrl+C, Ctrl+Q, ESC  Quit application",
        "  1, 2, 3, 4           Switch to tabs (Query, Ingest, Settings, Help)",
        // F1 toggle removed
        "",
        "QUERY TAB:",
        "  Enter question in the text area",
        "  Tab/Shift+Tab        Navigate between fields",
        "  Ctrl+Enter/Enter     Submit query",
        "  ←/→                  Adjust numeric parameters",
        "  Space                Toggle boolean options",
        "  Page Up/Down         Scroll response",
        "",
        "INGEST TAB:",
        "  Enter path/URL to ingest",
        "  Add comma-separated tags",
        "  Space                Toggle overwrite option",
        "  Enter                Start ingestion",
        "",
        "SETTINGS TAB:",
        "  Configure API URL and key",
        "  Set default query parameters",
        "  ←/→                  Adjust numeric values",
        "  Space                Toggle boolean options",
        "  Enter                Save/Reset configuration",
        "",
        "FEATURES:",
        "  • Hybrid search with BM25 + dense vectors",
        "  • Multiple reranking models (BGE local, Cohere)",
        "  • Query enhancement (HyDE, RAG-Fusion)",
        "  • Real-time ingestion progress tracking",
        "  • Configurable output formats (Text, JSON, Markdown)",
        "  • Persistent configuration with auto-save",
        "",
        "API ENDPOINTS:",
        "  POST /query/         Submit search queries",
        "  POST /ingest/        Start document ingestion",
        "  GET /ingest/status/  Check ingestion progress",
        "  GET /health/         API health check",
        "",
        "ENVIRONMENT VARIABLES:",
        "  ASKME_API_URL        Default API base URL",
        "  ASKME_API_KEY        API authentication key",
        "",
        "For more information, visit: https://github.com/your-repo/askme",
    ];

    let help_widget = Paragraph::new(help_text.join("\n"))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Help & Documentation")
                .title_alignment(Alignment::Center),
        )
        .wrap(Wrap { trim: true })
        .scroll((0, 0));

    frame.render_widget(help_widget, area);
}

/// Render the footer status bar
fn render_footer(app: &App, frame: &mut Frame, area: Rect) {
    let status_text = app.display_status();

    let footer_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(80), Constraint::Percentage(20)])
        .split(area);

    // Status message
    let status_widget = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::ALL).title("Status"))
        .wrap(Wrap { trim: true });

    frame.render_widget(status_widget, footer_layout[0]);

    // Shortcuts
    let shortcuts = vec!["Ctrl+Q: Quit", "1-4: Tabs"];

    let shortcuts_widget = Paragraph::new(shortcuts.join(" | "))
        .block(Block::default().borders(Borders::ALL).title("Shortcuts"))
        .alignment(Alignment::Center);

    frame.render_widget(shortcuts_widget, footer_layout[1]);
}

/// Render error popup overlay
fn render_error_popup(app: &App, frame: &mut Frame, area: Rect) {
    if let Some(error_msg) = app.display_error() {
        let popup_area = centered_rect(60, 20, area);

        // Clear the area
        frame.render_widget(Clear, popup_area);

        // Error content
        let error_text = format!("Error: {}\n\nPress any key to dismiss", error_msg);

        let error_widget = Paragraph::new(error_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Error")
                    .title_alignment(Alignment::Center)
                    .border_style(Style::default().fg(Color::Red)),
            )
            .style(Style::default().fg(Color::Red))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });

        frame.render_widget(error_widget, popup_area);
    }
}

/// Render a modal processing popup with a dynamic progress bar
fn render_processing_popup(app: &mut App, frame: &mut Frame, area: Rect) {
    // Shrink popup to half of previous size: 30% x 10%
    let popup_area = centered_rect(30, 10, area);
    frame.render_widget(Clear, popup_area);

    use ratatui::widgets::{Block, Borders, Gauge};
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let tick = (now.as_millis() / 100) as u16; // 10 Hz
    let percent = (tick % 100) as u16; // loop 0..99

    // Outer block
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Rounded)
        .title("Processing Query");

    // Render container
    frame.render_widget(block.clone(), popup_area);

    // Compute inner area with padding so content never touches borders
    let inner_area = block.inner(popup_area).inner(Margin { horizontal: 2, vertical: 1 });

    // Use a modern spinner (throbber) + a sleek line gauge
    let spinner = Throbber::default().label("Working...");

    // Vertical stack inside popup: spinner, progress bar, subtext
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner_area);

    // Spinner row: center horizontally with left/right padding
    let row0 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Length(10), Constraint::Percentage(45)])
        .split(v[0]);
    frame.render_stateful_widget(spinner, row0[1], &mut app.throbber);

    // Progress bar row: center bar and keep it narrower than popup width
    let row1 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(15), // left padding
            Constraint::Percentage(70), // gauge area (<= inner width)
            Constraint::Percentage(15), // right padding
        ])
        .split(v[1]);
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(percent as f64 / 100.0)
        .label(format!("{}%", percent));
    frame.render_widget(gauge, row1[1]);

    // Subtext
    let text = Paragraph::new("Please wait...")
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center);
    frame.render_widget(text, v[2]);
}

/// Create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Create a loading spinner text (test-only)
#[cfg(test)]
pub fn loading_spinner(tick: usize) -> &'static str {
    const SPINNER_CHARS: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    SPINNER_CHARS[tick % SPINNER_CHARS.len()]
}

/// Create a progress bar text representation (test-only)
#[cfg(test)]
pub fn progress_bar(progress: f64, width: usize) -> String {
    let filled = ((progress / 100.0) * width as f64) as usize;
    let empty = width.saturating_sub(filled);

    format!(
        "[{}{}] {:.1}%",
        "█".repeat(filled),
        "░".repeat(empty),
        progress
    )
}

// Removed unused color/style helper modules

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centered_rect() {
        let area = Rect::new(0, 0, 100, 100);
        let centered = centered_rect(50, 50, area);

        // Should be centered
        assert_eq!(centered.x, 25);
        assert_eq!(centered.y, 25);
        assert_eq!(centered.width, 50);
        assert_eq!(centered.height, 50);
    }

    #[test]
    fn test_loading_spinner() {
        // Test that spinner cycles through different characters
        let first = loading_spinner(0);
        let second = loading_spinner(1);
        assert_ne!(first, second);

        // Test that it wraps around
        let wrapped = loading_spinner(10); // Should be same as tick 0
        assert_eq!(loading_spinner(0), wrapped);
    }

    #[test]
    fn test_progress_bar() {
        let bar_0 = progress_bar(0.0, 10);
        assert!(bar_0.contains("0.0%"));
        assert!(bar_0.contains("░░░░░░░░░░"));

        let bar_50 = progress_bar(50.0, 10);
        assert!(bar_50.contains("50.0%"));
        assert!(bar_50.contains("█████░░░░░"));

        let bar_100 = progress_bar(100.0, 10);
        assert!(bar_100.contains("100.0%"));
        assert!(bar_100.contains("██████████"));
    }

    #[test]
    fn test_progress_bar_edge_cases() {
        // Test with 0 width (shouldn't panic)
        let bar_zero_width = progress_bar(50.0, 0);
        assert!(bar_zero_width.contains("50.0%"));

        // Test with very high progress (should cap at 100%)
        let bar_over = progress_bar(150.0, 10);
        assert!(bar_over.contains("150.0%"));
    }
}
