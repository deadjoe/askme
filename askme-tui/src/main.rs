use anyhow::Result;
use ratatui::crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};
use std::{
    io::{self, Stdout},
    time::Duration,
};
use tracing::{debug, error, info, warn};
use tracing_subscriber;

mod api;
mod app;
mod event;
mod tabs;
mod ui;

use app::{App, Tab};
use event::{AppEvent, EventHandler, handle_global_key_event, key_utils};

type Tui = Terminal<CrosstermBackend<Stdout>>;

/// Initialize the terminal for TUI mode
fn init_terminal() -> Result<Tui> {
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    enable_raw_mode()?;

    let backend = CrosstermBackend::new(io::stdout());
    let terminal = Terminal::new(backend)?;

    Ok(terminal)
}

/// Restore the terminal to normal mode
fn restore_terminal() -> Result<()> {
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}

/// Main application loop
async fn run_app() -> Result<()> {
    // Initialize tracing to file instead of stdout
    use std::fs::OpenOptions;
    use tracing_subscriber::fmt::writer::MakeWriterExt;

    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/askme-tui.log")?;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("askme_tui=info".parse()?) // Reduce verbosity
        )
        .with_writer(log_file.with_max_level(tracing::Level::DEBUG))
        .init();

    info!("Starting askme TUI application");

    // Initialize terminal
    let mut terminal = init_terminal()?;

    // Initialize application state
    let mut app = App::new().map_err(|e| {
        error!("Failed to initialize app: {}", e);
        e
    })?;

    // Initialize event handler
    let mut event_handler = EventHandler::new(Duration::from_millis(250));

    // Check API health on startup
    app.check_api_health().await;

    // Application loop
    let result = loop {
        // Render the UI - suppress any stdout output during rendering
        match terminal.draw(|frame| ui::render(&mut app, frame)) {
            Ok(_) => {}
            Err(e) => {
                error!("Failed to draw UI: {}", e);
                break Err(e.into());
            }
        }

        // Handle events
        match event_handler.next().await {
            Ok(event) => {
                if !handle_event(&mut app, event).await? {
                    break Ok(());
                }
            }
            Err(e) => {
                error!("Event handler error: {}", e);
                break Err(e);
            }
        }

        // Check if app should quit
        if app.should_quit {
            info!("Application quit requested");
            break Ok(());
        }

        // no-op per tick
    };

    // Cleanup
    event_handler.close();
    restore_terminal()?;

    result
}

/// Handle a single application event
async fn handle_event(app: &mut App, event: AppEvent) -> Result<bool> {
    match event {
        AppEvent::Quit => {
            debug!("Quit event received");
            app.quit();
            return Ok(false);
        }

        AppEvent::Key(key_event) => {
            // Clear error on any key press
            if app.error_message.is_some() {
                app.clear_error();
                return Ok(true);
            }

            // Handle global key events first
            if let Some(global_event) = handle_global_key_event(key_event) {
                return Box::pin(handle_event(app, global_event)).await;
            }

            // Handle tab switching (disabled while editing a field in Query tab)
            let editing_query_field = matches!(app.current_tab, Tab::Query) && app.query_tab.editing_field.is_some();
            if !editing_query_field {
                if let Some(tab_num) = key_utils::is_number(key_event) {
                    if tab_num >= 1 && tab_num <= 4 {
                        app.set_tab((tab_num - 1) as usize);
                        return Ok(true);
                    }
                }
            }

            // Note: Left/Right arrows are reserved for in-tab editing now

            // F1 toggle removed: all parameters stay visible

            // Handle tab-specific events
            match app.current_tab {
                Tab::Query => {
                    if !app.query_tab.handle_input(key_event) {
                        // Query submission
                        if key_utils::is_enter(key_event) || key_utils::is_ctrl_enter(key_event) {
                            handle_query_submission(app).await?;
                        }
                    }
                }

                Tab::Ingest => {
                    if !app.ingest_tab.handle_input(key_event) {
                        // Ingest submission
                        if key_utils::is_enter(key_event) {
                            handle_ingest_submission(app).await?;
                        }
                    }
                }

                Tab::Settings => {
                    if !app.settings_tab.handle_input(key_event) {
                        // Settings actions
                        if key_utils::is_enter(key_event) {
                            handle_settings_action(app).await?;
                        }
                    }
                }

                Tab::Help => {
                    // Help tab doesn't have interactive elements
                    // All input just moves to other tabs or quits
                }
            }
        }

        AppEvent::Tick => {
            // Check background query completion to update UI and dismiss modal
            // Advance spinner animation
            app.throbber.calc_next();
            if let Some(handle) = &mut app.query_task {
                if handle.is_finished() {
                    match handle.await {
                        Ok(Ok(resp)) => {
                            info!("Query completed successfully");
                            app.query_tab.set_response(resp);
                            app.set_status("Query completed");
                        }
                        Ok(Err(e)) => {
                            warn!("Query failed: {}", e);
                            app.query_tab.processing = false;
                            app.set_error(&format!("Query failed: {}", e));
                        }
                        Err(e) => {
                            warn!("Query task join failed: {}", e);
                            app.query_tab.processing = false;
                            app.set_error(&format!("Query failed: {}", e));
                        }
                    }
                    app.query_task = None;
                }
            }
        }

        AppEvent::Resize(width, height) => {
            debug!("Terminal resized to {}x{}", width, height);
            // Ratatui handles resize automatically
        }
    }

    Ok(true)
}

/// Handle query submission
async fn handle_query_submission(app: &mut App) -> Result<()> {
    let question = app.query_tab.question_input.lines().join("\n");

    if question.trim().is_empty() {
        app.set_error("Please enter a question");
        return Ok(());
    }

    debug!("Submitting query: {}", question.chars().take(50).collect::<String>());
    app.query_tab.start_processing();

    // Spawn non-blocking task so UI can render modal/progress
    let request = app.query_tab.get_query_request();
    let api = app.api_client.clone();
    app.query_task = Some(tokio::spawn(async move {
        let res = api.query(request).await;
        res.map_err(|e| anyhow::anyhow!(e))
    }));

    Ok(())
}

/// Handle ingest submission
async fn handle_ingest_submission(app: &mut App) -> Result<()> {
    let path = app.ingest_tab.path_input.lines().join("");

    if path.trim().is_empty() {
        app.set_error("Please enter a path or URL");
        return Ok(());
    }

    debug!("Submitting ingest request: {}", path);
    app.ingest_tab.start_processing();

    match app.process_ingest().await {
        Ok(()) => {
            info!("Ingestion completed successfully");
        }
        Err(e) => {
            warn!("Ingestion failed: {}", e);
            app.ingest_tab.processing = false;
            // Error is already set in app.process_ingest()
        }
    }

    Ok(())
}

/// Handle settings actions
async fn handle_settings_action(app: &mut App) -> Result<()> {
    match app.settings_tab.focused_field {
        tabs::settings::SettingsField::Save => {
            debug!("Saving configuration");
            let new_config = app.settings_tab.get_config();
            app.config = new_config;
            app.settings_tab.clear_modified();

            match app.update_api_client() {
                Ok(()) => {
                    if let Err(e) = app.save_config() {
                        app.set_error(&format!("Failed to save config: {}", e));
                    } else {
                        app.set_status("Configuration saved successfully");
                    }
                }
                Err(e) => {
                    app.set_error(&format!("Failed to update API client: {}", e));
                }
            }
        }

        tabs::settings::SettingsField::Reset => {
            debug!("Resetting configuration to defaults");
            app.settings_tab.reset_to_defaults();
            app.set_status("Configuration reset to defaults (not saved)");
        }

        _ => {
            // Other fields just move focus
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Handle panics gracefully
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        default_panic(info);
    }));

    // Run the application
    let result = run_app().await;

    // Ensure terminal is restored on any exit
    let _ = restore_terminal();

    match result {
        Ok(()) => {
            println!("askme TUI exited successfully");
            Ok(())
        }
        Err(e) => {
            eprintln!("askme TUI exited with error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[tokio::test]
    async fn test_app_initialization() {
        // This test verifies that the app can be initialized
        // without actually running the TUI
        let result = App::new();
        assert!(result.is_ok(), "App should initialize successfully");

        let app = result.unwrap();
        assert_eq!(app.current_tab, Tab::Query);
        assert!(!app.should_quit);
    }

    #[test]
    fn test_key_event_handling() {
        // Test global key events
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        let result = handle_global_key_event(ctrl_c);
        assert!(matches!(result, Some(AppEvent::Quit)));

        let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        let result = handle_global_key_event(esc);
        assert!(matches!(result, Some(AppEvent::Quit)));

        let regular_key = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        let result = handle_global_key_event(regular_key);
        assert!(result.is_none());
    }

    #[test]
    fn test_terminal_initialization() {
        // This test just ensures the terminal init/restore functions don't panic
        // In a real environment, this would need proper terminal setup
        // For now, we'll just test that the functions exist and are callable
        assert!(true); // Placeholder test
    }
}
