use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use anyhow::Result;
use tracing::debug;

/// Application events
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// Key press event
    Key(KeyEvent),
    /// Application tick for periodic updates
    Tick,
    /// Application should quit
    Quit,
    /// Resize terminal
    Resize(u16, u16),
}

/// Event handler for the application
pub struct EventHandler {
    /// Event receiver
    receiver: mpsc::UnboundedReceiver<AppEvent>,
    /// Event sender (for external use)
    #[allow(dead_code)]
    sender: mpsc::UnboundedSender<AppEvent>,
    /// Handler for the event loop
    handler: tokio::task::JoinHandle<()>,
}

impl EventHandler {
    /// Create a new event handler
    pub fn new(tick_rate: Duration) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let handler = {
            let sender = sender.clone();
            tokio::spawn(async move {
                let mut last_tick = Instant::now();

                loop {
                    let timeout = tick_rate
                        .checked_sub(last_tick.elapsed())
                        .unwrap_or_else(|| Duration::from_secs(0));

                    if event::poll(timeout).unwrap_or(false) {
                        match event::read() {
                            Ok(Event::Key(key_event)) => {
                                if let Err(_) = sender.send(AppEvent::Key(key_event)) {
                                    break;
                                }
                            }
                            Ok(Event::Resize(width, height)) => {
                                if let Err(_) = sender.send(AppEvent::Resize(width, height)) {
                                    break;
                                }
                            }
                            Ok(_) => {}
                            Err(_) => {
                                if let Err(_) = sender.send(AppEvent::Quit) {
                                    break;
                                }
                            }
                        }
                    }

                    if last_tick.elapsed() >= tick_rate {
                        if let Err(_) = sender.send(AppEvent::Tick) {
                            break;
                        }
                        last_tick = Instant::now();
                    }
                }
            })
        };

        Self {
            receiver,
            sender,
            handler,
        }
    }

    /// Receive the next event
    pub async fn next(&mut self) -> Result<AppEvent> {
        self.receiver
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("Event channel closed"))
    }

    /// Get a sender for external event injection
    #[allow(dead_code)]
    pub fn sender(&self) -> mpsc::UnboundedSender<AppEvent> {
        self.sender.clone()
    }

    /// Close the event handler
    pub fn close(&mut self) {
        self.handler.abort();
    }
}

impl Drop for EventHandler {
    fn drop(&mut self) {
        self.close();
    }
}

/// Handle global key events that apply to the entire application
pub fn handle_global_key_event(key_event: KeyEvent) -> Option<AppEvent> {
    match key_event {
        // Ctrl+C or Ctrl+Q to quit
        KeyEvent {
            code: KeyCode::Char('c'),
            modifiers: KeyModifiers::CONTROL,
            ..
        }
        | KeyEvent {
            code: KeyCode::Char('q'),
            modifiers: KeyModifiers::CONTROL,
            ..
        } => {
            debug!("Global quit key pressed");
            Some(AppEvent::Quit)
        }

        // ESC to quit
        KeyEvent {
            code: KeyCode::Esc,
            modifiers: KeyModifiers::NONE,
            ..
        } => {
            debug!("ESC key pressed");
            Some(AppEvent::Quit)
        }

        _ => None,
    }
}

/// Helper functions for key event matching
pub mod key_utils {
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    /// Check if key event is a specific character
    #[allow(dead_code)]
    pub fn is_char(key_event: KeyEvent, c: char) -> bool {
        matches!(
            key_event,
            KeyEvent {
                code: KeyCode::Char(ch),
                modifiers: KeyModifiers::NONE,
                ..
            } if ch == c
        )
    }

    /// Check if key event is a specific character with Ctrl
    #[allow(dead_code)]
    pub fn is_ctrl_char(key_event: KeyEvent, c: char) -> bool {
        matches!(
            key_event,
            KeyEvent {
                code: KeyCode::Char(ch),
                modifiers: KeyModifiers::CONTROL,
                ..
            } if ch == c
        )
    }

    /// Check if key event is Enter
    pub fn is_enter(key_event: KeyEvent) -> bool {
        matches!(
            key_event,
            KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::NONE,
                ..
            }
        )
    }

    /// Check if key event is Ctrl+Enter
    pub fn is_ctrl_enter(key_event: KeyEvent) -> bool {
        matches!(
            key_event,
            KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::CONTROL,
                ..
            }
        )
    }

    /// Check if key event is Tab
    #[allow(dead_code)]
    pub fn is_tab(key_event: KeyEvent) -> bool {
        matches!(
            key_event,
            KeyEvent {
                code: KeyCode::Tab,
                modifiers: KeyModifiers::NONE,
                ..
            }
        )
    }

    /// Check if key event is Shift+Tab
    #[allow(dead_code)]
    pub fn is_shift_tab(key_event: KeyEvent) -> bool {
        matches!(
            key_event,
            KeyEvent {
                code: KeyCode::BackTab,
                ..
            }
        )
    }

    /// Check if key event is F1
    #[allow(dead_code)]
    pub fn is_f1(key_event: KeyEvent) -> bool {
        matches!(
            key_event,
            KeyEvent {
                code: KeyCode::F(1),
                modifiers: KeyModifiers::NONE,
                ..
            }
        )
    }

    /// Check if key event is arrow key
    #[allow(dead_code)]
    pub fn is_arrow(key_event: KeyEvent) -> Option<Direction> {
        match key_event {
            KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::NONE,
                ..
            } => Some(Direction::Up),
            KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::NONE,
                ..
            } => Some(Direction::Down),
            KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::NONE,
                ..
            } => Some(Direction::Left),
            KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::NONE,
                ..
            } => Some(Direction::Right),
            _ => None,
        }
    }

    /// Direction enum for arrow keys
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Direction {
        Up,
        Down,
        Left,
        Right,
    }

    /// Check if key event is a number
    pub fn is_number(key_event: KeyEvent) -> Option<u8> {
        match key_event {
            KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::NONE,
                ..
            } if c.is_ascii_digit() => c.to_digit(10).map(|d| d as u8),
            _ => None,
        }
    }

    /// Check if key event is Page Up/Down
    #[allow(dead_code)]
    pub fn is_page(key_event: KeyEvent) -> Option<Direction> {
        match key_event {
            KeyEvent {
                code: KeyCode::PageUp,
                modifiers: KeyModifiers::NONE,
                ..
            } => Some(Direction::Up),
            KeyEvent {
                code: KeyCode::PageDown,
                modifiers: KeyModifiers::NONE,
                ..
            } => Some(Direction::Down),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::key_utils::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn test_global_key_handling() {
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(matches!(handle_global_key_event(ctrl_c), Some(AppEvent::Quit)));

        let ctrl_q = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::CONTROL);
        assert!(matches!(handle_global_key_event(ctrl_q), Some(AppEvent::Quit)));

        let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert!(matches!(handle_global_key_event(esc), Some(AppEvent::Quit)));

        let regular_key = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        assert!(handle_global_key_event(regular_key).is_none());
    }

    #[test]
    fn test_key_utils() {
        let char_a = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        assert!(is_char(char_a, 'a'));
        assert!(!is_char(char_a, 'b'));

        let ctrl_a = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL);
        assert!(is_ctrl_char(ctrl_a, 'a'));
        assert!(!is_char(ctrl_a, 'a'));

        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert!(is_enter(enter));

        let ctrl_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL);
        assert!(is_ctrl_enter(ctrl_enter));
        assert!(!is_enter(ctrl_enter));

        let tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        assert!(is_tab(tab));

        let shift_tab = KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT);
        assert!(is_shift_tab(shift_tab));

        let f1 = KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE);
        assert!(is_f1(f1));

        let up = KeyEvent::new(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(is_arrow(up), Some(Direction::Up));

        let digit_5 = KeyEvent::new(KeyCode::Char('5'), KeyModifiers::NONE);
        assert_eq!(is_number(digit_5), Some(5));

        let page_up = KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE);
        assert_eq!(is_page(page_up), Some(Direction::Up));
    }

    #[tokio::test]
    async fn test_event_handler_creation() {
        let tick_rate = Duration::from_millis(100);
        let handler = EventHandler::new(tick_rate);

        // Test that we can get a sender
        let sender = handler.sender();

        // Test sending a custom event
        sender.send(AppEvent::Quit).unwrap();

        // This test just ensures the handler can be created and basic operations work
        // More comprehensive testing would require mocking crossterm events
    }
}
