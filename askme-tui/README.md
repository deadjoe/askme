# askme-tui

A modern Terminal User Interface (TUI) client for the askme RAG system, built with Rust and ratatui.

## Features

- **🔍 Query Interface**: Interactive question answering with configurable parameters
- **📄 Document Ingestion**: Support for files, directories, and URLs with real-time progress
- **⚙️ Settings Management**: Persistent configuration with API key management
- **📊 Real-time Status**: Live progress tracking and task monitoring
- **🎨 Modern UI**: Clean, responsive interface with keyboard navigation
- **🔧 Advanced Options**: HyDE, RAG-Fusion, multiple rerankers, output formats

## Installation

### From Source

```bash
# Clone the repository (if not already in askme project)
cd askme-tui/

# Build and install
cargo build --release

# Run the TUI
cargo run --release
```

### System Requirements

- Rust 1.75 or later
- Terminal with Unicode support
- askme API server running (default: http://localhost:8080)

## Quick Start

1. **Start askme API server**:
   ```bash
   # From the main askme directory
   uvicorn askme.api.main:app --port 8080
   ```

2. **Launch TUI client**:
   ```bash
   cd askme-tui/
   cargo run --release
   ```

3. **Navigate and use**:
   - Use `1-4` keys or Tab to switch between tabs
   - Follow on-screen prompts for each tab
   - Press `Ctrl+Q` or `ESC` to quit

## Usage Guide

### Tab Navigation

- **1 or Query Tab**: Ask questions and view answers
- **2 or Ingest Tab**: Add documents to your knowledge base
- **3 or Settings Tab**: Configure API settings and defaults
- **4 or Help Tab**: View comprehensive help information

### Query Tab

#### Basic Usage
1. Enter your question in the text area
2. Adjust parameters as needed (TopK, Alpha)
3. Press `Enter` or `Ctrl+Enter` to submit
4. View results in the response panel

#### Advanced Parameters (Press F1)
- **TopK**: Number of documents to retrieve (1-100)
- **Alpha**: Hybrid search balance (0.0=sparse, 1.0=dense)
- **RRF**: Use Reciprocal Rank Fusion
- **HyDE**: Enable Hypothetical Document Embeddings
- **RAG-Fusion**: Enable multi-query generation
- **Reranker**: Choose between Qwen local or BGE local
- **Max Passages**: Limit passages for generation (1-20)

#### Keyboard Controls
- `Tab/Shift+Tab`: Navigate fields
- `←/→`: Adjust numeric values
- `Space`: Toggle boolean options
- `Enter/Ctrl+Enter`: Submit query
- `Page Up/Down`: Scroll response

### Ingest Tab

#### Supported Sources
- **Files**: PDF, text, markdown, etc.
- **Directories**: Batch process multiple files
- **URLs**: Web pages and documents

#### Process
1. Enter file path, directory, or URL
2. Add optional comma-separated tags
3. Toggle overwrite option if needed
4. Press `Enter` to start ingestion
5. Monitor progress in real-time

#### Example Paths
```
/path/to/document.pdf
/path/to/documents/
https://example.com/api-docs
```

### Settings Tab

#### Configuration Options
- **API URL**: askme server address
- **API Key**: Authentication token (optional)
- **Default Query Parameters**: Set defaults for new queries
- **Auto Save**: Automatically save configuration changes

#### Actions
- **Save Config**: Apply and persist changes
- **Reset to Defaults**: Restore factory settings

### Global Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit application |
| `Ctrl+C` | Quit application |
| `ESC` | Quit application |
| `1-4` | Switch to tab (Query/Ingest/Settings/Help) |
| `F1` | Toggle advanced options |
| `Tab` | Next field |
| `Shift+Tab` | Previous field |

## Configuration

### Environment Variables

```bash
# Set default API URL
export ASKME_API_URL="http://localhost:8080"

# Set API key for authentication
export ASKME_API_KEY="your-api-key-here"
```

### Configuration File

Settings are automatically saved to:
- **Linux/macOS**: `~/.config/askme-tui/config.toml`
- **Windows**: `%APPDATA%\askme-tui\config.toml`

Example configuration:
```toml
api_url = "http://localhost:8080"
api_key = "your-api-key"
auto_save = true
output_format = "text"

[default_query]
topk = 50
alpha = 0.5
use_rrf = true
rrf_k = 60
use_hyde = false
use_rag_fusion = false
reranker = "qwen_local"
max_passages = 8
include_debug = false
```

## API Compatibility

The TUI client is compatible with askme API endpoints:

- `POST /query/` - Submit search queries
- `POST /ingest/` - Start document ingestion
- `GET /ingest/status/{task_id}` - Check ingestion progress
- `GET /health/` - API health check

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test module
cargo test api::tests
```

### Project Structure

```
src/
├── main.rs           # Entry point and main loop
├── app.rs            # Application state and logic
├── ui.rs             # UI rendering
├── event.rs          # Event handling
├── api/              # API client and types
│   ├── mod.rs
│   ├── client.rs
│   └── types.rs
└── tabs/             # Tab implementations
    ├── mod.rs
    ├── query.rs
    ├── ingest.rs
    └── settings.rs
```

### Dependencies

- **ratatui**: Modern TUI framework
- **crossterm**: Cross-platform terminal manipulation
- **tokio**: Async runtime
- **reqwest**: HTTP client
- **serde**: Serialization framework
- **tui-textarea**: Text input widget

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Ensure askme API server is running
   - Check API URL in settings
   - Verify network connectivity

2. **Authentication Errors**
   - Set API key in settings or environment
   - Verify key is correct

3. **Ingestion Stuck**
   - Check source path exists
   - Verify permissions
   - Monitor logs for errors

4. **UI Display Issues**
   - Ensure terminal supports Unicode
   - Try resizing terminal window
   - Check terminal color support

### Logging

Enable debug logging:
```bash
RUST_LOG=askme_tui=debug cargo run
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run `cargo test` and `cargo fmt`
6. Submit a pull request

## License

This project is part of the askme system and follows the same license terms.

## Related Projects

- [askme](../): Main RAG system
- [askme API](../askme/api/): REST API server
- [askme Scripts](../scripts/): Command-line tools

For more information, see the main askme documentation.
