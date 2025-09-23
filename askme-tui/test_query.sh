#!/bin/bash

# Simple test script to verify the TUI client works

echo "Testing askme TUI client..."

# Build the client
echo "Building askme-tui..."
cargo build --release

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build successful"

# Test if we can show help without crashing
echo "Testing help display..."
timeout 2s cargo run --release &
sleep 1
pkill -f askme-tui

if [ $? -eq 0 ]; then
    echo "✅ TUI client starts without errors"
else
    echo "❌ TUI client failed to start"
    exit 1
fi

echo "🎉 Basic tests passed!"
echo ""
echo "Manual testing checklist:"
echo "1. ↑↓ keys navigate between fields"
echo "2. ←→ keys adjust numeric values"
echo "3. Space toggles boolean options"
echo "4. F1 toggles advanced options"
echo "5. Number keys set values directly"
echo "6. Progress bars show during processing"
echo "7. No debug output appears in TUI"
echo ""
echo "To test query functionality:"
echo "1. Start the askme API server: uvicorn askme.api.main:app --port 8080"
echo "2. Run: cargo run --release"
echo "3. Navigate to Query tab, enter a question, and submit"
