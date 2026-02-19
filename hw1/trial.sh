#!/usr/bin/env bash

# Generate unique session ID
SESSION_ID=$(date +"%Y-%m-%d_%H-%M-%S")

# EXPORT the variable so Python's os.environ can see it!
export SESSION_DIR="logs/session_$SESSION_ID"

# Create the session directory
mkdir -p "$SESSION_DIR"
echo "Starting training session. Logs and artifacts will be saved to $SESSION_DIR"

# Run Python, piping stdout to the session's console log
python3 -u myrunner.py 2>&1 | tee -i "$SESSION_DIR/console.log"

# Capture Python's exit code
py_exit_code=${PIPESTATUS[0]}

# Cleanup or Finalize
if [ $py_exit_code -eq 5 ]; then
    echo "!!! Execution aborted (Exit Code 5). Cleaning up temporary files..."
    # Deleting the dir also deletes the partially written decisions.log and history.log
    rm -rf "$SESSION_DIR"
    echo "Cleanup complete. Discarded run."
else
    echo "Process finished normally. Session assets secured at $SESSION_DIR"
fi
