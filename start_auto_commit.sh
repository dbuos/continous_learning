#!/bin/bash
# Auto-Commit Manager Script
# Controls the continuous git monitoring process

SCRIPT_DIR="/content"
PYTHON_SCRIPT="$SCRIPT_DIR/auto_commit.py"
PID_FILE="$SCRIPT_DIR/.auto_commit.pid"
LOG_FILE="$SCRIPT_DIR/auto_commit.log"

case "$1" in
  start)
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      if ps -p "$PID" > /dev/null 2>&1; then
        echo "Auto-commit is already running (PID: $PID)"
        exit 1
      else
        rm -f "$PID_FILE"
      fi
    fi

    echo "Starting auto-commit monitoring..."
    nohup python3 "$PYTHON_SCRIPT" --interval 30 > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Auto-commit started (PID: $(cat $PID_FILE))"
    echo "Logs: $LOG_FILE"
    ;;

  stop)
    if [ ! -f "$PID_FILE" ]; then
      echo "Auto-commit is not running"
      exit 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
      kill "$PID"
      rm -f "$PID_FILE"
      echo "Auto-commit stopped"
    else
      rm -f "$PID_FILE"
      echo "Auto-commit was not running (stale PID file removed)"
    fi
    ;;

  status)
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      if ps -p "$PID" > /dev/null 2>&1; then
        echo "Auto-commit is running (PID: $PID)"
        echo "Last 10 log entries:"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "No logs available"
      else
        echo "Auto-commit is not running (stale PID file)"
        rm -f "$PID_FILE"
      fi
    else
      echo "Auto-commit is not running"
    fi
    ;;

  restart)
    $0 stop
    sleep 2
    $0 start
    ;;

  logs)
    if [ -f "$LOG_FILE" ]; then
      tail -f "$LOG_FILE"
    else
      echo "No log file found"
    fi
    ;;

  *)
    echo "Usage: $0 {start|stop|status|restart|logs}"
    echo ""
    echo "Commands:"
    echo "  start   - Start the auto-commit monitoring process"
    echo "  stop    - Stop the auto-commit monitoring process"
    echo "  status  - Check if auto-commit is running"
    echo "  restart - Restart the auto-commit process"
    echo "  logs    - Follow the log file (Ctrl+C to exit)"
    exit 1
    ;;
esac

exit 0
