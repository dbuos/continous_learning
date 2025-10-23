# Continuous Auto-Commit System

This repository is equipped with an automatic git monitoring system that continuously watches for changes and commits/pushes them automatically.

## Overview

The system consists of two main components:

1. **`auto_commit.py`** - Python script that monitors the repository for changes
2. **`start_auto_commit.sh`** - Shell script to manage the monitoring process

## Quick Start

### Start Monitoring
```bash
./start_auto_commit.sh start
```

### Stop Monitoring
```bash
./start_auto_commit.sh stop
```

### Check Status
```bash
./start_auto_commit.sh status
```

### View Live Logs
```bash
./start_auto_commit.sh logs
```

### Restart
```bash
./start_auto_commit.sh restart
```

## How It Works

1. **Continuous Monitoring**: The script checks for file changes every 30 seconds
2. **Automatic Staging**: All changes (new, modified, deleted files) are automatically staged
3. **Smart Commit Messages**: Generates meaningful commit messages based on what changed
4. **Auto-Push**: Automatically pushes commits to the remote repository

## Commit Message Format

The system generates structured commit messages like:

```
Add 3 new files and Update 2 files

Added files:
- new_experiment.py
- results.json
- analysis.md

Modified files:
- CLAUDE.md
- README.md

🤖 Auto-committed at 2025-10-23 12:34:56

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

## Configuration

### Change Check Interval

Edit the interval when starting (default is 30 seconds):
```bash
python auto_commit.py --interval 60  # Check every 60 seconds
```

### Verbose Mode

Enable verbose logging to see all activity:
```bash
python auto_commit.py --verbose --interval 30
```

### Test Mode

Run a single check without continuous monitoring:
```bash
python auto_commit.py --once --verbose
```

## Excluded Files

The system automatically excludes:
- `.pyc` files
- `__pycache__/` directories
- `.git/` directory
- The `auto_commit.py` script itself (to avoid recursion)

## Logs

All activity is logged to:
```
/content/auto_commit.log
```

View logs in real-time:
```bash
tail -f /content/auto_commit.log
```

Or using the manager script:
```bash
./start_auto_commit.sh logs
```

## Process Management

The monitoring process runs in the background. Process ID is stored in:
```
/content/.auto_commit.pid
```

To manually check if running:
```bash
ps aux | grep auto_commit.py
```

## Remote Repository

Current remote:
```
git@github.com:dbuos/continous_learning.git
```

## Troubleshooting

### Process Not Starting

1. Check if already running:
   ```bash
   ./start_auto_commit.sh status
   ```

2. Check for errors in logs:
   ```bash
   tail -50 /content/auto_commit.log
   ```

### Push Failures

If pushes fail (e.g., permission issues), check:
1. SSH keys are configured correctly
2. Remote repository is accessible
3. Network connectivity

### Manual Override

To manually commit and push at any time:
```bash
git add -A
git commit -m "Your message"
git push
```

## Features

✓ **Automatic Detection**: Monitors all file changes
✓ **Smart Messages**: Context-aware commit messages
✓ **Background Process**: Runs continuously without user interaction
✓ **Safe**: Won't commit if nothing changed
✓ **Efficient**: Minimal CPU/memory usage
✓ **Reliable**: Handles errors gracefully

## Advanced Usage

### Run as a Service

To ensure it starts on system boot, you could add to crontab:
```bash
@reboot /content/start_auto_commit.sh start
```

### Custom Intervals for Different Times

You can modify the script to use different intervals based on time of day, activity patterns, etc.

## Repository Structure

```
/content/
├── auto_commit.py              # Main monitoring script
├── start_auto_commit.sh        # Process manager
├── AUTO_COMMIT_README.md       # This file
├── auto_commit.log             # Activity logs
├── .auto_commit.pid            # Process ID (when running)
└── [your project files]
```

## Current Status

The monitoring system is **ACTIVE** and running with:
- Check interval: 30 seconds
- Remote: git@github.com:dbuos/continous_learning.git
- Branch: master

Any changes you make will be automatically committed and pushed within 30 seconds.

## Stop Auto-Commit

If you need to make manual commits or want to stop auto-commit:

```bash
./start_auto_commit.sh stop
```

To resume:
```bash
./start_auto_commit.sh start
```

---

**Note**: This system is ideal for development, experiments, and continuous documentation. For production code, you may want to manually review commits before pushing.
