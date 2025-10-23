#!/usr/bin/env python3
"""
Continuous Git Auto-Commit and Push Script

Monitors the repository for changes and automatically:
1. Detects file changes
2. Generates meaningful commit messages
3. Commits changes
4. Pushes to remote repository

Usage:
    python auto_commit.py [--interval SECONDS] [--verbose]
"""

import subprocess
import time
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set


class GitAutoCommit:
    def __init__(self, repo_path: str = "/content", interval: int = 30, verbose: bool = False):
        self.repo_path = Path(repo_path)
        self.interval = interval
        self.verbose = verbose
        self.last_commit_time = None

        # Verify git repo
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"{repo_path} is not a git repository")

    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = "🔍" if level == "INFO" else "✓" if level == "SUCCESS" else "⚠"
        print(f"[{timestamp}] {prefix} {message}")

    def run_git_command(self, command: List[str]) -> tuple:
        """Run git command and return (stdout, stderr, returncode)"""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1

    def get_git_status(self) -> Dict[str, List[str]]:
        """Get current git status categorized by change type"""
        stdout, stderr, code = self.run_git_command(["git", "status", "--porcelain"])

        if code != 0:
            self.log(f"Git status failed: {stderr}", "ERROR")
            return {}

        changes = {
            "added": [],
            "modified": [],
            "deleted": [],
            "untracked": []
        }

        for line in stdout.split("\n"):
            if not line:
                continue

            status = line[:2]
            filepath = line[3:].strip()

            # Skip certain files
            if any(skip in filepath for skip in [".pyc", "__pycache__", ".git/", "auto_commit.py"]):
                continue

            if status == "??":
                changes["untracked"].append(filepath)
            elif status[0] == "M" or status[1] == "M":
                changes["modified"].append(filepath)
            elif status[0] == "A" or status[1] == "A":
                changes["added"].append(filepath)
            elif status[0] == "D" or status[1] == "D":
                changes["deleted"].append(filepath)

        return changes

    def generate_commit_message(self, changes: Dict[str, List[str]]) -> str:
        """Generate a meaningful commit message based on changes"""

        # Count changes
        total_changes = sum(len(files) for files in changes.values())

        if total_changes == 0:
            return None

        # Build commit message parts
        parts = []

        # Determine primary action
        if changes["added"]:
            if len(changes["added"]) == 1:
                parts.append(f"Add {changes['added'][0]}")
            else:
                parts.append(f"Add {len(changes['added'])} new files")

        if changes["modified"]:
            if len(changes["modified"]) == 1:
                parts.append(f"Update {changes['modified'][0]}")
            else:
                parts.append(f"Update {len(changes['modified'])} files")

        if changes["deleted"]:
            if len(changes["deleted"]) == 1:
                parts.append(f"Delete {changes['deleted'][0]}")
            else:
                parts.append(f"Delete {len(changes['deleted'])} files")

        if changes["untracked"]:
            if len(changes["untracked"]) == 1:
                parts.append(f"Add {changes['untracked'][0]}")
            else:
                parts.append(f"Add {len(changes['untracked'])} new files")

        # Create title
        title = " and ".join(parts)

        # Add detailed body
        body_lines = []

        if changes["added"]:
            body_lines.append("\nAdded files:")
            for f in changes["added"][:5]:  # Limit to 5
                body_lines.append(f"- {f}")
            if len(changes["added"]) > 5:
                body_lines.append(f"- ... and {len(changes['added']) - 5} more")

        if changes["modified"]:
            body_lines.append("\nModified files:")
            for f in changes["modified"][:5]:
                body_lines.append(f"- {f}")
            if len(changes["modified"]) > 5:
                body_lines.append(f"- ... and {len(changes['modified']) - 5} more")

        if changes["deleted"]:
            body_lines.append("\nDeleted files:")
            for f in changes["deleted"][:5]:
                body_lines.append(f"- {f}")
            if len(changes["deleted"]) > 5:
                body_lines.append(f"- ... and {len(changes['deleted']) - 5} more")

        if changes["untracked"]:
            body_lines.append("\nNew files:")
            for f in changes["untracked"][:5]:
                body_lines.append(f"- {f}")
            if len(changes["untracked"]) > 5:
                body_lines.append(f"- ... and {len(changes['untracked']) - 5} more")

        # Combine
        message = title
        if body_lines:
            message += "\n" + "\n".join(body_lines)

        # Add footer
        message += f"\n\n🤖 Auto-committed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        message += "\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"
        message += "\nCo-Authored-By: Claude <noreply@anthropic.com>"

        return message

    def commit_and_push(self, changes: Dict[str, List[str]]) -> bool:
        """Commit and push changes"""

        # Generate commit message
        commit_msg = self.generate_commit_message(changes)
        if not commit_msg:
            return False

        self.log("Changes detected, creating commit...")

        # Add all changes
        stdout, stderr, code = self.run_git_command(["git", "add", "-A"])
        if code != 0:
            self.log(f"Git add failed: {stderr}", "ERROR")
            return False

        # Commit
        stdout, stderr, code = self.run_git_command(["git", "commit", "-m", commit_msg])
        if code != 0:
            if "nothing to commit" in stderr:
                if self.verbose:
                    self.log("Nothing to commit after staging")
                return False
            self.log(f"Git commit failed: {stderr}", "ERROR")
            return False

        self.log(f"Committed: {commit_msg.split(chr(10))[0]}", "SUCCESS")

        # Push
        self.log("Pushing to remote...")
        stdout, stderr, code = self.run_git_command(["git", "push"])
        if code != 0:
            self.log(f"Git push failed: {stderr}", "ERROR")
            return False

        self.log("Pushed to remote successfully", "SUCCESS")
        self.last_commit_time = datetime.now()
        return True

    def check_and_commit(self):
        """Check for changes and commit if any"""
        changes = self.get_git_status()

        # Count total changes
        total = sum(len(files) for files in changes.values())

        if total == 0:
            if self.verbose:
                self.log("No changes detected")
            return

        # Log changes found
        change_summary = []
        if changes["added"]:
            change_summary.append(f"{len(changes['added'])} added")
        if changes["modified"]:
            change_summary.append(f"{len(changes['modified'])} modified")
        if changes["deleted"]:
            change_summary.append(f"{len(changes['deleted'])} deleted")
        if changes["untracked"]:
            change_summary.append(f"{len(changes['untracked'])} untracked")

        self.log(f"Found changes: {', '.join(change_summary)}")

        # Commit and push
        self.commit_and_push(changes)

    def run(self):
        """Main monitoring loop"""
        self.log(f"Starting continuous monitoring of {self.repo_path}")
        self.log(f"Check interval: {self.interval} seconds")
        self.log("Press Ctrl+C to stop")

        try:
            while True:
                self.check_and_commit()
                time.sleep(self.interval)

        except KeyboardInterrupt:
            self.log("\nMonitoring stopped by user")
        except Exception as e:
            self.log(f"Error in monitoring loop: {e}", "ERROR")
            raise


def main():
    parser = argparse.ArgumentParser(description="Continuous Git Auto-Commit and Push")
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for testing)"
    )

    args = parser.parse_args()

    auto_commit = GitAutoCommit(
        repo_path="/content",
        interval=args.interval,
        verbose=args.verbose
    )

    if args.once:
        auto_commit.log("Running in test mode (one-time check)")
        auto_commit.check_and_commit()
        auto_commit.log("Test complete")
    else:
        auto_commit.run()


if __name__ == "__main__":
    main()
