#!/usr/bin/env python3
"""
Release Script for LLMSummarizer Plugin

Manages semantic versioning, git tagging, and GitHub releases.

Usage:
    python scripts/release.py patch    # 0.1.0 -> 0.1.1
    python scripts/release.py minor    # 0.1.0 -> 0.2.0
    python scripts/release.py major    # 0.1.0 -> 1.0.0
    python scripts/release.py --current  # Show current version

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SemVer:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    
    @classmethod
    def parse(cls, version_str: str) -> "SemVer":
        """Parse version string like 'v1.2.3' or '1.2.3-beta'."""
        version_str = version_str.lstrip("v")
        
        # Match semver pattern
        match = re.match(
            r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$",
            version_str
        )
        
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
        )
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version
    
    @property
    def tag(self) -> str:
        return f"v{self}"
    
    def bump_major(self) -> "SemVer":
        return SemVer(self.major + 1, 0, 0)
    
    def bump_minor(self) -> "SemVer":
        return SemVer(self.major, self.minor + 1, 0)
    
    def bump_patch(self) -> "SemVer":
        return SemVer(self.major, self.minor, self.patch + 1)


def run_command(
    cmd: list[str],
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command."""
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}")
        if result.stderr:
            print(f"  {result.stderr}")
        sys.exit(1)
    return result


def get_current_version() -> SemVer | None:
    """Get current version from git tags."""
    result = run_command(
        ["git", "tag", "--list", "v*", "--sort=-v:refname"],
        check=False,
    )
    
    if result.returncode != 0 or not result.stdout.strip():
        return None
    
    # Get the latest version tag
    tags = result.stdout.strip().split("\n")
    
    for tag in tags:
        tag = tag.strip()
        if re.match(r"^v\d+\.\d+\.\d+", tag):
            try:
                return SemVer.parse(tag)
            except ValueError:
                continue
    
    return None


def check_git_status() -> bool:
    """Check if working directory is clean."""
    result = run_command(["git", "status", "--porcelain"], check=False)
    return not result.stdout.strip()


def check_on_main_branch() -> bool:
    """Check if on main/master branch."""
    result = run_command(["git", "branch", "--show-current"], check=False)
    branch = result.stdout.strip()
    return branch in ("main", "master")


def get_remote_name() -> str:
    """Get the remote name (usually 'origin')."""
    result = run_command(["git", "remote"], check=False)
    remotes = result.stdout.strip().split("\n")
    return remotes[0] if remotes else "origin"


def create_release(
    version: SemVer,
    push: bool = True,
    create_github_release: bool = False,
) -> None:
    """Create a new release."""
    tag = version.tag
    
    print(f"\nCreating release {tag}...")
    
    # Create annotated tag
    print(f"  Creating tag: {tag}")
    run_command([
        "git", "tag", "-a", tag,
        "-m", f"Release {tag}"
    ])
    
    if push:
        remote = get_remote_name()
        print(f"  Pushing tag to {remote}...")
        run_command(["git", "push", remote, tag])
    
    if create_github_release:
        print("  Creating GitHub release...")
        
        # Generate release notes from recent commits
        result = run_command([
            "git", "log", "--oneline", "-10",
            "--pretty=format:- %s"
        ], check=False)
        
        notes = f"## What's Changed\n\n{result.stdout}\n"
        
        run_command([
            "gh", "release", "create", tag,
            "--title", f"Release {tag}",
            "--notes", notes,
        ])
    
    print(f"\n✓ Released {tag}")


def main():
    parser = argparse.ArgumentParser(
        description="Release management with semantic versioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/release.py patch          # Bump patch version
  python scripts/release.py minor          # Bump minor version
  python scripts/release.py major          # Bump major version
  python scripts/release.py --current      # Show current version
  python scripts/release.py patch --github # Create GitHub release
        """
    )
    
    parser.add_argument(
        "bump",
        nargs="?",
        choices=["major", "minor", "patch"],
        help="Version component to bump"
    )
    parser.add_argument(
        "--current", "-c",
        action="store_true",
        help="Show current version and exit"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Create tag locally without pushing"
    )
    parser.add_argument(
        "--github", "-g",
        action="store_true",
        help="Create GitHub release (requires gh CLI)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip safety checks"
    )
    parser.add_argument(
        "--set-version",
        help="Set specific version (e.g., 1.0.0)"
    )
    
    args = parser.parse_args()
    
    # Get current version
    current = get_current_version()
    
    if args.current:
        if current:
            print(f"Current version: {current.tag}")
        else:
            print("No version tags found. First release will be v0.1.0")
        sys.exit(0)
    
    if not args.bump and not args.set_version:
        parser.print_help()
        sys.exit(1)
    
    # Safety checks
    if not args.force:
        if not check_git_status():
            print("Error: Working directory has uncommitted changes.")
            print("Commit your changes first, or use --force to skip this check.")
            sys.exit(1)
        
        if not check_on_main_branch():
            print("Warning: Not on main/master branch.")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != "y":
                sys.exit(1)
    
    # Determine new version
    if args.set_version:
        new_version = SemVer.parse(args.set_version)
    elif current is None:
        # First release
        new_version = SemVer(0, 1, 0)
    else:
        if args.bump == "major":
            new_version = current.bump_major()
        elif args.bump == "minor":
            new_version = current.bump_minor()
        else:
            new_version = current.bump_patch()
    
    # Confirm
    if current:
        print(f"Current version: {current.tag}")
    else:
        print("No previous version found.")
    
    print(f"New version: {new_version.tag}")
    
    if not args.force:
        response = input("\nProceed with release? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(1)
    
    # Create release
    create_release(
        version=new_version,
        push=not args.no_push,
        create_github_release=args.github,
    )


if __name__ == "__main__":
    main()
