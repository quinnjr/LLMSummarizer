#!/usr/bin/env python3
"""
Upload RAG Database to GitHub Releases

Packages the ChromaDB RAG database and uploads it as a GitHub release artifact
for easy distribution and download by users.

Prerequisites:
    - GitHub CLI (gh) installed and authenticated
    - Repository with push access

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def check_gh_cli() -> bool:
    """Check if GitHub CLI is installed and authenticated."""
    try:
        result = run_command(["gh", "auth", "status"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_repo_info() -> tuple[str, str] | None:
    """Get GitHub owner/repo from git remote."""
    try:
        result = run_command(["git", "remote", "get-url", "origin"], check=False)
        if result.returncode != 0:
            return None
        
        url = result.stdout.strip()
        
        # Parse GitHub URL (handles both HTTPS and SSH)
        if "github.com" in url:
            # Remove .git suffix
            url = url.rstrip(".git")
            
            if url.startswith("git@"):
                # SSH format: git@github.com:owner/repo
                parts = url.split(":")[-1].split("/")
            else:
                # HTTPS format: https://github.com/owner/repo
                parts = url.split("github.com/")[-1].split("/")
            
            if len(parts) >= 2:
                return parts[0], parts[1]
        
        return None
    except Exception:
        return None


def compute_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def package_database(db_path: Path, output_dir: Path) -> tuple[Path, dict]:
    """
    Package the RAG database into a compressed archive.
    
    Args:
        db_path: Path to ChromaDB database directory
        output_dir: Directory for output archive
        
    Returns:
        Tuple of (archive_path, metadata_dict)
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load database metadata
    metadata_file = db_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            db_metadata = json.load(f)
    else:
        db_metadata = {}
    
    # Generate version string
    timestamp = datetime.now().strftime("%Y%m%d")
    papers_count = db_metadata.get("papers_count", "unknown")
    version = f"v{timestamp}-{papers_count}papers"
    
    # Create archive
    archive_name = f"pd_literature_db_{version}.tar.gz"
    archive_path = output_dir / archive_name
    
    print(f"  Creating archive: {archive_name}")
    
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add database files
        tar.add(db_path, arcname="pd_literature_db")
    
    # Compute checksum
    checksum = compute_checksum(archive_path)
    
    # Get file size
    size_bytes = archive_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    # Build release metadata
    release_metadata = {
        "version": version,
        "archive_name": archive_name,
        "checksum_sha256": checksum,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "created": datetime.now().isoformat(),
        "database_metadata": db_metadata,
    }
    
    # Write checksum file
    checksum_file = output_dir / f"{archive_name}.sha256"
    with open(checksum_file, "w") as f:
        f.write(f"{checksum}  {archive_name}\n")
    
    # Write release metadata
    release_meta_file = output_dir / f"{archive_name}.json"
    with open(release_meta_file, "w") as f:
        json.dump(release_metadata, f, indent=2)
    
    return archive_path, release_metadata


def create_github_release(
    repo: str,
    version: str,
    archive_path: Path,
    metadata: dict,
    prerelease: bool = False,
) -> str:
    """
    Create a GitHub release and upload the archive.
    
    Args:
        repo: GitHub repo in owner/repo format
        version: Release version tag
        archive_path: Path to archive file
        metadata: Release metadata
        prerelease: Whether this is a prerelease
        
    Returns:
        Release URL
    """
    # Build release notes
    db_meta = metadata.get("database_metadata", {})
    
    release_notes = f"""## PD Literature RAG Database

Pre-built ChromaDB database containing curated Parkinson's disease research findings for RAG-enhanced summarization.

### Database Statistics
- **Papers**: {db_meta.get('papers_count', 'N/A')}
- **Findings**: {db_meta.get('findings_count', 'N/A')}
- **Min Year**: {db_meta.get('min_year', 'N/A')}
- **LLM Model Used**: {db_meta.get('llm_model', 'N/A')}
- **Build Date**: {db_meta.get('created', 'N/A')}

### Download & Install

```bash
# Download the database
curl -L -o pd_literature_db.tar.gz \\
  https://github.com/{repo}/releases/download/{version}/{archive_path.name}

# Verify checksum
echo "{metadata['checksum_sha256']}  pd_literature_db.tar.gz" | sha256sum -c

# Extract to LLMSummarizer data directory
mkdir -p data
tar -xzf pd_literature_db.tar.gz -C data/
```

### File Info
- **Size**: {metadata['size_mb']} MB
- **SHA256**: `{metadata['checksum_sha256']}`

### Usage
Enable RAG in your LLMSummarizer parameters:
```
use_rag    true
literature_db    data/pd_literature_db
```
"""
    
    # Write release notes to temp file
    notes_file = archive_path.parent / "release_notes.md"
    with open(notes_file, "w") as f:
        f.write(release_notes)
    
    # Create release
    print(f"  Creating GitHub release: {version}")
    
    cmd = [
        "gh", "release", "create", version,
        "--repo", repo,
        "--title", f"PD Literature RAG Database {version}",
        "--notes-file", str(notes_file),
    ]
    
    if prerelease:
        cmd.append("--prerelease")
    
    # Add files to upload
    cmd.append(str(archive_path))
    cmd.append(str(archive_path.with_suffix(".gz.sha256")))
    cmd.append(str(archive_path.with_suffix(".gz.json")))
    
    run_command(cmd)
    
    # Get release URL
    result = run_command([
        "gh", "release", "view", version,
        "--repo", repo,
        "--json", "url",
        "-q", ".url"
    ])
    
    return result.stdout.strip()


def download_database(
    repo: str,
    output_dir: Path,
    version: str | None = None,
) -> Path:
    """
    Download RAG database from GitHub releases.
    
    Args:
        repo: GitHub repo in owner/repo format
        output_dir: Directory to extract database to
        version: Specific version to download (default: latest)
        
    Returns:
        Path to extracted database
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get release info
    if version:
        release_cmd = ["gh", "release", "view", version, "--repo", repo, "--json", "assets"]
    else:
        release_cmd = ["gh", "release", "view", "--repo", repo, "--json", "assets,tagName"]
    
    result = run_command(release_cmd)
    release_info = json.loads(result.stdout)
    
    if not version:
        version = release_info.get("tagName", "latest")
    
    # Find the tar.gz asset
    assets = release_info.get("assets", [])
    archive_asset = None
    checksum_asset = None
    
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(".tar.gz"):
            archive_asset = asset
        elif name.endswith(".sha256"):
            checksum_asset = asset
    
    if not archive_asset:
        raise RuntimeError("No database archive found in release")
    
    archive_name = archive_asset["name"]
    archive_path = output_dir / archive_name
    
    # Download archive
    print(f"  Downloading: {archive_name}")
    run_command([
        "gh", "release", "download", version,
        "--repo", repo,
        "--pattern", archive_name,
        "--dir", str(output_dir),
        "--clobber"
    ])
    
    # Verify checksum if available
    if checksum_asset:
        checksum_path = output_dir / checksum_asset["name"]
        run_command([
            "gh", "release", "download", version,
            "--repo", repo,
            "--pattern", checksum_asset["name"],
            "--dir", str(output_dir),
            "--clobber"
        ])
        
        print("  Verifying checksum...")
        with open(checksum_path) as f:
            expected_checksum = f.read().split()[0]
        
        actual_checksum = compute_checksum(archive_path)
        
        if actual_checksum != expected_checksum:
            raise RuntimeError(f"Checksum mismatch! Expected {expected_checksum}, got {actual_checksum}")
        
        print("  Checksum verified!")
    
    # Extract archive
    print("  Extracting database...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(output_dir)
    
    db_path = output_dir / "pd_literature_db"
    
    # Cleanup archive
    archive_path.unlink()
    if checksum_asset:
        checksum_path.unlink()
    
    print(f"  Database extracted to: {db_path}")
    return db_path


def main():
    parser = argparse.ArgumentParser(
        description="Upload/Download RAG database to/from GitHub releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload database to GitHub release
  python upload_rag_to_github.py upload --db data/pd_literature_db
  
  # Upload to specific repo
  python upload_rag_to_github.py upload --db data/pd_literature_db --repo owner/repo
  
  # Download latest database
  python upload_rag_to_github.py download --output data/
  
  # Download specific version
  python upload_rag_to_github.py download --output data/ --version v20240115-500papers
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload database to GitHub")
    upload_parser.add_argument(
        "--db", "-d",
        default="data/pd_literature_db",
        help="Path to ChromaDB database directory"
    )
    upload_parser.add_argument(
        "--repo", "-r",
        help="GitHub repo (owner/repo). Auto-detected from git remote if not specified."
    )
    upload_parser.add_argument(
        "--output", "-o",
        default="dist",
        help="Directory for packaged artifacts (default: dist)"
    )
    upload_parser.add_argument(
        "--prerelease",
        action="store_true",
        help="Mark as prerelease"
    )
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download database from GitHub")
    download_parser.add_argument(
        "--repo", "-r",
        help="GitHub repo (owner/repo). Auto-detected from git remote if not specified."
    )
    download_parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory (default: data)"
    )
    download_parser.add_argument(
        "--version", "-v",
        help="Specific version to download (default: latest)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Check GitHub CLI
    if not check_gh_cli():
        print("Error: GitHub CLI (gh) is not installed or not authenticated.")
        print("Install: https://cli.github.com/")
        print("Authenticate: gh auth login")
        sys.exit(1)
    
    # Get repo
    repo = args.repo
    if not repo:
        repo_info = get_repo_info()
        if repo_info:
            repo = f"{repo_info[0]}/{repo_info[1]}"
        else:
            print("Error: Could not detect GitHub repo. Please specify with --repo")
            sys.exit(1)
    
    print(f"Repository: {repo}")
    
    if args.command == "upload":
        db_path = Path(args.db)
        output_dir = Path(args.output)
        
        print("\n[1/2] Packaging database...")
        archive_path, metadata = package_database(db_path, output_dir)
        
        print(f"\n[2/2] Uploading to GitHub...")
        release_url = create_github_release(
            repo=repo,
            version=metadata["version"],
            archive_path=archive_path,
            metadata=metadata,
            prerelease=args.prerelease,
        )
        
        print(f"\n{'=' * 60}")
        print("Upload complete!")
        print(f"{'=' * 60}")
        print(f"Release URL: {release_url}")
        print(f"Archive: {archive_path}")
        print(f"Size: {metadata['size_mb']} MB")
        print(f"SHA256: {metadata['checksum_sha256']}")
        
    elif args.command == "download":
        output_dir = Path(args.output)
        
        print("\nDownloading database from GitHub...")
        db_path = download_database(
            repo=repo,
            output_dir=output_dir,
            version=args.version,
        )
        
        print(f"\n{'=' * 60}")
        print("Download complete!")
        print(f"{'=' * 60}")
        print(f"Database path: {db_path}")


if __name__ == "__main__":
    main()
