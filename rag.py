"""
RAG (Retrieval-Augmented Generation) database management for LLMSummarizer.

Handles downloading, verifying, and querying literature databases
for augmenting LLM summaries with relevant research findings.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .domains import DomainConfig


def ensure_rag_database(
    literature_db: str | None,
    auto_download: bool,
    rag_repo: str | None,
    domain_config: DomainConfig,
) -> bool:
    """
    Ensure the RAG database is available, downloading if necessary.
    
    Args:
        literature_db: Path to the database directory
        auto_download: Whether to auto-download if missing
        rag_repo: GitHub repo to download from
        domain_config: Domain configuration for archive pattern
        
    Returns:
        True if database is available, False otherwise
    """
    if not literature_db:
        return False
    
    db_path = Path(literature_db)
    
    # Check if database already exists
    if db_path.exists() and (db_path / "chroma.sqlite3").exists():
        return True
    
    # Database doesn't exist - try to download
    if not auto_download:
        print(f"RAG database not found at {db_path}")
        print("Set rag_auto_download=true or run scripts/build_rag_database.py")
        return False
    
    print(f"RAG database not found at {db_path}. Attempting to download...")
    
    # Try to download from GitHub
    if download_rag_database(db_path, rag_repo, domain_config):
        return True
    
    print("RAG database download failed. Continuing without RAG augmentation.")
    print("To build locally, run: python scripts/build_rag_database.py --email your@email.com")
    return False


def download_rag_database(
    db_path: Path,
    rag_repo: str | None,
    domain_config: DomainConfig,
) -> bool:
    """
    Download RAG database from GitHub releases.
    
    Args:
        db_path: Path where database should be stored
        rag_repo: GitHub repo (owner/repo format)
        domain_config: Domain configuration for archive pattern
        
    Returns:
        True if download successful, False otherwise
    """
    # Determine repo to download from
    repo = rag_repo
    if not repo:
        # Try to detect from git remote
        repo = detect_github_repo()
    
    if not repo:
        print("No GitHub repo configured for RAG database download.")
        print("Set rag_repo parameter (e.g., 'owner/LLMSummarizer')")
        return False
    
    print(f"Checking GitHub releases for {repo}...")
    
    try:
        # Get latest release info via GitHub API
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "LLMSummarizer"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            release_info = json.loads(response.read().decode("utf-8"))
        
        # Find the database archive asset
        assets = release_info.get("assets", [])
        archive_asset = None
        checksum_asset = None
        
        archive_pattern = domain_config.rag_db_archive_pattern
        for asset in assets:
            name = asset.get("name", "")
            if name.endswith(".tar.gz") and archive_pattern in name:
                archive_asset = asset
            elif name.endswith(".sha256") and archive_pattern in name:
                checksum_asset = asset
        
        if not archive_asset:
            print("No RAG database found in latest release.")
            return False
        
        # Download archive
        archive_url = archive_asset["browser_download_url"]
        archive_name = archive_asset["name"]
        archive_size = archive_asset.get("size", 0)
        
        print(f"Downloading {archive_name} ({archive_size / 1024 / 1024:.1f} MB)...")
        
        # Create parent directory
        db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_archive = db_path.parent / archive_name
        
        # Download with progress
        download_with_progress(archive_url, temp_archive, archive_size)
        
        # Verify checksum if available
        if checksum_asset:
            print("Verifying checksum...")
            checksum_url = checksum_asset["browser_download_url"]
            
            req = urllib.request.Request(checksum_url, headers={"User-Agent": "LLMSummarizer"})
            with urllib.request.urlopen(req, timeout=30) as response:
                expected_checksum = response.read().decode("utf-8").split()[0]
            
            actual_checksum = compute_file_checksum(temp_archive)
            
            if actual_checksum != expected_checksum:
                print(f"Checksum mismatch! Expected {expected_checksum[:16]}...")
                temp_archive.unlink()
                return False
            
            print("Checksum verified.")
        
        # Extract archive
        print("Extracting database...")
        with tarfile.open(temp_archive, "r:gz") as tar:
            tar.extractall(db_path.parent)
        
        # Clean up archive
        temp_archive.unlink()
        
        # Verify extraction
        if db_path.exists():
            print(f"RAG database installed at {db_path}")
            return True
        else:
            print("Extraction failed - database directory not found")
            return False
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"No releases found for {repo}")
        else:
            print(f"HTTP error downloading database: {e}")
        return False
    except urllib.error.URLError as e:
        print(f"Network error downloading database: {e}")
        return False
    except Exception as e:
        print(f"Error downloading RAG database: {e}")
        return False


def download_with_progress(url: str, dest_path: Path, total_size: int) -> None:
    """
    Download a file with progress indication.
    
    Args:
        url: URL to download
        dest_path: Destination file path
        total_size: Expected file size in bytes
    """
    req = urllib.request.Request(url, headers={"User-Agent": "LLMSummarizer"})
    
    with urllib.request.urlopen(req, timeout=300) as response:
        with open(dest_path, "wb") as f:
            downloaded = 0
            chunk_size = 8192
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Progress: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="", flush=True)
            
            print()  # Newline after progress


def compute_file_checksum(filepath: Path) -> str:
    """
    Compute SHA256 checksum of a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Hex-encoded SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def detect_github_repo() -> str | None:
    """
    Try to detect GitHub repo from git remote.
    
    Returns:
        Repo in "owner/repo" format, or None if not detected
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        url = result.stdout.strip()
        
        if "github.com" in url:
            url = url.rstrip(".git")
            
            if url.startswith("git@"):
                parts = url.split(":")[-1].split("/")
            else:
                parts = url.split("github.com/")[-1].split("/")
            
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        
        return None
    except Exception:
        return None


def query_literature_database(
    literature_db: str,
    collection_name: str,
    key_terms: list[str],
    max_results_per_term: int = 3,
    max_total_findings: int = 8,
) -> list[dict[str, Any]]:
    """
    Query the literature database for relevant findings.
    
    Args:
        literature_db: Path to ChromaDB database
        collection_name: Name of the findings collection
        key_terms: Search terms for retrieval
        max_results_per_term: Max results per search term
        max_total_findings: Maximum total findings to return
        
    Returns:
        List of finding dictionaries
    """
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Connect to database
        client = chromadb.PersistentClient(
            path=literature_db,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            findings_collection = client.get_collection(collection_name)
        except Exception:
            print(f"Warning: '{collection_name}' collection not found in database")
            return []
        
        # Query for relevant findings
        all_findings = []
        for term in key_terms[:3]:  # Limit queries
            results = findings_collection.query(
                query_texts=[term],
                n_results=max_results_per_term
            )
            
            if results and results["documents"] and results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    finding_entry = {
                        "finding": doc,
                        "type": meta.get("finding_type", "unknown"),
                        "evidence": meta.get("evidence_strength", "unknown"),
                        "citation": meta.get("citation", "Unknown source"),
                    }
                    # Avoid duplicates
                    if finding_entry not in all_findings:
                        all_findings.append(finding_entry)
        
        return all_findings[:max_total_findings]
        
    except ImportError:
        print("Warning: chromadb not installed. RAG query disabled.")
        return []
    except Exception as e:
        print(f"Warning: RAG query failed: {e}")
        return []


def format_literature_context(findings: list[dict[str, Any]]) -> str:
    """
    Format literature findings as markdown context.
    
    Args:
        findings: List of finding dictionaries
        
    Returns:
        Formatted markdown string
    """
    if not findings:
        return ""
    
    literature_context = "\n\n## Related Literature Findings\n"
    literature_context += "The following findings from published research may provide context:\n"
    
    # Group by finding type
    findings_by_type: dict[str, list] = {}
    for f in findings:
        ftype = f["type"]
        if ftype not in findings_by_type:
            findings_by_type[ftype] = []
        findings_by_type[ftype].append(f)
    
    for ftype, type_findings in findings_by_type.items():
        literature_context += f"\n### {ftype.title()} Findings\n"
        for f in type_findings[:3]:
            literature_context += f"- {f['finding']}\n"
            literature_context += f"  *{f['citation']}* (Evidence: {f['evidence']})\n"
    
    return literature_context
