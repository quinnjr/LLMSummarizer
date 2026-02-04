#!/usr/bin/env python3
"""
Query the PD Literature RAG Database

Simple CLI tool to test and explore the RAG database.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

import argparse
import json
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Error: chromadb is required. Install with: pip install chromadb")
    exit(1)


def query_database(
    db_path: str,
    query: str,
    collection: str = "findings",
    n_results: int = 5,
    finding_type: str | None = None,
) -> None:
    """
    Query the RAG database.
    
    Args:
        db_path: Path to ChromaDB database
        query: Search query
        collection: Collection to query ("findings" or "papers")
        n_results: Number of results
        finding_type: Optional finding type filter
    """
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection_name = f"pd_{collection}"
    
    try:
        coll = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Could not open collection '{collection_name}': {e}")
        return
    
    # Build query
    where_filter = None
    if finding_type and collection == "findings":
        where_filter = {"finding_type": finding_type}
    
    results = coll.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )
    
    # Display results
    print(f"\nQuery: \"{query}\"")
    print(f"Collection: {collection_name} ({coll.count()} total items)")
    print("=" * 60)
    
    if not results["documents"][0]:
        print("No results found.")
        return
    
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0] if results["distances"] else [None] * len(results["documents"][0])
    )):
        print(f"\n[{i + 1}] Score: {1 - dist:.3f}" if dist else f"\n[{i + 1}]")
        
        if collection == "findings":
            print(f"    Type: {meta.get('finding_type', 'unknown')}")
            print(f"    Evidence: {meta.get('evidence_strength', 'unknown')}")
            print(f"    Finding: {doc}")
            print(f"    Citation: {meta.get('citation', 'N/A')}")
            if meta.get("entities"):
                print(f"    Entities: {meta['entities']}")
        else:
            print(f"    Title: {meta.get('title', 'Unknown')[:80]}")
            print(f"    Authors: {meta.get('authors', 'Unknown')[:60]}")
            print(f"    Year: {meta.get('year', 'Unknown')}")
            print(f"    Journal: {meta.get('journal', 'Unknown')}")
            print(f"    Abstract: {doc[:200]}...")


def show_stats(db_path: str) -> None:
    """Show database statistics."""
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    print("\nDatabase Statistics")
    print("=" * 40)
    
    # Load metadata if available
    metadata_path = Path(db_path) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Created: {metadata.get('created', 'Unknown')}")
        print(f"Min Year: {metadata.get('min_year', 'Unknown')}")
        print(f"LLM Model: {metadata.get('llm_model', 'Unknown')}")
        print()
    
    for name in ["pd_papers", "pd_findings"]:
        try:
            coll = client.get_collection(name)
            print(f"{name}: {coll.count()} items")
        except Exception:
            print(f"{name}: Not found")
    
    # Finding type distribution
    try:
        findings = client.get_collection("pd_findings")
        all_findings = findings.get(include=["metadatas"])
        
        type_counts = {}
        for meta in all_findings["metadatas"]:
            ftype = meta.get("finding_type", "unknown")
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        
        print("\nFindings by Type:")
        for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {ftype}: {count}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Query the PD Literature RAG Database"
    )
    
    parser.add_argument(
        "--db", "-d",
        default="data/pd_literature_db",
        help="Path to ChromaDB database"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the database")
    query_parser.add_argument("text", help="Search query text")
    query_parser.add_argument(
        "--collection", "-c",
        choices=["findings", "papers"],
        default="findings",
        help="Collection to query"
    )
    query_parser.add_argument(
        "--n", "-n",
        type=int,
        default=5,
        help="Number of results"
    )
    query_parser.add_argument(
        "--type", "-t",
        help="Filter by finding type"
    )
    
    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")
    
    args = parser.parse_args()
    
    if args.command == "query":
        query_database(
            db_path=args.db,
            query=args.text,
            collection=args.collection,
            n_results=args.n,
            finding_type=args.type,
        )
    elif args.command == "stats":
        show_stats(args.db)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
