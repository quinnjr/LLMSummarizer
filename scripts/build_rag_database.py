#!/usr/bin/env python3
"""
Literature RAG Database Builder

Scrapes publicly available research from PubMed Central Open Access subset,
extracts key findings using LLM, and builds a ChromaDB vector database for
RAG-enhanced summarization.

Supports multiple research domains:
- Parkinson's disease
- Alzheimer's disease
- Cancer
- Microbiome
- Custom domains

Data Sources:
- PubMed Central Open Access Subset (freely redistributable)
- PubMed Abstracts (freely accessible)

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Optional imports with graceful fallback
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# =============================================================================
# Domain Configurations
# =============================================================================

DOMAIN_CONFIGS = {
    "parkinsons": {
        "name": "Parkinson's Disease",
        "collection_name": "pd_findings",
        "db_path": "data/pd_literature_db",
        "search_queries": [
            # Core PD research
            '"Parkinson disease"[MeSH] AND "biomarker"[tiab]',
            '"Parkinson disease"[MeSH] AND ("gut microbiome"[tiab] OR "microbiota"[tiab])',
            '"Parkinson disease"[MeSH] AND ("alpha-synuclein"[tiab] OR "SNCA"[tiab])',
            # Genetics
            '"Parkinson disease"[MeSH] AND ("LRRK2"[tiab] OR "GBA"[tiab] OR "genetic"[tiab])',
            # Transcriptomics
            '"Parkinson disease"[MeSH] AND ("transcriptom"[tiab] OR "RNA-seq"[tiab] OR "gene expression"[tiab])',
            # Metabolomics
            '"Parkinson disease"[MeSH] AND ("metabolom"[tiab] OR "metabolite"[tiab])',
        ],
        "finding_types": ["biomarker", "mechanism", "clinical", "microbiome", "genetic", "therapeutic"],
    },
    "alzheimers": {
        "name": "Alzheimer's Disease",
        "collection_name": "ad_findings",
        "db_path": "data/ad_literature_db",
        "search_queries": [
            # Core AD research
            '"Alzheimer disease"[MeSH] AND "biomarker"[tiab]',
            '"Alzheimer disease"[MeSH] AND ("amyloid"[tiab] OR "amyloid beta"[tiab])',
            '"Alzheimer disease"[MeSH] AND ("tau protein"[tiab] OR "neurofibrillary"[tiab])',
            # Genetics
            '"Alzheimer disease"[MeSH] AND ("APOE"[tiab] OR "genetic risk"[tiab])',
            # Transcriptomics
            '"Alzheimer disease"[MeSH] AND ("transcriptom"[tiab] OR "RNA-seq"[tiab] OR "gene expression"[tiab])',
            # Neuroinflammation
            '"Alzheimer disease"[MeSH] AND ("neuroinflammation"[tiab] OR "microglia"[tiab])',
        ],
        "finding_types": ["biomarker", "mechanism", "clinical", "genetic", "therapeutic", "imaging"],
    },
    "cancer": {
        "name": "Cancer",
        "collection_name": "cancer_findings",
        "db_path": "data/cancer_literature_db",
        "search_queries": [
            # Multi-omics cancer
            '"Neoplasms"[MeSH] AND "multi-omics"[tiab]',
            '"Neoplasms"[MeSH] AND "biomarker"[tiab] AND "genomics"[tiab]',
            '"Neoplasms"[MeSH] AND ("tumor microenvironment"[tiab] OR "immune infiltration"[tiab])',
            # Transcriptomics
            '"Neoplasms"[MeSH] AND ("transcriptom"[tiab] OR "RNA-seq"[tiab]) AND "biomarker"[tiab]',
            # Mutations
            '"Neoplasms"[MeSH] AND ("driver mutation"[tiab] OR "somatic mutation"[tiab])',
            # Proteomics
            '"Neoplasms"[MeSH] AND "proteomics"[tiab] AND "biomarker"[tiab]',
        ],
        "finding_types": ["biomarker", "mutation", "therapeutic", "diagnostic", "prognostic", "immune"],
    },
    "microbiome": {
        "name": "Microbiome",
        "collection_name": "microbiome_findings",
        "db_path": "data/microbiome_literature_db",
        "search_queries": [
            # General microbiome
            '"gastrointestinal microbiome"[MeSH] AND "biomarker"[tiab]',
            '"gastrointestinal microbiome"[MeSH] AND "dysbiosis"[tiab]',
            '"gastrointestinal microbiome"[MeSH] AND ("metabolomics"[tiab] OR "metabolite"[tiab])',
            # Host interactions
            '"gastrointestinal microbiome"[MeSH] AND ("host-microbiome"[tiab] OR "immune"[tiab])',
            # Multi-omics
            '"gastrointestinal microbiome"[MeSH] AND ("metagenom"[tiab] OR "16S"[tiab])',
            # Disease associations
            '"gastrointestinal microbiome"[MeSH] AND "disease"[tiab] AND "association"[tiab]',
        ],
        "finding_types": ["biomarker", "dysbiosis", "metabolite", "host-interaction", "therapeutic", "mechanism"],
    },
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Paper:
    """Represents a research paper."""
    
    pmid: str
    pmcid: str | None
    title: str
    authors: list[str]
    journal: str
    year: int
    abstract: str
    full_text: str | None = None
    doi: str | None = None
    keywords: list[str] = field(default_factory=list)
    
    @property
    def citation(self) -> str:
        """Generate citation string."""
        author_str = self.authors[0] if self.authors else "Unknown"
        if len(self.authors) > 1:
            author_str += " et al."
        return f"{author_str} ({self.year}). {self.title}. {self.journal}."
    
    @property
    def doc_id(self) -> str:
        """Generate unique document ID."""
        return hashlib.md5(f"{self.pmid}:{self.title}".encode()).hexdigest()[:16]


@dataclass
class ExtractedFinding:
    """A curated finding extracted from a paper."""
    
    paper_id: str
    finding_type: str  # e.g., "biomarker", "mechanism", "clinical", "microbiome"
    finding: str
    evidence_strength: str  # "strong", "moderate", "preliminary"
    entities: list[str]  # genes, taxa, proteins mentioned
    context: str  # surrounding text
    citation: str


# =============================================================================
# PubMed API Client
# =============================================================================

class PubMedClient:
    """Client for PubMed E-utilities API."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    
    def __init__(self, email: str, api_key: str | None = None):
        """
        Initialize PubMed client.
        
        Args:
            email: Required by NCBI for API usage
            api_key: Optional API key for higher rate limits
        """
        self.email = email
        self.api_key = api_key
        self.rate_limit_delay = 0.34 if api_key else 1.0  # requests per second
        self._last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, url: str) -> str:
        """Make HTTP request with rate limiting and error handling."""
        self._rate_limit()
        
        try:
            req = Request(url, headers={"User-Agent": f"PDLitScraper/1.0 ({self.email})"})
            with urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8")
        except (URLError, HTTPError) as e:
            print(f"Request failed: {e}")
            raise
    
    def search(
        self,
        query: str,
        max_results: int = 1000,
        min_date: str | None = None,
        max_date: str | None = None,
    ) -> list[str]:
        """
        Search PubMed for articles.
        
        Args:
            query: Search query (PubMed syntax)
            max_results: Maximum number of results
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
            
        Returns:
            List of PMIDs
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        if min_date:
            params["mindate"] = min_date
            params["datetype"] = "pdat"
        
        if max_date:
            params["maxdate"] = max_date
        
        url = f"{self.BASE_URL}/esearch.fcgi?{urlencode(params)}"
        response = self._make_request(url)
        data = json.loads(response)
        
        return data.get("esearchresult", {}).get("idlist", [])
    
    def fetch_details(self, pmids: list[str]) -> list[Paper]:
        """
        Fetch paper details for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of Paper objects
        """
        if not pmids:
            return []
        
        papers = []
        
        # Process in batches of 200
        batch_size = 200
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "email": self.email,
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            url = f"{self.BASE_URL}/efetch.fcgi?{urlencode(params)}"
            response = self._make_request(url)
            
            papers.extend(self._parse_pubmed_xml(response))
            
            print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} papers...")
        
        return papers
    
    def _parse_pubmed_xml(self, xml_content: str) -> list[Paper]:
        """Parse PubMed XML response into Paper objects."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue
        
        return papers
    
    def _parse_article(self, article: ET.Element) -> Paper | None:
        """Parse a single PubMed article element."""
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None
        
        # PMID
        pmid_elem = medline.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""
        
        # Article info
        article_elem = medline.find(".//Article")
        if article_elem is None:
            return None
        
        # Title
        title_elem = article_elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""
        
        # Abstract
        abstract_parts = []
        for abstract_text in article_elem.findall(".//AbstractText"):
            label = abstract_text.get("Label", "")
            text = abstract_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)
        
        # Authors
        authors = []
        for author in article_elem.findall(".//Author"):
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname is not None:
                name = lastname.text or ""
                if forename is not None and forename.text:
                    name = f"{forename.text} {name}"
                authors.append(name)
        
        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""
        
        # Year
        year_elem = article_elem.find(".//PubDate/Year")
        if year_elem is None:
            year_elem = article_elem.find(".//PubDate/MedlineDate")
        year = 0
        if year_elem is not None and year_elem.text:
            match = re.search(r"(\d{4})", year_elem.text)
            if match:
                year = int(match.group(1))
        
        # DOI
        doi = None
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break
        
        # PMC ID
        pmcid = None
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.get("IdType") == "pmc":
                pmcid = id_elem.text
                break
        
        # Keywords
        keywords = []
        for kw in medline.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)
        
        return Paper(
            pmid=pmid,
            pmcid=pmcid,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            abstract=abstract,
            doi=doi,
            keywords=keywords,
        )
    
    def fetch_pmc_fulltext(self, pmcid: str) -> str | None:
        """
        Fetch full text from PubMed Central Open Access.
        
        Args:
            pmcid: PMC ID (e.g., "PMC1234567")
            
        Returns:
            Full text content or None if not available
        """
        if not pmcid:
            return None
        
        # Normalize PMC ID
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        
        # Try to get the article XML
        url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid[3:]}&metadataPrefix=pmc"
        
        try:
            response = self._make_request(url)
            return self._extract_pmc_text(response)
        except Exception as e:
            print(f"Could not fetch PMC full text for {pmcid}: {e}")
            return None
    
    def _extract_pmc_text(self, xml_content: str) -> str | None:
        """Extract text content from PMC XML."""
        try:
            root = ET.fromstring(xml_content)
            
            # Find the body element
            body = root.find(".//{http://dtd.nlm.nih.gov/ns/archiving/2.0/}body")
            if body is None:
                body = root.find(".//body")
            
            if body is None:
                return None
            
            # Extract all text
            text_parts = []
            for elem in body.iter():
                if elem.text:
                    text_parts.append(elem.text)
                if elem.tail:
                    text_parts.append(elem.tail)
            
            return " ".join(text_parts)
            
        except ET.ParseError:
            return None


# =============================================================================
# Finding Extractor (LLM-based)
# =============================================================================

class FindingExtractor:
    """Extracts curated findings from papers using LLM."""
    
    EXTRACTION_PROMPT = """You are a biomedical research curator specializing in Parkinson's disease.
Analyze the following research paper and extract key findings.

For each finding, provide:
1. finding_type: One of ["biomarker", "mechanism", "microbiome", "genetic", "clinical", "therapeutic", "metabolic"]
2. finding: A concise statement of the finding (1-2 sentences)
3. evidence_strength: One of ["strong", "moderate", "preliminary"] based on study design and results
4. entities: List of specific genes, proteins, taxa, metabolites, or pathways mentioned

Paper Title: {title}
Authors: {authors}
Year: {year}

Abstract:
{abstract}

{fulltext_section}

Return your response as a JSON array of findings. Example:
[
  {{
    "finding_type": "microbiome",
    "finding": "Reduced abundance of Prevotella species was associated with PD severity.",
    "evidence_strength": "moderate",
    "entities": ["Prevotella", "gut microbiome"]
  }}
]

Extract 3-7 key findings from this paper. Focus on findings relevant to:
- Gut microbiome changes in PD
- Biomarkers (genetic, protein, metabolic)
- Disease mechanisms
- Therapeutic targets

JSON findings:"""

    def __init__(self, model: str = "llama3"):
        """
        Initialize finding extractor.
        
        Args:
            model: Ollama model to use for extraction
        """
        self.model = model
    
    def extract_findings(self, paper: Paper) -> list[ExtractedFinding]:
        """
        Extract findings from a paper using LLM.
        
        Args:
            paper: Paper to analyze
            
        Returns:
            List of extracted findings
        """
        if not OLLAMA_AVAILABLE:
            # Fallback: extract from abstract using simple heuristics
            return self._extract_simple(paper)
        
        # Build prompt
        fulltext_section = ""
        if paper.full_text:
            # Include relevant sections (truncated)
            fulltext_section = f"Full Text (excerpt):\n{paper.full_text[:3000]}..."
        
        prompt = self.EXTRACTION_PROMPT.format(
            title=paper.title,
            authors=", ".join(paper.authors[:5]),
            year=paper.year,
            abstract=paper.abstract,
            fulltext_section=fulltext_section,
        )
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 1500}
            )
            
            findings_json = self._parse_json_response(response["response"])
            
            return [
                ExtractedFinding(
                    paper_id=paper.doc_id,
                    finding_type=f.get("finding_type", "other"),
                    finding=f.get("finding", ""),
                    evidence_strength=f.get("evidence_strength", "preliminary"),
                    entities=f.get("entities", []),
                    context=paper.abstract[:500],
                    citation=paper.citation,
                )
                for f in findings_json
                if f.get("finding")
            ]
            
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return self._extract_simple(paper)
    
    def _parse_json_response(self, response: str) -> list[dict]:
        """Parse JSON from LLM response."""
        # Find JSON array in response
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []
    
    def _extract_simple(self, paper: Paper) -> list[ExtractedFinding]:
        """Simple heuristic-based extraction fallback."""
        findings = []
        
        # Keywords that indicate findings
        finding_patterns = [
            (r"(increased|elevated|higher).{0,50}(in PD|Parkinson)", "biomarker"),
            (r"(decreased|reduced|lower).{0,50}(in PD|Parkinson)", "biomarker"),
            (r"(associated with|correlated with).{0,50}(PD|Parkinson)", "biomarker"),
            (r"(gut microbiome|microbiota|bacteria).{0,100}(PD|Parkinson)", "microbiome"),
            (r"(SNCA|LRRK2|PINK1|Parkin|GBA).{0,50}(mutation|variant)", "genetic"),
            (r"(alpha-synuclein|α-synuclein).{0,50}(aggregat|accumulat)", "mechanism"),
        ]
        
        text = f"{paper.title} {paper.abstract}"
        
        for pattern, finding_type in finding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                findings.append(ExtractedFinding(
                    paper_id=paper.doc_id,
                    finding_type=finding_type,
                    finding=f"Study found evidence related to: {matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])}",
                    evidence_strength="preliminary",
                    entities=[],
                    context=paper.abstract[:300],
                    citation=paper.citation,
                ))
        
        return findings[:5]  # Limit to 5 findings


# =============================================================================
# ChromaDB Database Builder
# =============================================================================

class RAGDatabaseBuilder:
    """Builds and manages the ChromaDB RAG database."""
    
    def __init__(self, db_path: str, collection_name: str = "findings", domain_name: str = "research"):
        """
        Initialize database builder.
        
        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name for the findings collection
            domain_name: Human-readable domain name for metadata
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required: pip install chromadb")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.domain_name = domain_name
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collections with domain-specific names
        papers_collection_name = collection_name.replace("_findings", "_papers").replace("findings", "papers")
        if not papers_collection_name.endswith("_papers"):
            papers_collection_name = f"{collection_name}_papers"
        
        self.papers_collection = self.client.get_or_create_collection(
            name=papers_collection_name,
            metadata={"description": f"{domain_name} research papers"}
        )
        
        self.findings_collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Curated findings from {domain_name} research"}
        )
    
    def add_paper(self, paper: Paper) -> None:
        """Add a paper to the database."""
        # Check if already exists
        existing = self.papers_collection.get(ids=[paper.doc_id])
        if existing["ids"]:
            return
        
        self.papers_collection.add(
            ids=[paper.doc_id],
            documents=[f"{paper.title}\n\n{paper.abstract}"],
            metadatas=[{
                "pmid": paper.pmid,
                "pmcid": paper.pmcid or "",
                "title": paper.title,
                "authors": "; ".join(paper.authors[:5]),
                "journal": paper.journal,
                "year": paper.year,
                "doi": paper.doi or "",
                "citation": paper.citation,
            }]
        )
    
    def add_finding(self, finding: ExtractedFinding) -> None:
        """Add a finding to the database."""
        finding_id = hashlib.md5(
            f"{finding.paper_id}:{finding.finding}".encode()
        ).hexdigest()[:16]
        
        # Check if already exists
        existing = self.findings_collection.get(ids=[finding_id])
        if existing["ids"]:
            return
        
        self.findings_collection.add(
            ids=[finding_id],
            documents=[finding.finding],
            metadatas=[{
                "paper_id": finding.paper_id,
                "finding_type": finding.finding_type,
                "evidence_strength": finding.evidence_strength,
                "entities": "; ".join(finding.entities),
                "context": finding.context[:500],
                "citation": finding.citation,
            }]
        )
    
    def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        return {
            "papers": self.papers_collection.count(),
            "findings": self.findings_collection.count(),
        }
    
    def query_findings(
        self,
        query: str,
        n_results: int = 5,
        finding_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Query findings database.
        
        Args:
            query: Search query
            n_results: Number of results to return
            finding_types: Optional filter by finding type
            
        Returns:
            List of matching findings with metadata
        """
        where_filter = None
        if finding_types:
            where_filter = {"finding_type": {"$in": finding_types}}
        
        results = self.findings_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )
        
        findings = []
        for i, doc in enumerate(results["documents"][0]):
            findings.append({
                "finding": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        
        return findings


# =============================================================================
# Main Scraper Pipeline
# =============================================================================

def build_database(
    domain: str,
    output_dir: str | None,
    email: str,
    api_key: str | None = None,
    max_papers: int = 500,
    min_year: int = 2015,
    llm_model: str = "llama3",
    fetch_fulltext: bool = False,
    custom_queries: list[str] | None = None,
) -> None:
    """
    Build a domain-specific RAG database.
    
    Args:
        domain: Research domain (parkinsons, alzheimers, cancer, microbiome)
        output_dir: Directory for ChromaDB database (uses domain default if None)
        email: Email for PubMed API
        api_key: Optional PubMed API key
        max_papers: Maximum number of papers to fetch
        min_year: Minimum publication year
        llm_model: Ollama model for finding extraction
        fetch_fulltext: Whether to fetch full text from PMC
        custom_queries: Custom PubMed queries (overrides domain defaults)
    """
    # Get domain configuration
    if domain not in DOMAIN_CONFIGS:
        print(f"Error: Unknown domain '{domain}'")
        print(f"Available domains: {', '.join(DOMAIN_CONFIGS.keys())}")
        sys.exit(1)
    
    config = DOMAIN_CONFIGS[domain]
    domain_name = config["name"]
    
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = config["db_path"]
    
    # Use custom queries or domain defaults
    search_queries = custom_queries if custom_queries else config["search_queries"]
    
    print("=" * 60)
    print(f"{domain_name} Literature RAG Database Builder")
    print("=" * 60)
    
    # Initialize components
    pubmed = PubMedClient(email=email, api_key=api_key)
    extractor = FindingExtractor(model=llm_model)
    db = RAGDatabaseBuilder(
        output_dir, 
        collection_name=config["collection_name"],
        domain_name=domain_name
    )
    
    all_pmids = set()
    
    # Search for papers
    print(f"\n[1/4] Searching PubMed for {domain_name} research...")
    
    for query in search_queries:
        print(f"  Query: {query[:60]}...")
        pmids = pubmed.search(
            query=query,
            max_results=max_papers // len(search_queries),
            min_date=f"{min_year}/01/01",
        )
        all_pmids.update(pmids)
        print(f"    Found {len(pmids)} papers")
    
    pmids_list = list(all_pmids)[:max_papers]
    print(f"\n  Total unique papers: {len(pmids_list)}")
    
    # Fetch paper details
    print(f"\n[2/4] Fetching paper details...")
    papers = pubmed.fetch_details(pmids_list)
    print(f"  Retrieved {len(papers)} papers with abstracts")
    
    # Optionally fetch full text
    if fetch_fulltext:
        print(f"\n[2b/4] Fetching full text for Open Access papers...")
        fulltext_count = 0
        for i, paper in enumerate(papers):
            if paper.pmcid:
                fulltext = pubmed.fetch_pmc_fulltext(paper.pmcid)
                if fulltext:
                    paper.full_text = fulltext
                    fulltext_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(papers)} papers ({fulltext_count} with full text)")
        
        print(f"  Retrieved full text for {fulltext_count} papers")
    
    # Extract findings
    print(f"\n[3/4] Extracting findings using LLM ({llm_model})...")
    
    all_findings = []
    for i, paper in enumerate(papers):
        findings = extractor.extract_findings(paper)
        all_findings.extend(findings)
        
        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(papers)} papers ({len(all_findings)} findings)")
    
    print(f"  Extracted {len(all_findings)} findings")
    
    # Build database
    print(f"\n[4/4] Building ChromaDB database...")
    
    for paper in papers:
        db.add_paper(paper)
    
    for finding in all_findings:
        db.add_finding(finding)
    
    stats = db.get_stats()
    
    # Save metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "domain": domain,
        "domain_name": domain_name,
        "collection_name": config["collection_name"],
        "papers_count": stats["papers"],
        "findings_count": stats["findings"],
        "min_year": min_year,
        "search_queries": search_queries,
        "llm_model": llm_model,
    }
    
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Database Build Complete!")
    print("=" * 60)
    print(f"  Location: {output_dir}")
    print(f"  Papers: {stats['papers']}")
    print(f"  Findings: {stats['findings']}")
    print(f"  Metadata: {metadata_path}")
    
    # Finding type distribution
    print("\n  Findings by type:")
    finding_types = {}
    for f in all_findings:
        finding_types[f.finding_type] = finding_types.get(f.finding_type, 0) + 1
    for ftype, count in sorted(finding_types.items(), key=lambda x: -x[1]):
        print(f"    {ftype}: {count}")


def main():
    """Main entry point."""
    available_domains = ", ".join(DOMAIN_CONFIGS.keys())
    
    parser = argparse.ArgumentParser(
        description="Build literature RAG database for various research domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available domains: {available_domains}

Examples:
  # Build Parkinson's disease database
  python build_rag_database.py --domain parkinsons --email your@email.com
  
  # Build Alzheimer's disease database
  python build_rag_database.py --domain alzheimers --email your@email.com
  
  # Build cancer database with more papers
  python build_rag_database.py --domain cancer --email your@email.com --max-papers 1000
  
  # With API key for faster downloads
  python build_rag_database.py --domain parkinsons --email your@email.com --api-key YOUR_KEY
  
  # Custom output directory
  python build_rag_database.py --domain parkinsons --email your@email.com --output my_db/
  
  # Include full text from Open Access papers
  python build_rag_database.py --domain parkinsons --email your@email.com --fetch-fulltext
        """
    )
    
    parser.add_argument(
        "--domain", "-d",
        default="parkinsons",
        choices=list(DOMAIN_CONFIGS.keys()),
        help=f"Research domain ({available_domains}). Default: parkinsons"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for ChromaDB database (uses domain default if not specified)"
    )
    parser.add_argument(
        "--email", "-e",
        required=True,
        help="Email address (required by PubMed API)"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="PubMed API key (optional, increases rate limit)"
    )
    parser.add_argument(
        "--max-papers", "-n",
        type=int,
        default=500,
        help="Maximum number of papers to fetch (default: 500)"
    )
    parser.add_argument(
        "--min-year", "-y",
        type=int,
        default=2015,
        help="Minimum publication year (default: 2015)"
    )
    parser.add_argument(
        "--model", "-m",
        default="llama3",
        help="Ollama model for finding extraction (default: llama3)"
    )
    parser.add_argument(
        "--fetch-fulltext", "-f",
        action="store_true",
        help="Fetch full text from PMC Open Access (slower)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not CHROMADB_AVAILABLE:
        print("Error: chromadb is required. Install with: pip install chromadb")
        sys.exit(1)
    
    if not OLLAMA_AVAILABLE:
        print("Warning: ollama not available. Using simple heuristic extraction.")
    
    build_database(
        domain=args.domain,
        output_dir=args.output,
        email=args.email,
        api_key=args.api_key,
        max_papers=args.max_papers,
        min_year=args.min_year,
        llm_model=args.model,
        fetch_fulltext=args.fetch_fulltext,
    )


if __name__ == "__main__":
    main()
