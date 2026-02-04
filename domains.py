"""
Domain configurations for LLMSummarizer.

Defines research domain-specific settings for prompts, RAG databases,
and output formatting.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainConfig:
    """Configuration for a specific research domain."""
    
    # Domain identification
    name: str
    display_name: str
    description: str
    
    # RAG database settings
    rag_collection_name: str
    rag_db_path: str
    rag_db_archive_pattern: str  # Pattern to match in GitHub releases
    
    # Prompt customization
    expert_role: str  # e.g., "neurology expert", "oncologist"
    research_focus: str  # e.g., "biomarker discovery", "treatment response"
    
    # Key term extraction patterns
    context_queries: list[str] = field(default_factory=list)
    feature_suffix: str = ""  # Appended to feature names in RAG queries
    
    # Output customization
    summary_title: str = "Multi-Omics Analysis Summary"
    summary_subtitle: str = "Biomarker Discovery"


# Pre-defined domain configurations
DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    "parkinsons": DomainConfig(
        name="parkinsons",
        display_name="Parkinson's Disease",
        description="Parkinson's disease multi-omics analysis",
        rag_collection_name="pd_findings",
        rag_db_path="data/pd_literature_db",
        rag_db_archive_pattern="pd_literature_db",
        expert_role="bioinformatics expert specializing in Parkinson's disease research",
        research_focus="identifying biomarkers and understanding disease mechanisms",
        context_queries=[
            "gut microbiome Parkinson's disease biomarker",
            "alpha-synuclein gut-brain axis",
            "Parkinson's disease transcriptomics gene expression",
        ],
        feature_suffix="Parkinson's disease",
        summary_title="Multi-Omics Analysis Summary",
        summary_subtitle="Parkinson's Disease Biomarker Discovery",
    ),
    "alzheimers": DomainConfig(
        name="alzheimers",
        display_name="Alzheimer's Disease",
        description="Alzheimer's disease multi-omics analysis",
        rag_collection_name="ad_findings",
        rag_db_path="data/ad_literature_db",
        rag_db_archive_pattern="ad_literature_db",
        expert_role="bioinformatics expert specializing in Alzheimer's disease research",
        research_focus="identifying biomarkers and understanding neurodegeneration",
        context_queries=[
            "amyloid beta Alzheimer's disease biomarker",
            "tau protein neurodegeneration",
            "Alzheimer's disease transcriptomics gene expression",
        ],
        feature_suffix="Alzheimer's disease",
        summary_title="Multi-Omics Analysis Summary",
        summary_subtitle="Alzheimer's Disease Biomarker Discovery",
    ),
    "cancer": DomainConfig(
        name="cancer",
        display_name="Cancer",
        description="Cancer multi-omics analysis",
        rag_collection_name="cancer_findings",
        rag_db_path="data/cancer_literature_db",
        rag_db_archive_pattern="cancer_literature_db",
        expert_role="bioinformatics expert specializing in cancer genomics",
        research_focus="identifying tumor biomarkers and therapeutic targets",
        context_queries=[
            "cancer driver mutation biomarker",
            "tumor microenvironment immunotherapy",
            "cancer transcriptomics gene expression signature",
        ],
        feature_suffix="cancer",
        summary_title="Multi-Omics Analysis Summary",
        summary_subtitle="Cancer Biomarker Discovery",
    ),
    "microbiome": DomainConfig(
        name="microbiome",
        display_name="Microbiome",
        description="Microbiome multi-omics analysis",
        rag_collection_name="microbiome_findings",
        rag_db_path="data/microbiome_literature_db",
        rag_db_archive_pattern="microbiome_literature_db",
        expert_role="bioinformatics expert specializing in microbiome research",
        research_focus="understanding microbial community dynamics and host interactions",
        context_queries=[
            "gut microbiome dysbiosis biomarker",
            "microbiome metabolomics short-chain fatty acids",
            "host-microbiome interaction immune response",
        ],
        feature_suffix="microbiome",
        summary_title="Multi-Omics Analysis Summary",
        summary_subtitle="Microbiome Analysis",
    ),
    "generic": DomainConfig(
        name="generic",
        display_name="Generic Analysis",
        description="Multi-omics analysis",
        rag_collection_name="findings",
        rag_db_path="data/literature_db",
        rag_db_archive_pattern="literature_db",
        expert_role="bioinformatics expert",
        research_focus="identifying significant biological patterns",
        context_queries=[],
        feature_suffix="",
        summary_title="Multi-Omics Analysis Summary",
        summary_subtitle="Biomarker Discovery",
    ),
}


def get_domain_config(domain: str) -> DomainConfig:
    """
    Get domain configuration by name.
    
    Args:
        domain: Domain name (parkinsons, alzheimers, cancer, microbiome, generic)
        
    Returns:
        DomainConfig for the specified domain
        
    Raises:
        ValueError: If domain is not found
    """
    if domain not in DOMAIN_CONFIGS:
        available = ", ".join(DOMAIN_CONFIGS.keys())
        raise ValueError(f"Unknown domain '{domain}'. Available: {available}")
    return DOMAIN_CONFIGS[domain]


def list_domains() -> list[str]:
    """Return list of available domain names."""
    return list(DOMAIN_CONFIGS.keys())
