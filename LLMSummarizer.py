"""
LLMSummarizer PluMA Plugin

Local LLM-based summarization of pipeline findings for multi-omics analysis.

This plugin uses locally-hosted large language models to:
1. Generate natural language summaries of statistical results
2. Interpret feature importance rankings in biological context
3. Compare integration method performance
4. Optionally contextualize findings with domain-specific literature (RAG)

Supports configurable research domains (Parkinson's, Alzheimer's, cancer, etc.)
with pre-built configurations and customizable RAG databases.

Uses local models via Ollama for data privacy.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .domains import DomainConfig, DOMAIN_CONFIGS
from .hardware import (
    HardwareInfo,
    MODEL_RECOMMENDATIONS,
    detect_hardware,
    select_best_model,
    get_model_size_estimate,
)
from .ollama_client import (
    OllamaClient,
    ensure_ollama_running,
    list_local_models,
    is_model_available,
    download_model,
)
from .rag import (
    ensure_rag_database,
    query_literature_database,
    format_literature_context,
)


class LLMSummarizer:
    """
    PluMA plugin for LLM-based summarization of pipeline results.
    
    Generates human-readable summaries of multi-omics analysis findings
    using locally-hosted language models, ensuring data privacy.
    
    Supports multiple research domains (Parkinson's, Alzheimer's, cancer, etc.)
    with domain-specific prompts and RAG databases.
    
    Parameters (via input file):
        feature_importance: Path to SHAP feature importance CSV
        cv_results: Path to cross-validation results CSV
        de_results: Path to differential expression results CSV
        cluster_results: Path to clustering results CSV
        model_metrics: Path to model evaluation metrics CSV
        
        domain: Research domain (parkinsons, alzheimers, cancer, microbiome, generic)
                Default: generic. Use 'custom' with custom_domain_* params for custom domains.
        
        model_name: Ollama model identifier (e.g., "llama3", "mistral", "phi3")
        temperature: Sampling temperature (default: 0.3)
        max_tokens: Maximum tokens in response (default: 1024)
        
        use_rag: Enable retrieval-augmented generation (default: true)
        literature_db: Path to literature vector database (overrides domain default)
        rag_collection: Collection name in database (overrides domain default)
        rag_auto_download: Auto-download database from GitHub if missing (default: true)
        rag_repo: GitHub repo for RAG database (auto-detected if not set)
        
        Custom domain parameters (when domain=custom):
            custom_domain_name: Display name for the domain
            custom_domain_expert_role: Expert role for prompts
            custom_domain_research_focus: Research focus description
            custom_domain_feature_suffix: Suffix for feature RAG queries
    
    Prerequisites:
        1. Install Ollama: https://ollama.com
        2. Start the server: ollama serve
        3. Pull a model: ollama pull llama3
    
    Outputs:
        - Natural language summary (text and markdown)
        - Structured findings JSON
        - Key takeaways for clinical/research audiences
    
    Available Domains:
        - parkinsons: Parkinson's disease research
        - alzheimers: Alzheimer's disease research
        - cancer: Cancer genomics
        - microbiome: Microbiome analysis
        - generic: General multi-omics (no domain-specific context)
        - custom: User-defined domain configuration
    """
    
    def __init__(self) -> None:
        """Initialize plugin state."""
        self.parameters: dict[str, str] = {}
        
        # Input data
        self.feature_importance: pd.DataFrame | None = None
        self.cv_results: pd.DataFrame | None = None
        self.de_results: pd.DataFrame | None = None
        self.cluster_results: pd.DataFrame | None = None
        self.model_metrics: pd.DataFrame | None = None
        
        # Results
        self.summary_text: str = ""
        self.structured_findings: dict[str, Any] = {}
        self.key_takeaways: list[str] = []
        
        # LLM client
        self.llm_client: Any = None
        
        # Hardware info (populated during initialization)
        self.hardware_info: HardwareInfo | None = None
        
        # Domain configuration (default to generic)
        self.domain: str = "generic"
        self.domain_config: DomainConfig = DOMAIN_CONFIGS["generic"]
        
        # Default parameters (model_name=None triggers auto-selection)
        self.model_name: str | None = None
        self.temperature: float = 0.3
        self.max_tokens: int = 1024
        self.use_rag: bool = True
        self.literature_db: str | None = None  # Will use domain default if not set
        self.rag_collection: str | None = None  # Will use domain default if not set
        self.rag_repo: str | None = None  # GitHub repo for RAG database (owner/repo)
        self.rag_auto_download: bool = True  # Auto-download RAG database if missing
    
    def input(self, filename: str) -> None:
        """
        Load input data and configuration parameters.
        
        Args:
            filename: Path to parameter file with key-value pairs
        """
        param_path = Path(filename)
        
        # Parse parameter file
        with param_path.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        self.parameters[parts[0]] = parts[1]
        
        # Load feature importance
        if "feature_importance" in self.parameters:
            self.feature_importance = pd.read_csv(
                self.parameters["feature_importance"]
            )
        
        # Load CV results
        if "cv_results" in self.parameters:
            self.cv_results = pd.read_csv(self.parameters["cv_results"])
        
        # Load differential expression results
        if "de_results" in self.parameters:
            self.de_results = pd.read_csv(self.parameters["de_results"])
        
        # Load clustering results
        if "cluster_results" in self.parameters:
            self.cluster_results = pd.read_csv(self.parameters["cluster_results"])
        
        # Load model metrics
        if "model_metrics" in self.parameters:
            self.model_metrics = pd.read_csv(self.parameters["model_metrics"])
        
        # Parse Ollama configuration (model_name is optional - auto-selects if not specified)
        if "model_name" in self.parameters:
            model_value = self.parameters["model_name"].strip()
            # Allow explicit "auto" or empty to trigger auto-selection
            if model_value.lower() not in ("auto", ""):
                self.model_name = model_value
        
        if "temperature" in self.parameters:
            self.temperature = float(self.parameters["temperature"])
        
        if "max_tokens" in self.parameters:
            self.max_tokens = int(self.parameters["max_tokens"])
        
        if "use_rag" in self.parameters:
            self.use_rag = self.parameters["use_rag"].lower() == "true"
        
        if "rag_repo" in self.parameters:
            self.rag_repo = self.parameters["rag_repo"]
        
        if "rag_auto_download" in self.parameters:
            self.rag_auto_download = self.parameters["rag_auto_download"].lower() == "true"
        
        # Parse domain configuration
        self._configure_domain()
        
        # Override domain defaults with explicit parameters (after domain is set)
        if "literature_db" in self.parameters:
            self.literature_db = self.parameters["literature_db"]
        
        if "rag_collection" in self.parameters:
            self.rag_collection = self.parameters["rag_collection"]
    
    def _configure_domain(self) -> None:
        """
        Configure domain-specific settings based on parameters.
        
        Handles built-in domains and custom domain configuration.
        """
        if "domain" in self.parameters:
            self.domain = self.parameters["domain"].lower().strip()
        
        if self.domain == "custom":
            # Build custom domain configuration
            self.domain_config = DomainConfig(
                name="custom",
                display_name=self.parameters.get(
                    "custom_domain_name", "Custom Analysis"
                ),
                description=self.parameters.get(
                    "custom_domain_description", "Custom multi-omics analysis"
                ),
                rag_collection_name=self.parameters.get(
                    "custom_domain_collection", "findings"
                ),
                rag_db_path=self.parameters.get(
                    "custom_domain_db_path", "data/literature_db"
                ),
                rag_db_archive_pattern=self.parameters.get(
                    "custom_domain_archive_pattern", "literature_db"
                ),
                expert_role=self.parameters.get(
                    "custom_domain_expert_role", "bioinformatics expert"
                ),
                research_focus=self.parameters.get(
                    "custom_domain_research_focus", 
                    "identifying significant biological patterns"
                ),
                context_queries=self.parameters.get(
                    "custom_domain_context_queries", ""
                ).split(",") if self.parameters.get("custom_domain_context_queries") else [],
                feature_suffix=self.parameters.get(
                    "custom_domain_feature_suffix", ""
                ),
                summary_title=self.parameters.get(
                    "custom_domain_summary_title", "Multi-Omics Analysis Summary"
                ),
                summary_subtitle=self.parameters.get(
                    "custom_domain_summary_subtitle", "Biomarker Discovery"
                ),
            )
        elif self.domain in DOMAIN_CONFIGS:
            self.domain_config = DOMAIN_CONFIGS[self.domain]
        else:
            print(f"Warning: Unknown domain '{self.domain}', using 'generic'")
            self.domain = "generic"
            self.domain_config = DOMAIN_CONFIGS["generic"]
        
        # Set defaults from domain config if not explicitly specified
        if self.literature_db is None:
            self.literature_db = self.domain_config.rag_db_path
        
        if self.rag_collection is None:
            self.rag_collection = self.domain_config.rag_collection_name
        
        print(f"Domain: {self.domain_config.display_name}")
    
    def run(self) -> None:
        """
        Execute LLM summarization pipeline.
        
        Steps:
        1. Initialize LLM client
        2. Prepare structured context from pipeline results
        3. Generate summary using LLM
        4. Extract key takeaways
        5. Optionally augment with literature context (RAG)
        """
        # Step 1: Initialize LLM client
        self.llm_client = self._initialize_llm()
        
        # Step 2: Prepare context
        context = self._prepare_context()
        
        # Step 3: Generate summary
        self.summary_text = self._generate_summary(context)
        
        # Step 4: Extract structured findings
        self.structured_findings = self._extract_structured_findings(context)
        
        # Step 5: Generate key takeaways
        self.key_takeaways = self._generate_takeaways(context)
        
        # Step 6: RAG augmentation (if enabled)
        if self.use_rag and self.literature_db:
            # Ensure RAG database is available (download if needed)
            if self._ensure_rag_database():
                self.summary_text = self._augment_with_literature(self.summary_text)
    
    def output(self, filename: str) -> None:
        """
        Write output files.
        
        Args:
            filename: Base path for output files
        """
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write text summary
        summary_path = output_path.with_suffix(".txt")
        with summary_path.open("w") as f:
            f.write(self.summary_text)
        
        # Write markdown summary
        md_path = output_path.with_suffix(".md")
        with md_path.open("w") as f:
            f.write(self._format_as_markdown())
        
        # Write structured findings as JSON
        json_path = output_path.with_suffix(".json")
        output_data = {
            "summary": self.summary_text,
            "structured_findings": self.structured_findings,
            "key_takeaways": self.key_takeaways,
            "domain": self.domain,
            "model": self.model_name,
        }
        with json_path.open("w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Summary written to: {summary_path}")
        print(f"Markdown summary: {md_path}")
        print(f"Structured findings: {json_path}")
    
    def _initialize_llm(self) -> OllamaClient:
        """
        Initialize the LLM client with appropriate model.
        
        Handles:
        - Hardware detection for model selection
        - Ollama server management
        - Model downloading if needed
        
        Returns:
            Configured OllamaClient instance
        """
        # Detect hardware capabilities
        self.hardware_info = detect_hardware()
        print(f"Detected hardware: {self.hardware_info}")
        
        # Ensure Ollama server is running
        ensure_ollama_running()
        
        # Get list of locally available models
        local_models = list_local_models()
        
        # Auto-select model if not configured
        if self.model_name is None:
            self.model_name = select_best_model(
                self.hardware_info,
                local_models,
                is_model_available
            )
            print(f"Auto-selected model: {self.model_name}")
        
        # Check if configured model needs to be downloaded
        if not is_model_available(self.model_name, local_models):
            print(f"Model '{self.model_name}' not found locally. Downloading...")
            size_estimate = get_model_size_estimate(self.model_name)
            download_model(self.model_name, size_estimate)
        else:
            print(f"Using locally available model: {self.model_name}")
        
        return OllamaClient(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _prepare_context(self) -> dict[str, Any]:
        """
        Prepare structured context from pipeline results.
        
        Returns:
            Dictionary with summarized results for LLM context
        """
        context = {
            "analysis_type": self.domain_config.description,
            "sections": []
        }
        
        # Feature importance section
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            context["feature_importance"] = {
                "n_features": len(self.feature_importance),
                "top_features": top_features.to_dict(orient="records"),
                "modality_breakdown": self._summarize_modality_importance()
            }
            context["sections"].append("feature_importance")
        
        # Model performance section
        if self.cv_results is not None or self.model_metrics is not None:
            context["model_performance"] = self._summarize_model_performance()
            context["sections"].append("model_performance")
        
        # Differential expression section
        if self.de_results is not None:
            sig_genes = self.de_results[self.de_results["padj"] < 0.05]
            context["differential_expression"] = {
                "total_genes": len(self.de_results),
                "significant_genes": len(sig_genes),
                "top_upregulated": sig_genes[sig_genes["log2FoldChange"] > 0].head(5).to_dict(orient="records"),
                "top_downregulated": sig_genes[sig_genes["log2FoldChange"] < 0].head(5).to_dict(orient="records")
            }
            context["sections"].append("differential_expression")
        
        # Clustering section
        if self.cluster_results is not None:
            context["clustering"] = {
                "n_samples": len(self.cluster_results),
                "n_clusters": self.cluster_results["cluster"].nunique(),
                "cluster_sizes": self.cluster_results["cluster"].value_counts().to_dict()
            }
            context["sections"].append("clustering")
        
        return context
    
    def _summarize_modality_importance(self) -> dict[str, float]:
        """
        Summarize feature importance by modality.
        
        Returns:
            Dictionary of modality -> total importance
        """
        if self.feature_importance is None or "modality" not in self.feature_importance.columns:
            return {}
        
        modality_sums = self.feature_importance.groupby("modality")["mean_abs_shap"].sum()
        total = modality_sums.sum()
        
        return {
            mod: float(val / total * 100) 
            for mod, val in modality_sums.items()
        }
    
    def _summarize_model_performance(self) -> dict[str, Any]:
        """
        Summarize model performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = {}
        
        if self.cv_results is not None:
            # Extract CV metrics
            for col in self.cv_results.columns:
                if col.startswith("test_"):
                    metric_name = col.replace("test_", "")
                    values = self.cv_results[col].values
                    metrics[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values))
                    }
        
        if self.model_metrics is not None:
            # Add any additional metrics
            for col in self.model_metrics.columns:
                if col not in metrics:
                    metrics[col] = float(self.model_metrics[col].iloc[0])
        
        return metrics
    
    def _generate_summary(self, context: dict[str, Any]) -> str:
        """
        Generate natural language summary using LLM.
        
        Args:
            context: Structured context dictionary
            
        Returns:
            Generated summary text
        """
        prompt = self._build_summary_prompt(context)
        response = self.llm_client.generate(prompt)
        return response
    
    def _build_summary_prompt(self, context: dict[str, Any]) -> str:
        """
        Build prompt for summary generation.
        
        Args:
            context: Structured context dictionary
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a {self.domain_config.expert_role} summarizing multi-omics analysis results 
focused on {self.domain_config.research_focus}. Generate a clear, scientifically accurate summary 
of the following analysis results. Focus on:
1. Key findings and their biological significance
2. Most important features from each data modality
3. Model performance and reliability
4. Potential clinical or research implications

Be precise with statistics but explain them in accessible terms.

Research Context: {self.domain_config.description}

ANALYSIS RESULTS:
"""
        
        # Add feature importance
        if "feature_importance" in context:
            fi = context["feature_importance"]
            prompt += f"\n## Feature Importance\n"
            prompt += f"Total features analyzed: {fi['n_features']}\n"
            prompt += f"\nTop 10 most important features:\n"
            for feat in fi["top_features"]:
                prompt += f"- {feat.get('feature', 'Unknown')}: importance = {feat.get('mean_abs_shap', 0):.4f} (modality: {feat.get('modality', 'unknown')})\n"
            
            prompt += f"\nModality contributions:\n"
            for mod, pct in fi.get("modality_breakdown", {}).items():
                prompt += f"- {mod}: {pct:.1f}%\n"
        
        # Add model performance
        if "model_performance" in context:
            perf = context["model_performance"]
            prompt += f"\n## Model Performance\n"
            for metric, values in perf.items():
                if isinstance(values, dict):
                    prompt += f"- {metric}: {values['mean']:.3f} (+/- {values['std']:.3f})\n"
                else:
                    prompt += f"- {metric}: {values:.3f}\n"
        
        # Add differential expression
        if "differential_expression" in context:
            de = context["differential_expression"]
            prompt += f"\n## Differential Expression\n"
            prompt += f"Total genes analyzed: {de['total_genes']}\n"
            prompt += f"Significantly differentially expressed (padj < 0.05): {de['significant_genes']}\n"
        
        # Add clustering
        if "clustering" in context:
            cl = context["clustering"]
            prompt += f"\n## Clustering Results\n"
            prompt += f"Samples: {cl['n_samples']}, Clusters identified: {cl['n_clusters']}\n"
            prompt += f"Cluster sizes: {cl['cluster_sizes']}\n"
        
        prompt += "\n\nPlease provide a comprehensive summary of these findings:"
        
        return prompt
    
    def _extract_structured_findings(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Extract structured findings from context.
        
        Args:
            context: Analysis context
            
        Returns:
            Structured findings dictionary
        """
        findings = {
            "analysis_type": context.get("analysis_type", "Unknown"),
            "data_overview": {},
            "key_results": {},
            "top_biomarkers": [],
            "model_reliability": {}
        }
        
        # Extract feature importance findings
        if "feature_importance" in context:
            fi = context["feature_importance"]
            findings["data_overview"]["n_features"] = fi["n_features"]
            findings["data_overview"]["modality_breakdown"] = fi.get("modality_breakdown", {})
            
            # Top biomarkers
            for feat in fi["top_features"][:5]:
                findings["top_biomarkers"].append({
                    "name": feat.get("feature", "Unknown"),
                    "modality": feat.get("modality", "unknown"),
                    "importance": feat.get("mean_abs_shap", 0)
                })
        
        # Extract model performance
        if "model_performance" in context:
            perf = context["model_performance"]
            findings["model_reliability"] = perf
            
            # Key metrics
            for metric in ["accuracy", "roc_auc", "f1"]:
                if metric in perf:
                    val = perf[metric]
                    if isinstance(val, dict):
                        findings["key_results"][metric] = val["mean"]
                    else:
                        findings["key_results"][metric] = val
        
        # Extract DE findings
        if "differential_expression" in context:
            de = context["differential_expression"]
            findings["key_results"]["significant_genes"] = de["significant_genes"]
            findings["key_results"]["total_genes"] = de["total_genes"]
        
        # Extract clustering
        if "clustering" in context:
            cl = context["clustering"]
            findings["key_results"]["n_clusters"] = cl["n_clusters"]
            findings["key_results"]["cluster_sizes"] = cl["cluster_sizes"]
        
        return findings
    
    def _generate_takeaways(self, context: dict[str, Any]) -> list[str]:
        """
        Generate key takeaways from the analysis.
        
        Args:
            context: Analysis context
            
        Returns:
            List of key takeaway strings
        """
        prompt = self._build_takeaways_prompt(context)
        response = self.llm_client.generate(prompt)
        
        # Parse response into list
        takeaways = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering and bullet points
            if line:
                line = line.lstrip("0123456789.-) ")
                if line:
                    takeaways.append(line)
        
        return takeaways[:5]  # Limit to 5 takeaways
    
    def _build_takeaways_prompt(self, context: dict[str, Any]) -> str:
        """
        Build prompt for takeaways generation.
        
        Args:
            context: Analysis context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Based on the following {self.domain_config.description} results, 
provide exactly 5 key takeaways for researchers. Be specific and actionable.

"""
        
        prompt += f"Analysis included: {', '.join(context.get('sections', []))}\n"
        
        if "model_performance" in context:
            perf = context["model_performance"]
            acc = perf.get("accuracy", {}).get("mean", 0)
            prompt += f"Model accuracy: {acc:.1%}\n"
        
        if "feature_importance" in context:
            fi = context["feature_importance"]
            prompt += f"Top feature: {fi['top_features'][0].get('feature', 'Unknown')}\n"
        
        prompt += "\nProvide 5 key takeaways (one per line):"
        
        return prompt
    
    def _ensure_rag_database(self) -> bool:
        """
        Ensure the RAG database is available, downloading if necessary.
        
        Returns:
            True if database is available, False otherwise
        """
        return ensure_rag_database(
            literature_db=self.literature_db,
            auto_download=self.rag_auto_download,
            rag_repo=self.rag_repo,
            domain_config=self.domain_config,
        )
    
    def _augment_with_literature(self, summary: str) -> str:
        """
        Augment summary with relevant literature findings using RAG.
        
        Args:
            summary: Generated summary text
            
        Returns:
            Summary augmented with literature references
        """
        if not self.literature_db:
            return summary
        
        # Extract key terms for retrieval
        key_terms = self._extract_key_terms(summary)
        
        # Query database
        collection_name = self.rag_collection or self.domain_config.rag_collection_name
        findings = query_literature_database(
            literature_db=self.literature_db,
            collection_name=collection_name,
            key_terms=key_terms,
        )
        
        # Format and append to summary
        if findings:
            literature_context = format_literature_context(findings)
            summary += literature_context
        
        return summary
    
    def _extract_key_terms(self, text: str) -> list[str]:
        """
        Extract key terms from text and analysis results for RAG retrieval.
        
        Args:
            text: Input text (summary)
            
        Returns:
            List of key terms/queries for database search
        """
        key_terms = []
        feature_suffix = self.domain_config.feature_suffix
        
        # Extract from feature importance if available
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(5)
            for _, row in top_features.iterrows():
                feature_name = row.get("feature", "")
                # Remove modality prefix (MG_, TX_)
                clean_name = re.sub(r'^(MG_|TX_|PT_|MT_)', '', str(feature_name))
                if clean_name:
                    if feature_suffix:
                        key_terms.append(f"{clean_name} {feature_suffix}")
                    else:
                        key_terms.append(clean_name)
        
        # Look for gene/protein names in text (typically uppercase)
        gene_pattern = r'\b([A-Z][A-Z0-9]{2,}[a-z]*)\b'
        genes = re.findall(gene_pattern, text)
        # Filter common non-gene words
        skip_words = {"THE", "AND", "FOR", "WITH", "FROM", "THAT", "THIS", "ARE", "WAS", "RNA", "DNA"}
        genes = [g for g in genes if g not in skip_words]
        for gene in genes[:5]:
            if feature_suffix:
                key_terms.append(f"{gene} {feature_suffix}")
            else:
                key_terms.append(gene)
        
        # Look for bacterial taxa names (genus names typically capitalized)
        taxa_pattern = r'\b([A-Z][a-z]+(?:aceae|ales|ota|coccus|bacillus|bacterium)?)\b'
        taxa = re.findall(taxa_pattern, text)
        for taxon in taxa[:3]:
            if len(taxon) > 4:  # Filter short matches
                key_terms.append(f"{taxon} microbiome")
        
        # Add domain-specific context queries
        if self.domain_config.context_queries:
            key_terms.extend(self.domain_config.context_queries)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms[:10]  # Limit to 10 queries
    
    def _format_as_markdown(self) -> str:
        """
        Format summary as markdown document.
        
        Returns:
            Markdown-formatted summary
        """
        md = f"# {self.domain_config.summary_title}\n\n"
        md += f"## {self.domain_config.summary_subtitle}\n\n"
        
        md += self.summary_text
        
        md += "\n\n---\n\n"
        md += "## Key Takeaways\n\n"
        for i, takeaway in enumerate(self.key_takeaways, 1):
            md += f"{i}. {takeaway}\n"
        
        if self.structured_findings.get("top_biomarkers"):
            md += "\n## Top Biomarkers\n\n"
            md += "| Rank | Biomarker | Modality | Importance |\n"
            md += "|------|-----------|----------|------------|\n"
            for i, bio in enumerate(self.structured_findings["top_biomarkers"], 1):
                md += f"| {i} | {bio['name']} | {bio['modality']} | {bio['importance']:.4f} |\n"
        
        md += "\n---\n*Generated using local LLM summarization*\n"
        
        return md
