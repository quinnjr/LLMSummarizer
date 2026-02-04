"""
LLMSummarizer PluMA Plugin

Local LLM-based summarization of pipeline findings for Parkinson's disease
multi-omics analysis.

This plugin uses locally-hosted large language models to:
1. Generate natural language summaries of statistical results
2. Interpret feature importance rankings in biological context
3. Compare integration method performance
4. Optionally contextualize findings with PD literature (RAG)

Uses local models via Ollama or llama-cpp-python for data privacy.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HardwareInfo:
    """System hardware information for model selection."""
    
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    
    def __str__(self) -> str:
        gpu_str = f"{self.gpu_name} ({self.gpu_vram_gb:.1f}GB)" if self.gpu_available else "None"
        return (
            f"CPU: {self.cpu_cores} cores | "
            f"RAM: {self.ram_gb:.1f}GB | "
            f"GPU: {gpu_str}"
        )


@dataclass
class ModelRecommendation:
    """Recommended model based on hardware capabilities."""
    
    name: str
    size_gb: float
    description: str
    min_ram_gb: float
    gpu_recommended: bool


# Model recommendations ordered by capability (best first)
MODEL_RECOMMENDATIONS: list[ModelRecommendation] = [
    # High-end GPU models (requires good GPU with VRAM)
    ModelRecommendation(
        name="llama3.1:70b",
        size_gb=40.0,
        description="Llama 3.1 70B - Best quality, requires high-end GPU",
        min_ram_gb=48.0,
        gpu_recommended=True
    ),
    ModelRecommendation(
        name="llama3.1:8b",
        size_gb=4.7,
        description="Llama 3.1 8B - Excellent balance of quality and speed",
        min_ram_gb=8.0,
        gpu_recommended=True
    ),
    ModelRecommendation(
        name="llama3:8b",
        size_gb=4.7,
        description="Llama 3 8B - Strong general-purpose model",
        min_ram_gb=8.0,
        gpu_recommended=True
    ),
    ModelRecommendation(
        name="mistral:7b",
        size_gb=4.1,
        description="Mistral 7B - Fast and capable",
        min_ram_gb=8.0,
        gpu_recommended=True
    ),
    # Mid-range models (GPU helpful but not required)
    ModelRecommendation(
        name="phi3:medium",
        size_gb=7.9,
        description="Phi-3 Medium - Microsoft's efficient model",
        min_ram_gb=12.0,
        gpu_recommended=False
    ),
    ModelRecommendation(
        name="phi3:mini",
        size_gb=2.2,
        description="Phi-3 Mini - Compact but capable",
        min_ram_gb=4.0,
        gpu_recommended=False
    ),
    ModelRecommendation(
        name="gemma2:2b",
        size_gb=1.6,
        description="Gemma 2 2B - Google's efficient small model",
        min_ram_gb=4.0,
        gpu_recommended=False
    ),
    # Lightweight models (CPU-friendly)
    ModelRecommendation(
        name="tinyllama:1.1b",
        size_gb=0.6,
        description="TinyLlama 1.1B - Minimal resources, basic capability",
        min_ram_gb=2.0,
        gpu_recommended=False
    ),
]


class LLMSummarizer:
    """
    PluMA plugin for LLM-based summarization of pipeline results.
    
    Generates human-readable summaries of multi-omics analysis findings
    using locally-hosted language models, ensuring data privacy.
    
    Parameters (via input file):
        feature_importance: Path to SHAP feature importance CSV
        cv_results: Path to cross-validation results CSV
        de_results: Path to differential expression results CSV
        cluster_results: Path to clustering results CSV
        model_metrics: Path to model evaluation metrics CSV
        
        model_name: Ollama model identifier (e.g., "llama3", "mistral", "phi3")
        temperature: Sampling temperature (default: 0.3)
        max_tokens: Maximum tokens in response (default: 1024)
        
        use_rag: Enable retrieval-augmented generation (default: true)
        literature_db: Path to literature vector database (for RAG)
        rag_auto_download: Auto-download database from GitHub if missing (default: true)
        rag_repo: GitHub repo for RAG database (auto-detected if not set)
    
    Prerequisites:
        1. Install Ollama: https://ollama.com
        2. Start the server: ollama serve
        3. Pull a model: ollama pull llama3
    
    Outputs:
        - Natural language summary (text and markdown)
        - Structured findings JSON
        - Key takeaways for clinical/research audiences
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
        
        # Default parameters (model_name=None triggers auto-selection)
        self.model_name: str | None = None
        self.temperature: float = 0.3
        self.max_tokens: int = 1024
        self.use_rag: bool = True
        self.literature_db: str | None = "data/pd_literature_db"
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
        
        if "literature_db" in self.parameters:
            self.literature_db = self.parameters["literature_db"]
        
        if "rag_repo" in self.parameters:
            self.rag_repo = self.parameters["rag_repo"]
        
        if "rag_auto_download" in self.parameters:
            self.rag_auto_download = self.parameters["rag_auto_download"].lower() == "true"
    
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
        Write results to output files.
        
        Args:
            filename: Base path for output files
        """
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write plain text summary
        with output_path.with_suffix(".summary.txt").open("w") as f:
            f.write(self.summary_text)
        
        # Write markdown summary
        markdown_summary = self._format_as_markdown()
        with output_path.with_suffix(".summary.md").open("w") as f:
            f.write(markdown_summary)
        
        # Write structured findings as JSON
        with output_path.with_suffix(".findings.json").open("w") as f:
            json.dump(self.structured_findings, f, indent=2, default=str)
        
        # Write key takeaways
        with output_path.with_suffix(".takeaways.txt").open("w") as f:
            f.write("Key Takeaways\n")
            f.write("=" * 40 + "\n\n")
            for i, takeaway in enumerate(self.key_takeaways, 1):
                f.write(f"{i}. {takeaway}\n\n")
    
    def _initialize_llm(self) -> Any:
        """
        Initialize Ollama LLM client.
        
        Automatically:
        - Detects system hardware (CPU, RAM, GPU)
        - Starts the Ollama server if not running
        - Selects an appropriate model based on hardware if not configured
        - Downloads the model if not available locally
        
        Returns:
            Ollama client instance
        """
        import ollama
        
        # Detect hardware capabilities
        self.hardware_info = self._detect_hardware()
        print(f"Detected hardware: {self.hardware_info}")
        
        # Ensure Ollama server is running
        if not self._is_ollama_running():
            print("Ollama server not running. Starting...")
            self._start_ollama_server()
        
        # Get list of locally available models
        available_models = ollama.list()
        local_models = self._parse_local_models(available_models)
        
        # Auto-select model if not configured
        if self.model_name is None:
            self.model_name = self._select_best_model(local_models)
            print(f"Auto-selected model: {self.model_name}")
        
        # Check if configured model needs to be downloaded
        if not self._is_model_available(self.model_name, local_models):
            print(f"Model '{self.model_name}' not found locally. Downloading...")
            self._download_model(self.model_name)
        else:
            print(f"Using locally available model: {self.model_name}")
        
        return OllamaClient(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _detect_hardware(self) -> HardwareInfo:
        """
        Detect system hardware capabilities.
        
        Returns:
            HardwareInfo with CPU, RAM, and GPU details
        """
        # CPU cores
        cpu_cores = os.cpu_count() or 1
        
        # RAM
        ram_gb = self._get_system_ram()
        
        # GPU detection
        gpu_available, gpu_name, gpu_vram = self._detect_gpu()
        
        return HardwareInfo(
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram
        )
    
    def _get_system_ram(self) -> float:
        """
        Get total system RAM in GB.
        
        Returns:
            RAM in gigabytes
        """
        try:
            # Try psutil first (most accurate)
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
        
        # Platform-specific fallbacks
        system = platform.system()
        
        if system == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # Value is in kB
                            kb = int(line.split()[1])
                            return kb / (1024**2)
            except (IOError, ValueError):
                pass
        
        elif system == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return int(result.stdout.strip()) / (1024**3)
            except (subprocess.SubprocessError, ValueError):
                pass
        
        elif system == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "totalphysicalmemory"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) >= 2:
                        return int(lines[1].strip()) / (1024**3)
            except (subprocess.SubprocessError, ValueError):
                pass
        
        # Conservative fallback
        return 8.0
    
    def _detect_gpu(self) -> tuple[bool, str | None, float | None]:
        """
        Detect GPU availability and specifications.
        
        Returns:
            Tuple of (gpu_available, gpu_name, vram_gb)
        """
        # Try NVIDIA GPU first (most common for ML)
        nvidia_result = self._detect_nvidia_gpu()
        if nvidia_result[0]:
            return nvidia_result
        
        # Try AMD GPU (ROCm)
        amd_result = self._detect_amd_gpu()
        if amd_result[0]:
            return amd_result
        
        # Try Apple Silicon (Metal)
        apple_result = self._detect_apple_gpu()
        if apple_result[0]:
            return apple_result
        
        return False, None, None
    
    def _detect_nvidia_gpu(self) -> tuple[bool, str | None, float | None]:
        """
        Detect NVIDIA GPU using nvidia-smi.
        
        Returns:
            Tuple of (available, name, vram_gb)
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse first GPU (may have multiple)
                line = result.stdout.strip().split("\n")[0]
                parts = line.split(", ")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    vram_mb = float(parts[1].strip())
                    vram_gb = vram_mb / 1024
                    return True, name, vram_gb
                    
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass
        
        return False, None, None
    
    def _detect_amd_gpu(self) -> tuple[bool, str | None, float | None]:
        """
        Detect AMD GPU using rocm-smi.
        
        Returns:
            Tuple of (available, name, vram_gb)
        """
        try:
            # Check for ROCm installation
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Get GPU name
                name = "AMD GPU"
                for line in result.stdout.split("\n"):
                    if "GPU" in line and ":" in line:
                        name = line.split(":")[-1].strip()
                        break
                
                # Try to get VRAM
                vram_result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True, text=True, timeout=10
                )
                
                vram_gb = None
                if vram_result.returncode == 0:
                    for line in vram_result.stdout.split("\n"):
                        if "Total" in line:
                            # Parse VRAM (usually in bytes or MB)
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.isdigit():
                                    val = int(part)
                                    # Heuristic: if > 1000, probably MB
                                    vram_gb = val / 1024 if val > 1000 else val
                                    break
                
                return True, name, vram_gb
                
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass
        
        return False, None, None
    
    def _detect_apple_gpu(self) -> tuple[bool, str | None, float | None]:
        """
        Detect Apple Silicon GPU (unified memory).
        
        Returns:
            Tuple of (available, name, vram_gb)
        """
        if platform.system() != "Darwin":
            return False, None, None
        
        try:
            # Check for Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                cpu_brand = result.stdout.strip()
                
                # Apple Silicon has unified memory (GPU shares RAM)
                if "Apple" in cpu_brand:
                    # Get chip name
                    chip_result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType"],
                        capture_output=True, text=True, timeout=10
                    )
                    
                    chip_name = "Apple Silicon"
                    if chip_result.returncode == 0:
                        for line in chip_result.stdout.split("\n"):
                            if "Chip:" in line:
                                chip_name = line.split(":")[-1].strip()
                                break
                    
                    # Unified memory - GPU can use most of system RAM
                    # Estimate ~75% available for GPU
                    ram_gb = self._get_system_ram()
                    gpu_vram = ram_gb * 0.75
                    
                    return True, f"{chip_name} GPU", gpu_vram
                    
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return False, None, None
    
    def _parse_local_models(self, ollama_response: dict) -> dict[str, dict]:
        """
        Parse Ollama list response into model info dict.
        
        Args:
            ollama_response: Response from ollama.list()
            
        Returns:
            Dict mapping model names to their info
        """
        models = {}
        for model in ollama_response.get("models", []):
            name = model.get("name", "")
            # Store both full name and base name
            models[name] = model
            
            # Also index by base name (without tag)
            base_name = name.split(":")[0]
            if base_name not in models:
                models[base_name] = model
        
        return models
    
    def _is_model_available(self, model_name: str, local_models: dict) -> bool:
        """
        Check if a model is available locally.
        
        Args:
            model_name: Model name to check
            local_models: Dict of locally available models
            
        Returns:
            True if model is available locally
        """
        # Check exact match
        if model_name in local_models:
            return True
        
        # Check with :latest tag
        if f"{model_name}:latest" in local_models:
            return True
        
        # Check base name match
        base_name = model_name.split(":")[0]
        if base_name in local_models:
            return True
        
        return False
    
    def _select_best_model(self, local_models: dict) -> str:
        """
        Select the best model based on hardware and local availability.
        
        Prefers locally available models, then recommends based on hardware.
        
        Args:
            local_models: Dict of locally available models
            
        Returns:
            Selected model name
        """
        hw = self.hardware_info
        assert hw is not None
        
        # First, check if any recommended models are already downloaded
        for rec in MODEL_RECOMMENDATIONS:
            if self._is_model_available(rec.name, local_models):
                # Verify hardware can run it
                if hw.ram_gb >= rec.min_ram_gb:
                    print(f"Found locally available model: {rec.name}")
                    return rec.name
        
        # No suitable local model, select best for hardware
        print("No suitable local model found. Selecting based on hardware...")
        
        # Filter models that hardware can support
        suitable_models = []
        for rec in MODEL_RECOMMENDATIONS:
            # Check RAM requirement
            if hw.ram_gb < rec.min_ram_gb:
                continue
            
            # For GPU-recommended models, prefer if GPU available
            score = 0
            if rec.gpu_recommended and hw.gpu_available:
                # Check if GPU VRAM is sufficient (rough estimate)
                if hw.gpu_vram_gb and hw.gpu_vram_gb >= rec.size_gb:
                    score = 100  # High priority for GPU-capable
                else:
                    score = 50  # Can still run on CPU
            elif not rec.gpu_recommended:
                score = 75  # CPU-friendly models score well
            else:
                score = 25  # GPU model without GPU
            
            suitable_models.append((score, rec))
        
        if not suitable_models:
            # Fallback to smallest model
            print("Warning: Limited resources detected. Using minimal model.")
            return "tinyllama:1.1b"
        
        # Sort by score (descending) and return best
        suitable_models.sort(key=lambda x: x[0], reverse=True)
        selected = suitable_models[0][1]
        
        print(f"Recommended: {selected.name} - {selected.description}")
        return selected.name
    
    def _download_model(self, model_name: str) -> None:
        """
        Download a model using Ollama.
        
        Args:
            model_name: Name of model to download
        """
        import ollama
        
        # Find recommendation for size estimate
        size_str = "unknown size"
        for rec in MODEL_RECOMMENDATIONS:
            if rec.name == model_name or rec.name.split(":")[0] == model_name.split(":")[0]:
                size_str = f"~{rec.size_gb:.1f}GB"
                break
        
        print(f"Downloading {model_name} ({size_str})...")
        print("This may take several minutes depending on your connection...")
        
        try:
            # Stream progress
            for progress in ollama.pull(model_name, stream=True):
                status = progress.get("status", "")
                completed = progress.get("completed", 0)
                total = progress.get("total", 0)
                
                if total > 0:
                    pct = (completed / total) * 100
                    print(f"\r  {status}: {pct:.1f}%", end="", flush=True)
                else:
                    print(f"\r  {status}...", end="", flush=True)
            
            print("\nModel downloaded successfully.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model '{model_name}': {e}")
    
    def _is_ollama_running(self) -> bool:
        """
        Check if Ollama server is running.
        
        Returns:
            True if server is responding, False otherwise
        """
        import urllib.request
        import urllib.error
        
        try:
            # Try to connect to Ollama API
            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return False
    
    def _start_ollama_server(self, timeout: int = 30) -> None:
        """
        Start Ollama server as a background process.
        
        Args:
            timeout: Maximum seconds to wait for server to start
            
        Raises:
            RuntimeError: If Ollama executable not found or server fails to start
        """
        # Find ollama executable
        ollama_path = shutil.which("ollama")
        if ollama_path is None:
            raise RuntimeError(
                "Ollama executable not found. Please install Ollama:\n"
                "  https://ollama.com/download"
            )
        
        # Start ollama serve as background process
        try:
            subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )
        except OSError as e:
            raise RuntimeError(f"Failed to start Ollama server: {e}")
        
        # Wait for server to become available
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_ollama_running():
                print("Ollama server started successfully.")
                return
            time.sleep(0.5)
        
        raise RuntimeError(
            f"Ollama server did not start within {timeout} seconds. "
            "Try starting manually: ollama serve"
        )
    
    def _prepare_context(self) -> dict[str, Any]:
        """
        Prepare structured context from pipeline results.
        
        Returns:
            Dictionary with summarized results for LLM context
        """
        context = {
            "analysis_type": "Multi-omics integration for Parkinson's disease",
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
        prompt = """You are a bioinformatics expert summarizing multi-omics analysis results 
for Parkinson's disease research. Generate a clear, scientifically accurate summary 
of the following analysis results. Focus on:
1. Key findings and their biological significance
2. Most important features from each data modality
3. Model performance and reliability
4. Potential clinical or research implications

Be precise with statistics but explain them in accessible terms.

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
        
        # Extract top biomarkers
        if "feature_importance" in context:
            for feat in context["feature_importance"]["top_features"][:10]:
                findings["top_biomarkers"].append({
                    "name": feat.get("feature", "Unknown"),
                    "importance": feat.get("mean_abs_shap", 0),
                    "modality": feat.get("modality", "unknown"),
                    "type": "metagenomics" if "MG_" in str(feat.get("feature", "")) else "transcriptomics"
                })
        
        # Model reliability
        if "model_performance" in context:
            perf = context["model_performance"]
            findings["model_reliability"] = {
                "accuracy": perf.get("accuracy", {}).get("mean", 0),
                "f1_score": perf.get("f1", {}).get("mean", 0),
                "roc_auc": perf.get("roc_auc", {}).get("mean", 0),
                "stability": 1 - perf.get("accuracy", {}).get("std", 0) if "accuracy" in perf else 0
            }
        
        return findings
    
    def _generate_takeaways(self, context: dict[str, Any]) -> list[str]:
        """
        Generate key takeaways from analysis.
        
        Args:
            context: Analysis context
            
        Returns:
            List of key takeaway strings
        """
        prompt = self._build_takeaways_prompt(context)
        
        response = self.llm_client.generate(prompt)
        
        # Parse response into list
        takeaways = []
        for line in response.split("\n"):
            line = line.strip()
            # Remove numbering and bullet points
            if line and not line.startswith("#"):
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
        prompt = """Based on the following multi-omics Parkinson's disease analysis results, 
provide exactly 5 key takeaways. Each takeaway should be a single sentence that 
a researcher or clinician could immediately understand and act upon.

Focus on:
- Most important biomarkers discovered
- Clinical relevance of findings
- Strengths and limitations of the analysis
- Suggested next steps

"""
        # Add context summary
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
        if not self.literature_db:
            return False
        
        db_path = Path(self.literature_db)
        
        # Check if database already exists
        if db_path.exists() and (db_path / "chroma.sqlite3").exists():
            return True
        
        # Database doesn't exist - try to download
        if not self.rag_auto_download:
            print(f"RAG database not found at {db_path}")
            print("Set rag_auto_download=true or run scripts/build_rag_database.py")
            return False
        
        print(f"RAG database not found at {db_path}. Attempting to download...")
        
        # Try to download from GitHub
        if self._download_rag_database(db_path):
            return True
        
        print("RAG database download failed. Continuing without RAG augmentation.")
        print("To build locally, run: python scripts/build_rag_database.py --email your@email.com")
        return False
    
    def _download_rag_database(self, db_path: Path) -> bool:
        """
        Download RAG database from GitHub releases.
        
        Args:
            db_path: Path where database should be stored
            
        Returns:
            True if download successful, False otherwise
        """
        import hashlib
        import tarfile
        import urllib.request
        import urllib.error
        
        # Determine repo to download from
        repo = self.rag_repo
        if not repo:
            # Try to detect from git remote
            repo = self._detect_github_repo()
        
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
            
            for asset in assets:
                name = asset.get("name", "")
                if name.endswith(".tar.gz") and "pd_literature_db" in name:
                    archive_asset = asset
                elif name.endswith(".sha256"):
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
            self._download_with_progress(archive_url, temp_archive, archive_size)
            
            # Verify checksum if available
            if checksum_asset:
                print("Verifying checksum...")
                checksum_url = checksum_asset["browser_download_url"]
                
                req = urllib.request.Request(checksum_url, headers={"User-Agent": "LLMSummarizer"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    expected_checksum = response.read().decode("utf-8").split()[0]
                
                actual_checksum = self._compute_file_checksum(temp_archive)
                
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
    
    def _download_with_progress(
        self,
        url: str,
        dest_path: Path,
        total_size: int,
    ) -> None:
        """
        Download a file with progress indication.
        
        Args:
            url: URL to download
            dest_path: Destination file path
            total_size: Expected file size in bytes
        """
        import urllib.request
        
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
    
    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA256 checksum of a file."""
        import hashlib
        
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _detect_github_repo(self) -> str | None:
        """Try to detect GitHub repo from git remote."""
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
    
    def _augment_with_literature(self, summary: str) -> str:
        """
        Augment summary with relevant literature findings using RAG.
        
        Queries the PD literature database for findings related to the
        biomarkers and features identified in the analysis.
        
        Args:
            summary: Generated summary text
            
        Returns:
            Summary augmented with literature references
        """
        if not self.literature_db:
            return summary
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Connect to literature database
            client = chromadb.PersistentClient(
                path=self.literature_db,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Query the findings collection
            try:
                findings_collection = client.get_collection("pd_findings")
            except Exception:
                print("Warning: pd_findings collection not found in database")
                return summary
            
            # Extract key terms from summary for retrieval
            key_terms = self._extract_key_terms(summary)
            
            # Query for relevant findings
            all_findings = []
            for term in key_terms[:3]:  # Limit queries
                results = findings_collection.query(
                    query_texts=[term],
                    n_results=3
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
            
            # Add literature context to summary
            if all_findings:
                literature_context = "\n\n## Related Literature Findings\n"
                literature_context += "The following findings from published research may provide context:\n"
                
                # Group by finding type
                findings_by_type: dict[str, list] = {}
                for f in all_findings[:8]:  # Limit total findings
                    ftype = f["type"]
                    if ftype not in findings_by_type:
                        findings_by_type[ftype] = []
                    findings_by_type[ftype].append(f)
                
                for ftype, findings in findings_by_type.items():
                    literature_context += f"\n### {ftype.title()} Findings\n"
                    for f in findings[:3]:
                        literature_context += f"- {f['finding']}\n"
                        literature_context += f"  *{f['citation']}* (Evidence: {f['evidence']})\n"
                
                summary += literature_context
            
            return summary
            
        except ImportError:
            print("Warning: chromadb not installed. RAG augmentation disabled.")
            return summary
        except Exception as e:
            print(f"Warning: RAG augmentation failed: {e}")
            return summary
    
    def _extract_key_terms(self, text: str) -> list[str]:
        """
        Extract key terms from text and analysis results for RAG retrieval.
        
        Args:
            text: Input text (summary)
            
        Returns:
            List of key terms/queries for database search
        """
        import re
        
        key_terms = []
        
        # Extract from feature importance if available
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(5)
            for _, row in top_features.iterrows():
                feature_name = row.get("feature", "")
                # Remove modality prefix (MG_, TX_)
                clean_name = re.sub(r'^(MG_|TX_|PT_|MT_)', '', str(feature_name))
                if clean_name:
                    key_terms.append(f"{clean_name} Parkinson's disease")
        
        # Look for gene/protein names in text (typically uppercase)
        gene_pattern = r'\b([A-Z][A-Z0-9]{2,}[a-z]*)\b'
        genes = re.findall(gene_pattern, text)
        # Filter common non-gene words
        skip_words = {"THE", "AND", "FOR", "WITH", "FROM", "THAT", "THIS", "ARE", "WAS", "RNA", "DNA"}
        genes = [g for g in genes if g not in skip_words]
        for gene in genes[:5]:
            key_terms.append(f"{gene} Parkinson's disease")
        
        # Look for bacterial taxa names (genus names typically capitalized)
        taxa_pattern = r'\b([A-Z][a-z]+(?:aceae|ales|ota|coccus|bacillus|bacterium)?)\b'
        taxa = re.findall(taxa_pattern, text)
        for taxon in taxa[:3]:
            if len(taxon) > 4:  # Filter short matches
                key_terms.append(f"{taxon} gut microbiome")
        
        # Add general context queries
        key_terms.extend([
            "gut microbiome Parkinson's disease biomarker",
            "alpha-synuclein gut-brain axis",
            "Parkinson's disease transcriptomics gene expression",
        ])
        
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
        md = "# Multi-Omics Analysis Summary\n\n"
        md += "## Parkinson's Disease Biomarker Discovery\n\n"
        
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


# Ollama Client

class OllamaClient:
    """Client for Ollama LLM inference."""
    
    def __init__(self, model: str, temperature: float, max_tokens: int):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama3", "mistral", "phi3")
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
        """
        import ollama
        self.client = ollama
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str) -> str:
        """
        Generate text completion using Ollama.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Generated text response
        """
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        return response["response"]
    
    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Chat completion using Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Assistant response text
        """
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        return response["message"]["content"]
