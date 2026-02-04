"""
Unit tests for LLMSummarizer PluMA Plugin.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pytest

from LLMSummarizer import LLMSummarizer
from domains import DomainConfig, DOMAIN_CONFIGS, get_domain_config, list_domains
from hardware import (
    HardwareInfo,
    ModelRecommendation,
    MODEL_RECOMMENDATIONS,
    detect_hardware,
    get_system_ram,
    detect_gpu,
    select_best_model,
    get_model_size_estimate,
)
from ollama_client import (
    is_ollama_running,
    is_model_available,
)


@pytest.fixture
def sample_feature_importance() -> pd.DataFrame:
    """Create sample SHAP feature importance data."""
    return pd.DataFrame({
        "feature": [f"MG_Taxa_{i}" for i in range(5)] + [f"TX_Gene_{i}" for i in range(5)],
        "mean_abs_shap": [0.15, 0.12, 0.10, 0.08, 0.06, 0.14, 0.11, 0.09, 0.07, 0.05],
        "modality": ["metagenomics"] * 5 + ["transcriptomics"] * 5
    })


@pytest.fixture
def sample_cv_results() -> pd.DataFrame:
    """Create sample cross-validation results."""
    return pd.DataFrame({
        "test_accuracy": [0.85, 0.87, 0.83, 0.86, 0.84],
        "test_f1": [0.84, 0.86, 0.82, 0.85, 0.83],
        "test_roc_auc": [0.90, 0.92, 0.88, 0.91, 0.89]
    })


@pytest.fixture
def sample_de_results() -> pd.DataFrame:
    """Create sample differential expression results."""
    return pd.DataFrame({
        "gene": [f"Gene_{i}" for i in range(20)],
        "log2FoldChange": np.random.randn(20),
        "pvalue": np.random.uniform(0, 1, 20),
        "padj": np.random.uniform(0, 1, 20)
    })


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def plugin_with_data(temp_dir, sample_feature_importance, sample_cv_results):
    """Create plugin instance with loaded test data."""
    fi_path = temp_dir / "feature_importance.csv"
    cv_path = temp_dir / "cv_results.csv"
    
    sample_feature_importance.to_csv(fi_path, index=False)
    sample_cv_results.to_csv(cv_path, index=False)
    
    param_path = temp_dir / "params.txt"
    param_path.write_text(f"""feature_importance\t{fi_path}
cv_results\t{cv_path}
model_name\tllama3
temperature\t0.3
max_tokens\t512
use_rag\tfalse
""")
    
    plugin = LLMSummarizer()
    plugin.input(str(param_path))
    
    return plugin


# =============================================================================
# Domain Configuration Tests
# =============================================================================

class TestDomainConfig:
    """Tests for domain configuration."""
    
    def test_domain_configs_not_empty(self):
        """Test that domain configs exist."""
        assert len(DOMAIN_CONFIGS) > 0
    
    def test_all_domains_have_required_fields(self):
        """Test that all domains have required fields."""
        for name, config in DOMAIN_CONFIGS.items():
            assert config.name == name
            assert config.display_name
            assert config.description
            assert config.rag_collection_name
            assert config.rag_db_path
            assert config.expert_role
            assert config.research_focus
    
    def test_parkinsons_config(self):
        """Test Parkinson's domain config."""
        config = DOMAIN_CONFIGS["parkinsons"]
        assert config.display_name == "Parkinson's Disease"
        assert config.rag_collection_name == "pd_findings"
        assert "Parkinson" in config.feature_suffix
    
    def test_get_domain_config(self):
        """Test get_domain_config helper."""
        config = get_domain_config("parkinsons")
        assert config.name == "parkinsons"
    
    def test_get_domain_config_invalid(self):
        """Test get_domain_config with invalid domain."""
        with pytest.raises(ValueError):
            get_domain_config("invalid_domain")
    
    def test_list_domains(self):
        """Test list_domains helper."""
        domains = list_domains()
        assert "parkinsons" in domains
        assert "generic" in domains
        assert len(domains) == len(DOMAIN_CONFIGS)


# =============================================================================
# Hardware Detection Tests
# =============================================================================

class TestHardwareInfo:
    """Tests for HardwareInfo dataclass."""
    
    def test_hardware_info_str_with_gpu(self):
        """Test string representation with GPU."""
        info = HardwareInfo(
            cpu_cores=8,
            ram_gb=32.0,
            gpu_available=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_vram_gb=24.0
        )
        
        result = str(info)
        
        assert "8 cores" in result
        assert "32.0GB" in result
        assert "NVIDIA RTX 4090" in result
        assert "24.0GB" in result
    
    def test_hardware_info_str_without_gpu(self):
        """Test string representation without GPU."""
        info = HardwareInfo(
            cpu_cores=4,
            ram_gb=16.0,
            gpu_available=False,
            gpu_name=None,
            gpu_vram_gb=None
        )
        
        result = str(info)
        
        assert "4 cores" in result
        assert "16.0GB" in result
        assert "None" in result


class TestModelRecommendations:
    """Tests for model recommendations."""
    
    def test_recommendations_not_empty(self):
        """Test that model recommendations exist."""
        assert len(MODEL_RECOMMENDATIONS) > 0
    
    def test_recommendations_have_required_fields(self):
        """Test that all recommendations have required fields."""
        for rec in MODEL_RECOMMENDATIONS:
            assert rec.name
            assert rec.size_gb > 0
            assert rec.description
            assert rec.min_ram_gb > 0
            assert isinstance(rec.gpu_recommended, bool)
    
    def test_recommendations_include_lightweight(self):
        """Test that there's at least one model for low-resource systems."""
        lightweight = [r for r in MODEL_RECOMMENDATIONS if r.min_ram_gb <= 4]
        assert len(lightweight) > 0


class TestHardwareDetection:
    """Tests for hardware detection functions."""
    
    def test_detect_hardware_returns_info(self):
        """Test that detect_hardware returns HardwareInfo."""
        info = detect_hardware()
        assert isinstance(info, HardwareInfo)
        assert info.cpu_cores > 0
        assert info.ram_gb > 0
    
    def test_get_system_ram_positive(self):
        """Test that get_system_ram returns positive value."""
        ram = get_system_ram()
        assert ram > 0
    
    def test_detect_gpu_returns_tuple(self):
        """Test that detect_gpu returns proper tuple."""
        result = detect_gpu()
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], bool)


class TestModelSelection:
    """Tests for model selection functions."""
    
    def test_is_model_available_exact_match(self):
        """Test exact model name matching."""
        local_models = {"llama3:8b": {}, "mistral:7b": {}}
        assert is_model_available("llama3:8b", local_models)
    
    def test_is_model_available_with_latest_tag(self):
        """Test model matching with :latest tag."""
        local_models = {"llama3:latest": {}}
        assert is_model_available("llama3", local_models)
    
    def test_is_model_not_available(self):
        """Test model not found."""
        local_models = {"mistral:7b": {}}
        assert not is_model_available("llama3", local_models)
    
    def test_select_best_model_with_local_available(self):
        """Test model selection prefers local models."""
        hw = HardwareInfo(
            cpu_cores=8, ram_gb=32.0,
            gpu_available=True, gpu_name="NVIDIA RTX 4090", gpu_vram_gb=24.0
        )
        local_models = {"llama3.1:8b": {}}
        
        selected = select_best_model(hw, local_models, is_model_available)
        assert selected == "llama3.1:8b"
    
    def test_select_best_model_for_low_ram(self):
        """Test model selection for low RAM system."""
        hw = HardwareInfo(
            cpu_cores=2, ram_gb=3.0,
            gpu_available=False, gpu_name=None, gpu_vram_gb=None
        )
        local_models = {}
        
        selected = select_best_model(hw, local_models, is_model_available)
        # Should select a small model
        assert "tiny" in selected or "gemma" in selected or "phi" in selected
    
    def test_get_model_size_estimate(self):
        """Test model size estimation."""
        size = get_model_size_estimate("llama3.1:8b")
        assert "GB" in size
        assert "4" in size or "5" in size  # ~4.7GB


# =============================================================================
# Ollama Client Tests
# =============================================================================

class TestOllamaServerManagement:
    """Tests for Ollama server management."""
    
    def test_is_ollama_running_returns_bool(self):
        """Test that is_ollama_running returns boolean."""
        result = is_ollama_running()
        assert isinstance(result, bool)


# =============================================================================
# LLMSummarizer Plugin Tests
# =============================================================================

class TestLLMSummarizerInit:
    """Tests for LLMSummarizer initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        plugin = LLMSummarizer()
        
        assert plugin.model_name is None  # Auto-select
        assert plugin.temperature == 0.3
        assert plugin.max_tokens == 1024
        assert plugin.domain == "generic"
    
    def test_load_feature_importance(self, plugin_with_data):
        """Test loading feature importance data."""
        assert plugin_with_data.feature_importance is not None
        assert len(plugin_with_data.feature_importance) == 10
    
    def test_parse_model_name(self, plugin_with_data):
        """Test model name parsing."""
        assert plugin_with_data.model_name == "llama3"
    
    def test_domain_configuration(self, temp_dir, sample_feature_importance):
        """Test domain configuration."""
        fi_path = temp_dir / "fi.csv"
        sample_feature_importance.to_csv(fi_path, index=False)
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"""feature_importance\t{fi_path}
domain\tparkinsons
use_rag\tfalse
""")
        
        plugin = LLMSummarizer()
        plugin.input(str(param_path))
        
        assert plugin.domain == "parkinsons"
        assert plugin.domain_config.display_name == "Parkinson's Disease"


class TestContextPreparation:
    """Tests for context preparation methods."""
    
    def test_prepare_context_with_feature_importance(self, plugin_with_data):
        """Test context preparation with feature importance."""
        context = plugin_with_data._prepare_context()
        
        assert "feature_importance" in context
        assert context["feature_importance"]["n_features"] == 10
        assert len(context["feature_importance"]["top_features"]) == 10
    
    def test_prepare_context_with_cv_results(self, plugin_with_data):
        """Test context preparation with CV results."""
        context = plugin_with_data._prepare_context()
        
        assert "model_performance" in context
        assert "accuracy" in context["model_performance"]
    
    def test_summarize_modality_importance(self, plugin_with_data):
        """Test modality importance summarization."""
        breakdown = plugin_with_data._summarize_modality_importance()
        
        assert "metagenomics" in breakdown
        assert "transcriptomics" in breakdown
        assert abs(sum(breakdown.values()) - 100) < 0.1


class TestPromptBuilding:
    """Tests for prompt building methods."""
    
    def test_build_summary_prompt(self, plugin_with_data):
        """Test summary prompt building."""
        context = plugin_with_data._prepare_context()
        prompt = plugin_with_data._build_summary_prompt(context)
        
        assert "bioinformatics expert" in prompt
        assert "Feature Importance" in prompt
        assert "Model Performance" in prompt
    
    def test_build_summary_prompt_parkinsons(self, temp_dir, sample_feature_importance):
        """Test summary prompt with Parkinson's domain."""
        fi_path = temp_dir / "fi.csv"
        sample_feature_importance.to_csv(fi_path, index=False)
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"""feature_importance\t{fi_path}
domain\tparkinsons
use_rag\tfalse
""")
        
        plugin = LLMSummarizer()
        plugin.input(str(param_path))
        
        context = plugin._prepare_context()
        prompt = plugin._build_summary_prompt(context)
        
        assert "Parkinson" in prompt
    
    def test_build_takeaways_prompt(self, plugin_with_data):
        """Test takeaways prompt building."""
        context = plugin_with_data._prepare_context()
        prompt = plugin_with_data._build_takeaways_prompt(context)
        
        assert "5 key takeaways" in prompt


class TestStructuredFindings:
    """Tests for structured findings extraction."""
    
    def test_extract_structured_findings(self, plugin_with_data):
        """Test structured findings extraction."""
        context = plugin_with_data._prepare_context()
        findings = plugin_with_data._extract_structured_findings(context)
        
        assert "analysis_type" in findings
        assert "top_biomarkers" in findings
        assert "model_reliability" in findings
    
    def test_top_biomarkers_structure(self, plugin_with_data):
        """Test top biomarkers have correct structure."""
        context = plugin_with_data._prepare_context()
        findings = plugin_with_data._extract_structured_findings(context)
        
        for biomarker in findings["top_biomarkers"]:
            assert "name" in biomarker
            assert "modality" in biomarker
            assert "importance" in biomarker


class TestMarkdownFormatting:
    """Tests for markdown output formatting."""
    
    def test_format_as_markdown(self, plugin_with_data):
        """Test markdown formatting."""
        plugin_with_data.summary_text = "Test summary"
        plugin_with_data.key_takeaways = ["Takeaway 1", "Takeaway 2"]
        plugin_with_data.structured_findings = {
            "top_biomarkers": [
                {"name": "Gene1", "modality": "tx", "importance": 0.5}
            ]
        }
        
        md = plugin_with_data._format_as_markdown()
        
        assert "# " in md  # Has header
        assert "Test summary" in md
        assert "Takeaway 1" in md
        assert "Gene1" in md


class TestOutput:
    """Tests for output methods."""
    
    def test_output_creates_files(self, plugin_with_data, temp_dir):
        """Test that output creates expected files."""
        plugin_with_data.summary_text = "Test summary"
        plugin_with_data.key_takeaways = ["Takeaway"]
        plugin_with_data.structured_findings = {"test": "data"}
        
        output_path = temp_dir / "output"
        plugin_with_data.output(str(output_path))
        
        assert output_path.with_suffix(".txt").exists()
        assert output_path.with_suffix(".md").exists()
        assert output_path.with_suffix(".json").exists()
    
    def test_output_json_valid(self, plugin_with_data, temp_dir):
        """Test that output JSON is valid."""
        plugin_with_data.summary_text = "Test summary"
        plugin_with_data.key_takeaways = ["Takeaway"]
        plugin_with_data.structured_findings = {"test": "data"}
        
        output_path = temp_dir / "output"
        plugin_with_data.output(str(output_path))
        
        json_path = output_path.with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)
        
        assert "summary" in data
        assert "key_takeaways" in data


class TestKeyTermExtraction:
    """Tests for key term extraction for RAG."""
    
    def test_extract_key_terms(self, plugin_with_data):
        """Test key term extraction."""
        terms = plugin_with_data._extract_key_terms("SNCA and GBA mutations in study")
        
        assert isinstance(terms, list)
        assert len(terms) > 0
    
    def test_extract_key_terms_from_features(self, plugin_with_data):
        """Test that features are included in key terms."""
        plugin_with_data.domain_config = DOMAIN_CONFIGS["parkinsons"]
        terms = plugin_with_data._extract_key_terms("Analysis results")
        
        # Should include terms from feature importance
        assert any("Taxa" in t or "Gene" in t or "Parkinson" in t for t in terms)


# =============================================================================
# Integration Tests
# =============================================================================

class TestOllamaClientIntegration:
    """Integration tests for OllamaClient (requires ollama module)."""
    
    @pytest.mark.skipif(True, reason="Requires ollama module")
    def test_ollama_client_init(self):
        """Test OllamaClient initialization."""
        from ollama_client import OllamaClient
        client = OllamaClient(model="llama3", temperature=0.3, max_tokens=100)
        assert client.model == "llama3"
