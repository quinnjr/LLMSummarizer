"""
Unit tests for LLMSummarizer PluMA Plugin.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from LLMSummarizer import (
    LLMSummarizer,
    HardwareInfo,
    ModelRecommendation,
    MODEL_RECOMMENDATIONS,
    OllamaClient,
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
""")
    
    plugin = LLMSummarizer()
    plugin.input(str(param_path))
    
    return plugin


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
        assert "RTX 4090" in result
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
    """Tests for model recommendations list."""
    
    def test_recommendations_not_empty(self):
        """Test that recommendations list is not empty."""
        assert len(MODEL_RECOMMENDATIONS) > 0
    
    def test_recommendations_have_required_fields(self):
        """Test all recommendations have required fields."""
        for rec in MODEL_RECOMMENDATIONS:
            assert rec.name
            assert rec.size_gb > 0
            assert rec.description
            assert rec.min_ram_gb > 0
    
    def test_recommendations_include_lightweight(self):
        """Test that lightweight models are included."""
        lightweight = [r for r in MODEL_RECOMMENDATIONS if r.min_ram_gb <= 4]
        assert len(lightweight) > 0


class TestLLMSummarizerInit:
    """Tests for plugin initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        plugin = LLMSummarizer()
        
        assert plugin.model_name is None  # Auto-select
        assert plugin.temperature == 0.3
        assert plugin.max_tokens == 1024
        assert plugin.use_rag is False
    
    def test_load_feature_importance(self, temp_dir, sample_feature_importance):
        """Test loading feature importance data."""
        fi_path = temp_dir / "fi.csv"
        sample_feature_importance.to_csv(fi_path, index=False)
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"feature_importance\t{fi_path}\n")
        
        plugin = LLMSummarizer()
        plugin.input(str(param_path))
        
        assert plugin.feature_importance is not None
        assert len(plugin.feature_importance) == 10
    
    def test_parse_model_name(self, temp_dir):
        """Test parsing model name parameter."""
        param_path = temp_dir / "params.txt"
        param_path.write_text("model_name\tmistral\n")
        
        plugin = LLMSummarizer()
        plugin.input(str(param_path))
        
        assert plugin.model_name == "mistral"
    
    def test_auto_model_name(self, temp_dir):
        """Test auto model selection."""
        param_path = temp_dir / "params.txt"
        param_path.write_text("model_name\tauto\n")
        
        plugin = LLMSummarizer()
        plugin.input(str(param_path))
        
        assert plugin.model_name is None  # Should trigger auto-selection


class TestHardwareDetection:
    """Tests for hardware detection methods."""
    
    def test_detect_hardware_returns_info(self):
        """Test that _detect_hardware returns HardwareInfo."""
        plugin = LLMSummarizer()
        
        info = plugin._detect_hardware()
        
        assert isinstance(info, HardwareInfo)
        assert info.cpu_cores > 0
        assert info.ram_gb > 0
    
    def test_get_system_ram_positive(self):
        """Test that RAM detection returns positive value."""
        plugin = LLMSummarizer()
        
        ram = plugin._get_system_ram()
        
        assert ram > 0
    
    def test_detect_gpu_returns_tuple(self):
        """Test that GPU detection returns 3-tuple."""
        plugin = LLMSummarizer()
        
        result = plugin._detect_gpu()
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], bool)


class TestModelSelection:
    """Tests for model selection logic."""
    
    def test_select_model_with_local_available(self):
        """Test model selection prefers local models."""
        plugin = LLMSummarizer()
        plugin.hardware_info = HardwareInfo(
            cpu_cores=8,
            ram_gb=16.0,
            gpu_available=False,
            gpu_name=None,
            gpu_vram_gb=None
        )
        
        # Mock local models including phi3:mini
        local_models = {
            "phi3:mini": {"name": "phi3:mini"},
            "phi3:mini:latest": {"name": "phi3:mini:latest"}
        }
        
        selected = plugin._select_best_model(local_models)
        
        # Should select the available local model
        assert selected == "phi3:mini"
    
    def test_select_model_for_low_ram(self):
        """Test model selection for low RAM systems."""
        plugin = LLMSummarizer()
        plugin.hardware_info = HardwareInfo(
            cpu_cores=2,
            ram_gb=4.0,
            gpu_available=False,
            gpu_name=None,
            gpu_vram_gb=None
        )
        
        local_models = {}  # No local models
        
        selected = plugin._select_best_model(local_models)
        
        # Should select a lightweight model
        rec = next((r for r in MODEL_RECOMMENDATIONS if r.name == selected), None)
        assert rec is not None
        assert rec.min_ram_gb <= 4.0
    
    def test_select_model_for_gpu_system(self):
        """Test model selection for GPU-equipped systems."""
        plugin = LLMSummarizer()
        plugin.hardware_info = HardwareInfo(
            cpu_cores=8,
            ram_gb=32.0,
            gpu_available=True,
            gpu_name="NVIDIA RTX 3080",
            gpu_vram_gb=10.0
        )
        
        local_models = {}
        
        selected = plugin._select_best_model(local_models)
        
        # Should select a capable model
        assert selected is not None
    
    def test_is_model_available_exact_match(self):
        """Test model availability check with exact match."""
        plugin = LLMSummarizer()
        
        local_models = {"llama3:8b": {"name": "llama3:8b"}}
        
        assert plugin._is_model_available("llama3:8b", local_models)
    
    def test_is_model_available_with_latest_tag(self):
        """Test model availability with :latest tag."""
        plugin = LLMSummarizer()
        
        local_models = {"llama3:latest": {"name": "llama3:latest"}}
        
        assert plugin._is_model_available("llama3", local_models)
    
    def test_is_model_not_available(self):
        """Test model not available check."""
        plugin = LLMSummarizer()
        
        local_models = {"mistral:7b": {"name": "mistral:7b"}}
        
        assert not plugin._is_model_available("llama3", local_models)


class TestOllamaServerManagement:
    """Tests for Ollama server management."""
    
    def test_is_ollama_running_returns_bool(self):
        """Test that server check returns boolean."""
        plugin = LLMSummarizer()
        
        result = plugin._is_ollama_running()
        
        assert isinstance(result, bool)
    
    @patch("shutil.which")
    def test_start_ollama_server_not_found(self, mock_which):
        """Test error when Ollama not installed."""
        mock_which.return_value = None
        
        plugin = LLMSummarizer()
        
        with pytest.raises(RuntimeError, match="not found"):
            plugin._start_ollama_server()


class TestContextPreparation:
    """Tests for context preparation."""
    
    def test_prepare_context_with_feature_importance(self, plugin_with_data):
        """Test context includes feature importance."""
        plugin = plugin_with_data
        
        context = plugin._prepare_context()
        
        assert "feature_importance" in context
        assert "top_features" in context["feature_importance"]
    
    def test_prepare_context_with_cv_results(self, plugin_with_data):
        """Test context includes model performance."""
        plugin = plugin_with_data
        
        context = plugin._prepare_context()
        
        assert "model_performance" in context
    
    def test_summarize_modality_importance(self, plugin_with_data):
        """Test modality importance summarization."""
        plugin = plugin_with_data
        
        summary = plugin._summarize_modality_importance()
        
        assert "metagenomics" in summary or "MG" in str(summary)


class TestPromptBuilding:
    """Tests for prompt construction."""
    
    def test_build_summary_prompt(self, plugin_with_data):
        """Test summary prompt construction."""
        plugin = plugin_with_data
        context = plugin._prepare_context()
        
        prompt = plugin._build_summary_prompt(context)
        
        assert "Parkinson" in prompt
        assert "Feature Importance" in prompt
    
    def test_build_takeaways_prompt(self, plugin_with_data):
        """Test takeaways prompt construction."""
        plugin = plugin_with_data
        context = plugin._prepare_context()
        
        prompt = plugin._build_takeaways_prompt(context)
        
        assert "5 key takeaways" in prompt.lower()


class TestStructuredFindings:
    """Tests for structured findings extraction."""
    
    def test_extract_structured_findings(self, plugin_with_data):
        """Test structured findings extraction."""
        plugin = plugin_with_data
        context = plugin._prepare_context()
        
        findings = plugin._extract_structured_findings(context)
        
        assert "analysis_type" in findings
        assert "top_biomarkers" in findings
        assert "model_reliability" in findings
    
    def test_top_biomarkers_structure(self, plugin_with_data):
        """Test top biomarkers have expected fields."""
        plugin = plugin_with_data
        context = plugin._prepare_context()
        
        findings = plugin._extract_structured_findings(context)
        
        if findings["top_biomarkers"]:
            biomarker = findings["top_biomarkers"][0]
            assert "name" in biomarker
            assert "importance" in biomarker
            assert "modality" in biomarker


class TestMarkdownFormatting:
    """Tests for markdown output formatting."""
    
    def test_format_as_markdown(self, plugin_with_data):
        """Test markdown formatting."""
        plugin = plugin_with_data
        plugin.summary_text = "Test summary"
        plugin.key_takeaways = ["Takeaway 1", "Takeaway 2"]
        plugin.structured_findings = {"top_biomarkers": []}
        
        markdown = plugin._format_as_markdown()
        
        assert "# " in markdown  # Has headers
        assert "Test summary" in markdown
        assert "Takeaway 1" in markdown


class TestOutput:
    """Tests for output generation."""
    
    def test_output_creates_files(self, plugin_with_data, temp_dir):
        """Test that output creates expected files."""
        plugin = plugin_with_data
        plugin.summary_text = "Test summary"
        plugin.key_takeaways = ["Takeaway 1"]
        plugin.structured_findings = {"test": "data"}
        
        output_path = temp_dir / "output"
        plugin.output(str(output_path))
        
        assert output_path.with_suffix(".summary.txt").exists()
        assert output_path.with_suffix(".summary.md").exists()
        assert output_path.with_suffix(".findings.json").exists()
        assert output_path.with_suffix(".takeaways.txt").exists()
    
    def test_output_json_valid(self, plugin_with_data, temp_dir):
        """Test that JSON output is valid."""
        plugin = plugin_with_data
        plugin.summary_text = "Test"
        plugin.key_takeaways = []
        plugin.structured_findings = {"key": "value", "number": 42}
        
        output_path = temp_dir / "output"
        plugin.output(str(output_path))
        
        json_path = output_path.with_suffix(".findings.json")
        with open(json_path) as f:
            data = json.load(f)
        
        assert data["key"] == "value"
        assert data["number"] == 42


class TestOllamaClient:
    """Tests for OllamaClient wrapper."""
    
    @patch("ollama.generate")
    def test_generate_method(self, mock_generate):
        """Test generate method calls Ollama correctly."""
        mock_generate.return_value = {"response": "Test response"}
        
        client = OllamaClient(
            model="llama3",
            temperature=0.3,
            max_tokens=100
        )
        
        result = client.generate("Test prompt")
        
        assert result == "Test response"
        mock_generate.assert_called_once()
    
    @patch("ollama.chat")
    def test_chat_method(self, mock_chat):
        """Test chat method calls Ollama correctly."""
        mock_chat.return_value = {"message": {"content": "Test chat response"}}
        
        client = OllamaClient(
            model="llama3",
            temperature=0.3,
            max_tokens=100
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat(messages)
        
        assert result == "Test chat response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
