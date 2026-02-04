# LLMSummarizer

A PluMA plugin for local LLM-based summarization of multi-omics pipeline findings with domain-specific RAG (Retrieval-Augmented Generation) support.

## Overview

This plugin uses locally-hosted large language models to:

1. Generate natural language summaries of statistical results
2. Interpret feature importance rankings in biological context
3. Compare integration method performance
4. Optionally contextualize findings with domain-specific literature (RAG)

Supports configurable research domains (Parkinson's, Alzheimer's, cancer, microbiome) with pre-built configurations and customizable RAG databases.

**Key Features:**
- 🔒 **Privacy-first**: All inference runs locally via Ollama
- 🧠 **Smart model selection**: Auto-detects hardware and selects optimal model
- 📚 **RAG augmentation**: Enhances summaries with relevant literature
- 🎯 **Multi-domain**: Pre-configured for multiple research areas
- 📦 **Modular design**: Clean separation of concerns

## Installation

```bash
# Clone the repository
git clone https://github.com/quinnjr/LLMSummarizer.git
cd LLMSummarizer

# Install dependencies
pip install -r requirements.txt

# Install Ollama (required)
# See: https://ollama.com/download
```

## Prerequisites

1. **Ollama** must be installed: https://ollama.com
2. The plugin will automatically:
   - Start the Ollama server if not running
   - Detect your hardware (CPU, RAM, GPU)
   - Download an appropriate model if needed

## Dependencies

### Required
- numpy
- pandas
- requests

### Optional
- chromadb (for RAG features)
- psutil (for accurate RAM detection)
- ollama (Python client)

## Usage

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `feature_importance` | Path to SHAP feature importance CSV | Optional |
| `cv_results` | Path to cross-validation results CSV | Optional |
| `de_results` | Path to differential expression results CSV | Optional |
| `cluster_results` | Path to clustering results CSV | Optional |
| `model_metrics` | Path to model evaluation metrics CSV | Optional |
| `domain` | Research domain (see below) | `generic` |
| `model_name` | Ollama model or `auto` | `auto` |
| `temperature` | Sampling temperature | `0.3` |
| `max_tokens` | Maximum tokens in response | `1024` |
| `use_rag` | Enable RAG augmentation | `true` |
| `literature_db` | Path to ChromaDB database | Domain default |
| `rag_auto_download` | Auto-download RAG database | `true` |
| `rag_repo` | GitHub repo for RAG database | Auto-detected |

### Available Domains

| Domain | Description | RAG Collection |
|--------|-------------|----------------|
| `parkinsons` | Parkinson's disease research | `pd_findings` |
| `alzheimers` | Alzheimer's disease research | `ad_findings` |
| `cancer` | Cancer genomics | `cancer_findings` |
| `microbiome` | Microbiome analysis | `microbiome_findings` |
| `generic` | General multi-omics | `findings` |
| `custom` | User-defined (see below) | Configurable |

### Example Parameter File

```
# Input files
feature_importance    CSV/shap_feature_importance.csv
cv_results            CSV/cv_results.csv

# Domain configuration
domain                parkinsons

# LLM settings (auto selects best model for your hardware)
model_name            auto
temperature           0.3
max_tokens            1024

# RAG settings
use_rag               true
rag_auto_download     true
```

### Custom Domain Configuration

```
domain                      custom
custom_domain_name          My Research Area
custom_domain_expert_role   bioinformatics expert in my field
custom_domain_research_focus identifying biomarkers
custom_domain_feature_suffix my disease
custom_domain_collection    my_findings
custom_domain_db_path       data/my_literature_db
```

### Outputs

- `summary.txt` - Plain text summary
- `summary.md` - Markdown-formatted summary with tables
- `summary.json` - Structured findings with metadata

## Package Structure

```
LLMSummarizer/
├── __init__.py          # Package exports
├── LLMSummarizer.py     # Main plugin class
├── domains.py           # Domain configurations
├── hardware.py          # Hardware detection & model selection
├── ollama_client.py     # Ollama client & server management
├── rag.py               # RAG database management
├── scripts/
│   ├── build_rag_database.py    # Build literature database
│   ├── query_rag_database.py    # Query database CLI
│   ├── upload_rag_to_github.py  # Upload to GitHub releases
│   └── release.py               # Version management
└── tests/
```

## Hardware Detection

The plugin automatically detects:
- **CPU**: Core count
- **RAM**: Total system memory
- **GPU**: NVIDIA (via nvidia-smi), AMD (via rocm-smi), Apple Silicon (Metal)

Based on detected hardware, it selects from:

| Model | Size | Min RAM | GPU Recommended |
|-------|------|---------|-----------------|
| llama3.1:70b | 40GB | 48GB | Yes |
| llama3.1:8b | 4.7GB | 8GB | Yes |
| mistral:7b | 4.1GB | 8GB | Yes |
| phi3:medium | 7.9GB | 12GB | No |
| phi3:mini | 2.2GB | 4GB | No |
| gemma2:2b | 1.6GB | 4GB | No |
| tinyllama:1.1b | 0.6GB | 2GB | No |

## RAG Database

### Building a Database

```bash
# Build Parkinson's disease database
python scripts/build_rag_database.py \
    --domain parkinsons \
    --email your@email.com \
    --max-papers 500

# Build for other domains
python scripts/build_rag_database.py --domain alzheimers --email your@email.com
python scripts/build_rag_database.py --domain cancer --email your@email.com
```

### Querying the Database

```bash
python scripts/query_rag_database.py \
    --db data/pd_literature_db \
    --query "alpha-synuclein gut microbiome"
```

### Uploading to GitHub Releases

```bash
python scripts/upload_rag_to_github.py \
    --db data/pd_literature_db \
    --repo quinnjr/LLMSummarizer
```

## API Usage

```python
from LLMSummarizer import LLMSummarizer

# Create plugin instance
plugin = LLMSummarizer()

# Load parameters
plugin.input("parameters.txt")

# Run summarization
plugin.run()

# Write outputs
plugin.output("output/summary")
```

### Using Individual Components

```python
from LLMSummarizer import (
    detect_hardware,
    OllamaClient,
    DOMAIN_CONFIGS,
    query_literature_database,
)

# Detect hardware
hw = detect_hardware()
print(f"RAM: {hw.ram_gb:.1f}GB, GPU: {hw.gpu_name}")

# Use Ollama client directly
client = OllamaClient(model="llama3", temperature=0.3, max_tokens=1024)
response = client.generate("Summarize this analysis...")

# Query RAG database
findings = query_literature_database(
    literature_db="data/pd_literature_db",
    collection_name="pd_findings",
    key_terms=["SNCA", "alpha-synuclein"],
)
```

## Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## References

### Large Language Models

1. **Touvron H, et al.** (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*
   - *Foundation for Llama models*

2. **Jiang AQ, et al.** (2023). Mistral 7B. *arXiv:2310.06825*
   - *Mistral model architecture*

### Retrieval-Augmented Generation

3. **Lewis P, et al.** (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
   - *RAG methodology*

4. **Guu K, et al.** (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML 2020*. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)
   - *Retrieval-augmented pre-training*

### Vector Databases

5. **Johnson J, Douze M, Jégou H** (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
   - *FAISS, foundation for vector search*

### LLMs in Bioinformatics

6. **Thirunavukarasu AJ, et al.** (2023). Large language models in medicine. *Nature Medicine*, 29:1930-1940. [doi:10.1038/s41591-023-02448-8](https://doi.org/10.1038/s41591-023-02448-8)
   - *LLMs in biomedical applications*

7. **Singhal K, et al.** (2023). Large language models encode clinical knowledge. *Nature*, 620:172-180. [doi:10.1038/s41586-023-06291-2](https://doi.org/10.1038/s41586-023-06291-2)
   - *Clinical knowledge in LLMs*

## License

MIT License

## Author

Joseph R. Quinn <quinn.josephr@protonmail.com>
