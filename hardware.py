"""
Hardware detection and model selection for LLMSummarizer.

Detects system hardware (CPU, RAM, GPU) and provides intelligent
model recommendations based on available resources.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass


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


def detect_hardware() -> HardwareInfo:
    """
    Detect system hardware capabilities.
    
    Returns:
        HardwareInfo with CPU, RAM, and GPU details
    """
    cpu_cores = os.cpu_count() or 1
    ram_gb = get_system_ram()
    gpu_available, gpu_name, gpu_vram = detect_gpu()
    
    return HardwareInfo(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram
    )


def get_system_ram() -> float:
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


def detect_gpu() -> tuple[bool, str | None, float | None]:
    """
    Detect GPU availability and specifications.
    
    Returns:
        Tuple of (gpu_available, gpu_name, vram_gb)
    """
    # Try NVIDIA GPU first (most common for ML)
    nvidia_result = detect_nvidia_gpu()
    if nvidia_result[0]:
        return nvidia_result
    
    # Try AMD GPU (ROCm)
    amd_result = detect_amd_gpu()
    if amd_result[0]:
        return amd_result
    
    # Try Apple Silicon (Metal)
    apple_result = detect_apple_gpu()
    if apple_result[0]:
        return apple_result
    
    return False, None, None


def detect_nvidia_gpu() -> tuple[bool, str | None, float | None]:
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


def detect_amd_gpu() -> tuple[bool, str | None, float | None]:
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
                        for part in parts:
                            if part.isdigit():
                                val = int(part)
                                # Heuristic: if > 1000, probably MB
                                vram_gb = val / 1024 if val > 1000 else val
                                break
            
            return True, name, vram_gb
            
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    
    return False, None, None


def detect_apple_gpu() -> tuple[bool, str | None, float | None]:
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
                ram_gb = get_system_ram()
                gpu_vram = ram_gb * 0.75
                
                return True, f"{chip_name} GPU", gpu_vram
                
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return False, None, None


def select_best_model(
    hardware_info: HardwareInfo,
    local_models: dict,
    is_model_available_fn: callable
) -> str:
    """
    Select the best model based on hardware and local availability.
    
    Prefers locally available models, then recommends based on hardware.
    
    Args:
        hardware_info: Detected hardware capabilities
        local_models: Dict of locally available models
        is_model_available_fn: Function to check if model is available
        
    Returns:
        Selected model name
    """
    hw = hardware_info
    
    # First, check if any recommended models are already downloaded
    for rec in MODEL_RECOMMENDATIONS:
        if is_model_available_fn(rec.name, local_models):
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


def get_model_size_estimate(model_name: str) -> str:
    """
    Get estimated size string for a model.
    
    Args:
        model_name: Model name to look up
        
    Returns:
        Size string like "~4.7GB" or "unknown size"
    """
    for rec in MODEL_RECOMMENDATIONS:
        if rec.name == model_name or rec.name.split(":")[0] == model_name.split(":")[0]:
            return f"~{rec.size_gb:.1f}GB"
    return "unknown size"
