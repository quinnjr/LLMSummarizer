"""
Ollama client and server management for LLMSummarizer.

Provides a client wrapper for Ollama LLM inference and utilities
for managing the Ollama server lifecycle.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.error
import urllib.request


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


def is_ollama_running() -> bool:
    """
    Check if Ollama server is running.
    
    Returns:
        True if server is responding, False otherwise
    """
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def start_ollama_server(timeout: int = 30) -> None:
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
        if is_ollama_running():
            print("Ollama server started successfully.")
            return
        time.sleep(0.5)
    
    raise RuntimeError(
        f"Ollama server did not start within {timeout} seconds. "
        "Try starting manually: ollama serve"
    )


def ensure_ollama_running() -> None:
    """
    Ensure Ollama server is running, starting it if necessary.
    
    Raises:
        RuntimeError: If server cannot be started
    """
    if not is_ollama_running():
        print("Ollama server not running. Starting...")
        start_ollama_server()


def list_local_models() -> dict[str, dict]:
    """
    Get list of locally available Ollama models.
    
    Returns:
        Dict mapping model names to their info
    """
    import ollama
    
    response = ollama.list()
    models = {}
    
    for model in response.get("models", []):
        name = model.get("name", "")
        # Store both full name and base name
        models[name] = model
        
        # Also index by base name (without tag)
        base_name = name.split(":")[0]
        if base_name not in models:
            models[base_name] = model
    
    return models


def is_model_available(model_name: str, local_models: dict) -> bool:
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


def download_model(model_name: str, size_estimate: str = "unknown size") -> None:
    """
    Download a model using Ollama.
    
    Args:
        model_name: Name of model to download
        size_estimate: Human-readable size estimate for display
    """
    import ollama
    
    print(f"Downloading {model_name} ({size_estimate})...")
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
