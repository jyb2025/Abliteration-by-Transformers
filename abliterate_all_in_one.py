#!/usr/bin/env python3
"""
Abliteration - All-in-one script for removing refusal directions from LLMs

This script combines all the functionality from the abliteration codebase into a single file.
It allows you to calculate the most significant refusal directions with harmful and harmless prompts,
remove them from the model, chat with the model, and compare original and abliterated models.

Usage:
    # Abliterate a model
    python abliterate_all_in_one.py abliterate -m <model_path> -o <output_dir>

    # Chat with a model
    python abliterate_all_in_one.py chat -m <model_path>

    # Compare original and abliterated models
    python abliterate_all_in_one.py compare --original <original_model_path> --abliterated <abliterated_model_path>

Examples:
    # Abliterate a model
    python abliterate_all_in_one.py abliterate -m meta-llama/Llama-3.2-3B-Instruct -o llama3.2-3b-abliterated

    # Chat with the abliterated model
    python abliterate_all_in_one.py chat -m llama3.2-3b-abliterated

    # Compare the original and abliterated models
    python abliterate_all_in_one.py compare --original meta-llama/Llama-3.2-3B-Instruct --abliterated llama3.2-3b-abliterated --output-file comparison_report.md
"""

import gc
import sys
import json
import torch
import random
import psutil
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any

# Import transformers components
try:
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        BitsAndBytesConfig,
        TextStreamer,
    )
except ImportError:
    print("Error: Required libraries not found. Please install them using:")
    print("pip install transformers datasets torch tqdm pandas psutil")
    sys.exit(1)

# ====================== GPU Utilities ======================

def get_gpu_info() -> List[Dict[str, Union[int, str, float]]]:
    """Get information about available GPUs and their memory"""
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            total_memory = gpu_props.total_memory / (1024 ** 3)  # Convert to GB
            # Get free memory
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Convert to GB

            gpu_info.append({
                'id': i,
                'name': gpu_props.name,
                'total_memory': total_memory,
                'free_memory': free_memory
            })
    return gpu_info

def find_best_gpu_for_tensor(tensor_size_gb: float, gpu_info: List[Dict[str, Union[int, str, float]]]) -> Optional[str]:
    """Find the best GPU to place a tensor based on available memory"""
    if not gpu_info:
        return None

    # Sort GPUs by free memory (descending)
    sorted_gpus = sorted(gpu_info, key=lambda x: x['free_memory'], reverse=True)

    # Find a GPU with enough free memory
    for gpu in sorted_gpus:
        if gpu['free_memory'] >= tensor_size_gb:
            return f"cuda:{gpu['id']}"

    # If no GPU has enough memory, use the one with the most free memory
    return f"cuda:{sorted_gpus[0]['id']}"


def print_system_info() -> None:
    """Print information about the system"""
    print("\n=== System Information ===")

    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"CPU: {cpu_count} physical cores, {cpu_count_logical} logical cores")

    # Memory info
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")

    # GPU info
    if torch.cuda.is_available():
        print(f"\n=== GPU Information ===")
        gpu_info = get_gpu_info()
        for gpu in gpu_info:
            print(f"GPU {gpu['id']}: {gpu['name']}")
            print(f"  Total memory: {gpu['total_memory']:.2f} GB")
            print(f"  Free memory: {gpu['free_memory']:.2f} GB")
    else:
        print("\nNo GPUs detected")
    print("========================\n")

# ====================== Data Loading ======================

def load_data(path: str) -> List[str]:
    """Load data from various file formats"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        data = df.get("text")
        if data is None:
            raise ValueError("No 'text' column found in parquet file")
        return data.tolist()
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list")
        return data
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

# ====================== Abliteration Core Functions ======================

def extract_hidden_states(raw_output: Any) -> Dict[str, List[List[torch.Tensor]]]:
    """Extract hidden states from model output and move them to CPU"""
    processed = {}

    assert hasattr(raw_output, "hidden_states"), "Model output doesn't have hidden_states"
    cpu_hidden = []
    for layer_output in raw_output.hidden_states:
        layer_tensors = []
        for tensor in layer_output:
            assert isinstance(tensor, torch.Tensor)
            layer_tensors.append(tensor.to("cpu"))
        cpu_hidden.append(layer_tensors)
    processed["hidden_states"] = cpu_hidden

    return processed

def compute_refusals(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    harmful_list: List[str],
    harmless_list: List[str],
    layer_fraction: float = 0.6,
) -> torch.Tensor:
    """Compute refusal directions using harmful and harmless prompts"""

    def welford(tokens_with_masks: List[Tuple[torch.Tensor, torch.Tensor]], desc: str) -> torch.Tensor:
        """Compute mean using Welford's online algorithm"""
        mean = None
        count = 0
        print(f"Processing {len(tokens_with_masks)} {desc} tokens...")

        for i, (token, attention_mask) in enumerate(tqdm(tokens_with_masks, desc=desc)):
            # Generate with attention mask
            raw_output = model.generate(
                token.to(model.device),
                attention_mask=attention_mask.to(model.device),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            cpu_output = extract_hidden_states(raw_output)

            # Debug the hidden states structure only for the first token
            if i == 0:
                print(f"Hidden states: {len(cpu_output['hidden_states'])} layers, using layer {layer_idx}")

            # Extract hidden states for the specified layer and position
            current_hidden = cpu_output["hidden_states"][0][layer_idx][:, pos, :]
            assert isinstance(current_hidden, torch.Tensor)
            current_hidden = current_hidden.detach()

            # Ensure consistent dtype (float32) for calculations
            if current_hidden.dtype != torch.float32:
                current_hidden = current_hidden.to(torch.float32)

            batch_size = current_hidden.size(dim=0)
            total_count = count + batch_size

            # Initialize mean or update using Welford's algorithm
            if mean is None:
                mean = current_hidden.mean(dim=0)
            else:
                # Ensure dimensions match before subtraction
                if mean.dim() == 1 and current_hidden.dim() == 2:
                    delta = current_hidden - mean.unsqueeze(0)
                else:
                    delta = current_hidden - mean
                mean = mean + (delta.sum(dim=0)) / total_count

            count = total_count

            # Clean up to save memory
            del raw_output, cpu_output, current_hidden
            torch.cuda.empty_cache()

        assert mean is not None

        # Check for NaN or Inf values
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"WARNING: NaN or Inf values detected in {desc} mean! Norm: {mean.norm()}")

        # Ensure the mean is properly shaped (should be 1D)
        if mean.dim() > 1:
            mean = mean.view(-1)

        return mean

    # Tokenize inputs with proper attention masks
    harmful_tokens = []
    for inst in harmful_list:
        inputs = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.ones_like(inputs)
        # Store both the input tokens and attention mask
        harmful_tokens.append((inputs, attention_mask))

    harmless_tokens = []
    for inst in harmless_list:
        inputs = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.ones_like(inputs)
        # Store both the input tokens and attention mask
        harmless_tokens.append((inputs, attention_mask))

    torch.cuda.empty_cache()
    gc.collect()

    # Determine which layer to use based on layer_fraction
    layer_idx = int(len(model.model.layers) * layer_fraction)
    pos = -1  # Use the last token position

    # Compute means for harmful and harmless prompts
    print("Computing harmful prompt representations...")
    harmful_mean = welford(harmful_tokens, "harmful")
    gc.collect()

    print("Computing harmless prompt representations...")
    harmless_mean = welford(harmless_tokens, "harmless")

    # Ensure both means have the same shape and are 1D
    if harmful_mean.dim() > 1:
        harmful_mean = harmful_mean.view(-1)
    if harmless_mean.dim() > 1:
        harmless_mean = harmless_mean.view(-1)

    # Ensure both means have the same dtype (float32)
    harmful_mean = harmful_mean.to(torch.float32)
    harmless_mean = harmless_mean.to(torch.float32)

    print(f"Mean vector dimensions: {harmful_mean.shape}")
    print(f"Mean vector dtype: {harmful_mean.dtype}")

    # Verify that the means are different
    mean_diff_norm = (harmful_mean - harmless_mean).norm().item()
    print(f"Difference between harmful and harmless means: {mean_diff_norm:.6f}")

    if mean_diff_norm < 1e-4:
        print("WARNING: The difference between harmful and harmless means is very small!")
        print("This may indicate that the model doesn't distinguish between these prompt types.")
        print("Try using more diverse prompts or adjusting the layer_fraction parameter.")

    # Compute refusal direction (harmful - harmless)
    refusal_dir = harmful_mean - harmless_mean

    # Check for NaN or Inf values
    if torch.isnan(refusal_dir).any() or torch.isinf(refusal_dir).any():
        print("WARNING: NaN or Inf values detected in refusal direction!")
        # Replace NaN/Inf values with zeros to prevent further issues
        refusal_dir = torch.nan_to_num(refusal_dir, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize the refusal direction
    refusal_norm = refusal_dir.norm()
    if refusal_norm > 0:
        refusal_dir = refusal_dir / refusal_norm
        print(f"Refusal direction computed and normalized. Norm: {refusal_dir.norm():.6f}")
    else:
        print("ERROR: Refusal direction has zero norm! Using random direction instead.")
        # Generate a random direction as fallback
        refusal_dir = torch.randn_like(refusal_dir)
        refusal_dir = refusal_dir / refusal_dir.norm()

    # Print information about the model architecture to help with debugging
    print("\nModel architecture information:")
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer0 = model.model.layers[0]
        if hasattr(layer0, "self_attn") and hasattr(layer0.self_attn, "o_proj"):
            weight_shape = layer0.self_attn.o_proj.weight.shape
            print(f"  Layer 0 attention output projection shape: {weight_shape}")
            print(f"  Refusal direction shape: {refusal_dir.shape}")
            print(f"  Refusal direction dtype: {refusal_dir.dtype}")

            # Check if we need to reshape the refusal direction to match the model architecture
            if weight_shape[0] != refusal_dir.shape[0]:
                print(f"  WARNING: Refusal direction shape ({refusal_dir.shape[0]}) doesn't match "
                      f"the model's hidden size ({weight_shape[0]})")
                print("  This may cause issues during abliteration.")

    # Ensure the refusal direction is in float32 format for consistent calculations
    refusal_dir = refusal_dir.to(torch.float32)
    print(f"Final refusal direction dtype: {refusal_dir.dtype}")

    return refusal_dir

def modify_tensor(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    """Modify a tensor by removing the refusal direction

    Args:
        tensor_data: The tensor to modify (e.g., a weight matrix)
        refusal_dir: The refusal direction tensor
        scale_factor: Scale factor for abliteration (higher = more abliteration)

    Returns:
        Modified tensor as a nn.Parameter
    """
    # Skip modification if scale factor is zero
    if scale_factor == 0.0:
        print("  Scale factor is zero, returning original tensor unchanged")
        return torch.nn.Parameter(tensor_data)

    # Get the original device and dtype
    original_device = tensor_data.device
    original_dtype = tensor_data.dtype

    # For multi-GPU setups, move both tensors to CPU first to avoid CUDA errors
    tensor_data_cpu = tensor_data.detach().to("cpu")
    refusal_dir_cpu = refusal_dir.detach().to("cpu")

    # Convert to float32 for better precision during computation
    # This is crucial for consistent results across different model dtypes
    tensor_float32 = tensor_data_cpu.to(torch.float32)
    refusal_dir_float32 = refusal_dir_cpu.to(torch.float32)

    # Ensure refusal_dir is a 1-dimensional tensor
    if refusal_dir_float32.dim() > 1:
        print(f"  Reshaping refusal direction from {refusal_dir_float32.shape} to 1D")
        refusal_dir_float32 = refusal_dir_float32.view(-1)

    # Store original tensor for verification
    original_tensor = tensor_float32.clone()

    # Get tensor dimensions
    tensor_shape = tensor_float32.shape

    # Print debug info
    print(f"  Tensor shape: {tensor_shape}, dtype: {original_dtype}")
    print(f"  Refusal direction shape: {refusal_dir_float32.shape}, norm: {refusal_dir_float32.norm().item():.6f}")
    print(f"  Scale factor: {scale_factor}")

    # Apply the abliteration formula on CPU
    try:
        # For 2D weight matrices (most common case)
        if tensor_float32.dim() == 2:
            # First try to match the first dimension (output dimension)
            if refusal_dir_float32.shape[0] == tensor_float32.shape[0]:
                # Case 1: Refusal direction matches the first dimension (output dimension)
                # This is the case for most models where the weight matrix is [hidden_size, intermediate_size]
                print(f"  Using row-wise abliteration (output dimension)")

                # Create projection matrix for row-wise abliteration
                outer_product = torch.outer(refusal_dir_float32, refusal_dir_float32)

                # Apply abliteration: W' = W - s * (r⊗r)W
                # This removes the component of each row in the direction of r
                abliteration_term = torch.matmul(outer_product, tensor_float32)
                tensor_modified = tensor_float32 - scale_factor * abliteration_term

            # Then try to match the second dimension (input dimension)
            elif refusal_dir_float32.shape[0] == tensor_float32.shape[1]:
                # Case 2: Refusal direction matches the second dimension (input dimension)
                print(f"  Using column-wise abliteration (input dimension)")

                # Create projection matrix for column-wise abliteration
                outer_product = torch.outer(refusal_dir_float32, refusal_dir_float32)

                # Apply abliteration: W' = W - s * W(r⊗r)
                # This removes the component of each column in the direction of r
                abliteration_term = torch.matmul(tensor_float32, outer_product)
                tensor_modified = tensor_float32 - scale_factor * abliteration_term

            # If neither dimension matches, resize the refusal direction
            else:
                print(f"  WARNING: Refusal direction shape {refusal_dir_float32.shape[0]} doesn't match "
                      f"either dimension of tensor {tensor_float32.shape}")

                # Prefer to match the first dimension (output dimension) if possible
                target_dim = tensor_float32.shape[0]
                print(f"  Resizing refusal direction to match first dimension ({target_dim})")

                # Create a new refusal direction with the target dimension
                resized_refusal = torch.zeros(target_dim, dtype=torch.float32)

                # Copy as much of the original refusal direction as possible
                min_size = min(refusal_dir_float32.shape[0], target_dim)
                resized_refusal[:min_size] = refusal_dir_float32[:min_size]

                # Normalize the resized refusal direction
                resized_norm = resized_refusal.norm()
                if resized_norm > 1e-6:  # Use a small threshold to avoid division by very small numbers
                    resized_refusal = resized_refusal / resized_norm
                    print(f"  Resized refusal direction normalized. New norm: {resized_refusal.norm().item():.6f}")
                else:
                    print("  ERROR: Resized refusal direction has near-zero norm! Using random direction instead.")
                    # Generate a random direction as fallback
                    resized_refusal = torch.randn(target_dim, dtype=torch.float32)
                    resized_refusal = resized_refusal / resized_refusal.norm()

                # Create projection matrix for row-wise abliteration with resized refusal direction
                outer_product = torch.outer(resized_refusal, resized_refusal)

                # Apply abliteration with resized refusal direction
                abliteration_term = torch.matmul(outer_product, tensor_float32)
                tensor_modified = tensor_float32 - scale_factor * abliteration_term

        # For tensors with other dimensions (3D, 4D, etc.)
        else:
            # For tensors with other dimensions, reshape, apply abliteration, then reshape back
            original_shape = tensor_float32.shape
            print(f"  Handling tensor with {len(original_shape)} dimensions")

            # Reshape to 2D: [*, last_dim]
            flattened = tensor_float32.reshape(-1, original_shape[-1])
            print(f"  Reshaped to {flattened.shape} for processing")

            # Check if refusal direction matches the last dimension
            if refusal_dir_float32.shape[0] != original_shape[-1]:
                print(f"  WARNING: Refusal direction shape {refusal_dir_float32.shape[0]} doesn't match "
                      f"last dimension of tensor {original_shape[-1]}")

                # Resize refusal direction to match the last dimension
                target_dim = original_shape[-1]
                print(f"  Resizing refusal direction to match last dimension ({target_dim})")

                # Create a new refusal direction with the target dimension
                resized_refusal = torch.zeros(target_dim, dtype=torch.float32)

                # Copy as much of the original refusal direction as possible
                min_size = min(refusal_dir_float32.shape[0], target_dim)
                if refusal_dir_float32.shape[0] > target_dim:
                    # Truncate
                    resized_refusal = refusal_dir_float32[:target_dim]
                else:
                    # Pad
                    resized_refusal[:min_size] = refusal_dir_float32[:min_size]

                # Normalize the resized refusal direction
                resized_norm = resized_refusal.norm()
                if resized_norm > 1e-6:
                    resized_refusal = resized_refusal / resized_norm
                    print(f"  Resized refusal direction normalized. New norm: {resized_refusal.norm().item():.6f}")
                else:
                    print("  ERROR: Resized refusal direction has near-zero norm! Using random direction instead.")
                    # Generate a random direction as fallback
                    resized_refusal = torch.randn(target_dim, dtype=torch.float32)
                    resized_refusal = resized_refusal / resized_refusal.norm()

                # Use the resized refusal direction
                refusal_dir_float32 = resized_refusal

            # Create projection matrix
            outer_product = torch.outer(refusal_dir_float32, refusal_dir_float32)

            # Apply abliteration to the flattened tensor
            # We're treating each row as a separate vector to project
            abliteration_term = torch.matmul(flattened, outer_product)
            modified_flattened = flattened - scale_factor * abliteration_term

            # Reshape back to original shape
            tensor_modified = modified_flattened.reshape(original_shape)

    except RuntimeError as e:
        print(f"  ERROR during abliteration: {e}")
        print("  Returning original tensor unchanged")
        return torch.nn.Parameter(tensor_data)

    # Check for NaN or Inf values and fix them
    if torch.isnan(tensor_modified).any() or torch.isinf(tensor_modified).any():
        print("  WARNING: NaN or Inf values detected in modified tensor! Fixing...")
        tensor_modified = torch.nan_to_num(tensor_modified, nan=0.0, posinf=1.0, neginf=-1.0)

    # Verify that the tensor actually changed
    diff_norm = (tensor_modified - original_tensor).norm().item()
    original_norm = original_tensor.norm().item()
    diff_relative = (diff_norm / original_norm) * 100 if original_norm > 0 else 0

    print(f"  Layer modified: Change magnitude = {diff_norm:.6f} ({diff_relative:.4f}%)")

    if diff_norm < 1e-6:
        print("  WARNING: Tensor barely changed! Abliteration may not be effective.")
        print("  Consider using a larger scale factor.")
    elif diff_relative > 50:
        print("  WARNING: Very large change detected! This might affect model performance.")
        print("  Consider using a smaller scale factor.")

    # Convert back to original dtype and device
    tensor_modified = tensor_modified.to(original_dtype).to(original_device)

    # Clean up
    del tensor_data_cpu, refusal_dir_cpu, tensor_float32, refusal_dir_float32
    del original_tensor
    if 'outer_product' in locals():
        del outer_product
    if 'abliteration_term' in locals():
        del abliteration_term
    torch.cuda.empty_cache()
    gc.collect()

    return torch.nn.Parameter(tensor_modified)

def analyze_layer_refusal_factors(
    model: PreTrainedModel,
    refusal_dir: torch.Tensor,
    num_layers_to_check: int = None,
) -> List[Tuple[int, float]]:
    """
    Analyze each layer to determine its refusal factor (how much it contributes to refusal)
    Returns a list of (layer_idx, refusal_factor) tuples sorted by refusal factor (highest first)
    """
    # Access the model's transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        lm_model = model.model
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # For models like Qwen, etc.
        lm_model = model.transformer
        lm_model.layers = lm_model.h  # Alias for compatibility
    else:
        raise ValueError("Model architecture not supported. Could not find layers.")

    num_layers = len(lm_model.layers)
    if num_layers_to_check is None:
        num_layers_to_check = num_layers
    else:
        num_layers_to_check = min(num_layers_to_check, num_layers)

    print(f"Analyzing refusal factors for {num_layers_to_check} layers...")

    # Ensure refusal_dir is on CPU and properly shaped and converted to float32
    refusal_dir_cpu = refusal_dir.detach().to("cpu").to(torch.float32)
    print(f"Refusal direction dtype after conversion: {refusal_dir_cpu.dtype}")

    if refusal_dir_cpu.dim() > 1:
        refusal_dir_cpu = refusal_dir_cpu.view(-1)

    layer_refusal_factors = []
    detailed_factors = {}  # Store detailed breakdown of factors

    for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
        try:
            layer = lm_model.layers[layer_idx]
            layer_factor = 0.0
            layer_details = {}

            # Check self-attention output projection
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                # Get weight matrix
                weight = layer.self_attn.o_proj.weight.data.detach().to("cpu").to(torch.float32)

                # For attention weights, we need to handle different model architectures
                # Some models have weights of shape [hidden_size, hidden_size] and others [hidden_size, intermediate_size]
                # The refusal direction should match the first dimension (output dimension)
                if refusal_dir_cpu.shape[0] == weight.shape[0]:
                    # For this case, we need to project each row of the weight matrix onto the refusal direction
                    # This gives us a measure of how much each output dimension aligns with the refusal direction
                    projection = torch.matmul(refusal_dir_cpu, weight)
                    # Sum of squared projections gives us a measure of alignment
                    attn_factor = (projection ** 2).sum().item()
                    layer_factor += attn_factor
                    layer_details["attn_factor"] = attn_factor
                else:
                    print(f"Warning: Dimension mismatch in layer {layer_idx} attention. "
                          f"Weight shape: {weight.shape}, Refusal dir shape: {refusal_dir_cpu.shape}")
                    print(f"  Expected refusal direction of size {weight.shape[0]}")
                    layer_details["attn_factor"] = 0.0

            # Check MLP down projection
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                # Get weight matrix
                weight = layer.mlp.down_proj.weight.data.detach().to("cpu").to(torch.float32)

                # For MLP weights, similar to attention, we need to match the output dimension
                if refusal_dir_cpu.shape[0] == weight.shape[0]:
                    # Project each row of the weight matrix onto the refusal direction
                    projection = torch.matmul(refusal_dir_cpu, weight)
                    # Sum of squared projections gives us a measure of alignment
                    mlp_factor = (projection ** 2).sum().item()
                    layer_factor += mlp_factor
                    layer_details["mlp_factor"] = mlp_factor
                else:
                    print(f"Warning: Dimension mismatch in layer {layer_idx} MLP. "
                          f"Weight shape: {weight.shape}, Refusal dir shape: {refusal_dir_cpu.shape}")
                    print(f"  Expected refusal direction of size {weight.shape[0]}")
                    layer_details["mlp_factor"] = 0.0

            # Alternative architectures
            if hasattr(layer, "attention") and hasattr(layer.attention, "dense"):
                weight = layer.attention.dense.weight.data.detach().to("cpu").to(torch.float32)
                if refusal_dir_cpu.shape[0] == weight.shape[0]:
                    projection = torch.matmul(refusal_dir_cpu, weight)
                    attn_factor = (projection ** 2).sum().item()
                    layer_factor += attn_factor
                    layer_details["alt_attn_factor"] = attn_factor
                else:
                    print(f"Warning: Dimension mismatch in layer {layer_idx} alternative attention. "
                          f"Weight shape: {weight.shape}, Refusal dir shape: {refusal_dir_cpu.shape}")
                    print(f"  Expected refusal direction of size {weight.shape[0]}")

            if hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"):
                weight = layer.mlp.c_proj.weight.data.detach().to("cpu").to(torch.float32)
                if refusal_dir_cpu.shape[0] == weight.shape[0]:
                    projection = torch.matmul(refusal_dir_cpu, weight)
                    mlp_factor = (projection ** 2).sum().item()
                    layer_factor += mlp_factor
                    layer_details["alt_mlp_factor"] = mlp_factor
                else:
                    print(f"Warning: Dimension mismatch in layer {layer_idx} alternative MLP. "
                          f"Weight shape: {weight.shape}, Refusal dir shape: {refusal_dir_cpu.shape}")
                    print(f"  Expected refusal direction of size {weight.shape[0]}")

            layer_refusal_factors.append((layer_idx, layer_factor))
            detailed_factors[layer_idx] = layer_details

            # Clean up to save memory
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error analyzing layer {layer_idx}: {e}")
            layer_refusal_factors.append((layer_idx, 0.0))
            detailed_factors[layer_idx] = {"error": str(e)}

    # Sort by refusal factor (highest first)
    layer_refusal_factors.sort(key=lambda x: x[1], reverse=True)

    # Print detailed breakdown for top layers
    print("\nDetailed refusal factor breakdown for top 10 layers:")
    for layer_idx, factor in layer_refusal_factors[:10]:
        details = detailed_factors[layer_idx]
        detail_str = ", ".join([f"{k}: {v:.6f}" for k, v in details.items() if not isinstance(v, str)])
        print(f"  Layer {layer_idx}: Total={factor:.6f} ({detail_str})")

    # Check if any layers have zero or very small factors
    zero_factors = sum(1 for _, factor in layer_refusal_factors if factor < 1e-6)
    all_negligible = all(factor < 1e-6 for _, factor in layer_refusal_factors)

    if zero_factors > 0:
        print(f"\nWARNING: {zero_factors} layers have negligible refusal factors.")
        print("This might indicate an issue with the refusal direction calculation.")

    if all_negligible:
        print("\nCRITICAL WARNING: ALL layers have negligible refusal factors!")
        print("This indicates a serious issue with the refusal direction calculation.")
        print("Abliteration should not proceed with these values as it may cause unintended changes.")

    return layer_refusal_factors

def apply_abliteration(
    model: PreTrainedModel,
    refusal_dir: torch.Tensor,
    skip_begin_layers: int = 1,
    skip_end_layers: int = 0,
    scale_factor: float = 1.0,
    top_refusal_layers: int = None,
    specific_layers: List[int] = None,
    min_refusal_factor: float = None,
    refusal_threshold: float = 1e-6,
    force_abliteration: bool = False,
    proportional_scaling: bool = False,
    max_scale_factor: float = None,
) -> PreTrainedModel:
    """Apply abliteration to model layers

    Args:
        model: The model to abliterate
        refusal_dir: The refusal direction tensor
        skip_begin_layers: Number of layers to skip at the beginning
        skip_end_layers: Number of layers to skip at the end
        scale_factor: Scale factor for abliteration
        top_refusal_layers: Only abliterate the N layers with highest refusal factors
        specific_layers: List of specific layer indices to abliterate (overrides top_refusal_layers)
        min_refusal_factor: Only abliterate layers with refusal factor >= this value
        refusal_threshold: Minimum refusal factor threshold to consider a layer for abliteration
        force_abliteration: Force abliteration even when refusal factors are negligible
        proportional_scaling: If True, scale the abliteration factor proportionally to each layer's refusal factor
        max_scale_factor: Maximum scale factor to use when proportional_scaling is True (defaults to scale_factor)
    """
    # Access the model's transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        lm_model = model.model
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # For models like GPT-2, Qwen, etc.
        lm_model = model.transformer
        lm_model.layers = lm_model.h  # Alias for compatibility
    else:
        raise ValueError("Model architecture not supported. Could not find layers.")

    num_layers = len(lm_model.layers)

    # Always analyze layers to find refusal factors
    print("Analyzing layer refusal factors...")
    layer_refusal_factors = analyze_layer_refusal_factors(model, refusal_dir)

    # Check if all refusal factors are below the threshold
    all_negligible = all(factor < refusal_threshold for _, factor in layer_refusal_factors)
    if all_negligible and not force_abliteration:
        print(f"\nABLITERATION ABORTED: All layers have refusal factors below threshold ({refusal_threshold}).")
        print("Making changes to the model when all refusal factors are negligible could result")
        print("in random, unintended modifications that don't target refusal behavior.")
        print("Please check your refusal direction calculation or lower the refusal threshold.")
        print("You can use --refusal-threshold to set a lower threshold if needed.")
        print("Alternatively, use --force-abliteration to proceed anyway (use with caution).")
        return model
    elif all_negligible and force_abliteration:
        print(f"\nWARNING: All layers have refusal factors below threshold ({refusal_threshold}).")
        print("Proceeding with abliteration anyway because --force-abliteration is enabled.")
        print("This may result in random, unintended modifications that don't target refusal behavior.")

    # Print refusal factors for all layers
    print("\nLayer refusal factors (higher = more contribution to refusal):")
    for layer_idx, factor in layer_refusal_factors[:20]:  # Show top 20
        print(f"  Layer {layer_idx}: {factor:.6f}")

    # Determine which layers to process based on the provided parameters
    if specific_layers is not None:
        # Use specific layers provided by the user
        print(f"\nUsing user-specified layers: {specific_layers}")
        # Filter out invalid layer indices
        layers_to_process = [layer_idx for layer_idx in specific_layers
                            if layer_idx >= skip_begin_layers and layer_idx < num_layers - skip_end_layers]

        if not layers_to_process:
            print("WARNING: No valid layers found in the specified list!")
            print("Falling back to using all layers in the specified range.")
            layers_to_process = list(range(skip_begin_layers, num_layers - skip_end_layers))
        else:
            print(f"Selected {len(layers_to_process)} valid layers for abliteration: {layers_to_process}")

    elif min_refusal_factor is not None:
        # Filter layers based on minimum refusal factor
        valid_layer_factors = [(layer_idx, factor) for layer_idx, factor in layer_refusal_factors
                              if layer_idx >= skip_begin_layers and
                                 layer_idx < num_layers - skip_end_layers and
                                 factor >= min_refusal_factor]

        layers_to_process = [layer_idx for layer_idx, _ in valid_layer_factors]

        if not layers_to_process:
            print(f"WARNING: No layers found with refusal factor >= {min_refusal_factor}!")
            print("Falling back to using top 5 layers by refusal factor.")
            # Use top 5 layers instead
            valid_layer_factors = [(layer_idx, factor) for layer_idx, factor in layer_refusal_factors
                                  if layer_idx >= skip_begin_layers and layer_idx < num_layers - skip_end_layers]
            top_valid_layers = valid_layer_factors[:min(5, len(valid_layer_factors))]
            layers_to_process = [layer_idx for layer_idx, _ in top_valid_layers]
        else:
            print(f"\nSelected {len(layers_to_process)} layers with refusal factor >= {min_refusal_factor}: {layers_to_process}")

    elif top_refusal_layers is not None and top_refusal_layers > 0:
        # Filter valid layers first (respect skip_begin and skip_end)
        valid_layer_factors = [(layer_idx, factor) for layer_idx, factor in layer_refusal_factors
                              if layer_idx >= skip_begin_layers and layer_idx < num_layers - skip_end_layers]

        # Take only the top N valid layers with highest refusal factors
        top_valid_layers = valid_layer_factors[:min(top_refusal_layers, len(valid_layer_factors))]

        # Extract just the layer indices
        layers_to_process = [layer_idx for layer_idx, _ in top_valid_layers]

        if not layers_to_process:
            print("WARNING: No valid layers with high refusal factors found!")
            print("Falling back to using all layers in the specified range.")
            layers_to_process = list(range(skip_begin_layers, num_layers - skip_end_layers))
        else:
            print(f"\nSelected {len(layers_to_process)} layers for abliteration: {layers_to_process}")
    else:
        # Process all layers in the range
        layers_to_process = list(range(skip_begin_layers, num_layers - skip_end_layers))
        print(f"Model has {num_layers} layers. Processing layers {skip_begin_layers} to {num_layers - skip_end_layers - 1}")

    if not layers_to_process:
        raise ValueError(f"No valid layers to process. Check your layer selection parameters.")

    # Track successful modifications
    successful_mods = 0
    total_attempted = 0

    # Create a dictionary of layer refusal factors for quick lookup
    layer_factor_dict = {layer_idx: factor for layer_idx, factor in layer_refusal_factors}

    # If using proportional scaling, calculate the scaling factors for each layer
    if proportional_scaling:
        # Set default max_scale_factor if not provided
        if max_scale_factor is None:
            max_scale_factor = scale_factor

        # Get the maximum refusal factor among the layers to process
        max_refusal = max(layer_factor_dict.get(layer_idx, 0.0) for layer_idx in layers_to_process)

        if max_refusal > 0:
            # Create a dictionary of proportional scale factors for each layer
            proportional_scale_factors = {}
            for layer_idx in layers_to_process:
                layer_factor = layer_factor_dict.get(layer_idx, 0.0)
                # Scale proportionally to the refusal factor, normalized by the maximum refusal factor
                # This ensures the layer with the highest refusal factor gets the max_scale_factor
                if layer_factor > 0:
                    proportional_scale_factors[layer_idx] = (layer_factor / max_refusal) * max_scale_factor
                else:
                    proportional_scale_factors[layer_idx] = 0.0

            print("\nUsing proportional scaling based on refusal factors:")
            for layer_idx in sorted(proportional_scale_factors.keys(), key=lambda x: proportional_scale_factors[x], reverse=True)[:10]:
                print(f"  Layer {layer_idx}: Refusal factor = {layer_factor_dict.get(layer_idx, 0.0):.6f}, Scale factor = {proportional_scale_factors[layer_idx]:.6f}")
        else:
            print("\nWARNING: Cannot use proportional scaling because all refusal factors are zero.")
            print("Using constant scale factor for all layers instead.")
            proportional_scaling = False

    # Process selected layers with error handling
    for layer_idx in tqdm(layers_to_process, desc="Applying abliteration"):
        print(f"\nProcessing layer {layer_idx}/{num_layers-1}:")

        # Check if this specific layer has a refusal factor below threshold
        layer_factor = layer_factor_dict.get(layer_idx, 0.0)
        if layer_factor < refusal_threshold and not force_abliteration:
            print(f"  SKIPPING: Layer {layer_idx} has refusal factor ({layer_factor:.8f}) below threshold ({refusal_threshold}).")
            print("  Modifying this layer would result in random, unintended changes.")
            continue
        elif layer_factor < refusal_threshold and force_abliteration:
            print(f"  WARNING: Layer {layer_idx} has refusal factor ({layer_factor:.8f}) below threshold ({refusal_threshold}).")
            print("  Proceeding anyway because --force-abliteration is enabled.")

        # Find the right attribute names for this model architecture
        try:
            layer = lm_model.layers[layer_idx]

            # Determine the scale factor to use for this layer
            current_scale_factor = scale_factor
            if proportional_scaling and layer_idx in proportional_scale_factors:
                current_scale_factor = proportional_scale_factors[layer_idx]
                print(f"  Using proportional scale factor: {current_scale_factor:.6f}")

            # Try to identify the output projection in self-attention
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                # Llama-style architecture
                print("  Modifying self-attention output projection...")
                total_attempted += 1
                layer.self_attn.o_proj.weight = modify_tensor(
                    layer.self_attn.o_proj.weight.data,
                    refusal_dir,
                    current_scale_factor,
                )
                successful_mods += 1
            elif hasattr(layer, "attention") and hasattr(layer.attention, "dense"):
                # GPT-2 style architecture
                print("  Modifying attention dense projection...")
                total_attempted += 1
                layer.attention.dense.weight = modify_tensor(
                    layer.attention.dense.weight.data,
                    refusal_dir,
                    current_scale_factor,
                )
                successful_mods += 1

            # Try to identify the MLP down projection
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                # Llama-style architecture
                print("  Modifying MLP down projection...")
                total_attempted += 1
                layer.mlp.down_proj.weight = modify_tensor(
                    layer.mlp.down_proj.weight.data,
                    refusal_dir,
                    current_scale_factor,
                )
                successful_mods += 1
            elif hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"):
                # GPT-2 style architecture
                print("  Modifying MLP c_proj...")
                total_attempted += 1
                layer.mlp.c_proj.weight = modify_tensor(
                    layer.mlp.c_proj.weight.data,
                    refusal_dir,
                    current_scale_factor,
                )
                successful_mods += 1

            # Free memory after each layer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")
            print("Skipping this layer and continuing...")
            continue

    print(f"\nAbliteration complete: {successful_mods}/{total_attempted} modifications successful")

    if successful_mods == 0:
        print("WARNING: No layers were successfully modified! Abliteration failed.")
    elif successful_mods < total_attempted * 0.5:
        print("WARNING: Less than half of attempted modifications succeeded. Abliteration may be incomplete.")

    torch.cuda.empty_cache()
    gc.collect()

    return model

# ====================== Compare Models Function ======================

def compare_models(
    original_model_path: str,
    abliterated_model_path: str,
    precision: str = "fp16",
    device: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    flash_attn: bool = False,
    multi_gpu: bool = False,
    max_memory: str = None,
    output_file: str = None,
    detailed: bool = False,
    threshold: float = 1e-6
) -> None:
    """Compare original and abliterated models to analyze differences

    Args:
        original_model_path: Path to the original model
        abliterated_model_path: Path to the abliterated model
        precision: Precision to use for loading models
        device: Device to load models on
        load_in_4bit: Whether to load models in 4-bit precision
        load_in_8bit: Whether to load models in 8-bit precision
        flash_attn: Whether to use flash attention
        multi_gpu: Whether to distribute models across multiple GPUs
        max_memory: Maximum memory to use per GPU
        output_file: Path to save the comparison report
        detailed: Whether to include detailed per-parameter statistics
        threshold: Threshold for considering a weight change significant
    """
    # Print system information
    print_system_info()

    # Get GPU information for tensor placement
    gpu_info = get_gpu_info()

    # Set precision
    if precision == "fp16":
        precision_dtype = torch.float16
    elif precision == "bf16":
        precision_dtype = torch.bfloat16
    else:
        precision_dtype = torch.float32

    # Configure quantization
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    # Configure device map for multi-GPU setup
    device_map = device
    max_memory_map = None

    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using multi-GPU setup with {torch.cuda.device_count()} GPUs")
        device_map = "auto"  # Always use auto for multi-GPU

        if max_memory:
            # Parse max memory string into a dictionary
            # Use integer indices for GPUs as required by Accelerate library
            memory_parts = max_memory.split(',')
            max_memory_map = {}
            for i, mem in enumerate(memory_parts):
                if i < torch.cuda.device_count():
                    max_memory_map[i] = mem.strip()  # Use integer index instead of "cuda:i"
            max_memory_map["cpu"] = "24GiB"  # Allow CPU offloading if needed
            print(f"Using custom memory limits: {max_memory_map}")
        else:
            # Create automatic memory map based on available GPU memory
            # Use integer indices for GPUs as required by Accelerate library
            max_memory_map = {gpu['id']: f"{int(gpu['free_memory'] * 0.9)}GiB" for gpu in gpu_info}
            max_memory_map["cpu"] = "24GiB"  # Allow CPU offloading if needed
            print(f"Using automatic memory limits: {max_memory_map}")

    # Common model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": precision_dtype,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
        "quantization_config": quant_config,
        "attn_implementation": "flash_attention_2" if flash_attn else None,
    }

    # Add max_memory if we're using multi-GPU
    if max_memory_map is not None:
        model_kwargs["max_memory"] = max_memory_map

    # Load the original model
    print(f"\nLoading original model from {original_model_path}...")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        **model_kwargs
    )
    original_model.requires_grad_(False)

    # Load the abliterated model
    print(f"\nLoading abliterated model from {abliterated_model_path}...")
    abliterated_model = AutoModelForCausalLM.from_pretrained(
        abliterated_model_path,
        **model_kwargs
    )
    abliterated_model.requires_grad_(False)

    # Verify that both models have the same architecture
    if not hasattr(original_model, "model") or not hasattr(original_model.model, "layers"):
        raise ValueError("Original model doesn't have the expected structure with model.layers")

    if not hasattr(abliterated_model, "model") or not hasattr(abliterated_model.model, "layers"):
        raise ValueError("Abliterated model doesn't have the expected structure with model.layers")

    orig_num_layers = len(original_model.model.layers)
    abl_num_layers = len(abliterated_model.model.layers)

    if orig_num_layers != abl_num_layers:
        print(f"WARNING: Models have different number of layers: Original={orig_num_layers}, Abliterated={abl_num_layers}")
        print("Will compare only the common layers.")
        num_layers = min(orig_num_layers, abl_num_layers)
    else:
        num_layers = orig_num_layers

    print(f"\nComparing {num_layers} layers between models...")

    # Initialize data structures for comparison results
    layer_diffs = {}
    total_params = 0
    total_changed_params = 0
    max_diff = 0.0
    max_diff_layer = None
    max_diff_component = None

    # Prepare report
    report = []
    report.append("# Model Comparison Report")
    report.append(f"\nOriginal model: {original_model_path}")
    report.append(f"Abliterated model: {abliterated_model_path}")
    report.append(f"\nNumber of layers: {num_layers}")

    # Compare each layer
    for layer_idx in tqdm(range(num_layers), desc="Comparing layers"):
        orig_layer = original_model.model.layers[layer_idx]
        abl_layer = abliterated_model.model.layers[layer_idx]

        layer_diffs[layer_idx] = {
            "components": {},
            "total_params": 0,
            "changed_params": 0,
            "max_diff": 0.0,
            "max_diff_component": None,
            "avg_diff": 0.0,
            "rel_diff_percent": 0.0
        }

        # Compare attention components
        if hasattr(orig_layer, "self_attn"):
            # Compare attention output projection
            if hasattr(orig_layer.self_attn, "o_proj") and hasattr(abl_layer.self_attn, "o_proj"):
                orig_weight = orig_layer.self_attn.o_proj.weight.data.cpu().float()
                abl_weight = abl_layer.self_attn.o_proj.weight.data.cpu().float()

                # Calculate differences
                diff_tensor = abl_weight - orig_weight
                abs_diff_tensor = torch.abs(diff_tensor)

                # Calculate statistics
                num_params = diff_tensor.numel()
                max_param_diff = abs_diff_tensor.max().item()
                avg_param_diff = abs_diff_tensor.mean().item()
                norm_diff = diff_tensor.norm().item()
                rel_diff = norm_diff / orig_weight.norm().item() * 100

                # Count changed parameters
                changed_mask = abs_diff_tensor > threshold
                num_changed = changed_mask.sum().item()

                # Store component results
                component_name = "self_attn.o_proj"
                layer_diffs[layer_idx]["components"][component_name] = {
                    "num_params": num_params,
                    "num_changed": num_changed,
                    "max_diff": max_param_diff,
                    "avg_diff": avg_param_diff,
                    "norm_diff": norm_diff,
                    "rel_diff_percent": rel_diff
                }

                # Update layer statistics
                layer_diffs[layer_idx]["total_params"] += num_params
                layer_diffs[layer_idx]["changed_params"] += num_changed

                if max_param_diff > layer_diffs[layer_idx]["max_diff"]:
                    layer_diffs[layer_idx]["max_diff"] = max_param_diff
                    layer_diffs[layer_idx]["max_diff_component"] = component_name

                # Update global statistics
                total_params += num_params
                total_changed_params += num_changed

                if max_param_diff > max_diff:
                    max_diff = max_param_diff
                    max_diff_layer = layer_idx
                    max_diff_component = component_name

        # Compare MLP components
        if hasattr(orig_layer, "mlp"):
            # Compare MLP down projection
            if hasattr(orig_layer.mlp, "down_proj") and hasattr(abl_layer.mlp, "down_proj"):
                orig_weight = orig_layer.mlp.down_proj.weight.data.cpu().float()
                abl_weight = abl_layer.mlp.down_proj.weight.data.cpu().float()

                # Calculate differences
                diff_tensor = abl_weight - orig_weight
                abs_diff_tensor = torch.abs(diff_tensor)

                # Calculate statistics
                num_params = diff_tensor.numel()
                max_param_diff = abs_diff_tensor.max().item()
                avg_param_diff = abs_diff_tensor.mean().item()
                norm_diff = diff_tensor.norm().item()
                rel_diff = norm_diff / orig_weight.norm().item() * 100

                # Count changed parameters
                changed_mask = abs_diff_tensor > threshold
                num_changed = changed_mask.sum().item()

                # Store component results
                component_name = "mlp.down_proj"
                layer_diffs[layer_idx]["components"][component_name] = {
                    "num_params": num_params,
                    "num_changed": num_changed,
                    "max_diff": max_param_diff,
                    "avg_diff": avg_param_diff,
                    "norm_diff": norm_diff,
                    "rel_diff_percent": rel_diff
                }

                # Update layer statistics
                layer_diffs[layer_idx]["total_params"] += num_params
                layer_diffs[layer_idx]["changed_params"] += num_changed

                if max_param_diff > layer_diffs[layer_idx]["max_diff"]:
                    layer_diffs[layer_idx]["max_diff"] = max_param_diff
                    layer_diffs[layer_idx]["max_diff_component"] = component_name

                # Update global statistics
                total_params += num_params
                total_changed_params += num_changed

                if max_param_diff > max_diff:
                    max_diff = max_param_diff
                    max_diff_layer = layer_idx
                    max_diff_component = component_name

            # Compare MLP c_proj (for GPT-2 style models)
            elif hasattr(orig_layer.mlp, "c_proj") and hasattr(abl_layer.mlp, "c_proj"):
                orig_weight = orig_layer.mlp.c_proj.weight.data.cpu().float()
                abl_weight = abl_layer.mlp.c_proj.weight.data.cpu().float()

                # Calculate differences
                diff_tensor = abl_weight - orig_weight
                abs_diff_tensor = torch.abs(diff_tensor)

                # Calculate statistics
                num_params = diff_tensor.numel()
                max_param_diff = abs_diff_tensor.max().item()
                avg_param_diff = abs_diff_tensor.mean().item()
                norm_diff = diff_tensor.norm().item()
                rel_diff = norm_diff / orig_weight.norm().item() * 100

                # Count changed parameters
                changed_mask = abs_diff_tensor > threshold
                num_changed = changed_mask.sum().item()

                # Store component results
                component_name = "mlp.c_proj"
                layer_diffs[layer_idx]["components"][component_name] = {
                    "num_params": num_params,
                    "num_changed": num_changed,
                    "max_diff": max_param_diff,
                    "avg_diff": avg_param_diff,
                    "norm_diff": norm_diff,
                    "rel_diff_percent": rel_diff
                }

                # Update layer statistics
                layer_diffs[layer_idx]["total_params"] += num_params
                layer_diffs[layer_idx]["changed_params"] += num_changed

                if max_param_diff > layer_diffs[layer_idx]["max_diff"]:
                    layer_diffs[layer_idx]["max_diff"] = max_param_diff
                    layer_diffs[layer_idx]["max_diff_component"] = component_name

                # Update global statistics
                total_params += num_params
                total_changed_params += num_changed

                if max_param_diff > max_diff:
                    max_diff = max_param_diff
                    max_diff_layer = layer_idx
                    max_diff_component = component_name

        # Calculate average difference for the layer
        if layer_diffs[layer_idx]["total_params"] > 0:
            layer_diffs[layer_idx]["avg_diff"] = layer_diffs[layer_idx]["changed_params"] / layer_diffs[layer_idx]["total_params"]

    # Generate summary report
    report.append("\n## Summary")
    report.append(f"\nTotal parameters compared: {total_params:,}")
    report.append(f"Parameters changed (diff > {threshold}): {total_changed_params:,} ({total_changed_params/total_params*100:.4f}%)")
    report.append(f"Maximum difference: {max_diff:.6f} in Layer {max_diff_layer}, Component {max_diff_component}")

    # Generate layer-by-layer report
    report.append("\n## Layer-by-Layer Analysis")

    # Sort layers by maximum difference
    sorted_layers = sorted(layer_diffs.items(), key=lambda x: x[1]["max_diff"], reverse=True)

    for layer_idx, layer_data in sorted_layers:
        if layer_data["total_params"] == 0:
            continue  # Skip layers with no parameters

        changed_percent = layer_data["changed_params"] / layer_data["total_params"] * 100
        report.append(f"\n### Layer {layer_idx}")
        report.append(f"- Parameters: {layer_data['total_params']:,}")
        report.append(f"- Changed parameters: {layer_data['changed_params']:,} ({changed_percent:.4f}%)")
        report.append(f"- Maximum difference: {layer_data['max_diff']:.6f} in {layer_data['max_diff_component']}")

        if detailed:
            report.append("\n#### Components")
            # Sort components by maximum difference
            sorted_components = sorted(layer_data["components"].items(), key=lambda x: x[1]["max_diff"], reverse=True)

            for component_name, component_data in sorted_components:
                changed_percent = component_data["num_changed"] / component_data["num_params"] * 100
                report.append(f"\n##### {component_name}")
                report.append(f"- Parameters: {component_data['num_params']:,}")
                report.append(f"- Changed parameters: {component_data['num_changed']:,} ({changed_percent:.4f}%)")
                report.append(f"- Maximum difference: {component_data['max_diff']:.6f}")
                report.append(f"- Average difference: {component_data['avg_diff']:.6f}")
                report.append(f"- Relative difference: {component_data['rel_diff_percent']:.4f}%")

    # Print report to console
    print("\n".join(report))

    # Save report to file if requested
    if output_file:
        print(f"\nSaving comparison report to {output_file}...")
        with open(output_file, "w") as f:
            f.write("\n".join(report))
        print(f"Report saved to {output_file}")

# ====================== Chat Function ======================

def chat_with_model(model_path: str, precision: str = "fp16", device: str = "auto",
                   load_in_4bit: bool = False, load_in_8bit: bool = False,
                   flash_attn: bool = False, max_new_tokens: int = 2048,
                   multi_gpu: bool = False, max_memory: str = None) -> None:
    """Interactive chat with a model"""
    # Print system information
    print_system_info()

    # Get GPU information for tensor placement
    gpu_info = get_gpu_info()

    # Set precision
    if precision == "fp16":
        precision_dtype = torch.float16
    elif precision == "bf16":
        precision_dtype = torch.bfloat16
    else:
        precision_dtype = torch.float32

    # Configure quantization
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    # Configure device map for multi-GPU setup
    device_map = device
    max_memory_map = None

    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using multi-GPU setup with {torch.cuda.device_count()} GPUs")
        device_map = "auto"  # Always use auto for multi-GPU

        if max_memory:
            # Parse max memory string into a dictionary
            # Use integer indices for GPUs as required by Accelerate library
            memory_parts = max_memory.split(',')
            max_memory_map = {}
            for i, mem in enumerate(memory_parts):
                if i < torch.cuda.device_count():
                    max_memory_map[i] = mem.strip()  # Use integer index instead of "cuda:i"
            max_memory_map["cpu"] = "24GiB"  # Allow CPU offloading if needed
            print(f"Using custom memory limits: {max_memory_map}")
        else:
            # Create automatic memory map based on available GPU memory
            # Use integer indices for GPUs as required by Accelerate library
            max_memory_map = {gpu['id']: f"{int(gpu['free_memory'] * 0.9)}GiB" for gpu in gpu_info}
            max_memory_map["cpu"] = "24GiB"  # Allow CPU offloading if needed
            print(f"Using automatic memory limits: {max_memory_map}")

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": precision_dtype,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
        "quantization_config": quant_config,
        "attn_implementation": "flash_attention_2" if flash_attn else None,
    }

    # Add max_memory if we're using multi-GPU
    if max_memory_map is not None:
        model_kwargs["max_memory"] = max_memory_map

    # Load the model with proper device distribution
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )

    # Disable gradient computation for inference
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        # If pad token is not set, use eos token as pad token but keep them distinct in the attention mask
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        # If no eos token either, use a special token
        else:
            tokenizer.pad_token = "[PAD]"

    print(f"Tokenizer: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")

    # Print model device information
    if multi_gpu:
        print("\nModel layer distribution:")
        if hasattr(model, "hf_device_map"):
            for layer, device in model.hf_device_map.items():
                print(f"  {layer}: {device}")
        else:
            print("  Device map not available")

    # Start chat loop
    conversation = []
    streamer = TextStreamer(tokenizer)
    print("Type /clear to clear history, /exit to quit.")
    while True:
        prompt = input("User> ")
        if prompt == "/clear":
            conversation = []
            print("! History cleared.")
            continue
        elif prompt == "/exit":
            break
        elif prompt == "":
            print("! Please type a message.")
            continue

        conversation.append({"role": "user", "content": prompt})

        # Apply chat template and create input tensors
        toks = tokenizer.apply_chat_template(
            conversation=conversation, add_generation_prompt=True, return_tensors="pt"
        )

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.ones_like(toks)

        # Determine the device to use for inputs
        input_device = model.device
        if multi_gpu and hasattr(model, "hf_device_map"):
            # In multi-GPU setup, use the device of the first layer
            for key, device in model.hf_device_map.items():
                if 'model.embed_tokens' in key or 'transformer.wte' in key or 'model.layers.0' in key:
                    input_device = device
                    break

        # Move inputs to the appropriate device
        toks = toks.to(input_device)
        attention_mask = attention_mask.to(input_device)

        # Generate response
        with torch.no_grad():
            gen = model.generate(
                toks,
                attention_mask=attention_mask,
                streamer=streamer,
                max_new_tokens=max_new_tokens
            )

        # Decode the generated tokens
        decoded = tokenizer.batch_decode(
            gen[0][len(toks[0]) :], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": "".join(decoded)})

# ====================== Command Line Interface ======================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Abliteration - Remove refusal directions from LLMs")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Abliterate command
    abliterate_parser = subparsers.add_parser("abliterate", help="Abliterate a model")

    # Basic options
    abliterate_parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Your model directory or huggingface model ID"
    )
    abliterate_parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory"
    )

    # Device options
    device_group = abliterate_parser.add_argument_group('Device Options')
    device_group.add_argument(
        "--device", "-d", type=str, choices=["auto", "cuda", "cpu", "last-gpu"],
        default="auto",
        help="Target device to process abliteration. 'last-gpu' will use the last available GPU."
    )
    device_group.add_argument(
        "--gpu-id", type=int, default=None,
        help="Specific GPU ID to use (overrides --device)"
    )
    device_group.add_argument(
        "--multi-gpu", action="store_true", default=False,
        help="Distribute model across multiple GPUs"
    )
    device_group.add_argument(
        "--max-memory", type=str, default=None,
        help="Maximum memory to use per GPU, e.g. '40GiB,40GiB,40GiB,40GiB,40GiB,40GiB'"
    )

    # Precision options
    abliterate_parser.add_argument(
        "--precision", "-p", type=str, choices=["fp16", "bf16", "fp32"],
        default="bf16", help="Precision to use for ablation, default is bf16"
    )

    # Model configuration
    abliterate_parser.add_argument(
        "--skip-begin", type=int, default=1,
        help="Number of layers to skip at the beginning. Defaults to 1 to avoid messing with the first layer"
    )
    abliterate_parser.add_argument(
        "--skip-end", type=int, default=0, help="Number of layers to skip at the end"
    )
    abliterate_parser.add_argument(
        "--layer-fraction", type=float, default=1.0,
        help="Fraction of layers to use for refusal_dir calculation"
    )
    abliterate_parser.add_argument(
        "--scale-factor", type=float, default=1.0,
        help="Scale factor for ablation. Use a negative scale-factor to encourage refusal."
    )
    abliterate_parser.add_argument(
        "--top-refusal-layers", type=int, default=None,
        help="Only abliterate the N layers with highest refusal factors. If not specified, all layers are processed."
    )
    abliterate_parser.add_argument(
        "--specific-layers", type=str, default=None,
        help="Comma-separated list of specific layer indices to abliterate (e.g., '5,10,15,20'). Overrides --top-refusal-layers."
    )
    abliterate_parser.add_argument(
        "--min-refusal-factor", type=float, default=None,
        help="Only abliterate layers with refusal factor >= this value."
    )
    abliterate_parser.add_argument(
        "--refusal-threshold", type=float, default=1e-6,
        help="Minimum refusal factor threshold to consider a layer for abliteration. Default: 1e-6"
    )
    abliterate_parser.add_argument(
        "--force-abliteration", action="store_true", default=False,
        help="Force abliteration even when refusal factors are negligible (use with caution)"
    )
    abliterate_parser.add_argument(
        "--proportional-scaling", action="store_true", default=False,
        help="Scale abliteration proportionally to each layer's refusal factor"
    )
    abliterate_parser.add_argument(
        "--max-scale-factor", type=float, default=None,
        help="Maximum scale factor to use when proportional-scaling is enabled (defaults to scale-factor)"
    )
    abliterate_parser.add_argument(
        "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
    )

    # Data options
    data_group = abliterate_parser.add_argument_group('Data Options')
    data_group.add_argument(
        "--data-harmful", "-dhf", type=str, default=None, help="Harmful prompts file"
    )
    data_group.add_argument(
        "--data-harmless", "-dhl", type=str, default=None, help="Harmless prompts file"
    )
    data_group.add_argument(
        "--deccp", action="store_true", default=False,
        help="For Chinese models, in specific topics"
    )
    data_group.add_argument(
        "--num-harmful", "-nhf", type=int, default=-1,
        help="Number of harmful calibrations to randomly select"
    )
    data_group.add_argument(
        "--num-harmless", "-nhl", type=int, default=-1,
        help="Number of harmless calibrations to randomly select"
    )

    # Refusal tensor options
    refusals = abliterate_parser.add_mutually_exclusive_group()
    refusals.add_argument(
        "--input-refusal", "-ir", type=str, default=None, help="Input tensor for refusal"
    )
    refusals.add_argument(
        "--output-refusal", "-or", type=str, default=None, help="Output tensor for refusal"
    )

    # Quantization options
    quant = abliterate_parser.add_mutually_exclusive_group()
    quant.add_argument(
        "--load-in-4bit", action="store_true", default=False,
        help="Load model in 4-bit precision using bitsandbytes"
    )
    quant.add_argument(
        "--load-in-8bit", action="store_true", default=False,
        help="Load model in 8-bit precision using bitsandbytes"
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare original and abliterated models")
    compare_parser.add_argument(
        "--original", "-o", type=str, required=True, help="Path to original model directory"
    )
    compare_parser.add_argument(
        "--abliterated", "-a", type=str, required=True, help="Path to abliterated model directory"
    )
    compare_parser.add_argument(
        "--output-file", "-f", type=str, default=None, help="Path to save comparison report"
    )
    compare_parser.add_argument(
        "--detailed", "-d", action="store_true", default=False,
        help="Include detailed per-parameter statistics"
    )
    compare_parser.add_argument(
        "--threshold", "-t", type=float, default=1e-6,
        help="Threshold for considering a weight change significant (default: 1e-6)"
    )
    compare_parser.add_argument(
        "--precision", "-p", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
        help="Precision of model"
    )
    compare_parser.add_argument(
        "--device", type=str, choices=["auto", "cuda", "cpu", "last-gpu"], default="auto",
        help="Target device for inference. 'last-gpu' will use the last available GPU."
    )
    compare_parser.add_argument(
        "--gpu-id", type=int, default=None,
        help="Specific GPU ID to use (overrides --device)"
    )
    compare_parser.add_argument(
        "--multi-gpu", action="store_true", default=False,
        help="Distribute model across multiple GPUs"
    )
    compare_parser.add_argument(
        "--max-memory", type=str, default=None,
        help="Maximum memory to use per GPU, e.g. '40GiB,40GiB,40GiB,40GiB,40GiB,40GiB'"
    )
    compare_quant = compare_parser.add_mutually_exclusive_group()
    compare_quant.add_argument(
        "--load-in-4bit", action="store_true", default=False,
        help="Load model in 4-bit precision using bitsandbytes"
    )
    compare_quant.add_argument(
        "--load-in-8bit", action="store_true", default=False,
        help="Load model in 8-bit precision using bitsandbytes"
    )
    compare_parser.add_argument(
        "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with a model")
    chat_parser.add_argument(
        "--model", "-m", type=str, required=True, help="Path to model directory"
    )
    chat_parser.add_argument(
        "--precision", "-p", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
        help="Precision of model"
    )
    chat_parser.add_argument(
        "--device", "-d", type=str, choices=["auto", "cuda", "cpu", "last-gpu"], default="auto",
        help="Target device for inference. 'last-gpu' will use the last available GPU."
    )
    chat_parser.add_argument(
        "--gpu-id", type=int, default=None,
        help="Specific GPU ID to use (overrides --device)"
    )
    chat_parser.add_argument(
        "--max-new-tokens", "-n", type=int, default=512, help="Max new tokens to generate"
    )
    chat_parser.add_argument(
        "--multi-gpu", action="store_true", default=False,
        help="Distribute model across multiple GPUs"
    )
    chat_parser.add_argument(
        "--max-memory", type=str, default=None,
        help="Maximum memory to use per GPU, e.g. '40GiB,40GiB,40GiB,40GiB,40GiB,40GiB'"
    )
    chat_quant = chat_parser.add_mutually_exclusive_group()
    chat_quant.add_argument(
        "--load-in-4bit", action="store_true", default=False,
        help="Load model in 4-bit precision using bitsandbytes"
    )
    chat_quant.add_argument(
        "--load-in-8bit", action="store_true", default=False,
        help="Load model in 8-bit precision using bitsandbytes"
    )
    chat_parser.add_argument(
        "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
    )

    # If no arguments are provided, default to abliterate
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # If no command is specified but model is provided, default to abliterate
    if args.command is None and hasattr(args, 'model'):
        args.command = 'abliterate'

    # Validate arguments for both commands
    if args.command in ['abliterate', 'chat']:
        # Common validations
        if args.load_in_4bit and args.load_in_8bit:
            parser.error("Do NOT use --load-in-4bit and --load-in-8bit simultaneously")

        # Handle device selection for both commands
        if args.gpu_id is not None:
            if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count():
                args.device = f"cuda:{args.gpu_id}"
            else:
                parser.error(f"GPU ID {args.gpu_id} is not available")
        elif args.device == "last-gpu" and torch.cuda.is_available():
            args.device = f"cuda:{torch.cuda.device_count() - 1}"

    # Validate arguments specific to abliterate command
    if args.command == 'abliterate':
        if args.output is None and args.output_refusal is None:
            parser.error("Either --output or --output-refusal must be specified")

        if args.input_refusal is not None and args.output_refusal is not None:
            parser.error("Do NOT use --input-refusal and --output-refusal simultaneously")

        if args.skip_begin < 1:
            parser.error("Do not mess with the first layer! --skip-begin must be >= 1")

        if args.layer_fraction < 0.0 or args.layer_fraction > 1.0:
            parser.error("--layer-fraction must be between 0.0 and 1.0")

    return args

# ====================== Main Function ======================

def main():
    """Main function"""
    args = parse_arguments()

    if args.command == 'chat':
        # Chat with a model
        chat_with_model(
            model_path=args.model,
            precision=args.precision,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            flash_attn=args.flash_attn,
            max_new_tokens=args.max_new_tokens,
            multi_gpu=args.multi_gpu,
            max_memory=args.max_memory
        )
    elif args.command == 'compare':
        # Compare original and abliterated models
        compare_models(
            original_model_path=args.original,
            abliterated_model_path=args.abliterated,
            precision=args.precision,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            flash_attn=args.flash_attn,
            multi_gpu=args.multi_gpu,
            max_memory=args.max_memory,
            output_file=args.output_file,
            detailed=args.detailed,
            threshold=args.threshold
        )
    elif args.command == 'abliterate':
        # Abliterate a model
        # Print system information
        print_system_info()

        # Get GPU information for tensor placement
        gpu_info = get_gpu_info()

        torch.inference_mode()
        torch.set_grad_enabled(False)

        # Set precision based on user input
        if args.precision == "fp16":
            precision = torch.float16
        elif args.precision == "bf16":
            precision = torch.bfloat16
        else:
            precision = torch.float32

        # Configure quantization if requested
        if args.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=precision,
                bnb_4bit_use_double_quant=True,
            )
        elif args.load_in_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=True,
            )
        else:
            quant_config = None

        # Load data files
        script_dir = Path(__file__).parent

        if args.data_harmful:
            harmful_list = load_data(args.data_harmful)
        else:
            harmful_path = script_dir / "data" / "harmful.parquet"
            if harmful_path.exists():
                harmful_list = load_data(harmful_path)
            else:
                print(f"Warning: Default harmful data file not found at {harmful_path}")
                print("Please provide a harmful data file with --data-harmful")
                sys.exit(1)

        if args.data_harmless:
            harmless_list = load_data(args.data_harmless)
        else:
            harmless_path = script_dir / "data" / "harmless.parquet"
            if harmless_path.exists():
                harmless_list = load_data(harmless_path)
            else:
                print(f"Warning: Default harmless data file not found at {harmless_path}")
                print("Please provide a harmless data file with --data-harmless")
                sys.exit(1)

        # Add DECCP data if requested
        if args.deccp:
            try:
                deccp_list = load_dataset("augmxnt/deccp", split="censored")
                harmful_list += deccp_list["text"]
            except Exception as e:
                print(f"Warning: Failed to load DECCP dataset: {e}")

        # Randomly select samples if requested
        if args.num_harmful > 0:
            if len(harmful_list) > args.num_harmful:
                print(f"Randomly selecting {args.num_harmful} harmful instructions from {len(harmful_list)} total")
                harmful_list = random.sample(harmful_list, args.num_harmful)
            else:
                print(f"Warning: Requested {args.num_harmful} harmful instructions but only {len(harmful_list)} are available")

        if args.num_harmless > 0:
            if len(harmless_list) > args.num_harmless:
                print(f"Randomly selecting {args.num_harmless} harmless instructions from {len(harmless_list)} total")
                harmless_list = random.sample(harmless_list, args.num_harmless)
            else:
                print(f"Warning: Requested {args.num_harmless} harmless instructions but only {len(harmless_list)} are available")

        # Load the model
        print(f"Loading model from {args.model}...")

        # Configure device map for multi-GPU setup
        device_map = args.device
        max_memory = None

        if args.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using multi-GPU setup with {torch.cuda.device_count()} GPUs")
            device_map = "auto"  # Always use auto for multi-GPU

            if args.max_memory:
                # Parse max memory string into a dictionary
                # Use integer indices for GPUs as required by Accelerate library
                memory_parts = args.max_memory.split(',')
                max_memory = {}
                for i, mem in enumerate(memory_parts):
                    if i < torch.cuda.device_count():
                        max_memory[i] = mem.strip()  # Use integer index instead of "cuda:i"
                max_memory["cpu"] = "24GiB"  # Allow CPU offloading if needed
                print(f"Using custom memory limits: {max_memory}")
            else:
                # Create automatic memory map based on available GPU memory
                # Use integer indices for GPUs as required by Accelerate library
                max_memory = {gpu['id']: f"{int(gpu['free_memory'] * 0.9)}GiB" for gpu in gpu_info}
                max_memory["cpu"] = "24GiB"  # Allow CPU offloading if needed
                print(f"Using automatic memory limits: {max_memory}")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": precision,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "quantization_config": quant_config,
            "attn_implementation": "flash_attention_2" if args.flash_attn else None,
        }

        # Add max_memory if we're using multi-GPU
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs
        )
        model.requires_grad_(False)

        if args.skip_begin + args.skip_end >= len(model.model.layers):
            raise ValueError("Too many layers to skip.")

        # Load the tokenizer with proper padding configuration
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, device_map=args.device
        )

        # Set padding token if not already set
        if tokenizer.pad_token is None:
            # If pad token is not set, use eos token as pad token but keep them distinct in the attention mask
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            # If no eos token either, use a special token
            else:
                tokenizer.pad_token = "[PAD]"

        print(f"Tokenizer: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")

        # Load or compute refusal tensor
        if args.input_refusal:
            print(f"Loading refusal tensor from {args.input_refusal}...")
            refusal_dir = torch.load(args.input_refusal)
        else:
            print("Computing refusal tensor...")
            refusal_dir = compute_refusals(
                model, tokenizer, harmful_list, harmless_list, args.layer_fraction
            )

        # If we have GPU info, try to place the refusal tensor on the best GPU
        if torch.cuda.is_available() and gpu_info:
            # Debug information about the refusal tensor
            print(f"DEBUG: Refusal tensor shape: {refusal_dir.shape}")
            print(f"DEBUG: Refusal tensor dimensions: {refusal_dir.dim()}")
            print(f"DEBUG: Refusal tensor dtype: {refusal_dir.dtype}")
            print(f"DEBUG: Refusal tensor element size: {refusal_dir.element_size()} bytes")
            print(f"DEBUG: Refusal tensor number of elements: {refusal_dir.nelement()}")

            # Estimate tensor size in GB (rough approximation)
            tensor_size_bytes = refusal_dir.element_size() * refusal_dir.nelement()
            tensor_size_mb = tensor_size_bytes / (1024**2)
            tensor_size_gb = tensor_size_bytes / (1024**3)

            print(f"DEBUG: Refusal tensor size: {tensor_size_bytes} bytes / {tensor_size_mb:.4f} MB / {tensor_size_gb:.6f} GB")

            best_device = find_best_gpu_for_tensor(tensor_size_gb, gpu_info)

            if best_device:
                print(f"Moving refusal tensor to {best_device} (size: {tensor_size_gb:.6f} GB)")
                refusal_dir = refusal_dir.to(best_device)

        # Save refusal tensor if requested
        if args.output_refusal:
            print(f"Saving refusal tensor to {args.output_refusal}...")
            torch.save(refusal_dir, args.output_refusal)

        if not args.output:
            sys.exit(0)

        print("Applying refusal tensor...")

        # Reload model if using quantization
        if args.load_in_4bit or args.load_in_8bit:
            print("Reloading model with bf16 precision...")
            del model
            torch.cuda.empty_cache()
            gc.collect()

            # Configure device map for multi-GPU setup
            device_map = "cpu"  # Default to CPU for this stage
            max_memory = None

            if args.multi_gpu and torch.cuda.device_count() > 1:
                print(f"Using multi-GPU setup for model reloading with {torch.cuda.device_count()} GPUs")
                device_map = "auto"  # Always use auto for multi-GPU

                # Refresh GPU info
                gpu_info = get_gpu_info()

                if args.max_memory:
                    # Parse max memory string into a dictionary
                    # Use integer indices for GPUs as required by Accelerate library
                    memory_parts = args.max_memory.split(',')
                    max_memory = {}
                    for i, mem in enumerate(memory_parts):
                        if i < torch.cuda.device_count():
                            max_memory[i] = mem.strip()  # Use integer index instead of "cuda:i"
                    max_memory["cpu"] = "256GiB"  # Allow CPU offloading if needed
                    print(f"Using custom memory limits: {max_memory}")
                else:
                    # Create automatic memory map based on available GPU memory
                    # Use integer indices for GPUs as required by Accelerate library
                    max_memory = {gpu['id']: f"{int(gpu['free_memory'] * 0.9)}GiB" for gpu in gpu_info}
                    max_memory["cpu"] = "256GiB"  # Allow CPU offloading if needed
                    print(f"Using automatic memory limits: {max_memory}")

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "device_map": device_map,
            }

            # Add max_memory if we're using multi-GPU
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory

            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                **model_kwargs
            )

        # Save original model weights for verification
        print("Saving a sample of original weights for verification...")
        original_weights = {}
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Sample a few layers to check later
            sample_layers = [args.skip_begin, len(model.model.layers) // 2, len(model.model.layers) - args.skip_end - 1]
            for layer_idx in sample_layers:
                if layer_idx >= 0 and layer_idx < len(model.model.layers):
                    if hasattr(model.model.layers[layer_idx], "self_attn") and hasattr(model.model.layers[layer_idx].self_attn, "o_proj"):
                        original_weights[f"layer_{layer_idx}_attn"] = model.model.layers[layer_idx].self_attn.o_proj.weight.data.clone().cpu()
                    if hasattr(model.model.layers[layer_idx], "mlp") and hasattr(model.model.layers[layer_idx].mlp, "down_proj"):
                        original_weights[f"layer_{layer_idx}_mlp"] = model.model.layers[layer_idx].mlp.down_proj.weight.data.clone().cpu()

        # Parse specific layers if provided
        specific_layers = None
        if args.specific_layers:
            try:
                specific_layers = [int(layer.strip()) for layer in args.specific_layers.split(',')]
                print(f"Using specific layers from command line: {specific_layers}")
            except ValueError:
                print(f"WARNING: Invalid format for specific layers: {args.specific_layers}")
                print("Expected comma-separated integers. Ignoring this parameter.")
                specific_layers = None

        # Apply abliteration
        model = apply_abliteration(
            model,
            refusal_dir,
            args.skip_begin,
            args.skip_end,
            args.scale_factor,
            args.top_refusal_layers,
            specific_layers,
            args.min_refusal_factor,
            args.refusal_threshold,
            args.force_abliteration,
            args.proportional_scaling,
            args.max_scale_factor,
        )

        # Verify that weights have changed
        print("\nVerifying weight changes...")
        changes_detected = False
        for key, original_weight in original_weights.items():
            layer_idx = int(key.split('_')[1])
            if "attn" in key and hasattr(model.model.layers[layer_idx].self_attn, "o_proj"):
                new_weight = model.model.layers[layer_idx].self_attn.o_proj.weight.data.cpu()
                diff = (new_weight - original_weight).norm().item()
                rel_diff = diff / original_weight.norm().item() * 100
                print(f"  {key}: Change magnitude = {diff:.6f} ({rel_diff:.4f}%)")
                if diff > 1e-6:
                    changes_detected = True
            elif "mlp" in key and hasattr(model.model.layers[layer_idx].mlp, "down_proj"):
                new_weight = model.model.layers[layer_idx].mlp.down_proj.weight.data.cpu()
                diff = (new_weight - original_weight).norm().item()
                rel_diff = diff / original_weight.norm().item() * 100
                print(f"  {key}: Change magnitude = {diff:.6f} ({rel_diff:.4f}%)")
                if diff > 1e-6:
                    changes_detected = True

        if not changes_detected:
            print("\nWARNING: No significant weight changes detected! Abliteration may have failed.")
            print("Try using a different scale factor or layer selection.")
        else:
            print("\nWeight changes confirmed. Abliteration appears successful.")

        # Save the abliterated model
        print(f"\nSaving abliterated model to {args.output}...")
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

        print(f"Abliteration complete! Model saved to {args.output}")
        print("You can chat with your abliterated model using:")
        print(f"  python {Path(__file__).name} chat -m {args.output}")

        if args.multi_gpu:
            print("To use multi-GPU distribution with your model:")
            print(f"  python {Path(__file__).name} chat -m {args.output} --multi-gpu")

if __name__ == "__main__":
    main()
