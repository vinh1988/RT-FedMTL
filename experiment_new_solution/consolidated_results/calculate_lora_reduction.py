#!/usr/bin/env python3
"""
Calculate LoRA parameter reduction for TinyBERT
"""

def calculate_lora_reduction():
    # TinyBERT (bert-tiny) has approximately 4.4 million parameters
    # Based on the model: prajjwal1/bert-tiny
    
    # Standard TinyBERT parameters (approximate)
    total_params = 4_400_000  # ~4.4M parameters for bert-tiny
    
    # LoRA configuration from the YAML file
    lora_rank = 8
    lora_alpha = 16.0
    
    # For BERT-based models, LoRA is typically applied to:
    # - query, key, value projections in attention layers
    # - output dense layer in attention
    # Each projection in bert-tiny has hidden_size = 128
    
    hidden_size = 128  # bert-tiny hidden size
    num_attention_heads = 2  # bert-tiny has 2 attention heads
    num_layers = 2  # bert-tiny has 2 layers
    
    # Calculate LoRA parameters
    # For each target module: (hidden_size * rank) + rank parameters for bias
    # LoRA matrices: A (rank x hidden_size) and B (hidden_size x rank)
    # Total per module: 2 * hidden_size * rank
    
    target_modules = 4  # query, key, value, output.dense
    lora_params_per_layer = target_modules * 2 * hidden_size * lora_rank
    total_lora_params = num_layers * lora_params_per_layer
    
    # Calculate reduction percentage
    reduction_percentage = (1 - total_lora_params / total_params) * 100
    
    print("LoRA Parameter Reduction Analysis for TinyBERT:")
    print("=" * 50)
    print(f"Total TinyBERT parameters: {total_params:,}")
    print(f"LoRA rank: {lora_rank}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Target modules per layer: {target_modules}")
    print()
    print(f"LoRA parameters per layer: {lora_params_per_layer:,}")
    print(f"Total LoRA parameters: {total_lora_params:,}")
    print(f"Parameter reduction: {reduction_percentage:.2f}%")
    print()
    
    # Alternative calculation with more conservative estimates
    # Some sources suggest bert-tiny has different parameter counts
    conservative_total = 4_384_000  # More conservative estimate
    conservative_reduction = (1 - total_lora_params / conservative_total) * 100
    
    print("Conservative Estimate:")
    print(f"Total parameters: {conservative_total:,}")
    print(f"Parameter reduction: {conservative_reduction:.2f}%")
    print()
    
    # Calculate for different ranks to show sensitivity
    print("Parameter Reduction for Different LoRA Ranks:")
    print("-" * 40)
    for rank in [4, 8, 16, 32]:
        lora_params_rank = num_layers * target_modules * 2 * hidden_size * rank
        reduction_rank = (1 - lora_params_rank / total_params) * 100
        print(f"Rank {rank:2d}: {lora_params_rank:6,} LoRA params, {reduction_rank:5.2f}% reduction")
    
    # Verify the "over 90%" claim
    print()
    print("Verification of 'over 90%' claim:")
    print(f"Current reduction (rank=8): {reduction_percentage:.2f}%")
    if reduction_percentage > 90:
        print("✅ The 'over 90%' claim is ACCURATE")
    else:
        print("❌ The 'over 90%' claim needs adjustment")
        print(f"   Should state: 'reduces communication volume by approximately {reduction_percentage:.1f}%'")

if __name__ == "__main__":
    calculate_lora_reduction()
