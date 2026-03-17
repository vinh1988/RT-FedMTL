#!/usr/bin/env python3
"""
Corrected MiniLM LoRA Parameter Reduction Analysis
"""

def correct_minilm_analysis():
    print("Corrected MiniLM LoRA Analysis")
    print("=" * 40)
    
    # MiniLM-L6-H384 specifications
    total_params = 22_000_000  # ~22M parameters
    hidden_size = 384
    num_layers = 6
    lora_rank = 8
    lora_alpha = 16.0
    target_modules = 5  # CORRECTED: attention.query, attention.key, attention.value, attention.output.dense, output.dense
    
    # Calculate LoRA parameters correctly
    # For each target module: 2 * hidden_size * rank parameters (A and B matrices)
    lora_params_per_layer = target_modules * 2 * hidden_size * lora_rank
    total_lora_params = num_layers * lora_params_per_layer
    
    # Calculate reduction percentage
    reduction_percentage = (1 - total_lora_params / total_params) * 100
    
    print(f"MiniLM-L6-H384 Corrected Analysis:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  LoRA Rank: {lora_rank}")
    print(f"  LoRA Alpha: {lora_alpha}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Number of Layers: {num_layers}")
    print(f"  Target Modules: {target_modules} (CORRECTED)")
    print(f"  LoRA Parameters per Layer: {lora_params_per_layer:,}")
    print(f"  Total LoRA Parameters: {total_lora_params:,}")
    print(f"  Parameter Reduction: {reduction_percentage:.2f}%")
    
    print(f"\nTarget Modules List:")
    print(f"  1. attention.query")
    print(f"  2. attention.key") 
    print(f"  3. attention.value")
    print(f"  4. attention.output.dense")
    print(f"  5. output.dense")
    
    print(f"\nComparison with Previous (Incorrect) Analysis:")
    print(f"  Previous target modules: 4")
    print(f"  Correct target modules: 5")
    print(f"  Previous LoRA params: 147,456")
    print(f"  Correct LoRA params: {total_lora_params:,}")
    print(f"  Difference: {total_lora_params - 147456:,} additional parameters")
    
    print(f"\nUpdated Reduction Percentage:")
    print(f"  Previous reduction: 99.33%")
    print(f"  Correct reduction: {reduction_percentage:.2f}%")
    print(f"  Difference: {99.33 - reduction_percentage:.2f}% points")
    
    # Verify if still >90%
    if reduction_percentage > 90:
        print(f"\n✅ Still achieves >90% reduction: {reduction_percentage:.2f}%")
    else:
        print(f"\n❌ No longer achieves >90% reduction: {reduction_percentage:.2f}%")

if __name__ == "__main__":
    correct_minilm_analysis()
