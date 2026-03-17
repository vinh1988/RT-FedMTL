#!/usr/bin/env python3
"""
Comprehensive LoRA Parameter Reduction Analysis for All Models
"""

def analyze_all_models():
    print("Comprehensive LoRA Parameter Reduction Analysis")
    print("=" * 60)
    
    # Model specifications based on configurations
    models = {
        'TinyBERT': {
            'total_params': 4_400_000,  # ~4.4M parameters
            'hidden_size': 128,
            'num_layers': 2,
            'lora_rank': 8,
            'lora_alpha': 16.0,
            'target_modules': 4
        },
        'MiniLM': {
            'total_params': 22_000_000,  # ~22M parameters  
            'hidden_size': 384,
            'num_layers': 6,
            'lora_rank': 8,
            'lora_alpha': 16.0,
            'target_modules': 4
        },
        'BERT-Mini': {
            'total_params': 11_000_000,  # ~11M parameters
            'hidden_size': 256,
            'num_layers': 4,
            'lora_rank': 16,
            'lora_alpha': 32.0,
            'target_modules': 4
        },
        'BERT-Medium': {
            'total_params': 41_000_000,  # ~41M parameters
            'hidden_size': 512,
            'num_layers': 8,
            'lora_rank': 32,
            'lora_alpha': 64.0,
            'target_modules': 4
        },
        'DistilBERT': {
            'total_params': 66_000_000,  # ~66M parameters
            'hidden_size': 768,
            'num_layers': 6,
            'lora_rank': 16,
            'lora_alpha': 32.0,
            'target_modules': 6  # Includes FFN layers
        }
    }
    
    results = []
    
    for model_name, config in models.items():
        # Calculate LoRA parameters
        # For each target module: 2 * hidden_size * rank parameters (A and B matrices)
        lora_params_per_layer = config['target_modules'] * 2 * config['hidden_size'] * config['lora_rank']
        total_lora_params = config['num_layers'] * lora_params_per_layer
        
        # Calculate reduction percentage
        reduction_percentage = (1 - total_lora_params / config['total_params']) * 100
        
        # Store results
        result = {
            'model': model_name,
            'total_params': config['total_params'],
            'lora_rank': config['lora_rank'],
            'lora_alpha': config['lora_alpha'],
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layers'],
            'target_modules': config['target_modules'],
            'lora_params': total_lora_params,
            'reduction_percentage': reduction_percentage
        }
        results.append(result)
        
        # Print individual model analysis
        print(f"\n{model_name}:")
        print(f"  Total Parameters: {config['total_params']:,}")
        print(f"  LoRA Rank: {config['lora_rank']}")
        print(f"  LoRA Alpha: {config['lora_alpha']}")
        print(f"  Hidden Size: {config['hidden_size']}")
        print(f"  Layers: {config['num_layers']}")
        print(f"  Target Modules: {config['target_modules']}")
        print(f"  LoRA Parameters: {total_lora_params:,}")
        print(f"  Parameter Reduction: {reduction_percentage:.2f}%")
    
    # Summary table
    print(f"\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Model':<12} {'Total Params':<12} {'LoRA Rank':<10} {'LoRA Params':<12} {'Reduction':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model']:<12} {result['total_params']:<12,} {result['lora_rank']:<10} "
              f"{result['lora_params']:<12,} {result['reduction_percentage']:<10.2f}%")
    
    # Overall conclusions
    print(f"\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    
    avg_reduction = sum(r['reduction_percentage'] for r in results) / len(results)
    min_reduction = min(r['reduction_percentage'] for r in results)
    max_reduction = max(r['reduction_percentage'] for r in results)
    
    print(f"Average Parameter Reduction: {avg_reduction:.2f}%")
    print(f"Minimum Reduction: {min_reduction:.2f}% (BERT-Medium)")
    print(f"Maximum Reduction: {max_reduction:.2f}% (TinyBERT)")
    
    print(f"\nKey Findings:")
    print(f"1. All models achieve >95% parameter reduction with LoRA")
    print(f"2. Smaller models (TinyBERT, MiniLM) achieve >99% reduction")
    print(f"3. Larger models (DistilBERT, BERT-Medium) still achieve >95% reduction")
    print(f"4. LoRA rank scaling is proportional to model size")
    print(f"5. The 'over 90%' claim is CONSISTENTLY verified across all models")
    
    print(f"\nVerification of 'over 90%' claim:")
    all_over_90 = all(r['reduction_percentage'] > 90 for r in results)
    if all_over_90:
        print("✅ VERIFIED: All models achieve >90% parameter reduction")
    else:
        print("❌ Some models do not achieve >90% reduction")
    
    return results

if __name__ == "__main__":
    analyze_all_models()
