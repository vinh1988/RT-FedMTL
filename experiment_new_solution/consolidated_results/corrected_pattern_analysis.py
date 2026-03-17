#!/usr/bin/env python3
"""
Corrected Pattern Analysis: layers_to_unfreeze configuration
"""

def corrected_pattern_analysis():
    print("Corrected Pattern Analysis: Partial Unfreezing Strategy")
    print("=" * 60)
    
    # Corrected model configurations
    models = {
        'TinyBERT': {
            'total_layers': 2,
            'has_partial_unfreeze': False,
            'strategy': 'Full LoRA (no partial unfreeze)',
            'layers_to_freeze': 0,
            'layers_to_unfreeze': 0
        },
        'MiniLM': {
            'total_layers': 6,
            'has_partial_unfreeze': True,
            'strategy': 'Partial unfreeze + LoRA',
            'layers_to_freeze': 2,
            'layers_to_unfreeze': 4
        },
        'BERT-Mini': {
            'total_layers': 4,
            'has_partial_unfreeze': True,
            'strategy': 'Partial unfreeze + LoRA',
            'layers_to_freeze': 2,
            'layers_to_unfreeze': 2
        },
        'BERT-Medium': {
            'total_layers': 8,
            'has_partial_unfreeze': True,
            'strategy': 'Partial unfreeze + LoRA',
            'layers_to_freeze': 4,
            'layers_to_unfreeze': 4
        },
        'DistilBERT': {
            'total_layers': 6,
            'has_partial_unfreeze': True,
            'strategy': 'Partial unfreeze + LoRA',
            'layers_to_freeze': 3,
            'layers_to_unfreeze': 3
        }
    }
    
    print("Model Configuration Analysis:")
    print("-" * 60)
    print(f"{'Model':<12} {'Total':<6} {'Strategy':<25} {'Freeze':<7} {'Unfreeze':<9} {'Half?'}")
    print("-" * 60)
    
    models_with_half_pattern = 0
    models_with_partial_unfreeze = 0
    
    for model_name, config in models.items():
        total_layers = config['total_layers']
        has_partial = config['has_partial_unfreeze']
        strategy = config['strategy']
        freeze = config['layers_to_freeze']
        unfreeze = config['layers_to_unfreeze']
        
        # Check half pattern only for models with partial unfreeze
        if has_partial:
            models_with_partial_unfreeze += 1
            half_of_total = total_layers / 2
            is_half = unfreeze == half_of_total
            if is_half:
                models_with_half_pattern += 1
            
            half_status = "✅" if is_half else "❌"
            print(f"{model_name:<12} {total_layers:<6} {strategy:<25} {freeze:<7} {unfreeze:<9} {half_status}")
            
            if not is_half:
                print(f"            Expected: {half_of_total}, Actual: {unfreeze}")
        else:
            print(f"{model_name:<12} {total_layers:<6} {strategy:<25} {'N/A':<7} {'N/A':<9} {'N/A'}")
    
    print("-" * 60)
    
    print(f"\nPattern Analysis Results:")
    print(f"Models with partial unfreeze: {models_with_partial_unfreeze}/5")
    print(f"Models following half pattern: {models_with_half_pattern}/{models_with_partial_unfreeze}")
    
    # Analyze the pattern for models that actually use partial unfreeze
    if models_with_partial_unfreeze > 0:
        percentage_following_pattern = (models_with_half_pattern / models_with_partial_unfreeze) * 100
        print(f"Percentage following half pattern: {percentage_following_pattern:.1f}%")
        
        if percentage_following_pattern == 100:
            print("✅ All models WITH partial unfreeze follow the half-layer pattern")
        else:
            print("❌ Some models WITH partial unfreeze don't follow the half-layer pattern")
    
    print(f"\nDetailed Strategy Breakdown:")
    for model_name, config in models.items():
        print(f"\n{model_name}:")
        print(f"  Strategy: {config['strategy']}")
        print(f"  Total layers: {config['total_layers']}")
        
        if config['has_partial_unfreeze']:
            freeze = config['layers_to_freeze']
            unfreeze = config['layers_to_unfreeze']
            total = config['total_layers']
            
            print(f"  Layers to freeze: {freeze}")
            print(f"  Layers to unfreeze: {unfreeze}")
            print(f"  Half of total: {total/2}")
            print(f"  Follows half pattern: {unfreeze == total/2}")
            print(f"  Freeze + Unfreeze = {freeze + unfreeze} (should equal {total})")
            
            if freeze + unfreeze == total:
                print(f"  ✅ Layer count consistent")
            else:
                print(f"  ❌ Layer count inconsistent")
        else:
            print(f"  ✅ Uses full LoRA approach (no partial unfreeze)")

if __name__ == "__main__":
    corrected_pattern_analysis()
