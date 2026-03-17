#!/usr/bin/env python3
"""
Verify the pattern: layers_to_unfreeze is half of total layers for all models
"""

def verify_layer_pattern():
    print("Verification: layers_to_unfreeze Pattern Analysis")
    print("=" * 50)
    
    # Model configurations based on YAML files
    models = {
        'TinyBERT': {
            'total_layers': 2,
            'layers_to_freeze': 0,  # Not specified in config
            'layers_to_unfreeze': 0,  # Not specified in config
            'config_file': 'tiny_bert/fl-mtl-slms-berttiny-stsb-qqp-sst2-lora/federated_config.yaml'
        },
        'MiniLM': {
            'total_layers': 6,
            'layers_to_freeze': 2,
            'layers_to_unfreeze': 4,
            'config_file': 'mini-lm/fl-mtl-slms-mini-lm-sts-qqp-sst2-lora/federated_config.yaml'
        },
        'BERT-Mini': {
            'total_layers': 4,
            'layers_to_freeze': 2,
            'layers_to_unfreeze': 2,
            'config_file': 'mini-bert/fl-mtl-slms-bertmini-stsb-qqp-sst2-lora/federated_config.yaml'
        },
        'BERT-Medium': {
            'total_layers': 8,
            'layers_to_freeze': 4,
            'layers_to_unfreeze': 4,
            'config_file': 'medium-bert/fl-mtl-slms-bertmedium-stsb-qqp-sst2-lora/federated_config.yaml'
        },
        'DistilBERT': {
            'total_layers': 6,
            'layers_to_freeze': 3,
            'layers_to_unfreeze': 3,
            'config_file': 'distil-bert/fl-mtl-slms-disti-bert-stsb-qqp-sst2-lora/federated_config.yaml'
        }
    }
    
    print("Model Configuration Analysis:")
    print("-" * 50)
    print(f"{'Model':<12} {'Total':<6} {'Freeze':<7} {'Unfreeze':<9} {'Half?':<6} {'Pattern'}")
    print("-" * 50)
    
    pattern_holds = True
    
    for model_name, config in models.items():
        total_layers = config['total_layers']
        layers_to_freeze = config['layers_to_freeze']
        layers_to_unfreeze = config['layers_to_unfreeze']
        
        # Check if layers_to_unfreeze equals half of total layers
        half_of_total = total_layers / 2
        is_half = layers_to_unfreeze == half_of_total
        
        if not is_half:
            pattern_holds = False
        
        # Calculate what half should be
        expected_unfreeze = total_layers // 2 if total_layers % 2 == 0 else total_layers / 2
        
        pattern_status = "✅" if is_half else "❌"
        
        print(f"{model_name:<12} {total_layers:<6} {layers_to_freeze:<7} {layers_to_unfreeze:<9} {is_half:<6} {pattern_status}")
        
        if not is_half:
            print(f"            Expected: {expected_unfreeze}, Actual: {layers_to_unfreeze}")
    
    print("-" * 50)
    
    print(f"\nPattern Verification Results:")
    print(f"Pattern: 'layers_to_unfreeze = half of total layers'")
    if pattern_holds:
        print("✅ PATTERN HOLDS: All models follow the half-layer pattern")
    else:
        print("❌ PATTERN BROKEN: Some models don't follow the half-layer pattern")
    
    print(f"\nDetailed Analysis:")
    for model_name, config in models.items():
        total = config['total_layers']
        freeze = config['layers_to_freeze']
        unfreeze = config['layers_to_unfreeze']
        
        print(f"\n{model_name}:")
        print(f"  Total layers: {total}")
        print(f"  Layers to freeze: {freeze}")
        print(f"  Layers to unfreeze: {unfreeze}")
        print(f"  Half of total: {total/2}")
        print(f"  Pattern matches: {unfreeze == total/2}")
        print(f"  Freeze + Unfreeze = {freeze + unfreeze} (should equal {total})")
        
        # Verify consistency
        if freeze + unfreeze == total:
            print(f"  ✅ Layer count consistent")
        else:
            print(f"  ❌ Layer count inconsistent")

if __name__ == "__main__":
    verify_layer_pattern()
