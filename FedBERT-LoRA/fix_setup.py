#!/usr/bin/env python3
"""
Fix common setup issues for FedBERT-LoRA
"""

import os
import sys
import subprocess

def fix_permissions():
    """Fix file permissions"""
    print("🔧 Fixing file permissions...")
    
    scripts = [
        "setup_environment.sh",
        "setup_environment_ubuntu.sh", 
        "run_experiment.sh",
        "activate_env.sh",
        "test_imports.py",
        "fix_setup.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f"✅ Fixed permissions for {script}")

def check_python_path():
    """Check and fix Python path issues"""
    print("\n🔍 Checking Python path...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, "src")
    
    if os.path.exists(src_dir):
        print(f"✅ Source directory found: {src_dir}")
        
        # Add to Python path if not already there
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            print(f"✅ Added {current_dir} to Python path")
    else:
        print(f"❌ Source directory not found: {src_dir}")
        return False
    
    return True

def test_imports():
    """Test critical imports"""
    print("\n🧪 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        import transformers
        print("✅ PyTorch and Transformers work")
        
        # Test our imports
        from src.models.federated_bert import FederatedBERTConfig
        from src.server.flower_server import create_flower_server
        from src.clients.flower_client import client_fn
        print("✅ FedBERT modules work")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_init_files():
    """Create missing __init__.py files"""
    print("\n📁 Checking __init__.py files...")
    
    init_dirs = [
        "src",
        "src/models",
        "src/server", 
        "src/clients",
        "src/aggregation",
        "src/utils"
    ]
    
    for dir_path in init_dirs:
        if os.path.exists(dir_path):
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Auto-generated __init__.py\n")
                print(f"✅ Created {init_file}")
            else:
                print(f"✅ {init_file} exists")

def main():
    """Main fix function"""
    print("🔧 FedBERT-LoRA Setup Fixer")
    print("=" * 30)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Run fixes
    fix_permissions()
    create_init_files()
    
    if check_python_path():
        if test_imports():
            print("\n🎉 Setup is working correctly!")
            print("\nNext steps:")
            print("1. source venv/bin/activate")
            print("2. python test_imports.py")
            print("3. python examples/run_simple_experiment.py")
        else:
            print("\n❌ Import issues remain. Try:")
            print("1. Reinstall dependencies: pip install -r requirements.txt")
            print("2. Install in development mode: pip install -e .")
    else:
        print("\n❌ Path issues found. Check your directory structure.")

if __name__ == "__main__":
    main()
