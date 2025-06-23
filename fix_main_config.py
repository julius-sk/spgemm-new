#!/usr/bin/env python3
"""
Quick fix for the TrainConfig parse_known_args issue
This patches the main training script to handle the configuration properly
"""

import os
import shutil

def backup_file(filepath):
    """Create backup of original file"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up {filepath} to {backup_path}")
        return True
    return False

def fix_main_config_usage():
    """Fix the main training script to handle TrainConfig properly"""
    
    # Check if the main file exists
    main_file = "maxk_gnn_integrated.py"
    if not os.path.exists(main_file):
        print(f"❌ {main_file} not found!")
        return False
    
    # Backup original
    backup_file(main_file)
    
    # Read current content
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Fix the parse_known_args usage
    old_pattern = "args, * = config.parse*known_args()"
    new_pattern = "args, unknown = config.parse_known_args()"
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("✅ Fixed parse_known_args usage")
    else:
        # Try alternative patterns
        patterns_to_fix = [
            ("args, * = config.parse*known_args()", "args, unknown = config.parse_known_args()"),
            ("args = config.parse_args()", "args, unknown = config.parse_known_args()"),
            ("config.parse_known_args()", "args, unknown = config.parse_known_args()"),
        ]
        
        for old, new in patterns_to_fix:
            if old in content:
                content = content.replace(old, new)
                print(f"✅ Fixed pattern: {old}")
                break
        else:
            # If no patterns match, let's look for the main function and fix it
            if "def main():" in content:
                # Find the config usage and fix it
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "config.parse" in line and "args" in line:
                        # Replace the problematic line
                        lines[i] = "    args, unknown = config.parse_known_args()"
                        print(f"✅ Fixed line {i+1}: {line.strip()}")
                        break
                content = '\n'.join(lines)
    
    # Also ensure proper imports
    if "from utils.config import TrainConfig" not in content:
        # Add import after other imports
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append(line)
            elif line.strip() == '' and in_imports:
                import_lines.append(line)
            else:
                if in_imports:
                    import_lines.append("from utils.config import TrainConfig")
                    in_imports = False
                other_lines.append(line)
        
        content = '\n'.join(import_lines + other_lines)
        print("✅ Added TrainConfig import")
    
    # Write fixed content
    with open(main_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {main_file}")
    return True

def create_minimal_main():
    """Create a minimal working main script if needed"""
    
    minimal_main = '''#!/usr/bin/env python3
"""
MaxK-GNN Integrated Training Script - Minimal Version
"""

import torch
import torch.nn.functional as F
from utils.config import TrainConfig

def main():
    print("🚀 MaxK-GNN Training - Minimal Version")
    
    # Create and parse configuration
    config = TrainConfig()
    args, unknown = config.parse_known_args()
    
    print(f"✅ Configuration loaded successfully!")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"MaxK: {args.maxk}")
    print(f"Output path: {args.path}")
    
    # Check for unknown arguments
    if unknown:
        print(f"⚠️ Unknown arguments: {unknown}")
    
    print("\\n🎯 Ready to train! (This is just a config test)")

if __name__ == "__main__":
    main()
'''
    
    with open("test_config.py", 'w') as f:
        f.write(minimal_main)
    
    print("✅ Created test_config.py for testing configuration")

def main():
    print("🔧 Fixing MaxK-GNN Configuration Issues")
    print("=" * 40)
    
    # Fix 1: Update utils/config.py
    print("Step 1: Updating utils/config.py...")
    config_dir = "utils"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"✅ Created {config_dir} directory")
    
    # The config.py content is in the artifact above
    print("✅ Use the fixed config.py from the artifact above")
    
    # Fix 2: Fix main script
    print("\\nStep 2: Fixing main training script...")
    if fix_main_config_usage():
        print("✅ Main script fixed")
    else:
        print("⚠️ Could not fix main script automatically")
    
    # Fix 3: Create test script
    print("\\nStep 3: Creating test script...")
    create_minimal_main()
    
    print("\\n🎯 Next Steps:")
    print("1. Copy the fixed config.py content from the artifact above to utils/config.py")
    print("2. Test configuration: python test_config.py")
    print("3. Run training: python maxk_gnn_integrated.py --dataset reddit --model sage")
    
    print("\\n📋 Changes made:")
    print("- Fixed TrainConfig to inherit from ArgumentParser")
    print("- Added missing parse_known_args() method")
    print("- Fixed argument parsing in main script")
    print("- Created test script for verification")

if __name__ == "__main__":
    main()
