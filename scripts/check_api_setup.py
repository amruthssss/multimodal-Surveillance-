"""
Check if API keys are properly configured for dataset download
"""

import os
from pathlib import Path

def check_kaggle_api():
    """Check if Kaggle API is configured"""
    print("\n" + "="*70)
    print("🔍 CHECKING KAGGLE API CONFIGURATION")
    print("="*70)
    
    # Check for kaggle.json in user directory
    kaggle_path = Path.home() / '.kaggle' / 'kaggle.json'
    
    print(f"\n📁 Looking for: {kaggle_path}")
    
    if kaggle_path.exists():
        print("   ✅ kaggle.json found!")
        print(f"   📍 Location: {kaggle_path}")
        
        # Check file size
        size = kaggle_path.stat().st_size
        print(f"   📏 Size: {size} bytes")
        
        if size < 50:
            print("   ⚠️  Warning: File seems too small, might be empty")
        
        # Try to read and validate
        try:
            import json
            with open(kaggle_path, 'r') as f:
                data = json.load(f)
                if 'username' in data and 'key' in data:
                    print(f"   ✅ Valid format: username = {data['username']}")
                    print("   ✅ API key present")
                else:
                    print("   ❌ Invalid format: missing 'username' or 'key'")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
        
        # Try to use Kaggle API
        try:
            import kaggle
            print("\n   Testing Kaggle API...")
            # This will fail if credentials are invalid
            kaggle.api.authenticate()
            print("   ✅ Kaggle API authenticated successfully!")
            return True
        except ImportError:
            print("   ⚠️  Kaggle package not installed")
            print("   💡 Run: pip install kaggle")
            return False
        except Exception as e:
            print(f"   ❌ Authentication failed: {e}")
            return False
    else:
        print("   ❌ kaggle.json NOT found!")
        print("\n   📋 TO FIX:")
        print("   1. Go to: https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Move downloaded kaggle.json to:")
        print(f"      {kaggle_path.parent}")
        print("\n   PowerShell commands:")
        print(f"      mkdir {kaggle_path.parent}")
        print(f"      Move-Item Downloads\\kaggle.json {kaggle_path.parent}\\")
        return False

def check_roboflow_api():
    """Check if Roboflow API is configured"""
    print("\n" + "="*70)
    print("🔍 CHECKING ROBOFLOW API CONFIGURATION")
    print("="*70)
    
    # Check for .env file
    env_path = Path('.env')
    
    print(f"\n📁 Looking for: {env_path.absolute()}")
    
    if env_path.exists():
        print("   ✅ .env file found!")
        
        # Check for API key
        with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if 'ROBOFLOW_API_KEY' in content:
                # Don't print the actual key
                print("   ✅ ROBOFLOW_API_KEY found in .env")
                
                # Try to use Roboflow API
                try:
                    from roboflow import Roboflow
                    # Extract key from .env
                    for line in content.split('\n'):
                        if 'ROBOFLOW_API_KEY' in line:
                            key = line.split('=')[1].strip()
                            rf = Roboflow(api_key=key)
                            print("   ✅ Roboflow API key valid!")
                            return True
                except ImportError:
                    print("   ⚠️  Roboflow package not installed")
                    print("   💡 Run: pip install roboflow")
                    return False
                except Exception as e:
                    print(f"   ❌ API key validation failed: {e}")
                    return False
            else:
                print("   ❌ ROBOFLOW_API_KEY not found in .env")
    else:
        print("   ℹ️  .env file not found (optional)")
        print("\n   📋 TO USE ROBOFLOW (OPTIONAL):")
        print("   1. Sign up: https://app.roboflow.com/")
        print("   2. Get API key: https://app.roboflow.com/settings/api")
        print("   3. Create .env file:")
        print("      echo 'ROBOFLOW_API_KEY=your_key_here' > .env")
        return False

def check_github_access():
    """Check if git is available"""
    print("\n" + "="*70)
    print("🔍 CHECKING GITHUB ACCESS")
    print("="*70)
    
    import subprocess
    
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print(f"   ✅ Git installed: {result.stdout.strip()}")
            print("   ✅ Can clone GitHub datasets directly!")
            return True
        else:
            print("   ❌ Git not working properly")
            return False
    except Exception as e:
        print("   ❌ Git not installed")
        print("   💡 Download: https://git-scm.com/download/win")
        return False

def main():
    print("\n" + "="*70)
    print("🔑 API KEY CONFIGURATION CHECKER")
    print("="*70)
    print("Checking which dataset sources are ready to use...")
    
    # Check all API configurations
    kaggle_ok = check_kaggle_api()
    roboflow_ok = check_roboflow_api()
    git_ok = check_github_access()
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    print("\n✅ Ready to use:")
    if kaggle_ok:
        print("   ✅ Kaggle (Recommended - 3.5 GB datasets)")
    if roboflow_ok:
        print("   ✅ Roboflow (Pre-labeled YOLO format)")
    if git_ok:
        print("   ✅ GitHub (Free datasets, no API needed)")
    
    if not any([kaggle_ok, roboflow_ok, git_ok]):
        print("   ❌ No dataset sources configured yet")
    
    print("\n📋 RECOMMENDATIONS:")
    
    if kaggle_ok:
        print("\n   🎯 YOU'RE READY! Start downloading:")
        print("      kaggle datasets download -d phylake1337/fire-dataset")
        print("      kaggle datasets download -d anshtanwar/car-crash-dataset")
    else:
        print("\n   🎯 SETUP KAGGLE (RECOMMENDED):")
        print("      1. Get API key: https://www.kaggle.com/account")
        print("      2. Save to: C:\\Users\\<YourName>\\.kaggle\\kaggle.json")
        print("      3. Install: pip install kaggle")
        print("      4. Download datasets (commands will work after setup)")
    
    if git_ok and not kaggle_ok:
        print("\n   💡 ALTERNATIVE: Use GitHub datasets:")
        print("      cd datasets")
        print("      git clone https://github.com/DeepQuestAI/Fire-Smoke-Dataset")
    
    print("\n📖 Full guide: API_SETUP_GUIDE.md")
    print("="*70)

if __name__ == "__main__":
    main()
