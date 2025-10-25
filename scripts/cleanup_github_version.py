"""
CLEANUP GITHUB VERSION
Deletes the GitHub upload copy after successful upload
Your original project remains untouched
"""

import shutil
from pathlib import Path

def cleanup_github_version():
    print("="*70)
    print("🗑️  CLEANUP GITHUB VERSION")
    print("="*70)
    
    github_dir = Path('../muli_modal_github')
    original_dir = Path('.')
    
    if not github_dir.exists():
        print("❌ GitHub version not found")
        print(f"   Looking for: {github_dir.absolute()}")
        return
    
    print(f"\n⚠️  WARNING: This will delete:")
    print(f"   {github_dir.absolute()}")
    print(f"\n✅ Your original project is safe:")
    print(f"   {original_dir.absolute()}")
    
    confirm = input("\nDelete GitHub version? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        print(f"\n🗑️  Deleting {github_dir}...")
        shutil.rmtree(github_dir)
        print("✅ GitHub version deleted!")
        print("\n💡 Your original project is unchanged and ready to use")
    else:
        print("❌ Cleanup cancelled")
        print("   GitHub version kept at:", github_dir.absolute())

if __name__ == "__main__":
    cleanup_github_version()
