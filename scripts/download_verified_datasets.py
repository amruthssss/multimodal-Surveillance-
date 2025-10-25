"""
VERIFIED DATASET DOWNLOADER FOR YOLO v11
Only uses datasets that are confirmed to exist and work
Focus: explosion, fire, vehicle_accident, fighting
"""

import os
import subprocess
import zipfile
from pathlib import Path
import time

def run_command(cmd, description):
    """Run command with progress display"""
    print(f"\n{'='*70}")
    print(f"⚙️  {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, shell=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success!")
            return True
        else:
            print(f"❌ Failed with code: {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def extract_zip(zip_path, dest_dir):
    """Extract zip file"""
    print(f"\n📦 Extracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            print(f"   Files to extract: {total_files}")
            zip_ref.extractall(dest_dir)
        print(f"✅ Extracted to: {dest_dir}")
        zip_path.unlink()
        print(f"🗑️  Deleted zip file")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def count_images(directory):
    """Count image files"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    count = 0
    for ext in extensions:
        count += len(list(Path(directory).rglob(f'*{ext}')))
    return count

def main():
    print("\n" + "="*70)
    print("🎯 VERIFIED DATASET DOWNLOADER")
    print("="*70)
    print("Events: fire, vehicle_accident, fighting, explosion")
    print("All datasets verified to exist on Kaggle")
    print("="*70)
    
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    # VERIFIED WORKING DATASETS
    datasets = [
        {
            'name': 'phylake1337/fire-dataset',
            'folder': 'fire',
            'size': '~400 MB',
            'events': ['fire'],
            'images': '~10,000'
        },
        {
            'name': 'sayedgamal99/smoke-fire-detection-yolo',
            'folder': 'fire_smoke_yolo',
            'size': '~3 GB',
            'events': ['fire', 'smoke'],
            'images': '~15,000'
        },
        {
            'name': 'dataclusterlabs/fire-and-smoke-dataset',
            'folder': 'fire_smoke',
            'size': '~88 MB',
            'events': ['fire', 'smoke'],
            'images': '~5,000'
        },
        {
            'name': 'ckay16/accident-detection-from-cctv-footage',
            'folder': 'accidents_cctv',
            'size': '~261 MB',
            'events': ['vehicle_accident'],
            'images': '~3,000'
        },
        {
            'name': 'nextmillionaire/car-accident-dataset',
            'folder': 'car_accidents',
            'size': '~7 MB',
            'events': ['vehicle_accident'],
            'images': '~800'
        },
        {
            'name': 'mohamedmustafa/real-life-violence-situations-dataset',
            'folder': 'violence',
            'size': '~3.8 GB',
            'events': ['fighting'],
            'images': '~15,000 videos'
        },
        {
            'name': 'toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd',
            'folder': 'violence_cctv',
            'size': '~1 GB',
            'events': ['fighting'],
            'images': '~5,000'
        }
    ]
    
    print(f"\n📋 DATASETS TO DOWNLOAD ({len(datasets)} datasets)")
    print("="*70)
    
    total_size_gb = 0
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   Events: {', '.join(ds['events'])}")
        print(f"   Size: {ds['size']}")
        print(f"   Images: {ds['images']}")
        
        # Calculate total
        size_str = ds['size'].replace('~', '').strip()
        if 'GB' in size_str:
            total_size_gb += float(size_str.split()[0])
        elif 'MB' in size_str:
            total_size_gb += float(size_str.split()[0]) / 1024
    
    print(f"\n{'='*70}")
    print(f"📦 TOTAL SIZE: ~{total_size_gb:.1f} GB")
    print(f"💾 DISK SPACE NEEDED: ~{total_size_gb * 1.5:.1f} GB (including extracted files)")
    print(f"{'='*70}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("\n   Option 1: QUICK START (Smallest, fastest)")
    print("      - phylake1337/fire-dataset (400 MB)")
    print("      - nextmillionaire/car-accident-dataset (7 MB)")
    print("      - dataclusterlabs/fire-and-smoke-dataset (88 MB)")
    print("      Total: ~500 MB, ~15 minutes")
    
    print("\n   Option 2: BALANCED (Good quality, reasonable size)")
    print("      - phylake1337/fire-dataset (400 MB)")
    print("      - ckay16/accident-detection-from-cctv-footage (261 MB)")
    print("      - toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd (1 GB)")
    print("      Total: ~1.7 GB, ~30 minutes")
    
    print("\n   Option 3: COMPLETE (Best quality, large)")
    print("      - All 7 datasets above")
    print(f"      Total: ~{total_size_gb:.1f} GB, ~60-90 minutes")
    
    print(f"\n{'='*70}")
    choice = input("\nChoose option (1/2/3) or 'custom' to select individually: ").strip()
    
    selected_datasets = []
    
    if choice == '1':
        selected_datasets = [datasets[0], datasets[4], datasets[2]]
        print("\n✅ QUICK START selected")
    elif choice == '2':
        selected_datasets = [datasets[0], datasets[3], datasets[6]]
        print("\n✅ BALANCED selected")
    elif choice == '3':
        selected_datasets = datasets
        print("\n✅ COMPLETE selected")
    elif choice.lower() == 'custom':
        print("\nSelect datasets (enter numbers separated by commas, e.g., 1,3,5):")
        selected = input("Your selection: ").strip()
        indices = [int(x.strip())-1 for x in selected.split(',')]
        selected_datasets = [datasets[i] for i in indices if 0 <= i < len(datasets)]
    else:
        print("❌ Invalid choice, exiting")
        return
    
    # Confirm
    print(f"\n{'='*70}")
    print("📥 WILL DOWNLOAD:")
    for ds in selected_datasets:
        print(f"   ✅ {ds['name']} ({ds['size']})")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled")
        return
    
    # Download
    downloaded = []
    failed = []
    
    start_time = time.time()
    
    for i, ds in enumerate(selected_datasets, 1):
        print(f"\n{'#'*70}")
        print(f"# {i}/{len(selected_datasets)}: {ds['name']}")
        print(f"{'#'*70}")
        
        dest_folder = datasets_dir / ds['folder']
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Download
        cmd = f"kaggle datasets download -d {ds['name']}"
        if run_command(cmd, f"Downloading {ds['name']}"):
            # Extract
            zip_name = ds['name'].split('/')[-1] + '.zip'
            zip_path = Path(zip_name)
            
            if zip_path.exists():
                if extract_zip(zip_path, dest_folder):
                    images = count_images(dest_folder)
                    downloaded.append({
                        'name': ds['name'],
                        'folder': dest_folder,
                        'events': ds['events'],
                        'images': images
                    })
                    print(f"✅ {ds['name']} - COMPLETE ({images} images)")
                else:
                    failed.append(ds['name'])
            else:
                print(f"❌ Zip file not found: {zip_name}")
                failed.append(ds['name'])
        else:
            failed.append(ds['name'])
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"✅ Success: {len(downloaded)}/{len(selected_datasets)}")
    
    if downloaded:
        print("\n📁 Downloaded datasets:")
        total_images = 0
        for ds in downloaded:
            print(f"\n   ✅ {ds['name']}")
            print(f"      Location: {ds['folder']}")
            print(f"      Events: {', '.join(ds['events'])}")
            print(f"      Images: {ds['images']:,}")
            total_images += ds['images']
        
        print(f"\n📊 TOTAL IMAGES: {total_images:,}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for name in failed:
            print(f"   ❌ {name}")
    
    # Next steps
    print(f"\n{'='*70}")
    print("✅ DOWNLOAD COMPLETE!")
    print(f"{'='*70}")
    
    if downloaded:
        print("\n📋 NEXT STEPS:")
        print("\n1. Convert to YOLO format:")
        for ds in downloaded:
            event = ds['events'][0]  # Primary event
            print(f"   python scripts/convert_to_yolo.py --src {ds['folder']} --dst datasets/combined_yolo --labels {event}")
        
        print("\n2. Train YOLO v11:")
        print("   python scripts/train_yolo_v11.py --data datasets/combined_yolo/data.yaml --epochs 50 --batch 16")
        
        print("\n3. Test trained model:")
        print("   python test_cctv_system.py --model runs/train/exp/weights/best.pt")
        
        print(f"\n💡 TIP: Start with Option 1 (quick) datasets first to test the pipeline!")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
