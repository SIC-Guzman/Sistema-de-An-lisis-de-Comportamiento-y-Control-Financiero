"""
Script to download the Sparkov Credit Card Fraud Detection dataset from Kaggle.

Requirements:
1. Kaggle account
2. Kaggle API credentials (~/.kaggle/kaggle.json)
3. kaggle package installed (pip install kaggle)

Instructions to get Kaggle API credentials:
1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)
5. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac only)
"""

import os
import sys
from pathlib import Path
import subprocess
import zipfile


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("❌ Kaggle credentials not found!")
        print("\nTo download the dataset, you need Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print(f"4. Save kaggle.json to {kaggle_dir}")
        print("\nOn Linux/Mac, also run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle credentials found")
    return True


def download_dataset(output_dir: str = "data/raw"):
    """
    Download Sparkov dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading dataset to {output_path}...")
    
    # Kaggle dataset identifier
    dataset = "kartik2112/fraud-detection"
    
    try:
        # Download using kaggle CLI
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_path)],
            check=True
        )
        
        print("✅ Dataset downloaded successfully!")
        
        # Unzip the dataset
        zip_file = output_path / "fraud-detection.zip"
        if zip_file.exists():
            print("\n📦 Extracting files...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            # Remove zip file
            zip_file.unlink()
            print("✅ Extraction complete!")
        
        # List downloaded files
        print("\n📄 Downloaded files:")
        for file in output_path.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def verify_dataset(data_dir: str = "data/raw"):
    """
    Verify that dataset files exist and are valid.
    
    Args:
        data_dir: Directory containing the dataset
    """
    data_path = Path(data_dir)
    
    # Expected files
    expected_files = ["fraudTrain.csv", "fraudTest.csv"]
    
    print("\n🔍 Verifying dataset...")
    
    all_found = True
    for filename in expected_files:
        file_path = data_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename} found ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {filename} not found")
            all_found = False
    
    if all_found:
        print("\n✅ Dataset verification successful!")
        print("\nNext steps:")
        print("1. Run: python scripts/preprocess_pipeline.py")
        print("2. Or explore: jupyter notebook notebooks/01_exploratory_analysis.ipynb")
    else:
        print("\n❌ Dataset verification failed. Please check the download.")
    
    return all_found


def main():
    """Main function."""
    print("=" * 60)
    print("  Sparkov Credit Card Fraud Detection Dataset Downloader")
    print("=" * 60)
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        print("\n❌ Cannot proceed without Kaggle credentials.")
        sys.exit(1)
    
    # Download dataset
    if download_dataset():
        # Verify dataset
        verify_dataset()
        print("\n✅ Setup complete! Dataset ready for use.")
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
