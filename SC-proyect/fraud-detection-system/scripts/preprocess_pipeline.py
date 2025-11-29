"""
Complete preprocessing pipeline for fraud detection system.
Loads, cleans, validates, and saves processed data.

Usage:
    python scripts/preprocess_pipeline.py
    python scripts/preprocess_pipeline.py --sample 0.1  # Use 10% sample
    python scripts/preprocess_pipeline.py --skip-download  # If data already downloaded
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.schema import get_schema
from src.utils.logger import setup_logger
from src.utils.config import get_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run complete preprocessing pipeline'
    )
    
    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Sample fraction of data (0-1). Default: use all data'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download (assumes data already exists)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['csv', 'parquet'],
        default='parquet',
        help='Output file format'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data without cleaning'
    )
    
    return parser.parse_args()


def download_dataset():
    """Download dataset from Kaggle."""
    logger = setup_logger("download")
    logger.info("Starting dataset download...")
    
    import subprocess
    
    try:
        subprocess.run(
            [sys.executable, "scripts/download_dataset.py"],
            check=True
        )
        logger.info("✅ Dataset downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        logger.error("❌ Dataset download failed")
        return False


def validate_data(df, logger):
    """Validate data quality."""
    logger.info("Validating data schema...")
    
    schema = get_schema()
    validation_report = schema.get_validation_report(df)
    
    print("\n" + validation_report)
    
    validation_results = schema.validate(df)
    
    return validation_results['valid']


def process_dataset(
    file_name: str,
    output_name: str,
    sample_frac: float = None,
    output_format: str = 'parquet',
    validate_only: bool = False,
    logger = None
):
    """
    Process a single dataset file.
    
    Args:
        file_name: Input file name
        output_name: Output file name
        sample_frac: Fraction to sample
        output_format: Output format
        validate_only: Only validate, don't clean
        logger: Logger instance
    """
    config = get_config()
    
    logger.info("=" * 60)
    logger.info(f"Processing: {file_name}")
    logger.info("=" * 60)
    
    # Load data
    loader = DataLoader()
    logger.info("Loading data...")
    
    df = loader.load_data(
        file_name,
        validate=False,  # We'll validate separately
        sample_frac=sample_frac
    )
    
    logger.info(f"Loaded {len(df):,} rows")
    
    # Display basic info
    logger.info(f"\nDataset Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Fraud rate: {df['is_fraud'].mean()*100:.4f}%")
    logger.info(f"  Date range: {df['trans_date_trans_time'].min()} to {df['trans_date_trans_time'].max()}")
    
    # Validate
    is_valid = validate_data(df, logger)
    
    if validate_only:
        logger.info("Validation complete (--validate-only mode)")
        return df
    
    if not is_valid:
        logger.warning("⚠️  Data validation failed, but proceeding with cleaning...")
    
    # Clean data
    cleaner = DataCleaner()
    logger.info("\nCleaning data...")
    
    df_clean = cleaner.clean_data(
        df,
        remove_duplicates=True,
        handle_missing=True,
        normalize_types=True,
        handle_outliers=False,
        validate_constraints=True
    )
    
    # Display cleaning report
    print("\n" + cleaner.get_cleaning_report())
    
    # Validate cleaned data
    logger.info("\nValidating cleaned data...")
    is_valid_after = validate_data(df_clean, logger)
    
    if is_valid_after:
        logger.info("✅ Cleaned data passes validation")
    else:
        logger.warning("⚠️  Cleaned data still has validation issues")
    
    # Save processed data
    processed_dir = Path(config.get('paths.data.processed'))
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / f"{output_name}.{output_format}"
    
    logger.info(f"\nSaving processed data to {output_file}...")
    loader.save_data(df_clean, output_file, file_type=output_format)
    
    logger.info(f"✅ Saved {len(df_clean):,} rows")
    
    return df_clean


def generate_summary_report(train_df, test_df, logger):
    """Generate summary report of preprocessing."""
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    
    logger.info("\nTraining Set:")
    logger.info(f"  Rows: {len(train_df):,}")
    logger.info(f"  Columns: {len(train_df.columns)}")
    logger.info(f"  Fraud cases: {train_df['is_fraud'].sum():,} ({train_df['is_fraud'].mean()*100:.4f}%)")
    logger.info(f"  Date range: {train_df['trans_date_trans_time'].min()} to {train_df['trans_date_trans_time'].max()}")
    
    logger.info("\nTest Set:")
    logger.info(f"  Rows: {len(test_df):,}")
    logger.info(f"  Columns: {len(test_df.columns)}")
    logger.info(f"  Fraud cases: {test_df['is_fraud'].sum():,} ({test_df['is_fraud'].mean()*100:.4f}%)")
    logger.info(f"  Date range: {test_df['trans_date_trans_time'].min()} to {test_df['trans_date_trans_time'].max()}")
    
    logger.info("\nData Quality:")
    logger.info(f"  Missing values (train): {train_df.isnull().sum().sum()}")
    logger.info(f"  Missing values (test): {test_df.isnull().sum().sum()}")
    logger.info(f"  Duplicates (train): {train_df.duplicated().sum()}")
    logger.info(f"  Duplicates (test): {test_df.duplicated().sum()}")
    
    logger.info("\n✅ Preprocessing pipeline complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Explore data: jupyter notebook notebooks/01_exploratory_analysis.ipynb")
    logger.info("  2. Feature engineering: Start Sprint 2")
    logger.info("  3. Model training: Sprint 2")


def main():
    """Main preprocessing pipeline."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger("preprocess", level="INFO")
    
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION - PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # Download dataset if needed
    if not args.skip_download:
        if not download_dataset():
            logger.error("Cannot proceed without dataset")
            sys.exit(1)
    
    config = get_config()
    raw_path = Path(config.get('paths.data.raw'))
    
    # Check if files exist
    train_file = raw_path / "fraudTrain.csv"
    test_file = raw_path / "fraudTest.csv"
    
    if not train_file.exists() or not test_file.exists():
        logger.error(f"Dataset files not found in {raw_path}")
        logger.error("Please run: python scripts/download_dataset.py")
        sys.exit(1)
    
    # Process training set
    train_df = process_dataset(
        train_file,
        "fraudTrain_processed",
        sample_frac=args.sample,
        output_format=args.output_format,
        validate_only=args.validate_only,
        logger=logger
    )
    
    # Process test set
    test_df = process_dataset(
        test_file,
        "fraudTest_processed",
        sample_frac=args.sample,
        output_format=args.output_format,
        validate_only=args.validate_only,
        logger=logger
    )
    
    # Generate summary
    generate_summary_report(train_df, test_df, logger)


if __name__ == "__main__":
    main()
