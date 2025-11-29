
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import json

from ..utils.logger import LoggerMixin
from ..utils.config import get_config
from ..utils.constants import Columns
from .schema import get_schema


class DataLoader(LoggerMixin):
    
    def __init__(self, config_path: Optional[str] = None):

        self.config = get_config(config_path)
        self.schema = get_schema()
    
    def load_csv(
        self,
        file_path: Union[str, Path],
        nrows: Optional[int] = None,
        usecols: Optional[list] = None,
        parse_dates: bool = True
    ) -> pd.DataFrame:

        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Loading CSV from {file_path}")
        
        # Determine date columns
        date_cols = [Columns.TRANS_DATE_TIME, Columns.DOB] if parse_dates else None
        
        try:
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                usecols=usecols,
                parse_dates=date_cols,
                low_memory=False
            )
            
            self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_json(
        self,
        file_path: Union[str, Path],
        orient: str = "records"
    ) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            orient: JSON orientation ('records', 'split', 'index', etc.)
            
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Loading JSON from {file_path}")
        
        try:
            df = pd.read_json(file_path, orient=orient)
            
            # Parse date columns
            if Columns.TRANS_DATE_TIME in df.columns:
                df[Columns.TRANS_DATE_TIME] = pd.to_datetime(
                    df[Columns.TRANS_DATE_TIME]
                )
            
            self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading JSON: {e}")
            raise
    
    def load_parquet(
        self,
        file_path: Union[str, Path],
        columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            columns: Columns to load (None = all)
            
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Loading Parquet from {file_path}")
        
        try:
            df = pd.read_parquet(file_path, columns=columns)
            
            self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading Parquet: {e}")
            raise
    
    def load_data(
        self,
        file_path: Optional[Union[str, Path]] = None,
        file_type: Optional[str] = None,
        validate: bool = True,
        sample_frac: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from file (auto-detects format).
        
        Args:
            file_path: Path to data file. If None, uses config.
            file_type: Force file type ('csv', 'json', 'parquet')
            validate: Whether to validate schema
            sample_frac: Fraction of data to sample (0-1)
            **kwargs: Additional arguments for specific loader
            
        Returns:
            DataFrame with loaded data
        """
        # Use default path from config if not provided
        if file_path is None:
            raw_path = Path(self.config.get('paths.data.raw'))
            dataset_name = self.config.get('dataset.name', 'fraudTrain.csv')
            file_path = raw_path / dataset_name
        else:
            file_path = Path(file_path)
        
        # Auto-detect file type
        if file_type is None:
            file_type = file_path.suffix.lower().lstrip('.')
        
        # Load based on file type
        if file_type == 'csv':
            df = self.load_csv(file_path, **kwargs)
        elif file_type == 'json':
            df = self.load_json(file_path, **kwargs)
        elif file_type in ['parquet', 'pq']:
            df = self.load_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Sample if requested
        if sample_frac is not None:
            if not 0 < sample_frac <= 1:
                raise ValueError("sample_frac must be between 0 and 1")
            
            original_size = len(df)
            df = df.sample(frac=sample_frac, random_state=42)
            self.logger.info(f"Sampled {len(df):,} rows ({sample_frac*100:.1f}%) from {original_size:,}")
        
        # Validate schema
        if validate:
            self.logger.info("Validating schema...")
            validation_results = self.schema.validate(df)
            
            if not validation_results["valid"]:
                self.logger.warning("Schema validation failed!")
                report = self.schema.get_validation_report(df)
                self.logger.warning(f"\n{report}")
            else:
                self.logger.info("✅ Schema validation passed")
        
        return df
    
    def load_train_test(
        self,
        train_file: str = "fraudTrain.csv",
        test_file: str = "fraudTest.csv",
        validate: bool = True,
        sample_frac: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test datasets.
        
        Args:
            train_file: Name of training file
            test_file: Name of test file
            validate: Whether to validate schema
            sample_frac: Fraction of data to sample
            
        Returns:
            Tuple of (train_df, test_df)
        """
        raw_path = Path(self.config.get('paths.data.raw'))
        
        self.logger.info("Loading training data...")
        train_df = self.load_data(
            raw_path / train_file,
            validate=validate,
            sample_frac=sample_frac
        )
        
        self.logger.info("Loading test data...")
        test_df = self.load_data(
            raw_path / test_file,
            validate=validate,
            sample_frac=sample_frac
        )
        
        self.logger.info(f"Train set: {len(train_df):,} rows, {train_df[Columns.IS_FRAUD].sum():,} frauds ({train_df[Columns.IS_FRAUD].mean()*100:.2f}%)")
        self.logger.info(f"Test set: {len(test_df):,} rows, {test_df[Columns.IS_FRAUD].sum():,} frauds ({test_df[Columns.IS_FRAUD].mean()*100:.2f}%)")
        
        return train_df, test_df
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            "shape": df.shape,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "fraud_rate": df[Columns.IS_FRAUD].mean() if Columns.IS_FRAUD in df.columns else None,
        }
        
        return info
    
    def save_data(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ):
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            file_type: File type ('csv', 'json', 'parquet')
            **kwargs: Additional arguments for saving
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect file type
        if file_type is None:
            file_type = file_path.suffix.lower().lstrip('.')
        
        self.logger.info(f"Saving data to {file_path}")
        
        try:
            if file_type == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_type == 'json':
                df.to_json(file_path, orient='records', **kwargs)
            elif file_type in ['parquet', 'pq']:
                df.to_parquet(file_path, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.logger.info(f"✅ Data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise


def load_data(*args, **kwargs) -> pd.DataFrame:
    """
    Convenience function to load data.
    
    Returns:
        DataFrame with loaded data
    """
    loader = DataLoader()
    return loader.load_data(*args, **kwargs)


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    
    print("Testing data loader...")
    print("Note: This requires the dataset to be downloaded first.")
    print("Run: python scripts/download_dataset.py")
