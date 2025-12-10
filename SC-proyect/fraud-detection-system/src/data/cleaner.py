
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from ..utils.logger import LoggerMixin
from ..utils.config import get_config
from ..utils.constants import Columns, Gender


class DataCleaner(LoggerMixin):

    def __init__(self, config_path: Optional[str] = None):

        self.config = get_config(config_path)
        self.cleaning_stats = {}
    
    def clean_data(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        normalize_types: bool = True,
        handle_outliers: bool = False,
        validate_constraints: bool = True
    ) -> pd.DataFrame:

        self.logger.info("Starting data cleaning pipeline...")
        original_shape = df.shape
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # 1. Remove duplicates
        if remove_duplicates:
            df = self.remove_duplicates(df)
        
        # 2. Handle missing values
        if handle_missing:
            df = self.handle_missing_values(df)
        
        # 3. Normalize data types
        if normalize_types:
            df = self.normalize_data_types(df)
        
        # 4. Validate and fix constraints
        if validate_constraints:
            df = self.validate_and_fix_constraints(df)
        
        # 5. Handle outliers (optional)
        if handle_outliers:
            df = self.handle_outliers(df)
        
        # 6. Sort by timestamp
        if Columns.TRANS_DATE_TIME in df.columns:
            df = df.sort_values(Columns.TRANS_DATE_TIME).reset_index(drop=True)
        
        final_shape = df.shape
        rows_removed = original_shape[0] - final_shape[0]
        
        self.logger.info(f"✅ Cleaning complete: {original_shape} → {final_shape} ({rows_removed:,} rows removed)")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate transactions.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame without duplicates
        """
        initial_size = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicates based on transaction number if available
        if Columns.TRANS_NUM in df.columns:
            df = df.drop_duplicates(subset=[Columns.TRANS_NUM], keep='first')
        
        duplicates_removed = initial_size - len(df)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed:,} duplicate rows")
            self.cleaning_stats['duplicates_removed'] = duplicates_removed
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            self.logger.info("No missing values found")
            return df
        
        self.logger.info(f"Found {missing_before:,} missing values")
        
        # Critical columns - drop rows if missing
        critical_cols = [
            Columns.TRANS_DATE_TIME,
            Columns.CC_NUM,
            Columns.AMT,
            Columns.LAT,
            Columns.LONG,
            Columns.IS_FRAUD
        ]
        
        for col in critical_cols:
            if col in df.columns and df[col].isnull().any():
                rows_before = len(df)
                df = df.dropna(subset=[col])
                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    self.logger.info(f"Dropped {rows_dropped:,} rows with missing {col}")
        
        # Categorical columns - fill with 'Unknown'
        categorical_cols = [Columns.CATEGORY, Columns.MERCHANT, Columns.GENDER]
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
                self.logger.info(f"Filled missing {col} with 'Unknown'")
        
        # Numeric columns - fill with median
        numeric_cols = [Columns.CITY_POP]
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.logger.info(f"Filled missing {col} with median: {median_val}")
        
        # Geographic merchant coordinates - fill with user coordinates
        if Columns.MERCH_LAT in df.columns and df[Columns.MERCH_LAT].isnull().any():
            df[Columns.MERCH_LAT] = df[Columns.MERCH_LAT].fillna(df[Columns.LAT])
        
        if Columns.MERCH_LONG in df.columns and df[Columns.MERCH_LONG].isnull().any():
            df[Columns.MERCH_LONG] = df[Columns.MERCH_LONG].fillna(df[Columns.LONG])
        
        missing_after = df.isnull().sum().sum()
        self.logger.info(f"Missing values: {missing_before:,} → {missing_after:,}")
        self.cleaning_stats['missing_values_handled'] = missing_before - missing_after
        
        return df
    
    def normalize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("Normalizing data types...")
        
        # Convert datetime
        if Columns.TRANS_DATE_TIME in df.columns:
            if df[Columns.TRANS_DATE_TIME].dtype == 'object':
                df[Columns.TRANS_DATE_TIME] = pd.to_datetime(
                    df[Columns.TRANS_DATE_TIME],
                    errors='coerce'
                )
        
        if Columns.DOB in df.columns:
            if df[Columns.DOB].dtype == 'object':
                df[Columns.DOB] = pd.to_datetime(
                    df[Columns.DOB],
                    errors='coerce'
                )
        
        # Ensure numeric columns are numeric
        numeric_cols = [
            Columns.AMT,
            Columns.LAT,
            Columns.LONG,
            Columns.MERCH_LAT,
            Columns.MERCH_LONG,
            Columns.CITY_POP,
            Columns.UNIX_TIME
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure categorical columns are strings
        categorical_cols = [
            Columns.CATEGORY,
            Columns.MERCHANT,
            Columns.GENDER,
            Columns.TRANS_NUM
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Ensure fraud label is integer
        if Columns.IS_FRAUD in df.columns:
            df[Columns.IS_FRAUD] = df[Columns.IS_FRAUD].astype(int)
        
        self.logger.info("✅ Data types normalized")
        
        return df
    
    def validate_and_fix_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix constraint violations.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with fixed constraints
        """
        self.logger.info("Validating constraints...")
        
        initial_size = len(df)
        
        # Amount constraints
        if Columns.AMT in df.columns:
            # Remove negative amounts
            invalid_amt = (df[Columns.AMT] < 0) | (df[Columns.AMT] > 100000)
            if invalid_amt.any():
                count = invalid_amt.sum()
                df = df[~invalid_amt]
                self.logger.info(f"Removed {count:,} rows with invalid amounts")
        
        # Geographic constraints
        if Columns.LAT in df.columns:
            invalid_lat = (df[Columns.LAT] < -90) | (df[Columns.LAT] > 90)
            if invalid_lat.any():
                count = invalid_lat.sum()
                df = df[~invalid_lat]
                self.logger.info(f"Removed {count:,} rows with invalid latitude")
        
        if Columns.LONG in df.columns:
            invalid_long = (df[Columns.LONG] < -180) | (df[Columns.LONG] > 180)
            if invalid_long.any():
                count = invalid_long.sum()
                df = df[~invalid_long]
                self.logger.info(f"Removed {count:,} rows with invalid longitude")
        
        # Gender constraints
        if Columns.GENDER in df.columns:
            valid_genders = [Gender.MALE.value, Gender.FEMALE.value, 'Unknown']
            invalid_gender = ~df[Columns.GENDER].isin(valid_genders)
            if invalid_gender.any():
                count = invalid_gender.sum()
                df = df[~invalid_gender]
                self.logger.info(f"Removed {count:,} rows with invalid gender")
        
        # Fraud label constraints
        if Columns.IS_FRAUD in df.columns:
            invalid_fraud = ~df[Columns.IS_FRAUD].isin([0, 1])
            if invalid_fraud.any():
                count = invalid_fraud.sum()
                df = df[~invalid_fraud]
                self.logger.info(f"Removed {count:,} rows with invalid fraud label")
        
        rows_removed = initial_size - len(df)
        if rows_removed > 0:
            self.logger.info(f"Total rows removed for constraint violations: {rows_removed:,}")
            self.cleaning_stats['constraint_violations'] = rows_removed
        
        return df
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.
        
        Args:
            df: DataFrame to process
            method: Method for outlier detection ('iqr' or 'z-score')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        self.logger.info(f"Handling outliers using {method} method...")
        
        # Only handle amount outliers for now
        if Columns.AMT not in df.columns:
            return df
        
        initial_size = len(df)
        
        if method == 'iqr':
            Q1 = df[Columns.AMT].quantile(0.25)
            Q3 = df[Columns.AMT].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[Columns.AMT] < lower_bound) | (df[Columns.AMT] > upper_bound)
            
        elif method == 'z-score':
            z_scores = np.abs((df[Columns.AMT] - df[Columns.AMT].mean()) / df[Columns.AMT].std())
            outliers = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Don't remove outliers, just log them
        outlier_count = outliers.sum()
        if outlier_count > 0:
            self.logger.info(f"Found {outlier_count:,} outliers in amount column (not removed)")
            self.cleaning_stats['outliers_found'] = outlier_count
        
        return df
    
    def get_cleaning_report(self) -> str:
        """
        Get cleaning statistics report.
        
        Returns:
            Formatted cleaning report
        """
        report = ["=" * 60]
        report.append("DATA CLEANING REPORT")
        report.append("=" * 60)
        
        if not self.cleaning_stats:
            report.append("\nNo cleaning statistics available")
        else:
            report.append("\nCleaning Statistics:")
            for key, value in self.cleaning_stats.items():
                report.append(f"  - {key.replace('_', ' ').title()}: {value:,}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:

    cleaner = DataCleaner()
    return cleaner.clean_data(df, **kwargs)


if __name__ == "__main__":
    # Test data cleaner
    print("Testing data cleaner...")
    
    # Create sample data with issues
    sample_data = {
        Columns.TRANS_DATE_TIME: ['2023-01-01 10:00:00', '2023-01-01 11:00:00', None],
        Columns.CC_NUM: [1234, 5678, 9012],
        Columns.AMT: [100.50, -50.00, 200.00],  # One negative
        Columns.LAT: [40.7128, 34.0522, 100.0],  # One invalid
        Columns.LONG: [-74.0060, -118.2437, -122.4194],
        Columns.IS_FRAUD: [0, 1, 0]
    }
    
    df_test = pd.DataFrame(sample_data)
    print("\nOriginal data:")
    print(df_test)
    
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_data(df_test)
    
    print("\nCleaned data:")
    print(df_cleaned)
    
    print("\n" + cleaner.get_cleaning_report())
