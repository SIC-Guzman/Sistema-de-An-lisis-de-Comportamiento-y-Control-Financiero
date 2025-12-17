import pandas as pd
import numpy as np
from typing import Optional
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
        df = df.copy()

        if remove_duplicates:
            df = self.remove_duplicates(df)

        if handle_missing:
            df = self.handle_missing_values(df)

        if normalize_types:
            df = self.normalize_data_types(df)

        if validate_constraints:
            df = self.validate_and_fix_constraints(df)

        if handle_outliers:
            df = self.handle_outliers(df)

        if Columns.TRANS_DATE_TIME in df.columns:
            df = df.sort_values(Columns.TRANS_DATE_TIME).reset_index(drop=True)

        self.logger.info(f"✅ Cleaning complete: {original_shape} → {df.shape}")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_size = len(df)
        df = df.drop_duplicates()

        if Columns.TRANS_NUM in df.columns:
            df = df.drop_duplicates(subset=[Columns.TRANS_NUM], keep="first")

        removed = initial_size - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed:,} duplicate rows")
            self.cleaning_stats["duplicates_removed"] = removed

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_before = df.isnull().sum().sum()
        if missing_before == 0:
            self.logger.info("No missing values found")
            return df

        self.logger.info(f"Found {missing_before:,} missing values")

        # CRÍTICOS REALES para que el modelo no se rompa:
        # (NO forzamos is_fraud, ni trans_date_trans_time)
        critical_cols = [Columns.AMT, Columns.LAT, Columns.LONG, Columns.UNIX_TIME]
        present_critical = [c for c in critical_cols if c in df.columns]

        if present_critical:
            before = len(df)
            df = df.dropna(subset=present_critical)
            dropped = before - len(df)
            if dropped > 0:
                self.logger.info(f"Dropped {dropped:,} rows with missing critical cols: {present_critical}")
                self.cleaning_stats["critical_missing_removed"] = dropped

        # Categóricas: tolerante
        for col in [Columns.CATEGORY, Columns.MERCHANT, Columns.GENDER]:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")

        # city_pop: mediana (si existe)
        if Columns.CITY_POP in df.columns and df[Columns.CITY_POP].isnull().any():
            median_val = df[Columns.CITY_POP].median()
            df[Columns.CITY_POP] = df[Columns.CITY_POP].fillna(median_val)

        # Merch coords: fallback seguro
        if Columns.MERCH_LAT in df.columns:
            df[Columns.MERCH_LAT] = df[Columns.MERCH_LAT].fillna(df.get(Columns.LAT))
        if Columns.MERCH_LONG in df.columns:
            df[Columns.MERCH_LONG] = df[Columns.MERCH_LONG].fillna(df.get(Columns.LONG))

        # Género: NORMALIZAR, NUNCA eliminar filas
        if Columns.GENDER in df.columns:
            df[Columns.GENDER] = df[Columns.GENDER].where(
                df[Columns.GENDER].isin([Gender.MALE.value, Gender.FEMALE.value]),
                "Unknown"
            )

        missing_after = df.isnull().sum().sum()
        self.logger.info(f"Missing values: {missing_before:,} → {missing_after:,}")
        self.cleaning_stats["missing_values_handled"] = missing_before - missing_after

        return df

    def normalize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Normalizing data types...")

        # Datetimes tolerantes
        if Columns.TRANS_DATE_TIME in df.columns:
            df[Columns.TRANS_DATE_TIME] = pd.to_datetime(df[Columns.TRANS_DATE_TIME], errors="coerce")
        if Columns.DOB in df.columns:
            df[Columns.DOB] = pd.to_datetime(df[Columns.DOB], errors="coerce")

        # Numéricos (esto maneja letras en números)
        numeric_cols = [
            Columns.AMT, Columns.LAT, Columns.LONG, Columns.MERCH_LAT, Columns.MERCH_LONG,
            Columns.CITY_POP, Columns.UNIX_TIME
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Categóricas a string
        for col in [Columns.CATEGORY, Columns.MERCHANT, Columns.GENDER, Columns.TRANS_NUM, Columns.JOB]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # is_fraud si existe, normalizar a int (pero NO hacerlo crítico)
        if Columns.IS_FRAUD in df.columns:
            df[Columns.IS_FRAUD] = pd.to_numeric(df[Columns.IS_FRAUD], errors="coerce")
            df[Columns.IS_FRAUD] = df[Columns.IS_FRAUD].fillna(0).astype(int)

        self.logger.info("✅ Data types normalized")
        return df

    def validate_and_fix_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Validating constraints...")
        initial_size = len(df)

        # Hard constraints (para no romper modelo)
        if Columns.AMT in df.columns:
            df = df[(df[Columns.AMT] >= 0) & (df[Columns.AMT] <= 100000)]

        if Columns.LAT in df.columns:
            df = df[df[Columns.LAT].between(-90, 90)]
        if Columns.LONG in df.columns:
            df = df[df[Columns.LONG].between(-180, 180)]

        # unix_time debe existir y ser numérico
        if Columns.UNIX_TIME in df.columns:
            df = df.dropna(subset=[Columns.UNIX_TIME])

        removed = initial_size - len(df)
        if removed > 0:
            self.logger.info(f"Total rows removed for constraint violations: {removed:,}")
            self.cleaning_stats["constraint_violations"] = removed

        return df

    def handle_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 3.0) -> pd.DataFrame:
        # Mantener compatibilidad: detectar/loggear, no eliminar
        if Columns.AMT not in df.columns:
            return df

        if method == "iqr":
            q1 = df[Columns.AMT].quantile(0.25)
            q3 = df[Columns.AMT].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = (df[Columns.AMT] < lower) | (df[Columns.AMT] > upper)
        elif method == "z-score":
            z = np.abs((df[Columns.AMT] - df[Columns.AMT].mean()) / df[Columns.AMT].std())
            outliers = z > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        count = int(outliers.sum())
        if count > 0:
            self.logger.info(f"Found {count:,} outliers in amount column (not removed)")
            self.cleaning_stats["outliers_found"] = count

        return df

    def get_cleaning_report(self) -> str:
        report = ["=" * 60, "DATA CLEANING REPORT", "=" * 60]
        if not self.cleaning_stats:
            report.append("\nNo cleaning statistics available")
        else:
            report.append("\nCleaning Statistics:")
            for k, v in self.cleaning_stats.items():
                report.append(f"  - {k.replace('_', ' ').title()}: {v:,}")
        report.append("\n" + "=" * 60)
        return "\n".join(report)


def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    cleaner = DataCleaner()
    return cleaner.clean_data(df, **kwargs)
