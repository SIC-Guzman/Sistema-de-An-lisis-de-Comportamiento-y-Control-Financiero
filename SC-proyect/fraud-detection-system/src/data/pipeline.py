"""
Lo que hace este módulo es definir EL único punto de entrada para cargar datasets desde data/raw/,
validarlos contra el schema, limpiarlos y devolver DataFrames listos para que lo usen

Creacion de fluho
- Cargar datasets desde disco (data/raw)
- Validar esquema
- Limpiar datos
- Devolver el DataFrame 
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from .loader import DataLoader
from .cleaner import DataCleaner
from .schema import get_schema
from ..utils.logger import LoggerMixin
from ..utils.constants import Columns


class DataPipeline(LoggerMixin):
    def __init__(self, base_path: Optional[Path] = None):
        # Raíz del proyecto (…/fraud-detection-system)
        if base_path is None:
            self.base_path = Path(__file__).resolve().parents[2]
        else:
            self.base_path = Path(base_path)

        # data/raw/
        self.raw_path = self.base_path / "data" / "raw"

        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.schema = get_schema()

        self.logger.info(f"DataPipeline initialized | raw_path={self.raw_path}")

    def _get_dataset_path(self, dataset: str) -> Path:
        #Devuelve la ruta al dataset solicitado.
        if dataset == "train":
            return self.raw_path / "fraudTrain.csv"
        elif dataset == "test":
            return self.raw_path / "fraudTest.csv"
        else:
            raise ValueError("dataset must be 'train' or 'test'")

    def load_and_prepare(
        self,
        dataset: str = "train",
        sample_frac: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Carga y prepara un dataset completo.
        """
        dataset_path = self._get_dataset_path(dataset)

        self.logger.info(f"Loading dataset: {dataset_path.name}")

        # 1. Cargar datos
        df = self.loader.load_data(
            file_path=dataset_path,
            validate=False,          # Validamos manualmente después
            sample_frac=sample_frac
        )

        self.logger.info(f"Raw data shape: {df.shape}")

        # 2. Validar esquema (informativo, no bloqueante)
        validation = self.schema.validate(df)
        if not validation["valid"]:
            self.logger.warning("Schema validation issues detected")
            report = self.schema.get_validation_report(df)
            self.logger.warning(f"\n{report}")
        else:
            self.logger.info("Schema validation passed")

        # 3. Limpieza de datos
        df_clean = self.cleaner.clean_data(
            df,
            remove_duplicates=True,
            handle_missing=True,
            normalize_types=True,
            validate_constraints=True,
            handle_outliers=False
        )

        self.logger.info(f"Cleaned data shape: {df_clean.shape}")

        # 4. Validación mínima post-limpieza (crítica)
        self._validate_minimal_requirements(df_clean)

        return df_clean

    def load_train_test(
        self,
        sample_frac: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga y prepara train y test.
        """
        train_df = self.load_and_prepare("train", sample_frac)
        test_df = self.load_and_prepare("test", sample_frac)

        self.logger.info(
            f"Train fraud rate: {train_df[Columns.IS_FRAUD].mean():.4f} | "
            f"Test fraud rate: {test_df[Columns.IS_FRAUD].mean():.4f}"
        )

        return train_df, test_df

    def _validate_minimal_requirements(self, df: pd.DataFrame):
        """Validaciones críticas para continuar el pipeline."""
        required = [
            Columns.TRANS_DATE_TIME,
            Columns.CC_NUM,
            Columns.AMT,
            Columns.LAT,
            Columns.LONG,
            Columns.MERCH_LAT,
            Columns.MERCH_LONG,
            Columns.UNIX_TIME,
            Columns.IS_FRAUD,
        ]

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing critical columns after cleaning: {missing}")

        if df.empty:
            raise ValueError("Dataset is empty after cleaning")

        self.logger.info("Minimal requirements validation passed")


# Funciones de conveniencia

def load_prepared_dataset(dataset: str = "train") -> pd.DataFrame:
    pipeline = DataPipeline()
    return pipeline.load_and_prepare(dataset)


def load_prepared_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = DataPipeline()
    return pipeline.load_train_test()
