"""
Data schema definitions and validation.
Defines expected structure and types for the fraud detection dataset.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import pandas as pd

from ..utils.constants import Columns, EXPECTED_DTYPES, REQUIRED_COLUMNS


@dataclass
class DataSchema:
    """Schema definition for fraud detection dataset."""
    
    # Column names
    columns: List[str] = None
    
    # Data types
    dtypes: Dict[str, str] = None
    
    # Required columns
    required_columns: List[str] = None
    
    # Value constraints
    constraints: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.columns is None:
            self.columns = list(EXPECTED_DTYPES.keys())
        
        if self.dtypes is None:
            self.dtypes = EXPECTED_DTYPES
        
        if self.required_columns is None:
            self.required_columns = REQUIRED_COLUMNS
        
        if self.constraints is None:
            self.constraints = {
                Columns.AMT: {
                    "min": 0,
                    "max": 100000,  # Reasonable maximum transaction
                },
                Columns.LAT: {
                    "min": -90,
                    "max": 90,
                },
                Columns.LONG: {
                    "min": -180,
                    "max": 180,
                },
                Columns.MERCH_LAT: {
                    "min": -90,
                    "max": 90,
                },
                Columns.MERCH_LONG: {
                    "min": -180,
                    "max": 180,
                },
                Columns.IS_FRAUD: {
                    "values": [0, 1],
                },
                Columns.GENDER: {
                    "values": ["M", "F"],
                }
            }
    
    def validate_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate that dataframe has required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with 'missing' and 'extra' column lists
        """
        df_columns = set(df.columns)
        expected_columns = set(self.required_columns)
        
        missing = list(expected_columns - df_columns)
        extra = list(df_columns - set(self.columns))
        
        return {
            "missing": missing,
            "extra": extra,
        }
    
    def validate_dtypes(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Validate data types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary of columns with incorrect types
        """
        incorrect_types = {}
        
        for col, expected_dtype in self.dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                # Allow int64 for int32, float64 for float32, etc.
                if not self._dtypes_compatible(actual_dtype, expected_dtype):
                    incorrect_types[col] = {
                        "expected": expected_dtype,
                        "actual": actual_dtype,
                    }
        
        return incorrect_types
    
    def _dtypes_compatible(self, actual: str, expected: str) -> bool:
        """Check if two dtypes are compatible."""
        # Exact match
        if actual == expected:
            return True
        
        # Int types are compatible
        if "int" in actual and "int" in expected:
            return True
        
        # Float types are compatible
        if "float" in actual and "float" in expected:
            return True
        
        # Object and string are compatible
        if (actual == "object" and expected == "object") or \
           (actual == "string" and expected == "object"):
            return True
        
        return False
    
    def validate_constraints(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Validate value constraints.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary of columns with rows violating constraints
        """
        violations = {}
        
        for col, constraint in self.constraints.items():
            if col not in df.columns:
                continue
            
            col_violations = []
            
            # Check min/max constraints
            if "min" in constraint:
                mask = df[col] < constraint["min"]
                col_violations.extend(df[mask].index.tolist())
            
            if "max" in constraint:
                mask = df[col] > constraint["max"]
                col_violations.extend(df[mask].index.tolist())
            
            # Check allowed values
            if "values" in constraint:
                mask = ~df[col].isin(constraint["values"])
                col_violations.extend(df[mask].index.tolist())
            
            if col_violations:
                violations[col] = list(set(col_violations))  # Remove duplicates
        
        return violations
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validations.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "columns": self.validate_columns(df),
            "dtypes": self.validate_dtypes(df),
            "constraints": self.validate_constraints(df),
        }
        
        # Check if any validation failed
        if results["columns"]["missing"]:
            results["valid"] = False
        
        if results["dtypes"]:
            results["valid"] = False
        
        if results["constraints"]:
            results["valid"] = False
        
        return results
    
    def get_validation_report(self, df: pd.DataFrame) -> str:
        """
        Get human-readable validation report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Formatted validation report
        """
        results = self.validate(df)
        
        report = ["=" * 60]
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "✅ PASSED" if results["valid"] else "❌ FAILED"
        report.append(f"\nStatus: {status}")
        report.append(f"Rows: {len(df):,}")
        report.append(f"Columns: {len(df.columns)}")
        
        # Column validation
        report.append("\n" + "-" * 60)
        report.append("COLUMN VALIDATION")
        report.append("-" * 60)
        
        if results["columns"]["missing"]:
            report.append(f"❌ Missing required columns: {results['columns']['missing']}")
        else:
            report.append("✅ All required columns present")
        
        if results["columns"]["extra"]:
            report.append(f"⚠️  Extra columns: {results['columns']['extra']}")
        
        # Data type validation
        report.append("\n" + "-" * 60)
        report.append("DATA TYPE VALIDATION")
        report.append("-" * 60)
        
        if results["dtypes"]:
            report.append("❌ Incorrect data types:")
            for col, info in results["dtypes"].items():
                report.append(f"  - {col}: expected {info['expected']}, got {info['actual']}")
        else:
            report.append("✅ All data types correct")
        
        # Constraint validation
        report.append("\n" + "-" * 60)
        report.append("CONSTRAINT VALIDATION")
        report.append("-" * 60)
        
        if results["constraints"]:
            report.append("❌ Constraint violations:")
            for col, violations in results["constraints"].items():
                report.append(f"  - {col}: {len(violations)} rows violate constraints")
        else:
            report.append("✅ All constraints satisfied")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Global schema instance
_schema = None


def get_schema() -> DataSchema:
    """Get global schema instance."""
    global _schema
    
    if _schema is None:
        _schema = DataSchema()
    
    return _schema


if __name__ == "__main__":
    # Test schema
    schema = get_schema()
    print("Schema columns:", len(schema.columns))
    print("Required columns:", schema.required_columns)
    print("\nConstraints:")
    for col, constraint in schema.constraints.items():
        print(f"  {col}: {constraint}")
