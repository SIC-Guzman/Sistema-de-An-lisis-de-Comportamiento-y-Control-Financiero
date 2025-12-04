from enum import Enum


# Dataset column names (Sparkov dataset)
class Columns:
    """Dataset column names."""
    TRANS_DATE_TIME = "trans_date_trans_time"
    CC_NUM = "cc_num"
    MERCHANT = "merchant"
    CATEGORY = "category"
    AMT = "amt"
    FIRST = "first"
    LAST = "last"
    GENDER = "gender"
    STREET = "street"
    CITY = "city"
    STATE = "state"
    ZIP = "zip"
    LAT = "lat"
    LONG = "long"
    CITY_POP = "city_pop"
    JOB = "job"
    DOB = "dob"
    TRANS_NUM = "trans_num"
    UNIX_TIME = "unix_time"
    MERCH_LAT = "merch_lat"
    MERCH_LONG = "merch_long"
    IS_FRAUD = "is_fraud"


# Feature names (to be created during feature engineering)
class Features:
    """Feature names."""
    # Temporal
    HOUR = "hour"
    DAY_OF_WEEK = "day_of_week"
    MONTH = "month"
    IS_WEEKEND = "is_weekend"
    IS_NIGHT = "is_night"
    TIME_SINCE_LAST = "time_since_last_trans"
    TRANS_VELOCITY = "trans_velocity"
    
    # Geographic
    DISTANCE_FROM_HOME = "distance_from_home"
    DISTANCE_FROM_LAST = "distance_from_last"
    GEOGRAPHIC_VELOCITY = "geographic_velocity"
    
    # Behavioral
    AMT_MEAN_7D = "amt_mean_7d"
    AMT_STD_7D = "amt_std_7d"
    TRANS_FREQ_7D = "trans_freq_7d"
    MERCHANT_DIVERSITY = "merchant_diversity"
    CATEGORY_ENTROPY = "category_entropy"
    
    # NLP
    MERCHANT_RISK_SCORE = "merchant_risk_score"


# Transaction categories
class TransactionCategory(str, Enum):
    """Transaction categories."""
    GROCERY = "grocery_pos"
    GAS = "gas_transport"
    MISC = "misc_pos"
    SHOPPING = "shopping_pos"
    ENTERTAINMENT = "entertainment"
    FOOD = "food_dining"
    PERSONAL_CARE = "personal_care"
    HEALTH = "health_fitness"
    MISC_NET = "misc_net"
    SHOPPING_NET = "shopping_net"
    KIDS = "kids_pets"
    HOME = "home"
    TRAVEL = "travel"


# Gender values
class Gender(str, Enum):
    """Gender values."""
    MALE = "M"
    FEMALE = "F"


# Risk levels
class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Suspicious keywords for NLP
SUSPICIOUS_KEYWORDS = [
    "international",
    "overseas",
    "wire",
    "crypto",
    "bitcoin",
    "atm",
    "withdrawal",
    "foreign",
    "exchange",
    "transfer",
    "online",
    "offshore"
]


# Time constants
HOURS_IN_DAY = 24
DAYS_IN_WEEK = 7
SECONDS_IN_HOUR = 3600
SECONDS_IN_DAY = 86400


# Geographic constants
EARTH_RADIUS_KM = 6371  # Earth's radius in kilometers
MAX_REASONABLE_VELOCITY = 800  # km/h (airplane speed)


# Model constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1


# Anomaly thresholds
class Thresholds:
    """Anomaly detection thresholds."""
    HIGH_RISK = 0.8
    MEDIUM_RISK = 0.5
    LOW_RISK = 0.3
    
    # Time-based
    NIGHT_START = 2  # 2 AM
    NIGHT_END = 6    # 6 AM
    
    # Amount-based
    HIGH_AMOUNT_PERCENTILE = 95
    LOW_AMOUNT = 0
    
    # Frequency-based
    MAX_TRANS_PER_HOUR = 10
    MAX_TRANS_PER_DAY = 50


# Data types for validation
EXPECTED_DTYPES = {
    Columns.TRANS_DATE_TIME: "datetime64[ns]",  # Updated: datetime is better than object
    Columns.CC_NUM: "int64",
    Columns.MERCHANT: "object",
    Columns.CATEGORY: "object",
    Columns.AMT: "float64",
    Columns.FIRST: "object",
    Columns.LAST: "object",
    Columns.GENDER: "object",
    Columns.STREET: "object",
    Columns.CITY: "object",
    Columns.STATE: "object",
    Columns.ZIP: "int64",
    Columns.LAT: "float64",
    Columns.LONG: "float64",
    Columns.CITY_POP: "int64",
    Columns.JOB: "object",
    Columns.DOB: "datetime64[ns]",  # Updated: datetime is better than object
    Columns.TRANS_NUM: "object",
    Columns.UNIX_TIME: "int64",
    Columns.MERCH_LAT: "float64",
    Columns.MERCH_LONG: "float64",
    Columns.IS_FRAUD: "int64"
}


# Required columns
REQUIRED_COLUMNS = [
    Columns.TRANS_DATE_TIME,
    Columns.CC_NUM,
    Columns.MERCHANT,
    Columns.CATEGORY,
    Columns.AMT,
    Columns.LAT,
    Columns.LONG,
    Columns.IS_FRAUD
]


if __name__ == "__main__":
    print("Available transaction categories:")
    for cat in TransactionCategory:
        print(f"  - {cat.value}")
    
    print("\nRisk levels:")
    for risk in RiskLevel:
        print(f"  - {risk.value}")