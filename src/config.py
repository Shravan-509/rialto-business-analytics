"""Configuration constants for the Rialto Business Analytics Platform.

All calculations are driven from the source CSV and these schema constants.
Replacing the CSV with another file that uses the same columns will refresh
the application outputs automatically.
"""

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
POWER_BI_DIR = DATA_DIR / "power_bi"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_FILE = RAW_DATA_DIR / "Rialto Data.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "rialto_cleaned.csv"

TRANSACTION_ID_COLUMN = "Transaction_ID"
CUSTOMER_ID_COLUMN = "Customer_ID"
DATE_COLUMN = "Transaction_Date"
REVENUE_COLUMN = "Amount"
RETURN_COLUMN = "Returned"
SATISFACTION_COLUMN = "Satisfaction_Score"
TEXT_COLUMN = "Feedback_Text"

REQUIRED_COLUMNS = [
    TRANSACTION_ID_COLUMN,
    CUSTOMER_ID_COLUMN,
    DATE_COLUMN,
    REVENUE_COLUMN,
    RETURN_COLUMN,
    SATISFACTION_COLUMN,
    TEXT_COLUMN,
]

LOW_SATISFACTION_THRESHOLD = 2
REVENUE_BAND_LABELS = ["Low", "Mid", "High", "Premium"]

APP_TITLE = "Rialto AI-Powered Business Analytics Platform"
APP_SUBTITLE = (
    "Interactive decision intelligence for revenue, customers, returns, "
    "and satisfaction"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "openai"
).strip().lower()

MAX_OUTPUT_TOKENS = 350
