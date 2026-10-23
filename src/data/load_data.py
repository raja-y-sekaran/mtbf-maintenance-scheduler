"""Load the raw CMMS extract and the equipment-class table."""
import pandas as pd

from src.utils.paths import DATA_RAW, GROUND_TRUTH


def load_data():
    """Returns (events, classes). Classes holds cost and current PM per class, never the true Weibull values."""
    return pd.read_csv(DATA_RAW / "cmms_events.csv"), pd.read_csv(DATA_RAW / "equipment_classes.csv")


def load_truth():
    """True Weibull parameters, for validation only (known because the data is synthetic)."""
    return pd.read_csv(GROUND_TRUTH / "true_weibull.csv")
