import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, ID_COL, TARGET_COL
from src.features import add_features


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])
    df = df.drop_duplicates(subset=[ID_COL])
    df = df.fillna({"TotalCharges": 0})
    return df


def main() -> None:
    df = load_raw_data()
    df = clean_data(df)
    df = add_features(df)
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"Saved processed data to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
