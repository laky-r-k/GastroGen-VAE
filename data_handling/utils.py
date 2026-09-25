import pandas as pd
from typing import Optional

def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads raw gastric cancer CSV dataset.
    """
   
    print(f"[DataLoader] Loading raw dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[DataLoader] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


