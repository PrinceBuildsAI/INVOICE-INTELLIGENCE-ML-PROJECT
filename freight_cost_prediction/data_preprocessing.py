import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str):
    """Load vendor invoice data from SQLite database."""

    conn = sqlite3.connect(db_path)

    try:
        query = "SELECT * FROM vendor_invoice"
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    return df


def prepare_feature(df: pd.DataFrame):
    """Select features and target variable."""

    X = df[["Quantity", "Dollars"]]
    y = df["Freight"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )