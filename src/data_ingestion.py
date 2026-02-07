import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging


# ------------------ LOGGING SETUP ------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(log_dir, "data_ingestion.log"))
file_handler.setFormatter(formatter)

# Avoid duplicate logs
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ------------------ FUNCTIONS ------------------
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
        df = df.rename(columns={"v1": "target", "v2": "text"})
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error("Missing column: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    try:
        raw_path = os.path.join(data_path, "raw")
        os.makedirs(raw_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_path, "test.csv"), index=False)

        logger.debug("Train and test data saved at %s", raw_path)
    except Exception as e:
        logger.error("Failed to save data: %s", e)
        raise


# ------------------ MAIN ------------------
def main():
    try:
        TEST_SIZE = 0.2
        DATA_URL = (
            "https://raw.githubusercontent.com/"
            "Vishnu02071999/MLOPS-simple-pipeline/"
            "refs/heads/main/spam.csv"
        )

        df = load_data(DATA_URL)
        df = preprocess_data(df)

        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=2
        )

        save_data(train_df, test_df, data_path="data")

    except Exception as e:
        logger.error("Data ingestion failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()