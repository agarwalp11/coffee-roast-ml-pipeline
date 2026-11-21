
from pathlib import Path
from google.cloud import storage

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_PATH = PROJECT_ROOT / "data" / "processed" / "coffee_downloaded.csv"

# Google Cloud Storage information
BUCKET_NAME = "coffee-roast-ml-pipeline-agarwalp11"
BLOB_NAME = "coffee_cleaned.csv"


def ingest_data():
    # 1. Connect to Google Cloud Storage
    client = storage.Client(project="coffee-roast-ml-pipeline")

    # 2. Access our storage bucket
    bucket = client.bucket(BUCKET_NAME)

    # 3. Locate the CSV file in the bucket
    # blob is our csv file
    blob = bucket.blob(BLOB_NAME)

    # 4. Create the local folder if needed
    DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 5. Download the dataset
    blob.download_to_filename(str(DOWNLOAD_PATH))

    print(f"Dataset downloaded successfully to: {DOWNLOAD_PATH}")

    return DOWNLOAD_PATH


if __name__ == "__main__":
    ingest_data()
