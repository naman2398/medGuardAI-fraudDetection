import requests
import pandas as pd
from google.cloud import storage
import logging
from time import sleep

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "https://data.cms.gov/data-api/v1/dataset/92396110-2aed-4d63-a6a2-5d6207d46a29/data"
API_SUMMARY_URL = "https://data.cms.gov/data-api/v1/dataset/8889d81e-2ee7-448f-8713-f071038289b5/data"
PROJECT_ID = "MedGuardAI"
BUCKET_NAME = "medguard_rawdata"
BATCH_SIZE = 5000
MAX_RETRIES = 3

def get_total_records():
    response = requests.get(f"{API_SUMMARY_URL}?size=1")
    response.raise_for_status()
    data = response.json()
    if 'meta' in data and 'totalCount' in data['meta']:
        return data['meta']['totalCount']
    return None

def fetch_batch(offset, size):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(API_SUMMARY_URL, params={'offset': offset, 'size': size}, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                sleep(2 ** attempt)
            else:
                raise

def upload_to_gcs(df, batch_num):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"raw/cms_partb_summary_2023/batch_{batch_num:06d}.parquet")
    
    parquet_bytes = df.to_parquet(index=False)
    blob.upload_from_string(parquet_bytes, content_type='application/octet-stream')
    logger.info(f"Uploaded batch {batch_num} ({len(df)} records) to GCS")

def ingest_cms_data():
    total = get_total_records()
    if total:
        logger.info(f"Total records: {total:,}")
    
    offset = 0
    batch_num = 1
    total_ingested = 0
    
    while True:
        logger.info(f"Fetching batch {batch_num} (offset: {offset})...")
        data = fetch_batch(offset, BATCH_SIZE)
        
        records = data if isinstance(data, list) else data.get('data', [])
        
        if not records:
            logger.info("No more data to fetch")
            break
        
        df = pd.DataFrame(records)
        upload_to_gcs(df, batch_num)
        
        total_ingested += len(records)
        if total:
            logger.info(f"Progress: {total_ingested:,}/{total:,} ({100*total_ingested/total:.1f}%)")
        
        if len(records) < BATCH_SIZE:
            break
        
        offset += BATCH_SIZE
        batch_num += 1
    
    logger.info(f"Ingestion complete. Total records: {total_ingested:,}")

if __name__ == "__main__":
    ingest_cms_data()