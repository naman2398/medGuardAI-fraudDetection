import requests
import pandas as pd
from google.cloud import storage
import logging
from time import sleep

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

API_ENDPOINTS_BY_YEAR = {
    2018: "https://data.cms.gov/data-api/v1/dataset/fb6d9fe8-38c1-4d24-83d4-0b7b291000b2/data",
    2019: "https://data.cms.gov/data-api/v1/dataset/c957b49e-1323-49e7-8678-c09da387551d/data",
    2020: "https://data.cms.gov/data-api/v1/dataset/c957b49e-1323-49e7-8678-c09da387551d/data",
    2021: "https://data.cms.gov/data-api/v1/dataset/31dc2c47-f297-4948-bfb4-075e1bec3a02/data",
    2022: "https://data.cms.gov/data-api/v1/dataset/e650987d-01b7-4f09-b75e-b0b075afbf98/data",
    2023: "https://data.cms.gov/data-api/v1/dataset/92396110-2aed-4d63-a6a2-5d6207d46a29/data"

}

# API_ENDPOINTS_SUMMARY_BY_YEAR = {
#     2018: "https://data.cms.gov/data-api/v1/dataset/900850df-c9a9-47ce-a9e0-d0ceae5a811f/data",
#     2019: "https://data.cms.gov/data-api/v1/dataset/29d799aa-c660-44fe-a51a-72c4b3e661ac/data",
#     2020: "https://data.cms.gov/data-api/v1/dataset/29d799aa-c660-44fe-a51a-72c4b3e661ac/data",
#     2021: "https://data.cms.gov/data-api/v1/dataset/21555c17-ec1b-4e74-b2c6-925c6cbb3147/data",
#     2022: "https://data.cms.gov/data-api/v1/dataset/21555c17-ec1b-4e74-b2c6-925c6cbb3147/data",
#     2023: "https://data.cms.gov/data-api/v1/dataset/8889d81e-2ee7-448f-8713-f071038289b5/data"
# }

PROJECT_ID = "involuted-fold-474521-h3"
BUCKET_NAME = "medguard_rawdata"
BATCH_SIZE = 5000
MAX_RETRIES = 3

INGEST_TYPE = "details"  # Options: "details" or "summary"

def get_total_records(api_endpoint):
    """Retrieve the total count of records from the API endpoint."""
    params = {'size': 1} 
    
    response = requests.get(api_endpoint, params=params) 
    response.raise_for_status()
    data = response.json()
    if 'meta' in data and 'totalCount' in data['meta']:
        return data['meta']['totalCount']
    return None

def fetch_batch(offset, size, api_endpoint):
    """Fetch a batch of records from the CMS API with retry logic."""
    params = {'offset': offset, 'size': size}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(api_endpoint, 
                                    params=params, 
                                    timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                sleep(2 ** attempt)
            else:
                raise

def create_year_folder(year, ingest_type):
    """Create the folder structure for a specific year in GCS."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    folder_path = f"raw/cms_partb_{ingest_type}_data/cms_partb_{ingest_type}_{year}/"
    blob = bucket.blob(f"{folder_path}.folder_placeholder")
    
    if not blob.exists():
        blob.upload_from_string('', content_type='text/plain')
        logger.info(f"Created folder structure: {folder_path}")
    else:
        logger.info(f"Folder already exists: {folder_path}")

def upload_to_gcs(df, batch_num, year, ingest_type):
    """Upload a DataFrame batch to GCS as a Parquet file."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    blob = bucket.blob(f"raw/cms_partb_{ingest_type}_data/cms_partb_{ingest_type}_{year}/batch_{batch_num:06d}.parquet") 
    
    parquet_bytes = df.to_parquet(index=False)
    blob.upload_from_string(parquet_bytes, content_type='application/octet-stream')
    logger.info(f"Uploaded batch {batch_num} ({len(df)} records) for year {year} ({ingest_type}) to GCS")

def ingest_cms_data(year, api_endpoint, ingest_type):
    """Main function to ingest CMS data for a specific year."""
    logger.info(f"Starting ingestion for year: {year} ({ingest_type})")
    
    create_year_folder(year, ingest_type)
    
    total = get_total_records(api_endpoint) 
    if total:
        logger.info(f"Total records for {year} ({ingest_type}): {total:,}")
    
    offset = 0
    batch_num = 1
    total_ingested = 0
    
    while True:
        logger.info(f"Fetching batch {batch_num} (offset: {offset}) for year {year} ({ingest_type})...")
        data = fetch_batch(offset, BATCH_SIZE, api_endpoint) 
        
        records = data if isinstance(data, list) else data.get('data', [])
        
        if not records:
            logger.info(f"No more data to fetch for year {year} ({ingest_type})")
            break
        
        df = pd.DataFrame(records)
        upload_to_gcs(df, batch_num, year, ingest_type) 
        
        total_ingested += len(records)
        if total:
            logger.info(f"Progress for {year} ({ingest_type}): {total_ingested:,}/{total:,} ({100*total_ingested/total:.1f}%)")
        
        if len(records) < BATCH_SIZE:
            break
        
        offset += BATCH_SIZE
        batch_num += 1
    
    logger.info(f"Ingestion complete for year {year} ({ingest_type}). Total records: {total_ingested:,}")

if __name__ == "__main__":
    if INGEST_TYPE == "summary":
        API_ENDPOINTS = API_ENDPOINTS_SUMMARY_BY_YEAR
        logger.info("Ingesting SUMMARY data")
    else:
        API_ENDPOINTS = API_ENDPOINTS_BY_YEAR
        logger.info("Ingesting DETAIL data")
    
    logger.info(f"Starting multi-year data ingestion for: {list(API_ENDPOINTS.keys())}")
    
    for year, api_endpoint in API_ENDPOINTS.items():
        ingest_cms_data(year, api_endpoint, INGEST_TYPE)
        
    logger.info(f"All years have been processed for {INGEST_TYPE} data.")