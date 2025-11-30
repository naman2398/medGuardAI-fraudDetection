from pyspark.sql import SparkSession 
from pyspark.sql.functions import (
    col, when, regexp_replace, lit,
    input_file_name, regexp_extract,
    min, max, mean, stddev, sum as sum_, percentile_approx, coalesce,
    year, to_date, lower, trim
)
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import floor
from pyspark.sql.window import Window
from google.cloud import bigquery
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
YEARS = ["2018", "2019", "2020", "2021", "2022", "2023"]
BUCKET_BASE_DETAILS = "gs://medguard_rawdata/raw/cms_partb_details_data"
BUCKET_BASE_SUMMARY = "gs://medguard_rawdata/raw/cms_partb_summary_data"
LEIE_PATH = "gs://medguard_rawdata/raw/fraud_labels/"

BIGQUERY_DATASET = "medguard_processed_all_years"
BIGQUERY_TABLE = "fraud_training_data_enhanced_v2"  # New table with Z-score & risk ratio features
BIGQUERY_LOCATION = "US"
STAGING_BUCKET = "medguard_rawdata"

FRAUD_EXCLTYPE_CODES = [
    '1128a1',
    '1128a2',
    '1128a3',
    '1128a4',
    '1128b4',
    '1128b7'
]

currency_columns = [
    'avg_sbmtd_chrg', 'avg_mdcr_alowd_amt',
    'avg_mdcr_pymt_amt', 'avg_mdcr_stdzd_amt'
]

pii_cols = [
    "rndrng_prvdr_last_org_name", "rndrng_prvdr_first_name", "rndrng_prvdr_mi",
    "rndrng_prvdr_crdntls", "rndrng_prvdr_st1", "rndrng_prvdr_st2", "rndrng_prvdr_city",
    "rndrng_prvdr_state_abrvtn", "rndrng_prvdr_state_fips", "rndrng_prvdr_zip5",
    "rndrng_prvdr_ruca", "rndrng_prvdr_ruca_desc", "rndrng_prvdr_cntry",
    "hcpcs_desc"
]

integer_columns = [
    'tot_benes', 'tot_srvcs', 'tot_bene_day_srvcs', 'hcpcs_cd', 'rndrng_npi'
]

string_columns = [
    'rndrng_prvdr_ent_cd', 'rndrng_prvdr_type', 'rndrng_prvdr_mdcr_prtcptg_ind',
    'hcpcs_drug_ind', 'place_of_srvc'
]

# Columns for Z-score computation (peer comparison features)
ZSCORE_NUMERIC_COLS = [
    'tot_srvcs',
    'avg_mdcr_pymt_amt',
    'tot_benes',
    'avg_sbmtd_chrg'
]


# Phase 0: Z-Score and Risk Ratio Features (NEW - Model Enhancement)
def compute_zscore_features(df, group_col='rndrng_prvdr_type', numeric_cols=None):
    """
    Compute peer-relative Z-scores and ratios for numeric columns.
    
    Z-scores normalize values relative to the provider's specialty group,
    enabling detection of outliers within peer groups.
    
    Args:
        df: PySpark DataFrame with aggregated features
        group_col: Column to partition by (default: provider specialty)
        numeric_cols: List of numeric columns to compute Z-scores for
        
    Returns:
        DataFrame with additional Z-score and ratio columns
    """
    if numeric_cols is None:
        numeric_cols = ZSCORE_NUMERIC_COLS
    
    logger.info(f"Computing Z-score features for {len(numeric_cols)} columns...")
    logger.info(f"Partitioning by: {group_col}")
    
    # Define window spec partitioned by provider specialty
    window_spec = Window.partitionBy(group_col)
    
    for col_name in numeric_cols:
        # Check if column exists (handle different naming from aggregation)
        # Try both raw name and aggregated name patterns
        actual_col = None
        for candidate in [col_name, f"{col_name}_sum", f"{col_name}_mean"]:
            if candidate in df.columns:
                actual_col = candidate
                break
        
        if actual_col is None:
            logger.warning(f"Column {col_name} not found, skipping Z-score computation")
            continue
            
        logger.info(f"  Computing Z-score for: {actual_col}")
        
        # Calculate group statistics
        group_mean_col = f"{col_name}_group_mean"
        group_std_col = f"{col_name}_group_std"
        
        df = df.withColumn(
            group_mean_col,
            mean(col(actual_col)).over(window_spec)
        )
        
        df = df.withColumn(
            group_std_col,
            stddev(col(actual_col)).over(window_spec)
        )
        
        # Handle NULL/zero stddev (single-row groups or identical values)
        # Use small epsilon (0.01) to avoid division by zero
        df = df.withColumn(
            group_std_col,
            when(
                (col(group_std_col).isNull()) | (col(group_std_col) == 0),
                lit(0.01)
            ).otherwise(col(group_std_col))
        )
        
        # Compute Z-score: (X - μ) / σ
        zscore_col = f"{col_name}_zscore"
        df = df.withColumn(
            zscore_col,
            (col(actual_col) - col(group_mean_col)) / col(group_std_col)
        )
        
        # Compute Ratio: X / μ (with safeguard for zero mean)
        ratio_col = f"{col_name}_ratio"
        df = df.withColumn(
            ratio_col,
            when(
                col(group_mean_col) == 0,
                lit(1.0)  # If group mean is 0, ratio is 1 (no deviation)
            ).otherwise(
                col(actual_col) / col(group_mean_col)
            )
        )
        
        # Drop intermediate columns (keep only zscore and ratio)
        df = df.drop(group_mean_col, group_std_col)
        
        logger.info(f"    Created: {zscore_col}, {ratio_col}")
    
    # Log summary of new features
    zscore_cols = [c for c in df.columns if '_zscore' in c or '_ratio' in c]
    logger.info(f"✓ Z-score feature computation complete. Added {len(zscore_cols)} columns")
    
    return df


def compute_risk_ratios(df):
    """
    Compute risk ratio features known to correlate with fraud.
    
    These are explicit interaction features derived from domain knowledge:
    - Billing Inflation: Ratio of submitted charges to Medicare payment
    - Service Density: Ratio of services to unique beneficiaries
    
    Args:
        df: PySpark DataFrame with aggregated features
        
    Returns:
        DataFrame with additional risk ratio columns
    """
    logger.info("Computing risk ratio features...")
    
    # Billing Inflation: avg_sbmtd_chrg / avg_mdcr_pymt_amt
    # High values indicate excessive upcoding or billing inflation
    # Find the actual column names (may be aggregated)
    sbmtd_col = None
    pymt_col = None
    srvcs_col = None
    benes_col = None
    
    # Check for aggregated column names
    for candidate in ['avg_sbmtd_chrg', 'average_submitted_chrg_amt_sum', 'average_submitted_chrg_amt_mean']:
        if candidate in df.columns:
            sbmtd_col = candidate
            break
            
    for candidate in ['avg_mdcr_pymt_amt', 'average_medicare_payment_amt_sum', 'average_medicare_payment_amt_mean']:
        if candidate in df.columns:
            pymt_col = candidate
            break
            
    for candidate in ['tot_srvcs', 'line_srvc_cnt_sum', 'line_srvc_cnt_mean']:
        if candidate in df.columns:
            srvcs_col = candidate
            break
            
    for candidate in ['tot_benes', 'bene_unique_cnt_sum', 'bene_unique_cnt_mean']:
        if candidate in df.columns:
            benes_col = candidate
            break
    
    if sbmtd_col and pymt_col:
        logger.info(f"  Computing billing_inflation from {sbmtd_col} / {pymt_col}")
        df = df.withColumn(
            'billing_inflation',
            when(
                (col(pymt_col).isNull()) | (col(pymt_col) == 0),
                lit(1.0)  # Default to 1.0 if payment is 0 or null
            ).otherwise(
                col(sbmtd_col) / col(pymt_col)
            )
        )
        logger.info("    Created: billing_inflation")
    else:
        logger.warning(f"Could not compute billing_inflation: sbmtd={sbmtd_col}, pymt={pymt_col}")
    
    # Service Density: tot_srvcs / tot_benes
    # High values indicate potential churning or unnecessary services
    if srvcs_col and benes_col:
        logger.info(f"  Computing service_density from {srvcs_col} / {benes_col}")
        df = df.withColumn(
            'service_density',
            when(
                (col(benes_col).isNull()) | (col(benes_col) == 0),
                lit(1.0)  # Default to 1.0 if no beneficiaries
            ).otherwise(
                col(srvcs_col) / col(benes_col)
            )
        )
        logger.info("    Created: service_density")
    else:
        logger.warning(f"Could not compute service_density: srvcs={srvcs_col}, benes={benes_col}")
    
    logger.info("✓ Risk ratio computation complete")
    
    return df


# Phase 1: Cleaning
def clean_dataframe(df):
    """Clean and standardize raw CMS Part B data."""
    logger.info("Starting data cleaning...")
    
    # Standardize column names to lowercase
    df = df.select([col(c).alias(c.lower()) for c in df.columns])
    
    # Rename specific column
    df = df.withColumnRenamed("rfrg_crdntls", "rfrg_prvdr_crdntls")
    
    # Drop PII columns
    df = df.drop(*pii_cols)
    logger.info(f"Dropped {len(pii_cols)} PII columns")
    
    # Handle CMS suppression: replace NULL tot_benes with 5
    df = df.withColumn(
        'tot_benes',
        when(col('tot_benes').isNull(), 5).otherwise(col('tot_benes'))
    )
    
    # Clean currency columns: remove $ and commas, cast to float
    for col_name in currency_columns:
        df = df.withColumn(
            col_name,
            regexp_replace(col(col_name), r'[\$,]', '').cast('float')
        )
    
    # Fix provider type inconsistencies
    df = df.withColumn(
        'rndrng_prvdr_type',
        when(col('rndrng_prvdr_type') == 'Allergy/ Immunology', 'Allergy/Immunology')
        .when(col('rndrng_prvdr_type') == 'CRNA', 'Certified Registered Nurse Anesthetist')
        .otherwise(col('rndrng_prvdr_type'))
    )
    
    # Remove duplicate rows
    df = df.dropDuplicates()
    
    # Cast year to string (already extracted from filename)
    df = df.withColumn("year", col("year").cast('string'))
    
    # Cast integer columns
    for col_name in integer_columns:
        df = df.withColumn(col_name, col(col_name).cast('int'))
    
    # Cast string columns
    for col_name in string_columns:
        df = df.withColumn(col_name, col(col_name).cast('string'))
    
    logger.info("Data cleaning completed")
    return df


# Phase 2: Aggregation
def aggregate_features(df):
    """Aggregate features by provider NPI, year, type, and place of service."""
    logger.info("Starting feature aggregation...")
    
    group_keys = [
        "rndrng_npi", "year", "rndrng_prvdr_type", "place_of_srvc"
    ]
    
    agg_targets = {
        "tot_srvcs": "line_srvc_cnt",
        "tot_benes": "bene_unique_cnt",
        "tot_bene_day_srvcs": "bene_day_srvc_cnt",
        "avg_sbmtd_chrg": "average_submitted_chrg_amt",
        "avg_mdcr_pymt_amt": "average_medicare_payment_amt"
    }
    
    for old, new in agg_targets.items():
        df = df.withColumnRenamed(old, new)
    
    agg_exprs = []
    for col_name in agg_targets.values():
        agg_exprs.extend([
            min(col_name).alias(f"{col_name}_min"),
            max(col_name).alias(f"{col_name}_max"),
            mean(col_name).alias(f"{col_name}_mean"),
            percentile_approx(col_name, 0.5).alias(f"{col_name}_median"),
            sum_(col_name).alias(f"{col_name}_sum"),
            stddev(col_name).alias(f"{col_name}_std")
        ])
    
    df_aggregated = df.groupBy(group_keys).agg(*agg_exprs)
    
    # Replace NULL stddev with 0.0 (happens when only 1 row per group)
    stddev_cols = [f"{v}_std" for v in agg_targets.values()]
    for c in stddev_cols:
        df_aggregated = df_aggregated.withColumn(c, coalesce(col(c), lit(0.0)))
    
    logger.info("Aggregation completed")
    return df_aggregated


# Phase 3: Enrichment and Join
def enrich_and_join(spark, df_aggregated):
    """Enrich aggregated data with provider-level summary statistics."""
    logger.info("Starting enrichment and join...")
    
    # Read all enrichment years at once
    enrich_path = f"{BUCKET_BASE_SUMMARY}/cms_partb_summary_*/*.parquet"
    logger.info(f"Reading all enrichment data from: {enrich_path}")
    
    df_prv_raw = spark.read.parquet(enrich_path)
    
    df_prv_raw = df_prv_raw.select([col(c).alias(c.lower()) for c in df_prv_raw.columns])
    
    # Extract year from file path
    df_prv_raw = df_prv_raw.withColumn("source_file", input_file_name())
    df_prv_raw = df_prv_raw.withColumn(
        "year", 
        regexp_extract("source_file", r'cms_partb_summary_(\d{4})', 1)
    )
    df_prv_raw = df_prv_raw.drop("source_file")
    
    logger.info("✓ Enrichment data loaded successfully from all years")
    
    drop_columns_prv = [
        "rndrng_prvdr_last_org_name", "rndrng_prvdr_first_name", "rndrng_prvdr_mi",
        "rndrng_prvdr_crdntls", "rndrng_prvdr_st1", "rndrng_prvdr_st2", "rndrng_prvdr_city",
        "rndrng_prvdr_state_abrvtn", "rndrng_prvdr_state_fips", "rndrng_prvdr_zip5",
        "rndrng_prvdr_ruca", "rndrng_prvdr_ruca_desc", "rndrng_prvdr_cntry",
        "drug_sprsn_ind", "med_sprsn_ind", "bene_race_wht_cnt", "bene_race_black_cnt", 
        "bene_race_api_cnt", "bene_race_hspnc_cnt", "bene_race_natind_cnt", "bene_race_othr_cnt"
    ]
    df_prv = df_prv_raw.drop(*[c for c in drop_columns_prv if c in df_prv_raw.columns])
    
    numeric_castable = ['string', 'int', 'bigint', 'float', 'double']
    for c in df_prv.columns:
        col_type = df_prv.schema[c].dataType.simpleString()
        if col_type == "string":
            df_prv = df_prv.withColumn(c, regexp_replace(col(c), '[\$,]', ''))
        if col_type in numeric_castable:
            try:
                df_prv = df_prv.withColumn(c, col(c).cast(DoubleType()))
            except Exception as e:
                logger.warning(f"Could not cast column {c} to double: {e}")
    
    df_prv = df_prv.fillna(0.0)
    df_prv = df_prv.withColumn("rndrng_npi", col("rndrng_npi").cast("int"))
    df_prv = df_prv.withColumn("year", col("year").cast("int").cast("string"))
    
    join_keys = ["rndrng_npi", "year"]
    overlapping_cols = [c for c in df_prv.columns if c in df_aggregated.columns and c not in join_keys]
    if overlapping_cols:
        logger.info(f"Dropping {len(overlapping_cols)} duplicate columns from enrichment")
        df_prv = df_prv.drop(*overlapping_cols)
    
    logger.info("Enrichment data prepared")
    
    df_enriched = df_aggregated.join(df_prv, on=join_keys, how="inner")
    
    logger.info("Join completed")
    
    return df_enriched


# Phase 4: Fraud Labeling
def load_leie_data(spark):
    """Load and process LEIE exclusion list from GCS."""
    logger.info("Loading LEIE exclusion data...")
    
    try:
        df_leie_raw = spark.read.parquet(
            LEIE_PATH,
            header=True,
            inferSchema=True
        )
        logger.info("Raw LEIE records loaded")
    except Exception as e:
        logger.error(f"Failed to load LEIE data from {LEIE_PATH}")
        logger.error(f"Error: {e}")
        raise RuntimeError(f"LEIE dataset not found at {LEIE_PATH}. Please verify GCS path.") from e
    
    df_leie = df_leie_raw.select([col(c).alias(c.lower()) for c in df_leie_raw.columns])
    
    df_leie = df_leie.filter(
        (col('npi').isNotNull()) & 
        (col('npi') != 0) & 
        (col('npi') != '0') &
        (col('npi') != '')
    )
    
    df_leie = df_leie.withColumn('npi', col('npi').cast('int'))
    
    df_leie = df_leie.withColumn(
        'excldate_parsed',
        to_date(col('excldate').cast('string'), 'yyyyMMdd')
    )
    
    df_leie = df_leie.withColumn(
        'excl_year',
        year(col('excldate_parsed'))
    )
    
    df_leie = df_leie.withColumn(
        'reindate_parsed',
        when((col('reindate') == 0) | (col('reindate') == '0'), lit(None))
        .otherwise(to_date(col('reindate').cast('string'), 'yyyyMMdd'))
    )
    
    df_leie = df_leie.withColumn(
        'excltype',
        lower(trim(col('excltype')))
    )
    
    logger.info("LEIE records after cleaning")
    
    return df_leie


def create_fraud_labels(df_leie, years_list):
    """Create fraud label lookup table from LEIE data for multiple years."""
    logger.info(f"Creating fraud labels for years: {years_list}...")
    
    # Filter to fraud-relevant exclusion types
    df_fraud = df_leie.filter(
        col('excltype').isin(FRAUD_EXCLTYPE_CODES)
    )
    
    # Select relevant columns for year-based filtering
    df_fraud_npis = df_fraud.select('npi', 'excl_year', 'reindate_parsed').distinct()
    
    logger.info(f"Total fraudulent NPIs in LEIE: {df_fraud_npis.count():,}")
    
    return df_fraud_npis


def label_fraud_cases(df_enriched, df_fraud_npis):
    """Join fraud labels to enriched dataset (no year filtering)."""
    logger.info("Labeling dataset with fraud indicators...")
    
    # Get distinct fraud NPIs
    df_fraud_distinct = df_fraud_npis.select('npi').distinct()
    
    # Left join for labeling
    df_labeled = df_enriched.join(
        df_fraud_distinct,
        df_enriched.rndrng_npi == df_fraud_distinct.npi,
        how='left'
    )
    
    # Create fraud label based on successful join
    df_labeled = df_labeled.withColumn(
        'fraud_label',
        when(col('npi').isNotNull(), 1).otherwise(0)
    )
    
    # Drop duplicate NPI column from join
    df_labeled = df_labeled.drop('npi')
    
    # Count fraud labels
    fraud_label_count = df_labeled.filter(col('fraud_label') == 1).select('rndrng_npi').distinct().count()
    logger.info(f"NPIs labeled as fraud: {fraud_label_count:,}")
    logger.info(f"Labeling completed. Dataset size: {df_labeled.count():,} records")
    
    return df_labeled


# Phase 5: One-Hot Encoding
def one_hot_encode_categoricals(spark, df_labeled):
    """Create binary columns for each unique categorical value."""
    logger.info("Performing SQL-based One-Hot Encoding...")
    
    categorical_cols = ['rndrng_prvdr_type', 'place_of_srvc']
    df_encoded = df_labeled
    
    for col_name in categorical_cols:
        logger.info(f"Encoding {col_name}...")
        
        unique_values = [
            row[0] for row in 
            df_encoded.select(col_name).distinct().collect()
            if row[0] is not None
        ]
        
        logger.info(f"  Found {len(unique_values)} unique values")
        
        for value in unique_values:
            safe_value = (
                str(value)
                .replace(' ', '_')
                .replace('/', '_')
                .replace('-', '_')
                .replace('(', '')
                .replace(')', '')
                .replace(',', '')
                .replace('.', '')
                .replace('&', 'and')
                .replace("'", '')
                [:50]
            )
            
            new_col_name = f"{col_name}_ohe_{safe_value}"
            
            df_encoded = df_encoded.withColumn(
                new_col_name,
                when(col(col_name) == value, 1).otherwise(0)
            )
        
        df_encoded = df_encoded.drop(col_name)
        
        logger.info(f"  Created {len(unique_values)} binary columns for {col_name}")
    
    ohe_cols = [c for c in df_encoded.columns if '_ohe_' in c]
    logger.info(f"✓ One-Hot Encoding completed")
    logger.info(f"  Total OHE columns created: {len(ohe_cols)}")
    
    if len(ohe_cols) > 0:
        logger.info(f"  Sample OHE columns: {ohe_cols[:5]}")
    
    return df_encoded


# Phase 6: Feature Selection & Preparation
def prepare_for_training(df_encoded):
    """Prepare final feature set for machine learning."""
    logger.info("Preparing dataset for ML training...")
    
    # Keep year column - it's valuable for temporal analysis
    feature_cols = [c for c in df_encoded.columns if c != 'fraud_label']
    df_train = df_encoded.select(*feature_cols, 'fraud_label')
    
    logger.info(f"Final feature count: {len(feature_cols)}")
    logger.info(f"Total columns: {len(df_train.columns)} (features + target)")
    
    agg_cols = [c for c in feature_cols if any(stat in c for stat in ['_min', '_max', '_mean', '_median', '_sum', '_std'])]
    ohe_cols = [c for c in feature_cols if '_ohe' in c]
    other_cols = [c for c in feature_cols if c not in agg_cols and c not in ohe_cols and c != 'year']
    
    logger.info(f"  - Aggregated features: {len(agg_cols)}")
    logger.info(f"  - Enrichment features: {len(other_cols)}")
    logger.info(f"  - OHE features: {len(ohe_cols)}")
    logger.info(f"  - Year column: included ✓")
    
    return df_train


# Phase 7: Output Persistence
def save_to_bigquery(df_final, dataset, table, temp_gcs_bucket, bq_location="US"):
    """Create BigQuery dataset if needed and save final training data."""
    try:
        logger.info(f"Ensuring BigQuery dataset '{dataset}' exists in {bq_location}...")
        client = bigquery.Client()
        dataset_ref = bigquery.Dataset(f"{client.project}.{dataset}")
        dataset_ref.location = bq_location
        client.create_dataset(dataset_ref, exists_ok=True)
        logger.info(f"✓ Dataset '{dataset}' is ready.")
    except Exception as e:
        logger.error(f"FATAL: Failed to create or verify BigQuery dataset: {e}")
        raise

    logger.info(f"Saving final dataset to BigQuery table: {dataset}.{table}...")
    
    df_final.write \
        .format("bigquery") \
        .option("table", f"{dataset}.{table}") \
        .option("temporaryGcsBucket", temp_gcs_bucket) \
        .mode("overwrite") \
        .save()
    
    row_count = df_final.count() 
    
    logger.info(f"✓ Successfully saved {row_count:,} rows to {dataset}.{table}")
    logger.info(f"  Mode: overwrite")
    logger.info(f"  Temp GCS Bucket: {temp_gcs_bucket}")





def main():
    """Main pipeline execution."""
    logger.info("="*60)
    logger.info("CMS Part B Data Processing Pipeline - Starting")
    logger.info("="*60)

    spark = (
        SparkSession.builder
        .appName("MedGuardAIFraudPrep")
        .getOrCreate()
    )
    logger.info("SparkSession initialized successfully")
    
    # Read all years at once using wildcard pattern
    raw_path = f"{BUCKET_BASE_DETAILS}/cms_partb_details_*/*.parquet"
    logger.info(f"Reading all years from: {raw_path}")
    
    df_raw = spark.read.parquet(raw_path)
    
    # Extract year from file path
    df_raw = df_raw.withColumn("source_file", input_file_name())
    df_raw = df_raw.withColumn(
        "year", 
        regexp_extract("source_file", r'cms_partb_details_(\d{4})', 1)
    )
    df_raw = df_raw.drop('source_file')
    
    logger.info("✓ Data loaded successfully from all years")
    
    # Phase 1: Cleaning
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Data Cleaning")
    logger.info("="*60)
    df_cleaned = clean_dataframe(df_raw)
    
    # Phase 2: Aggregation
    logger.info("="*60)
    logger.info("PHASE 2: Feature Aggregation")
    logger.info("="*60)
    df_aggregated = aggregate_features(df_cleaned)
    
    df_aggregated = df_aggregated.cache()
    logger.info(f"Aggregated data cached. Total rows: {df_aggregated.count()}")
    
    # Phase 2.5: Z-Score and Risk Ratio Features (NEW - Model Enhancement)
    logger.info("="*60)
    logger.info("PHASE 2.5: Z-Score and Risk Ratio Features")
    logger.info("="*60)
    
    # Compute Z-scores relative to provider specialty
    df_aggregated = compute_zscore_features(
        df_aggregated, 
        group_col='rndrng_prvdr_type',
        numeric_cols=['line_srvc_cnt', 'average_medicare_payment_amt', 
                      'bene_unique_cnt', 'average_submitted_chrg_amt']
    )
    
    # Compute risk ratio features
    df_aggregated = compute_risk_ratios(df_aggregated)
    
    # Re-cache after adding new features
    df_aggregated = df_aggregated.cache()
    new_feature_cols = [c for c in df_aggregated.columns if '_zscore' in c or '_ratio' in c 
                        or c in ['billing_inflation', 'service_density']]
    logger.info(f"New enhancement features added: {new_feature_cols}")
    
    # Phase 3: Enrichment
    logger.info("="*60)
    logger.info("PHASE 3: Enrichment and Join")
    logger.info("="*60)
    df_enriched = enrich_and_join(spark, df_aggregated)
    
    # Phase 4: Fraud Labeling
    logger.info("="*60)
    logger.info("PHASE 4: Fraud Labeling")
    logger.info("="*60)
    
    df_leie = load_leie_data(spark)
    df_fraud_npis = create_fraud_labels(df_leie, years_list=YEARS)
    df_labeled = label_fraud_cases(df_enriched, df_fraud_npis)
    
    df_labeled = df_labeled.cache()
    fraud_count = df_labeled.filter(col('fraud_label') == 1).count()
    total_count = df_labeled.count()
    fraud_pct = (fraud_count / total_count * 100) if total_count > 0 else 0
    logger.info(f"Labeled data cached. Total: {total_count:,}, Fraud: {fraud_count:,} ({fraud_pct:.2f}%)")
    
    # Phase 5: One-Hot Encoding
    logger.info("="*60)
    logger.info("PHASE 5: One-Hot Encoding")
    logger.info("="*60)
    df_encoded = one_hot_encode_categoricals(spark, df_labeled)
    
    # Phase 6: Prepare for Training
    logger.info("="*60)
    logger.info("PHASE 6: Feature Selection & Preparation")
    logger.info("="*60)
    df_final = prepare_for_training(df_encoded)
    
    # Phase 7: Save Output
    logger.info("="*60)
    logger.info("PHASE 7: Save to BigQuery")
    logger.info("="*60)
    
    save_to_bigquery(df_final, BIGQUERY_DATASET, BIGQUERY_TABLE, STAGING_BUCKET, BIGQUERY_LOCATION)
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Data available in BigQuery: {BIGQUERY_DATASET}.{BIGQUERY_TABLE}")
    logger.info("="*60)

    df_aggregated.unpersist()
    df_labeled.unpersist()


if __name__ == "__main__":
    main()