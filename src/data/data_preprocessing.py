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

ZSCORE_NUMERIC_COLS = [
    'tot_srvcs',
    'avg_mdcr_pymt_amt',
    'tot_benes',
    'avg_sbmtd_chrg'
]


def compute_zscore_features(df, group_col='rndrng_prvdr_type', numeric_cols=None):
    """Compute peer-relative Z-scores and ratios for numeric columns."""
    if numeric_cols is None:
        numeric_cols = ZSCORE_NUMERIC_COLS
    
    logger.info(f"Computing Z-score features for {len(numeric_cols)} columns...")
    window_spec = Window.partitionBy(group_col)
    
    for col_name in numeric_cols:
        actual_col = None
        for candidate in [col_name, f"{col_name}_sum", f"{col_name}_mean"]:
            if candidate in df.columns:
                actual_col = candidate
                break
        
        if actual_col is None:
            logger.warning(f"Column {col_name} not found, skipping Z-score computation")
            continue
        
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
        
        df = df.withColumn(
            group_std_col,
            when(
                (col(group_std_col).isNull()) | (col(group_std_col) == 0),
                lit(0.01)
            ).otherwise(col(group_std_col))
        )
        
        zscore_col = f"{col_name}_zscore"
        df = df.withColumn(
            zscore_col,
            (col(actual_col) - col(group_mean_col)) / col(group_std_col)
        )
        
        ratio_col = f"{col_name}_ratio"
        df = df.withColumn(
            ratio_col,
            when(
                col(group_mean_col) == 0,
                lit(1.0)
            ).otherwise(
                col(actual_col) / col(group_mean_col)
            )
        )
        
        df = df.drop(group_mean_col, group_std_col)
    
    zscore_cols = [c for c in df.columns if '_zscore' in c or '_ratio' in c]
    logger.info(f"Z-score features complete: {len(zscore_cols)} columns added")
    
    return df


def compute_risk_ratios(df):
    """Compute risk ratio features (billing inflation and service density)."""
    logger.info("Computing risk ratio features...")
    
    sbmtd_col = None
    pymt_col = None
    srvcs_col = None
    benes_col = None
    
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
        df = df.withColumn(
            'billing_inflation',
            when(
                (col(pymt_col).isNull()) | (col(pymt_col) == 0),
                lit(1.0)
            ).otherwise(
                col(sbmtd_col) / col(pymt_col)
            )
        )
    else:
        logger.warning(f"Could not compute billing_inflation: sbmtd={sbmtd_col}, pymt={pymt_col}")
    
    if srvcs_col and benes_col:
        df = df.withColumn(
            'service_density',
            when(
                (col(benes_col).isNull()) | (col(benes_col) == 0),
                lit(1.0)
            ).otherwise(
                col(srvcs_col) / col(benes_col)
            )
        )
    else:
        logger.warning(f"Could not compute service_density: srvcs={srvcs_col}, benes={benes_col}")
    
    return df


def clean_dataframe(df):
    """Clean and standardize raw CMS Part B data."""
    logger.info("Starting data cleaning...")
    
    df = df.select([col(c).alias(c.lower()) for c in df.columns])
    df = df.withColumnRenamed("rfrg_crdntls", "rfrg_prvdr_crdntls")
    df = df.drop(*pii_cols)
    
    df = df.withColumn(
        'tot_benes',
        when(col('tot_benes').isNull(), 5).otherwise(col('tot_benes'))
    )
    
    for col_name in currency_columns:
        df = df.withColumn(
            col_name,
            regexp_replace(col(col_name), r'[\$,]', '').cast('float')
        )
    
    df = df.withColumn(
        'rndrng_prvdr_type',
        when(col('rndrng_prvdr_type') == 'Allergy/ Immunology', 'Allergy/Immunology')
        .when(col('rndrng_prvdr_type') == 'CRNA', 'Certified Registered Nurse Anesthetist')
        .otherwise(col('rndrng_prvdr_type'))
    )
    
    df = df.dropDuplicates()
    df = df.withColumn("year", col("year").cast('string'))
    
    for col_name in integer_columns:
        df = df.withColumn(col_name, col(col_name).cast('int'))
    
    for col_name in string_columns:
        df = df.withColumn(col_name, col(col_name).cast('string'))
    
    logger.info("Data cleaning completed")
    return df


def aggregate_features(df):
    """Aggregate features by provider NPI, year, type, and place of service."""
    logger.info("Starting feature aggregation...")
    
    group_keys = ["rndrng_npi", "year", "rndrng_prvdr_type", "place_of_srvc"]
    
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
    
    stddev_cols = [f"{v}_std" for v in agg_targets.values()]
    for c in stddev_cols:
        df_aggregated = df_aggregated.withColumn(c, coalesce(col(c), lit(0.0)))
    
    logger.info("Aggregation completed")
    return df_aggregated


def enrich_and_join(spark, df_aggregated):
    """Enrich aggregated data with provider-level summary statistics."""
    logger.info("Starting enrichment and join...")
    
    enrich_path = f"{BUCKET_BASE_SUMMARY}/cms_partb_summary_*/*.parquet"
    df_prv_raw = spark.read.parquet(enrich_path)
    df_prv_raw = df_prv_raw.select([col(c).alias(c.lower()) for c in df_prv_raw.columns])
    
    df_prv_raw = df_prv_raw.withColumn("source_file", input_file_name())
    df_prv_raw = df_prv_raw.withColumn(
        "year", 
        regexp_extract("source_file", r'cms_partb_summary_(\d{4})', 1)
    )
    df_prv_raw = df_prv_raw.drop("source_file")
    
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
        df_prv = df_prv.drop(*overlapping_cols)
    
    df_enriched = df_aggregated.join(df_prv, on=join_keys, how="inner")
    logger.info("Enrichment join completed")
    
    return df_enriched


def load_leie_data(spark):
    """Load and process LEIE exclusion list from GCS."""
    logger.info("Loading LEIE exclusion data...")
    
    try:
        df_leie_raw = spark.read.parquet(
            LEIE_PATH,
            header=True,
            inferSchema=True
        )
    except Exception as e:
        logger.error(f"Failed to load LEIE data from {LEIE_PATH}: {e}")
        raise RuntimeError(f"LEIE dataset not found at {LEIE_PATH}.") from e
    
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
    
    return df_leie


def create_fraud_labels(df_leie, years_list):
    """Create fraud label lookup table from LEIE data."""
    df_fraud = df_leie.filter(
        col('excltype').isin(FRAUD_EXCLTYPE_CODES)
    )
    
    df_fraud_npis = df_fraud.select('npi', 'excl_year', 'reindate_parsed').distinct()
    logger.info(f"Total fraudulent NPIs in LEIE: {df_fraud_npis.count():,}")
    
    return df_fraud_npis


def label_fraud_cases(df_enriched, df_fraud_npis):
    """Join fraud labels to enriched dataset."""
    logger.info("Labeling dataset with fraud indicators...")
    
    df_fraud_distinct = df_fraud_npis.select('npi').distinct()
    
    df_labeled = df_enriched.join(
        df_fraud_distinct,
        df_enriched.rndrng_npi == df_fraud_distinct.npi,
        how='left'
    )
    
    df_labeled = df_labeled.withColumn(
        'fraud_label',
        when(col('npi').isNotNull(), 1).otherwise(0)
    )
    df_labeled = df_labeled.drop('npi')
    
    fraud_count = df_labeled.filter(col('fraud_label') == 1).select('rndrng_npi').distinct().count()
    logger.info(f"NPIs labeled as fraud: {fraud_count:,}")
    
    return df_labeled


def one_hot_encode_categoricals(spark, df_labeled):
    """Create binary columns for each unique categorical value."""
    logger.info("Performing One-Hot Encoding...")
    
    categorical_cols = ['rndrng_prvdr_type', 'place_of_srvc']
    df_encoded = df_labeled
    
    for col_name in categorical_cols:
        unique_values = [
            row[0] for row in 
            df_encoded.select(col_name).distinct().collect()
            if row[0] is not None
        ]
        
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
    
    ohe_cols = [c for c in df_encoded.columns if '_ohe_' in c]
    logger.info(f"One-Hot Encoding completed: {len(ohe_cols)} columns created")
    
    return df_encoded


def prepare_for_training(df_encoded):
    """Prepare final feature set for machine learning."""
    logger.info("Preparing dataset for ML training...")
    
    feature_cols = [c for c in df_encoded.columns if c != 'fraud_label']
    df_train = df_encoded.select(*feature_cols, 'fraud_label')
    
    logger.info(f"Final feature count: {len(feature_cols)}")
    return df_train


def save_to_bigquery(df_final, dataset, table, temp_gcs_bucket, bq_location="US"):
    """Create BigQuery dataset if needed and save final training data."""
    try:
        client = bigquery.Client()
        dataset_ref = bigquery.Dataset(f"{client.project}.{dataset}")
        dataset_ref.location = bq_location
        client.create_dataset(dataset_ref, exists_ok=True)
    except Exception as e:
        logger.error(f"Failed to create BigQuery dataset: {e}")
        raise

    logger.info(f"Saving to BigQuery: {dataset}.{table}...")
    
    df_final.write \
        .format("bigquery") \
        .option("table", f"{dataset}.{table}") \
        .option("temporaryGcsBucket", temp_gcs_bucket) \
        .mode("overwrite") \
        .save()
    
    logger.info(f"Saved {df_final.count():,} rows to {dataset}.{table}")





def main():
    """Main pipeline execution."""
    logger.info("CMS Part B Data Processing Pipeline - Starting")

    spark = (
        SparkSession.builder
        .appName("MedGuardAIFraudPrep")
        .getOrCreate()
    )
    
    raw_path = f"{BUCKET_BASE_DETAILS}/cms_partb_details_*/*.parquet"
    logger.info(f"Reading data from: {raw_path}")
    
    df_raw = spark.read.parquet(raw_path)
    df_raw = df_raw.withColumn("source_file", input_file_name())
    df_raw = df_raw.withColumn(
        "year", 
        regexp_extract("source_file", r'cms_partb_details_(\d{4})', 1)
    )
    df_raw = df_raw.drop('source_file')
    
    # Phase 1: Cleaning
    df_cleaned = clean_dataframe(df_raw)
    
    # Phase 2: Aggregation
    df_aggregated = aggregate_features(df_cleaned)
    df_aggregated = df_aggregated.cache()
    logger.info(f"Aggregated rows: {df_aggregated.count()}")
    
    # Phase 2.5: Z-Score and Risk Ratio Features
    df_aggregated = compute_zscore_features(
        df_aggregated, 
        group_col='rndrng_prvdr_type',
        numeric_cols=['line_srvc_cnt', 'average_medicare_payment_amt', 
                      'bene_unique_cnt', 'average_submitted_chrg_amt']
    )
    df_aggregated = compute_risk_ratios(df_aggregated)
    df_aggregated = df_aggregated.cache()
    
    # Phase 3: Enrichment
    df_enriched = enrich_and_join(spark, df_aggregated)
    
    # Phase 4: Fraud Labeling
    df_leie = load_leie_data(spark)
    df_fraud_npis = create_fraud_labels(df_leie, years_list=YEARS)
    df_labeled = label_fraud_cases(df_enriched, df_fraud_npis)
    df_labeled = df_labeled.cache()
    
    fraud_count = df_labeled.filter(col('fraud_label') == 1).count()
    total_count = df_labeled.count()
    logger.info(f"Labeled data: {total_count:,} total, {fraud_count:,} fraud ({fraud_count/total_count*100:.2f}%)")
    
    # Phase 5: One-Hot Encoding
    df_encoded = one_hot_encode_categoricals(spark, df_labeled)
    
    # Phase 6: Prepare for Training
    df_final = prepare_for_training(df_encoded)
    
    # Phase 7: Save Output
    save_to_bigquery(df_final, BIGQUERY_DATASET, BIGQUERY_TABLE, STAGING_BUCKET, BIGQUERY_LOCATION)
    
    logger.info(f"Pipeline completed. Data: {BIGQUERY_DATASET}.{BIGQUERY_TABLE}")

    df_aggregated.unpersist()
    df_labeled.unpersist()


if __name__ == "__main__":
    main()