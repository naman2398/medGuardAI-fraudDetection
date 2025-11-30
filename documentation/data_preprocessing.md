# PySpark Data Preprocessing Workflow

**Source File:** `src/data/data_preprocessing.py`  
**Execution Environment:** GCP Dataproc Serverless (PySpark)

---

## Overview

This document describes the data preprocessing pipeline that transforms raw CMS Medicare Part B data (2018-2023) into a model-ready feature table for fraud detection. The pipeline processes ~8 million provider records with a 0.04% fraud rate.

---

## Input Data Sources

| Source | Location | Description |
|--------|----------|-------------|
| **Part B Details (PRV_SVC)** | `gs://medguard_rawdata/raw/cms_partb_details_data/` | Claims-level data with service counts, charges, payments |
| **Part B Summary (PRV)** | `gs://medguard_rawdata/raw/cms_partb_summary_data/` | Provider-level summary with demographics, chronic conditions |
| **LEIE Exclusion List** | `gs://medguard_rawdata/raw/fraud_labels/` | OIG exclusion list for fraud labeling |

---

## Pipeline Phases

### Phase 1: Data Ingestion & Cleaning

**Function:** `clean_dataframe()`

| Step | Action | Details |
|------|--------|---------|
| 1. Load Raw Data | Read Parquet files from GCS | Multi-year files (2018-2023) consolidated into single DataFrame |
| 2. Extract Year | Parse year from filename | `regexp_extract(source_file, 'cms_partb_details_(\d{4})', 1)` |
| 3. Drop PII Columns | Remove identifying information | Names, addresses, credentials, geographic details |
| 4. Currency Cleaning | Strip `$` and `,` from amounts | Cast to float type |
| 5. Type Standardization | Fix provider types | Merge aliases (e.g., "CRNA" → "Certified Registered Nurse Anesthetist") |
| 6. Handle Missing Values | Impute `tot_benes` nulls | Default to 5 when null |
| 7. Type Casting | Convert columns to proper types | Integers: `tot_benes`, `tot_srvcs`, `rndrng_npi`; Strings: `rndrng_prvdr_type` |

**Columns Dropped (PII):**
- `rndrng_prvdr_last_org_name`, `rndrng_prvdr_first_name`, `rndrng_prvdr_mi`
- `rndrng_prvdr_crdntls`, `rndrng_prvdr_st1`, `rndrng_prvdr_st2`
- `rndrng_prvdr_city`, `rndrng_prvdr_state_abrvtn`, `rndrng_prvdr_zip5`
- `rndrng_prvdr_ruca`, `rndrng_prvdr_ruca_desc`, `rndrng_prvdr_cntry`
- `hcpcs_desc`

---

### Phase 2: Feature Aggregation

**Function:** `aggregate_features()`

**Grouping Keys:**
- `rndrng_npi` (Provider NPI)
- `year`
- `rndrng_prvdr_type` (Provider specialty)
- `place_of_srvc` (Facility/Non-facility)

**Aggregated Columns:**

| Original Column | Renamed To | Aggregations |
|-----------------|------------|--------------|
| `tot_srvcs` | `line_srvc_cnt` | min, max, mean, median, sum, std |
| `tot_benes` | `bene_unique_cnt` | min, max, mean, median, sum, std |
| `tot_bene_day_srvcs` | `bene_day_srvc_cnt` | min, max, mean, median, sum, std |
| `avg_sbmtd_chrg` | `average_submitted_chrg_amt` | min, max, mean, median, sum, std |
| `avg_mdcr_pymt_amt` | `average_medicare_payment_amt` | min, max, mean, median, sum, std |

**Output:** 30 numeric features (5 columns × 6 statistics) + 4 grouping keys

---

### Phase 3: Peer-Relative Feature Engineering

**Functions:** `compute_zscore_features()`, `compute_risk_ratios()`

#### 3.1 Z-Score Features

**Purpose:** Transform absolute values into peer-relative scores to detect anomalous billing patterns within specialty groups.

**Implementation:**
```
Window: Partition by rndrng_prvdr_type (provider specialty)

For each numeric column:
  - group_mean = mean(column) over specialty window
  - group_std = stddev(column) over specialty window (min 0.01 to avoid division by zero)
  - Z-Score = (value - group_mean) / group_std
  - Ratio = value / group_mean
```

**Z-Score Columns Generated:**

| Base Column | Z-Score Column | Ratio Column |
|-------------|----------------|--------------|
| `line_srvc_cnt` | `line_srvc_cnt_zscore` | `line_srvc_cnt_ratio` |
| `average_medicare_payment_amt` | `average_medicare_payment_amt_zscore` | `average_medicare_payment_amt_ratio` |
| `bene_unique_cnt` | `bene_unique_cnt_zscore` | `bene_unique_cnt_ratio` |
| `average_submitted_chrg_amt` | `average_submitted_chrg_amt_zscore` | `average_submitted_chrg_amt_ratio` |

#### 3.2 Risk Ratio Features

**Purpose:** Create explicit interaction features correlated with fraud patterns.

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `billing_inflation` | `avg_sbmtd_chrg / avg_mdcr_pymt_amt` | High values indicate potential upcoding |
| `service_density` | `tot_srvcs / tot_benes` | High values indicate potential churning/unnecessary services |

---

### Phase 4: Data Enrichment

**Function:** `enrich_and_join()`

**Action:** Inner join aggregated data with Provider Summary (PRV) dataset on `(rndrng_npi, year)`.

**Additional Features from PRV Source:**
- `tot_hcpcs_cds` - Total unique HCPCS codes billed
- `tot_sbmtd_chrg`, `tot_mdcr_alowd_amt`, `tot_mdcr_pymt_amt` - Total amounts
- `drug_*` and `med_*` - Drug vs medical service breakdowns
- `bene_avg_age`, `bene_age_*_cnt` - Beneficiary age distributions
- `bene_dual_cnt`, `bene_ndual_cnt` - Dual eligibility counts
- `bene_avg_risk_scre` - Average HCC risk score
- `bene_cc_*_pct` - Chronic condition percentages (diabetes, heart disease, etc.)

**Columns Dropped from PRV:**
- All PII/geographic fields
- Suppression indicators (`drug_sprsn_ind`, `med_sprsn_ind`)
- Race counts (high missing values)

---

### Phase 5: Fraud Labeling

**Functions:** `load_leie_data()`, `create_fraud_labels()`, `label_fraud_cases()`

**LEIE (List of Excluded Individuals/Entities) Processing:**

1. Load LEIE exclusion list from Parquet
2. Filter for fraud-related exclusion codes:
   - `1128a1` - Conviction of program-related crimes
   - `1128a2` - Conviction relating to patient abuse
   - `1128a3` - Felony conviction relating to healthcare fraud
   - `1128a4` - Felony conviction relating to controlled substances
   - `1128b4` - License revocation/suspension
   - `1128b7` - Fraud, kickbacks, other prohibited activities
3. Parse exclusion dates and extract year
4. Left join to enriched data on `rndrng_npi`
5. Create binary `fraud_label` (1 if NPI in LEIE fraud list, else 0)

**Expected Output:** ~0.04% positive class (fraud rate)

---

### Phase 6: Categorical Encoding

**Function:** `one_hot_encode_categoricals()`

**Encoded Columns:**
- `rndrng_prvdr_type` → ~101 binary columns (one per specialty)
- `place_of_srvc` → 2 binary columns (Facility/Non-facility)

**Naming Convention:** `{column}_ohe_{sanitized_value}`

---

### Phase 7: Output

**Function:** `save_to_bigquery()`

**Destination:** `medguard_processed_all_years.fraud_training_data_enhanced_v2`

| Parameter | Value |
|-----------|-------|
| Dataset | `medguard_processed_all_years` |
| Table | `fraud_training_data_enhanced_v2` |
| Location | `US` |
| Mode | Overwrite |

---

## Final Feature Summary

| Category | Count | Examples |
|----------|-------|----------|
| **Aggregated Statistics** | 30 | `*_min`, `*_max`, `*_mean`, `*_median`, `*_sum`, `*_std` |
| **Z-Scores & Ratios** | 8 | `*_zscore`, `*_ratio` |
| **Risk Ratios** | 2 | `billing_inflation`, `service_density` |
| **Provider Summary** | ~47 | `tot_hcpcs_cds`, `bene_avg_age`, `bene_cc_*_pct` |
| **One-Hot Encoded** | ~103 | `rndrng_prvdr_type_ohe_*`, `place_of_srvc_ohe_*` |
| **Target** | 1 | `fraud_label` |
| **Total** | ~190+ | — |

---

## Execution

**Script:** `src/data/preprocess.sh`

```bash
# Submit to Dataproc Serverless
gcloud dataproc batches submit pyspark \
    src/data/data_preprocessing.py \
    --region=us-central1 \
    --deps-bucket=gs://medguard_rawdata/dataproc-deps
```

---

## Data Quality Notes

1. **Missing Values:** Numeric columns imputed with 0.0; `tot_benes` nulls set to 5
2. **Duplicate Handling:** `dropDuplicates()` applied after cleaning
3. **Type Consistency:** All years cast to string for consistent joining
4. **Z-Score Edge Cases:** `group_std` floored at 0.01 to prevent division by zero
