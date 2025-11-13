# PySpark Preprocessing Workflow Plan: Medicare Part B Fraud Classification

## Goal
To transform raw 2022-2023 CMS Medicare Part B data (PRV_SVC source) and its Provider Summary (PRV source) into a single, enriched, provider-level feature table (Aggregated-Enriched) suitable for classification model training.

## Execution Environment
GCP Dataproc Serverless (PySpark)

## Input Data Sources
1. **Part B Summary by Provider and Service (PRV_SVC)**: Claims-level data (raw input).
2. **Part B Summary by Provider (PRV)**: Provider-level summary data (enrichment input).
3. **LEIE Exclusion List**: External file for fraud labeling (lookup data).

**Note:** This plan specifically excludes the use of gender attributes as requested by the user, diverging from the original paper's inclusion of that attribute.

---

## Phase 1: Data Ingestion and Cleaning (PRV_SVC Source)

This phase prepares the raw claims-level data for aggregation.

### A. Data Ingestion & Unification

| Step | Action | PRV_SVC Columns Involved | Output / Result |
|------|--------|-------------------------|-----------------|
| 1. Ingestion | Load 2013-2019 annual files into a single Spark DataFrame. | All columns (e.g., Rndrng_NPI, HCPCS_Cd, Rndrng_Prvdr_Type, Place_Of_Srvc). | Consolidated Spark DataFrame. |
| 2. Year Extraction | Add a Year column, essential for joining and time-series fraud labeling. | N/A (Derived from file metadata or path) | DF_PRV_SVC with new Year column. |
| 3. PII Exclusion | Drop columns identified as PII or irrelevant geographic details to prevent target leakage and simplify the model. | Rndrng_Prvdr_Last_Org_Name, Rndrng_Prvdr_First_Name, Rndrng_Prvdr_MI, Rndrng_Prvdr_Crdntls, Rndrng_Prvdr_St1, Rndrng_Prvdr_St2, Rndrng_Prvdr_City, Rndrng_Prvdr_State_Abrvtn, Rndrng_Prvdr_State_FIPS, Rndrng_Prvdr_Zip5, Rndrng_Prvdr_RUCA, Rndrng_Prvdr_RUCA_Desc, Rndrng_Prvdr_Cntry, HCPCS_Desc. | Reduced DF_PRV_SVC. |

### B. Cleaning and Type Casting

| Step | Action | Columns Involved (PRV_SVC) | Type Conversion / Note |
|------|--------|---------------------------|------------------------|
| 1. Numeric Casting | Convert all currency/count columns to appropriate numeric types (e.g., Decimal or Double). | Tot_Benes, Tot_Srvcs, Tot_Bene_Day_Srvcs, Avg_Sbmtd_Chrg, Avg_Mdcr_Alowd_Amt, Avg_Mdcr_Pymt_Amt, Avg_Mdcr_Stdzd_Amt. | Cast from String to Double. |
| 2. Categorical Cleaning | Standardize categorical values to handle inconsistencies and typos across years (simulating manual correction). | Rndrng_Prvdr_Type. | Apply a PySpark UDF for normalization (e.g., case standardization, merging known aliases) to reduce cardinality from 127 to ~102. |

### C. Feature Selection for Aggregation

| Column Type | Retained Columns (PRV_SVC) | Role in Next Phase |
|------------|---------------------------|-------------------|
| Identifier/Grouping | Rndrng_NPI, Year, Rndrng_Prvdr_Type, Place_Of_Srvc. | Group By Keys for aggregation. |
| Numeric/Aggregatable | Tot_Srvcs (Renamed to Line_srvc_cnt in source text context), Tot_Benes (Renamed to Bene_unique_cnt), Tot_Bene_Day_Srvcs (Renamed to Bene_day_srvc_cnt), Avg_Sbmtd_Chrg (Renamed to Average_submitted_chrg_amt), Avg_Mdcr_Pymt_Amt (Renamed to Average_medicare_payment_amt). | Value Columns for statistical summarization. |
| High-Dimensional (Dropped) | HCPCS_Cd. | Dropped to enable compression and provider-level focus. |

---

## Phase 2: Feature Engineering (Aggregation)

This phase transforms the row-per-service claims data into the provider-per-year format (Aggregated Data Set).

### A. Aggregation Operation

| Step | Action | Output Columns (Naming Convention: {Original_Col}_{Stat}) | Imputation |
|------|--------|----------------------------------------------------------|------------|
| 1. Grouping | Group DF_PRV_SVC by the key columns. | N/A | N/A |
| 2. Summary Statistics | Calculate six summary statistics for each of the 5 numeric columns. | 30 New Numeric Features: Tot_Srvcs_min, Tot_Srvcs_max, Tot_Srvcs_median, Tot_Srvcs_mean, Tot_Srvcs_sum, Tot_Srvcs_std (and 5 similar sets for the other numeric columns). | Std columns: Impute NaN or null resulting from single-row groups with 0.0. |
| 3. Final Aggregated DF | Resulting DataFrame (DF_Aggregated) with 30 new numeric features and the grouping keys. | DF_Aggregated (Final feature set: 30 numeric + 3 categorical: Rndrng_NPI, Year, Rndrng_Prvdr_Type, Place_Of_Srvc). | N/A |

---

## Phase 3: Data Enrichment and Finalization (Creating Aggregated-Enriched)

This phase integrates the Provider Summary data and applies the final pre-modeling steps.

### A. Summary Data Ingestion and Cleaning (PRV Source)

| Step | Action | PRV Source Columns Involved | Notes |
|------|--------|----------------------------|-------|
| 1. Ingestion | Load the Part B Summary by Provider data (PRV) for 2013-2019. | All columns (e.g., Rndrng_NPI, Tot_HCPCS_Cds, Bene_Avg_Age, Bene_CC_PH_Asthma_V2_Pct). | Add Year column derived from file metadata. |
| 2. PII/Suppression Exclusion | Exclude PII and suppressed fields to clean data. | Excluded: All PII/Geographic fields (similar to Phase 1.A.3), all Bene_Race_*_Cnt columns (due to high missing values/suppression). | The Drug_Sprsn_Ind and Med_Sprsn_Ind are also dropped as they are suppression flags. |
| 3. Numeric Casting & Cleaning | Convert currency/count fields to numeric types. | Currency: Tot_Sbmtd_Chrg, Tot_Mdcr_Alowd_Amt, Tot_Mdcr_Pymt_Amt, Tot_Mdcr_Stdzd_Amt, Drug_Sbmtd_Chrg, Drug_Mdcr_Alowd_Amt, Drug_Mdcr_Pymt_Amt, Drug_Mdcr_Stdzd_Amt, Med_Sbmtd_Chrg, Med_Mdcr_Alowd_Amt, Med_Mdcr_Pymt_Amt, Med_Mdcr_Stdzd_Amt. | Remove non-numeric characters (e.g., '$', ',') then cast to Double. |
| 4. Missing Value Imputation | Impute remaining missing numeric values (especially in count/cost fields). | All remaining numeric fields (e.g., CC percentages, counts). | Impute NaN or null with 0.0 (as per document context). |

### B. Feature Joining and Final Feature Set Construction

| Step | Action | Key Columns | Result |
|------|--------|-------------|--------|
| 1. Joining | Perform an INNER JOIN between DF_Aggregated (from Phase 2) and DF_PRV (from Phase 3.A) to create the final Aggregated-Enriched data set. | Rndrng_NPI and Year. | DF_AE (Aggregated-Enriched). |
| 2. Fraud Labeling | Join DF_AE with the pre-processed LEIE data (containing Rndrng_NPI, Year, and the Fraud_Label). | Rndrng_NPI and Year. | DF_AE_Labeled with the final binary classification target column (Fraud_Label). |
| 3. Final Feature Selection | Select all feature columns for the final model input, excluding the NPI and Year identifiers, but retaining the categorical and engineered features. | Final Features (approx. 80 columns total): <ul><li>**Identifiers (to drop before training):** Rndrng_NPI, Year</li><li>**Categorical:** Rndrng_Prvdr_Type, Place_Of_Srvc</li><li>**Numeric (30 from Aggregation):** All *_min, *_max, *_median, *_mean, *_sum, *_std features.</li><li>**Numeric (47 from Enrichment):** Tot_HCPCS_Cds, Tot_Benes, Tot_Srvcs, Tot_Sbmtd_Chrg, Tot_Mdcr_Alowd_Amt, Tot_Mdcr_Pymt_Amt, Tot_Mdcr_Stdzd_Amt, all Drug_* and Med_* counts/amounts, all Bene_Age_*_Cnt features, Bene_Avg_Age, Bene_Dual_Cnt, Bene_Ndual_Cnt, Bene_Avg_Risk_Scre, and all Bene_CC_*_Pct chronic condition features.</li></ul> | Final, production-ready feature table. |

### C. Pre-Modeling Steps

| Step | Action | Result |
|------|--------|--------|
| 1. Categorical Encoding | One-Hot Encode the remaining categorical features. | New binary feature columns for Rndrng_Prvdr_Type (101 new columns) and Place_Of_Srvc (2 new columns). |
| 2. Column Standardization | Although the original paper skipped scaling for tree-based models, it is good practice to keep the option open. No explicit scaling is required for PySpark data output if target model is XGBoost/RF. | Final DataFrame schema is confirmed (all features are numeric after OHE). |
| 3. Output/Persistence | Save the final feature table to a high-performance distributed store (e.g., Google Cloud Storage or BigQuery) for model training. | Persisted DF_AE_Labeled table. |
