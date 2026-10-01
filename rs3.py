import io
import requests
import pandas as pd

CARTO_API = "https://phl.carto.com/api/v2/sql"


def fetch_carto_dataset(description, queries):
    """Queries Carto API using requests.get, printing exact server errors if 400 occurs."""
    print(f"\n--- {description} ---")
    for idx, sql in enumerate(queries, 1):
        print(f"Attempt {idx}: Trying query: {sql[:70]}...")
        try:
            params = {"q": " ".join(sql.split()), "format": "csv"}
            response = requests.get(CARTO_API, params=params, timeout=30)

            if response.status_code == 200 and len(response.text.strip()) > 0:
                df = pd.read_csv(io.StringIO(response.text))
                if not df.empty:
                    print(f"   -> SUCCESS! Fetched {len(df):,} records.")
                    return df
            else:
                print(
                    f"   -> Carto returned status {response.status_code}: {response.text.strip()[:150]}"
                )
        except Exception as e:
            print(f"   -> Request error: {e}")

    print("   -> All query attempts failed for this dataset.")
    return pd.DataFrame()


# 1. Fetch L&I Violations (2024-2026)
violation_queries = [
    "SELECT * FROM violations WHERE violationdate >= '2024-01-01'",
    "SELECT * FROM violations WHERE violation_date >= '2024-01-01'",
    "SELECT * FROM li_violations WHERE violationdate >= '2024-01-01'",
]
df_violations = fetch_carto_dataset(
    "Fetching L&I Violations (2024-2026)", violation_queries
)

# 2. Fetch Real Estate Tax Delinquencies / Balances (2024-2026)
tax_queries = [
    "SELECT * FROM real_estate_tax_delinquencies WHERE tax_period IN (2024, 2025, 2026)",
    "SELECT * FROM real_estate_tax_delinquencies WHERE tax_period IN ('2024', '2025', '2026')",
    "SELECT * FROM real_estate_tax_balances WHERE tax_period IN (2024, 2025, 2026)",
    "SELECT * FROM real_estate_tax_delinquencies LIMIT 5000",
]
df_tax = fetch_carto_dataset(
    "Fetching Real Estate Tax Delinquencies", tax_queries
)


# 3. Processing & Merging Datasets
print("\n--- Processing and Merging Datasets ---")


def find_column(df, candidates):
    """Helper to locate available column names dynamically."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# Process Violations
violation_addr_col = find_column(
    df_violations, ["address", "street_address", "location", "situs_address"]
)
violation_date_col = find_column(
    df_violations, ["violationdate", "violation_date", "issue_date"]
)

if not df_violations.empty and violation_addr_col:
    df_violations["clean_address"] = (
        df_violations[violation_addr_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df_violations_grouped = (
        df_violations.groupby("clean_address")
        .agg(
            violation_count=(violation_addr_col, "count"),
            latest_violation_date=(
                violation_date_col if violation_date_col else violation_addr_col,
                "max",
            ),
        )
        .reset_index()
        .rename(columns={"clean_address": "address"})
    )
else:
    df_violations_grouped = pd.DataFrame(
        columns=["address", "violation_count", "latest_violation_date"]
    )

# Process Tax Delinquencies
tax_addr_col = find_column(
    df_tax, ["street_address", "address", "location", "situs_address"]
)

if not df_tax.empty and tax_addr_col:
    df_tax["clean_address"] = (
        df_tax[tax_addr_col].astype(str).str.strip().str.upper()
    )

    # Calculate total balance dynamically based on available columns
    due_cols = [
        c
        for c in ["principal_due", "penalty_due", "interest_due", "total_due"]
        if c in df_tax.columns
    ]
    if due_cols:
        for c in due_cols:
            df_tax[c] = pd.to_numeric(df_tax[c], errors="coerce").fillna(0)
        df_tax["computed_total"] = df_tax[due_cols].sum(axis=1)
    else:
        df_tax["computed_total"] = 0

    tax_period_col = find_column(df_tax, ["tax_period", "tax_year", "year"])

    df_tax_grouped = (
        df_tax.groupby("clean_address")
        .agg(
            unpaid_tax_total=("computed_total", "sum"),
            delinquent_years=(
                tax_period_col if tax_period_col else tax_addr_col,
                lambda x: ", ".join(map(str, sorted(set(x)))),
            ),
        )
        .reset_index()
        .rename(columns={"clean_address": "address"})
    )
else:
    df_tax_grouped = pd.DataFrame(
        columns=["address", "unpaid_tax_total", "delinquent_years"]
    )


# Merge on Standardized Address
df_combined = pd.merge(
    df_violations_grouped, df_tax_grouped, on="address", how="outer"
)

# Clean Output
df_combined["violation_count"] = (
    pd.to_numeric(df_combined["violation_count"], errors="coerce")
    .fillna(0)
    .astype(int)
)
df_combined["unpaid_tax_total"] = (
    pd.to_numeric(df_combined["unpaid_tax_total"], errors="coerce")
    .fillna(0)
    .round(2)
)
df_combined["delinquent_years"] = df_combined["delinquent_years"].fillna("None")
df_combined["latest_violation_date"] = df_combined[
    "latest_violation_date"
].fillna("None")

# Sort properties by total unpaid tax balance, then violation count
df_combined = df_combined.sort_values(
    by=["unpaid_tax_total", "violation_count"], ascending=[False, False]
)

# Save to CSV
output_file = "philly_problem_properties_2024_2026.csv"
df_combined.to_csv(output_file, index=False)

print(f"\nDone! Saved combined records to {output_file}.")
print(f"Total Unique Properties Found: {len(df_combined):,}")
