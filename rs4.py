from urllib.parse import urlencode
import io
import requests
import pandas as pd

CARTO_API = "https://phl.carto.com/api/v2/sql"


def fetch_carto_data(label, sql):
    print(f"Fetching {label}...")
    params = {"q": " ".join(sql.split()), "format": "csv"}
    try:
        # Request only essential columns to keep the download under 5MB
        response = requests.get(CARTO_API, params=params, timeout=60)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            print(f"  -> Success! Downloaded {len(df):,} records.")
            return df
        else:
            print(f"  -> Server Error ({response.status_code}): {response.text.strip()[:100]}")
    except Exception as e:
        print(f"  -> Request failed: {e}")

    return pd.DataFrame()


# 1. Fetch L&I Violations (Only light, essential columns)
sql_violations = """
    SELECT address, violationdate, casenumber 
    FROM violations 
    WHERE violationdate >= '2024-01-01'
"""
df_violations = fetch_carto_data("L&I Violations (2024-2026)", sql_violations)


# 2. Fetch Tax Delinquencies (Only light, essential columns)
sql_tax = """
    SELECT street_address, tax_period, principal_due, penalty_due, interest_due 
    FROM real_estate_tax_delinquencies 
    WHERE tax_period IN (2024, 2025, 2026)
"""
df_tax = fetch_carto_data("Real Estate Tax Delinquencies (2024-2026)", sql_tax)


# 3. Processing and Merging
print("\nProcessing and merging data...")

# Aggregate Violations
if not df_violations.empty and "address" in df_violations.columns:
    df_violations["address"] = df_violations["address"].astype(str).str.strip().str.upper()
    df_v_grouped = (
        df_violations.groupby("address")
        .agg(
            violation_count=("casenumber", "count"),
            latest_violation_date=("violationdate", "max"),
        )
        .reset_index()
    )
else:
    df_v_grouped = pd.DataFrame(columns=["address", "violation_count", "latest_violation_date"])

# Aggregate Taxes
if not df_tax.empty and "street_address" in df_tax.columns:
    df_tax["address"] = df_tax["street_address"].astype(str).str.strip().str.upper()

    for col in ["principal_due", "penalty_due", "interest_due"]:
        df_tax[col] = pd.to_numeric(df_tax[col], errors="coerce").fillna(0)

    df_tax["total_due"] = df_tax["principal_due"] + df_tax["penalty_due"] + df_tax["interest_due"]

    df_t_grouped = (
        df_tax.groupby("address")
        .agg(
            unpaid_tax_total=("total_due", "sum"),
            delinquent_years=("tax_period", lambda x: ", ".join(map(str, sorted(set(x))))),
        )
        .reset_index()
    )
else:
    df_t_grouped = pd.DataFrame(columns=["address", "unpaid_tax_total", "delinquent_years"])

# Merge Datasets
df_final = pd.merge(df_v_grouped, df_t_grouped, on="address", how="outer")

# Clean formatting
df_final["violation_count"] = pd.to_numeric(df_final["violation_count"], errors="coerce").fillna(0).astype(int)
df_final["unpaid_tax_total"] = pd.to_numeric(df_final["unpaid_tax_total"], errors="coerce").fillna(0).round(2)
df_final["delinquent_years"] = df_final["delinquent_years"].fillna("None")
df_final["latest_violation_date"] = df_final["latest_violation_date"].fillna("None")

# Sort by highest tax owed, then violation count
df_final = df_final.sort_values(by=["unpaid_tax_total", "violation_count"], ascending=[False, False])

# Export
output_filename = "philly_problem_properties_2024_2026.csv"
df_final.to_csv(output_filename, index=False)

print(f"\nDone! Exported {len(df_final):,} unique properties to {output_filename}")
