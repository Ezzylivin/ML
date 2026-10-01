from urllib.parse import urlencode
import io
import requests
import pandas as pd

CARTO_API = "https://phl.carto.com/api/v2/sql"


def fetch_carto_data(label, sql):
    print(f"Fetching {label}...")
    params = {"q": " ".join(sql.split()), "format": "csv"}
    try:
        response = requests.get(CARTO_API, params=params, timeout=90)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            print(f"  -> Success! Downloaded {len(df):,} records.")
            return df
        else:
            print(
                f"  -> Server Error ({response.status_code}):"
                f" {response.text.strip()[:150]}"
            )
    except Exception as e:
        print(f"  -> Request failed: {e}")
    return pd.DataFrame()


# 1. Fetch L&I Violations (2024-2026)
print("--- STEP 1: Fetching L&I Code Violations (2024-2026) ---")
sql_violations = """
    SELECT address, violationdate, casenumber 
    FROM violations 
    WHERE violationdate >= '2024-01-01'
"""
df_violations = fetch_carto_data("L&I Violations", sql_violations)


# 2. Fetch Active Real Estate Tax Balances (2024-2026)
print(
    "\n--- STEP 2: Fetching Current Real Estate Tax Balances (2024-2026) ---"
)
sql_tax_balances = """
    SELECT street_address, total_due, principal_due, penalty_due, interest_due, tax_period 
    FROM real_estate_tax_balances 
    WHERE total_due > 0 AND tax_period IN (2024, 2025, 2026)
"""

sql_tax_fallback = """
    SELECT street_address, total_due, principal_due, penalty_due, interest_due, oldest_year_owed, most_recent_year_owed 
    FROM real_estate_tax_delinquencies 
    WHERE total_due > 0
"""

df_tax = fetch_carto_data(
    "Real Estate Tax Balances (2024-2026)", sql_tax_balances
)

if df_tax.empty:
    print(
        "  -> 'real_estate_tax_balances' returned 0 records. Retrying with"
        " active delinquency table..."
    )
    df_tax = fetch_carto_data(
        "Real Estate Tax Delinquencies", sql_tax_fallback
    )


# 3. Fetch OPA Assessment & Ownership Data
print(
    "\n--- STEP 3: Fetching Property Assessment & Ownership Data (OPA) ---"
)
sql_opa = """
    SELECT location, owner_1, owner_2, mailing_address_1, mailing_street, mailing_city_state, mailing_zip, category_code_description 
    FROM opa_properties
"""
df_opa = fetch_carto_data("OPA Owner Data", sql_opa)


# 4. Processing, Filtering & Merging
print("\n--- STEP 4: Processing & Merging Datasets ---")

# A. Group Violations
if not df_violations.empty and "address" in df_violations.columns:
    df_violations["address"] = (
        df_violations["address"].astype(str).str.strip().str.upper()
    )
    df_v_grouped = (
        df_violations.groupby("address")
        .agg(
            violation_count=("casenumber", "count"),
            latest_violation_date=("violationdate", "max"),
        )
        .reset_index()
    )
else:
    df_v_grouped = pd.DataFrame(
        columns=["address", "violation_count", "latest_violation_date"]
    )

# B. Group Tax Debt
if not df_tax.empty:
    tax_addr_col = (
        "street_address" if "street_address" in df_tax.columns else "address"
    )
    df_tax["address"] = df_tax[tax_addr_col].astype(str).str.strip().str.upper()

    for col in ["total_due", "principal_due", "penalty_due", "interest_due"]:
        if col in df_tax.columns:
            df_tax[col] = pd.to_numeric(df_tax[col], errors="coerce").fillna(0)

    if "total_due" in df_tax.columns and df_tax["total_due"].sum() > 0:
        df_tax["computed_total"] = df_tax["total_due"]
    else:
        df_tax["computed_total"] = (
            df_tax["principal_due"]
            + df_tax["penalty_due"]
            + df_tax["interest_due"]
        )

    if "tax_period" in df_tax.columns:
        df_tax["delinquent_years"] = df_tax["tax_period"].astype(str)
    elif (
        "oldest_year_owed" in df_tax.columns
        and "most_recent_year_owed" in df_tax.columns
    ):
        df_tax["delinquent_years"] = (
            df_tax["oldest_year_owed"].astype(str)
            + " - "
            + df_tax["most_recent_year_owed"].astype(str)
        )
    else:
        df_tax["delinquent_years"] = "2024-2026"

    df_t_grouped = (
        df_tax.groupby("address")
        .agg(
            unpaid_tax_total=("computed_total", "sum"),
            delinquent_years=(
                "delinquent_years",
                lambda x: ", ".join(map(str, sorted(set(x)))),
            ),
        )
        .reset_index()
    )
else:
    df_t_grouped = pd.DataFrame(
        columns=["address", "unpaid_tax_total", "delinquent_years"]
    )

# C. Process OPA Data
if not df_opa.empty and "location" in df_opa.columns:
    df_opa["address"] = df_opa["location"].astype(str).str.strip().str.upper()
    df_opa = df_opa.drop_duplicates(subset=["address"])
    opa_cols = [
        "address",
        "owner_1",
        "owner_2",
        "mailing_address_1",
        "mailing_street",
        "mailing_city_state",
        "mailing_zip",
        "category_code_description",
    ]
    df_opa_clean = df_opa[[c for c in opa_cols if c in df_opa.columns]]
else:
    df_opa_clean = pd.DataFrame(columns=["address", "owner_1"])

# D. Merge All Datasets
df_merged = pd.merge(df_v_grouped, df_t_grouped, on="address", how="outer")
df_final = pd.merge(df_merged, df_opa_clean, on="address", how="left")

# E. Formatting and Clean Nulls
df_final["violation_count"] = (
    pd.to_numeric(df_final["violation_count"], errors="coerce")
    .fillna(0)
    .astype(int)
)
df_final["unpaid_tax_total"] = (
    pd.to_numeric(df_final["unpaid_tax_total"], errors="coerce")
    .fillna(0)
    .round(2)
)
df_final["delinquent_years"] = df_final["delinquent_years"].fillna("None")
df_final["latest_violation_date"] = df_final["latest_violation_date"].fillna(
    "None"
)

fill_opa_cols = [
    "owner_1",
    "owner_2",
    "mailing_address_1",
    "mailing_street",
    "mailing_city_state",
    "mailing_zip",
    "category_code_description",
]
for col in fill_opa_cols:
    if col in df_final.columns:
        df_final[col] = df_final[col].fillna("Unknown / Not Listed")

# F. Filter for Tax Owed <= $5,000 (and > $0)
df_final = df_final[
    (df_final["unpaid_tax_total"] > 0) & (df_final["unpaid_tax_total"] <= 5000)
]

# Sort by highest tax owed, then violation count
df_final = df_final.sort_values(
    by=["unpaid_tax_total", "violation_count"], ascending=[False, False]
)

# Export Output
output_file = "philly_problem_properties_2024_2026_with_owners.csv"
df_final.to_csv(output_file, index=False)

matched_count = len(df_final[df_final["owner_1"] != "Unknown / Not Listed"])
print(
    f"\nDone! Exported complete enriched dataset to {output_file}"
)
print(f"Total Properties Owed <= $5k: {len(df_final):,}")
print(
    f"Properties Matched with Owners: {matched_count:,}"
    f" ({(matched_count/len(df_final))*100 if len(df_final)>0 else 0:.1f}%)"
)
