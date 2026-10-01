import pandas as pd

CARTO_API = "https://phl.carto.com/api/v2/sql"

print("1. Fetching L&I Violations (2024-2026)...")
# Correct Carto table name is 'violations'
violations_sql = """
    SELECT address, violationdate, casenumber, violationdetails 
    FROM violations 
    WHERE violationdate >= '2024-01-01' 
      AND violationdate <= '2026-12-31'
"""

# Request directly as CSV for speed and stability
violations_url = f"{CARTO_API}?q={violations_sql}&format=csv"
try:
    df_violations = pd.read_csv(violations_url)
    print(f"   -> Found {len(df_violations):,} violation records.")
except Exception as e:
    print(f"   -> Error fetching violations: {e}")
    df_violations = pd.DataFrame()


print("\n2. Fetching Real Estate Tax Delinquencies (2024-2026)...")
# Tax period values are numeric integers in Carto
tax_sql = """
    SELECT street_address, tax_period, principal_due, penalty_due, interest_due
    FROM real_estate_tax_delinquencies
    WHERE tax_period IN (2024, 2025, 2026)
"""

tax_url = f"{CARTO_API}?q={tax_sql}&format=csv"
try:
    df_tax = pd.read_csv(tax_url)
    print(f"   -> Found {len(df_tax):,} tax delinquency records.")
except Exception as e:
    print(f"   -> Error fetching tax delinquencies: {e}")
    df_tax = pd.DataFrame()


print("\n3. Processing and Merging Datasets...")

# Process and group Violations
if not df_violations.empty and 'address' in df_violations.columns:
    df_violations['address'] = df_violations['address'].astype(str).str.strip().str.upper()
    df_violations_grouped = df_violations.groupby('address').agg(
        violation_count=('casenumber', 'count'),
        latest_violation_date=('violationdate', 'max')
    ).reset_index()
else:
    df_violations_grouped = pd.DataFrame(columns=['address', 'violation_count', 'latest_violation_date'])

# Process and group Tax Delinquencies
if not df_tax.empty and 'street_address' in df_tax.columns:
    df_tax['street_address'] = df_tax['street_address'].astype(str).str.strip().str.upper()
    df_tax['total_due'] = (
        df_tax['principal_due'].fillna(0) + 
        df_tax['penalty_due'].fillna(0) + 
        df_tax['interest_due'].fillna(0)
    )
    
    df_tax_grouped = df_tax.groupby('street_address').agg(
        unpaid_tax_total=('total_due', 'sum'),
        delinquent_years=('tax_period', lambda x: ', '.join(map(str, sorted(set(x)))))
    ).reset_index().rename(columns={'street_address': 'address'})
else:
    df_tax_grouped = pd.DataFrame(columns=['address', 'unpaid_tax_total', 'delinquent_years'])

# Merge datasets on standardized street address
df_combined = pd.merge(df_violations_grouped, df_tax_grouped, on='address', how='outer')

# Fill missing values for clean reporting
df_combined['violation_count'] = df_combined['violation_count'].fillna(0).astype(int)
df_combined['unpaid_tax_total'] = df_combined['unpaid_tax_total'].fillna(0).round(2)
df_combined['delinquent_years'] = df_combined['delinquent_years'].fillna('None')
df_combined['latest_violation_date'] = df_combined['latest_violation_date'].fillna('None')

# Sort by properties with highest unpaid tax balance, then highest violation count
df_combined = df_combined.sort_values(by=['unpaid_tax_total', 'violation_count'], ascending=[False, False])

# Export output
output_file = "philly_problem_properties_2024_2026.csv"
df_combined.to_csv(output_file, index=False)

print(f"\nDone! Output saved to {output_file}.")
print(f"Total Unique Properties Found: {len(df_combined):,}")
