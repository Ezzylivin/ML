import requests
import pandas as pd

# The base URL for the City of Philadelphia Carto SQL API
CARTO_API = "https://phl.carto.com/api/v2/sql"

print("Fetching L&I Violations (2024-2026)...")
# 1. Query L&I Violations between 2024 and 2026
# We pull the address, violation date, and case number
violations_query = """
    SELECT address, violationdate, casenumber, violationdetails 
    FROM li_violations 
    WHERE violationdate >= '2024-01-01' 
      AND violationdate <= '2026-12-31'
"""
res_violations = requests.get(CARTO_API, params={'q': violations_query}).json()
df_violations = pd.DataFrame(res_violations.get('rows', []))

# Clean up the violations dataframe
if not df_violations.empty:
    # Standardize the address column to uppercase for matching
    df_violations['address'] = df_violations['address'].str.strip().str.upper()
    # Group by address to count violations and keep a list of case numbers
    df_violations_grouped = df_violations.groupby('address').agg(
        violation_count=('casenumber', 'count'),
        latest_violation_date=('violationdate', 'max')
    ).reset_index()
else:
    df_violations_grouped = pd.DataFrame(columns=['address', 'violation_count'])


print("Fetching Real Estate Tax Delinquencies (2024-2026)...")
# 2. Query Real Estate Tax Delinquencies
# Tax delinquencies dataset uses 'street_address' and 'tax_period' (year)
tax_query = """
    SELECT street_address, tax_period, principal_due, penalty_due, interest_due, is_actionable
    FROM real_estate_tax_delinquencies
    WHERE tax_period IN ('2024', '2025', '2026')
"""
res_tax = requests.get(CARTO_API, params={'q': tax_query}).json()
df_tax = pd.DataFrame(res_tax.get('rows', []))

# Clean up the tax dataframe
if not df_tax.empty:
    df_tax['street_address'] = df_tax['street_address'].str.strip().str.upper()
    
    # Sum up the total due per address for the 2024-2026 period
    df_tax['total_due'] = df_tax['principal_due'] + df_tax['penalty_due'] + df_tax['interest_due']
    df_tax_grouped = df_tax.groupby('street_address').agg(
        unpaid_tax_total=('total_due', 'sum'),
        delinquent_years=('tax_period', lambda x: ', '.join(sorted(set(x))))
    ).reset_index()
    # Rename column to match violations dataframe for the merge
    df_tax_grouped = df_tax_grouped.rename(columns={'street_address': 'address'})
else:
    df_tax_grouped = pd.DataFrame(columns=['address', 'unpaid_tax_total'])


print("Merging datasets...")
# 3. Merge the two datasets on the Property Address
# An 'outer' merge ensures we get properties with violations, unpaid taxes, or both
df_combined = pd.merge(df_violations_grouped, df_tax_grouped, on='address', how='outer')

# Fill NaN values with 0 for counts/totals, and "None" for text
df_combined['violation_count'] = df_combined['violation_count'].fillna(0)
df_combined['unpaid_tax_total'] = df_combined['unpaid_tax_total'].fillna(0)

# Sort by properties that have the highest unpaid taxes and most violations
df_combined = df_combined.sort_values(by=['unpaid_tax_total', 'violation_count'], ascending=[False, False])

# 4. Save to CSV
output_file = "philly_problem_properties_2024_2026.csv"
df_combined.to_csv(output_file, index=False)
print(f"Done! Saved results to {output_file}. Found {len(df_combined)} unique properties.")
