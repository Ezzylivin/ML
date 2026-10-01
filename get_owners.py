import io import requests import pandas as 
pd CARTO_API = 
"https://phl.carto.com/api/v2/sql" print("1. 
Loading existing problem properties 
dataset...") input_file = 
"philly_problem_properties_2024_2026.csv" 
try:
    df_properties = pd.read_csv(input_file) 
    print(f" -> Loaded 
    {len(df_properties):,} properties from 
    {input_file}")
except Exception as e: print(f" -> Error 
    loading {input_file}: {e}") exit(1)
print("\n2. Fetching Property Assessment & 
Ownership Data (OPA)...")
# Query Philadelphia Office of Property 
# Assessment (OPA) dataset
opa_sql = """ SELECT location, owner_1, 
    owner_2, mailing_address_1, 
    mailing_street, mailing_city_state, 
    mailing_zip, category_code_description 
    FROM opa_properties
""" params = {"q": " 
".join(opa_sql.split()), "format": "csv"} 
try:
    response = requests.get(CARTO_API, 
    params=params, timeout=90) if 
    response.status_code == 200:
        df_opa = 
        pd.read_csv(io.StringIO(response.text)) 
        print(f" -> Success! Fetched 
        {len(df_opa):,} OPA assessment 
        records.")
    else: print(f" -> Server Error 
        ({response.status_code}): 
        {response.text.strip()[:150]}") 
        df_opa = pd.DataFrame()
except Exception as e: print(f" -> Request 
    failed: {e}") df_opa = pd.DataFrame()
print("\n3. Processing and Matching Owners 
to Addresses...") if not df_opa.empty and 
"location" in df_opa.columns:
    # Standardize location/address for 
    # merging
    df_opa["address"] = 
    df_opa["location"].astype(str).str.strip().str.upper()
    
    # Remove duplicate property entries if 
    # present
    df_opa = 
    df_opa.drop_duplicates(subset=["address"])
    
    # Select owner and mailing columns
    owner_cols = [ "address", "owner_1", 
        "owner_2", "mailing_address_1", 
        "mailing_street", 
        "mailing_city_state", "mailing_zip", 
        "category_code_description"
    ] df_opa_clean = df_opa[[c for c in 
    owner_cols if c in df_opa.columns]]
else: df_opa_clean = 
    pd.DataFrame(columns=["address", 
    "owner_1"])
# Standardize address in problem properties 
# dataframe
df_properties["address"] = 
df_properties["address"].astype(str).str.strip().str.upper()
# Join owner information onto the problem 
# properties list
df_merged = pd.merge(df_properties, 
df_opa_clean, on="address", how="left")
# Fill missing owner fields
for col in ["owner_1", "owner_2", 
"mailing_address_1", "mailing_street", 
"mailing_city_state", "mailing_zip", 
"category_code_description"]:
    if col in df_merged.columns: 
        df_merged[col] = 
        df_merged[col].fillna("Unknown / Not 
        Listed")
# Save updated dataset
output_file = 
"philly_problem_properties_with_owners.csv" 
df_merged.to_csv(output_file, index=False) 
matched_count = 
len(df_merged[df_merged["owner_1"] != 
"Unknown / Not Listed"]) print(f"\nDone! 
Exported enriched dataset to {output_file}") 
print(f"Total Properties: 
{len(df_merged):,}") print(f"Properties 
Matched with Owners: {matched_count:,} 
({(matched_count/len(df_merged))*100:.1f}%)")

