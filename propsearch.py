import time
import re
import requests
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# Handle playwright-stealth API version differences (1.x vs 2.x+)
try:
    from playwright_stealth import Stealth
    def apply_stealth(page_or_context):
        Stealth().apply_stealth_sync(page_or_context)
except ImportError:
    try:
        from playwright_stealth import stealth_sync
        def apply_stealth(page_or_context):
            stealth_sync(page_or_context)
    except ImportError:
        def apply_stealth(page_or_context):
            pass

# -------------------------------------------------------------------
# STEP 1: Scrape Bid4Assets (With Universal Stealth & Debugging)
# -------------------------------------------------------------------
def get_bid4assets_properties(sales_dates):
    properties = []
    base_url = "https://www.bid4assets.com/philaforeclosures"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-infobars',
                '--window-size=1920,1080',
                '--disable-dev-shm-usage',
            ]
        )
        
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York"
        )
        
        page = context.new_page()
        
        # Apply stealth patch to mask automated browser flags
        apply_stealth(page)

        for date in sales_dates:
            url = f"{base_url}?salesdate={date}"
            print(f"[*] Navigating to: {url}...")
            
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                
                # Brief pause for dynamic AJAX content to render
                time.sleep(5)
                
                # Check for Cloudflare challenge
                page_title = page.title()
                if "Just a moment" in page_title or "Attention Required" in page_title:
                    print(f"[!] Cloudflare challenge triggered on date {date}. Page Title: '{page_title}'")
                    page.screenshot(path=f"cf_block_{date}.png")
                    print(f"    -> Saved diagnostic screenshot to 'cf_block_{date}.png'")
                    continue

                # Verify if property table exists on the page
                if page.locator("table").count() == 0:
                    print(f"[-] No table found on page for date {date}. Page Title: '{page_title}'")
                    page.screenshot(path=f"no_table_{date}.png")
                    continue
                
                page.wait_for_selector("table", timeout=15000)
                rows = page.query_selector_all("table tr")
                found_count = 0
                
                for row in rows[1:]:  # Skip table header row
                    cols = [td.inner_text().strip() for td in row.query_selector_all("td, th")]
                    if len(cols) >= 4:
                        opa = cols[2] if len(cols) > 2 else ''
                        address = cols[3] if len(cols) > 3 else ''
                        
                        if address and not address.lower().startswith('address'):
                            properties.append({
                                'sales_date': date,
                                'opa_number': opa,
                                'raw_address': address,
                                'status': cols[-1] if cols else ''
                            })
                            found_count += 1
                            
                print(f"    -> Extracted {found_count} properties for date {date}.")
                
            except Exception as e:
                print(f"[!] Timeout or error loading date {date}: {e}")
                try:
                    page.screenshot(path=f"error_{date}.png")
                    print(f"    -> Saved error screenshot to 'error_{date}.png'")
                except Exception:
                    pass
                
        browser.close()
        
    print(f"\n[+] Total properties harvested: {len(properties)}\n")
    return properties

# -------------------------------------------------------------------
# STEP 2: Query Atlas via Philadelphia's Public APIs
# -------------------------------------------------------------------
def get_atlas_property_details(address, opa_number=None):
    details = {
        'owner_name': 'N/A',
        'mailing_address': 'N/A',
        'zip_code': 'N/A',
        'tax_balance': '$0.00'
    }
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    query = opa_number if (opa_number and opa_number != 'N/A') else address
    ais_url = f"https://api.phila.gov/ais/v1/search/{requests.utils.quote(str(query))}?gatekeeperKey=6ba4de64d6ca99aa4db3b9194e37adbf"
    
    try:
        ais_res = requests.get(ais_url, headers=headers, timeout=10)
        if ais_res.status_code == 200:
            data = ais_res.json()
            features = data.get('features', [])
            if features:
                props = features[0].get('properties', {})
                details['owner_name'] = props.get('opa_owners', 'N/A')
                details['mailing_address'] = props.get('mailing_address', props.get('opa_address', 'N/A'))
                details['zip_code'] = props.get('zip_code', 'N/A')
                fetched_opa = props.get('opa_account_num', opa_number)
                
                # Fetch Real Estate Tax Balance from Carto DB
                if fetched_opa:
                    carto_sql = f"SELECT SUM(total) as balance FROM real_estate_tax_balances WHERE opa_account_num = '{fetched_opa}'"
                    carto_url = f"https://phl.carto.com/api/v2/sql?q={carto_sql}"
                    tax_res = requests.get(carto_url, headers=headers, timeout=10)
                    if tax_res.status_code == 200:
                        tax_data = tax_res.json()
                        rows = tax_data.get('rows', [])
                        if rows and rows[0].get('balance') is not None:
                            bal = rows[0]['balance']
                            details['tax_balance'] = f"${bal:,.2f}"
    except Exception as e:
        print(f"[!] Atlas API error for {address}: {e}")
        
    return details

# -------------------------------------------------------------------
# STEP 3: Search Owner Phone Numbers
# -------------------------------------------------------------------
def search_phone_numbers(owner_name, zip_code):
    if not owner_name or owner_name == 'N/A':
        return []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    query = f'"{owner_name}" Philadelphia PA {zip_code} phone number'
    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    
    phone_numbers = []
    try:
        res = requests.get(search_url, headers=headers, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            text_content = soup.get_text()
            pattern = re.compile(r'\(?\b[2-9]\d{2}\)?[-. ]?\d{3}[-. ]?\d{4}\b')
            found_phones = pattern.findall(text_content)
            
            for phone in found_phones:
                clean_phone = re.sub(r'[^\d]', '', phone)
                if len(clean_phone) == 10:
                    formatted = f"({clean_phone[:3]}) {clean_phone[3:6]}-{clean_phone[6:]}"
                    if formatted not in phone_numbers:
                        phone_numbers.append(formatted)
                if len(phone_numbers) == 3:
                    break
    except Exception as e:
        print(f"[!] Phone search error for {owner_name}: {e}")
        
    return phone_numbers

# -------------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------------
def main():
    sales_dates = ['20261006', '20261103', '20261201']
    print("=== Philadelphia Foreclosure & Property Scraper (Stealth) ===\n")
    
    raw_properties = get_bid4assets_properties(sales_dates)
    
    if not raw_properties:
        print("[!] No properties retrieved. Check generated screenshots in your working directory.")
        return

    combined_results = []
    for idx, prop in enumerate(raw_properties, start=1):
        address = prop['raw_address']
        opa = prop['opa_number']
        print(f"[{idx}/{len(raw_properties)}] Processing: {address} (OPA: {opa})")
        
        atlas_data = get_atlas_property_details(address, opa)
        phones = search_phone_numbers(atlas_data['owner_name'], atlas_data['zip_code'])
        
        combined_results.append({
            'Auction Date': prop['sales_date'],
            'Property Address': address,
            'OPA Account #': opa,
            'Owner Name': atlas_data['owner_name'],
            'Mailing Address': atlas_data['mailing_address'],
            'Zip Code': atlas_data['zip_code'],
            'Current Tax Balance': atlas_data['tax_balance'],
            'Top 3 Phone Numbers': ", ".join(phones) if phones else "None found"
        })
        time.sleep(1)
        
    df = pd.DataFrame(combined_results)
    df.to_csv("philly_foreclosures_output.csv", index=False)
    print("\n[+] Done! Output saved to 'philly_foreclosures_output.csv'.")

if __name__ == '__main__':
    main()
