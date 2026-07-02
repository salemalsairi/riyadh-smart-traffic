


import csv
import os
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

def fetch_tomtom_routing_data(api_key: str, base_url: str, coord_pair: str) -> dict:


    url = f"{base_url}/{coord_pair}/json?key={api_key}&traffic=true"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            return response.json()
        else:
            print(f'error: {response.status_code}')
            return None
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None

if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv('TOMTOM_API_KEY')
    if not api_key:
        raise ValueError("Critical Error: TOMTOM_API_KEY is not set in the environment.")

    # Routing API
    BASE_URL = "https://api.tomtom.com/routing/1/calculateRoute"


    target_links = {

        # Zone A: شبكة جنوب شرق الرياض (البطحاء ومحيطها)

        "ZoneA_Batha_Link1": "24.629974303366414,46.71731321863171:24.610885148626895,46.72827825421672",
        "ZoneA_Batha_Link2": "24.610885148626895,46.72827825421672:24.594514401521195,46.73706569492821",
        "ZoneA_Firyan_Link1": "24.619069037350556,46.71424544862073:24.60721544790706,46.719323640082926",
        "ZoneA_Firyan_Link2": "24.60721544790706,46.719323640082926:24.60404309405173,46.72162718236526",
        "ZoneA_Firyan_Link3": "24.60404309405173,46.72162718236526:24.588275001670336,46.72480668238175",
        "ZoneA_PrinceMohd_Link1": "24.606137411863475,46.710746867710206:24.60721544790706,46.719323640082926",
        "ZoneA_PrinceMohd_Link2": "24.60721544790706,46.719323640082926:24.610885148626895,46.72827825421672",
        "ZoneA_PrinceMohd_Link3": "24.610885148626895,46.72827825421672:24.61628458053075,46.73753684000637",



    }



    file_exists = os.path.isfile('riyadh_traffic_links.csv')


    if not file_exists:
        with open('riyadh_traffic_links.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'link_name', 'length_meters', 'travel_time_seconds', 'traffic_delay_seconds'])

    while True:
        current_time = datetime.now(ZoneInfo("Asia/Riyadh"))
        real_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- Starting Data Extraction Cycle at {real_time} ---")

        for link_name, coord_pair in target_links.items():
            print(f"Fetching data for {link_name}...")
            traffic_data = fetch_tomtom_routing_data(api_key, BASE_URL, coord_pair)

            if traffic_data:
                try:

                    summary = traffic_data["routes"][0]["summary"]
                    length = summary["lengthInMeters"]
                    travel_time = summary["travelTimeInSeconds"]
                    delay = summary["trafficDelayInSeconds"]

                    data_row = [real_time, link_name, length, travel_time, delay]

                    with open('riyadh_traffic_links.csv', mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(data_row)

                    print(f"-> Successfully saved data for {link_name} (Delay: {delay}s)")
                except KeyError as e:
                    print(f"Data extraction failed. Missing key in JSON: {e}")

            time.sleep(1)

        print('Cycle complete. Pipeline sleeping for 15 minutes...')
        time.sleep(900)
