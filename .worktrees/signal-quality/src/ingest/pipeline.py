import time

import ingest.persistence as persistence
from core.db import init_db
from ingest.deduplication import fuse_data
from ingest.normalization import update_history
from ingest.persistence import store_aircraft
from sources.adsb import fetch as fetch_adsb

conn, cursor = init_db()
persistence.conn = conn
persistence.cursor = cursor

while True:
    adsb_data = fetch_adsb()
    aircraft = fuse_data([adsb_data])

    update_history(aircraft)
    store_aircraft(aircraft)

    print(f"\n--- Retrieved {len(aircraft)} aircraft (after fusion) ---")
    time.sleep(30)
