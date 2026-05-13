import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ingest.persistence as persistence
from core.db import init_db
from ingest.deduplication import fuse_data
from ingest.normalization import update_history
from ingest.persistence import store_aircraft
from intelligence.monitoring_backend import persist_source_runs
from sources.adsb import fetch_with_report as fetch_adsb_with_report


def main() -> None:
    conn, cursor = init_db()
    persistence.conn = conn
    persistence.cursor = cursor

    while True:
        adsb_data, adsb_report = fetch_adsb_with_report()
        aircraft = fuse_data([adsb_data])

        update_history(aircraft)
        store_aircraft(aircraft)
        persist_source_runs(cursor, [adsb_report])
        conn.commit()

        print(f"\n--- Retrieved {len(aircraft)} aircraft (after fusion) ---")
        time.sleep(30)


if __name__ == "__main__":
    main()
