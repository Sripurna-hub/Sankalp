import os
import json
from backend.config import DB_FILE


def load_db():
    """Load persistent DB state or initialize defaults."""
    default_db = {
        "drone_active_area": {
            "lat": 11.5324,
            "lon": 76.1512,
            "radius_km": 75.0,
            "location_name": "Wayanad Landslide Zone",
            "timestamp": "2026-08-30 00:00:00"
        },
        "public_uploads": [],
        "sos_signals": []
    }

    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump(default_db, f, indent=4)
        return default_db

    try:
        with open(DB_FILE, "r") as f:
            db = json.load(f)

            if db.get("drone_active_area", {}).get("radius_km", 0) < 75.0:
                db["drone_active_area"]["radius_km"] = 75.0

            # Backfill sos_signals for DBs saved before this feature existed
            db.setdefault("sos_signals", [])

            return db
    except Exception:
        return default_db


def save_db(data):
    """Write current state back to disk."""
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)