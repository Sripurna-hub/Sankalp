import time
import os
from PIL import Image

from backend.config import UPLOADS_DIR
from backend.geospatial import calculate_haversine_distance
from backend.csv_registry import lookup_coordinates_from_csv
from backend.media import save_uploaded_image, save_uploaded_video

def get_location_for_file(file_name, df_metadata, fallback_event):
    lat_val, lon_val, event_val = lookup_coordinates_from_csv(
        file_name, df_metadata
    )

    if "nepal" in file_name.lower():
        lat_val = 27.7172 if lat_val is None else lat_val
        lon_val = 85.3240 if lon_val is None else lon_val
        event_val = "Nepal Floods"

    if lat_val is None or lon_val is None:
        lat_val = 0.0
        lon_val = 0.0
        event_val = fallback_event

    return lat_val, lon_val, event_val

def make_verification(lat, lon, drone_info):
    dist = calculate_haversine_distance(
        lat, lon, drone_info["lat"], drone_info["lon"]
    )
    verified = dist <= drone_info["radius_km"]
    return (
        verified,
        dist,
        "VERIFIED (In Active Disaster Area)"
        if verified
        else f"REJECTED (Out of Geofence - {dist:.1f} km away)"
    )

def append_upload(db_data, item):
    db_data["public_uploads"].append(item)

def create_image_upload(
    db_data, uploaded_file, source, lat, lon, event, drone_info
):
    file_id = f"{int(time.time())}_{uploaded_file.name}"
    save_path = save_uploaded_image(uploaded_file, file_id)

    verified, dist, status_reason = make_verification(lat, lon, drone_info)

    item = {
        "id": file_id,
        "source": source,
        "filename": uploaded_file.name,
        "file_path": save_path,
        "annotated_path": save_path,
        "lat": lat,
        "lon": lon,
        "event": event,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verified": verified,
        "status_reason": status_reason,
        "distance_km": round(dist, 2),
        "analyzed": False,
        "category": None,
        "severity": None,
        "color": None,
        "media_type": "image"
    }
    append_upload(db_data, item)
    return item

def create_video_upload(
    db_data, uploaded_file, source, lat, lon, event, drone_info
):
    file_id = f"{int(time.time())}_{uploaded_file.name}"
    save_path = save_uploaded_video(uploaded_file, file_id)

    verified, dist, status_reason = make_verification(lat, lon, drone_info)

    item = {
        "id": file_id,
        "source": source,
        "filename": uploaded_file.name,
        "file_path": save_path,
        "annotated_path": save_path,
        "lat": lat,
        "lon": lon,
        "event": event,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verified": verified,
        "status_reason": status_reason,
        "distance_km": round(dist, 2),
        "analyzed": False,
        "category": None,
        "severity": None,
        "color": None,
        "media_type": "video"
    }
    append_upload(db_data, item)
    return item
