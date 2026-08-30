import folium
import streamlit as st
from streamlit_folium import st_folium

from backend.media import encode_image_to_base64


def render_original_map(
    uploads_list,
    drone_info,
    tab_key,
    sos_list=None
):
    """
    COMMON MAP FOR ALL THREE TABS.

    Public Tab
    Infrastructure Tab
    Rescue Team Tab

    sos_list is OPTIONAL and only passed by the Rescue Team tab.
    """

    st.subheader("GEOSPATIAL MAP VIEW")

    st.caption(
        "All validated and analyzed public + infrastructure "
        "locations are displayed on the common map."
    )

    map_center = [
        float(drone_info["lat"]),
        float(drone_info["lon"]),
    ]

    m = folium.Map(
        location=map_center,
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&scale=2&x={x}&y={y}&z={z}",
        attr="Google Maps High-Detail",
        name="Google Roadmap",
        max_zoom=22,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        max_zoom=20,
    ).add_to(m)

    folium.LayerControl(
        position="topleft",
        collapsed=False,
    ).add_to(m)

    folium.Circle(
        radius=float(drone_info["radius_km"]) * 1000,
        location=map_center,
        color="crimson",
        fill=True,
        fill_opacity=0.08,
        popup=f"Active Recon Zone ({drone_info['radius_km']} km Radius)",
    ).add_to(m)

    # =========================================================================
    # COMMON UPLOAD DATA
    # =========================================================================

    for item in uploads_list:

        if not item.get("analyzed", False):
            continue

        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (ValueError, TypeError, KeyError):
            continue

        if lat == 0.0 and lon == 0.0:
            continue

        severity = int(item.get("severity", 0) or 0)
        color = str(item.get("color", "GREEN")).upper()

        pin_color = (
            "red" if color == "RED"
            else "orange" if color == "YELLOW"
            else "green"
        )

        img_path = item.get("annotated_path") or item.get("file_path")
        b64_img = encode_image_to_base64(img_path)

        img_html = ""
        if b64_img:
            img_html = (
                f'<img src="data:image/jpeg;base64,{b64_img}" '
                'style="width:230px;border-radius:6px;'
                'margin-top:6px;margin-bottom:6px;" />'
            )

        tag_bg = (
            "#dc3545" if color == "RED"
            else "#fd7e14" if color == "YELLOW"
            else "#198754"
        )

        nav_url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"

        nav_button = (
            f'<a href="{nav_url}" target="_blank" '
            'style="background-color:#0d6efd;color:white;'
            'padding:5px 8px;text-decoration:none;'
            'border-radius:4px;font-size:11px;'
            'font-weight:bold;">📍 Launch Navigation Route</a>'
        )

        source = item.get("source", "Upload")
        category = item.get("category", "Infrastructure")
        filename = item.get("filename", "Unknown")
        event = item.get("event", "N/A")

        popup_html = (
            '<div style="font-family:Arial,sans-serif;width:240px;">'
            f'<h4 style="margin:0;padding-bottom:4px;border-bottom:1px solid #ccc;">{filename}</h4>'
            f'<div style="background-color:{tag_bg};color:white;padding:4px 6px;'
            f'border-radius:4px;font-weight:bold;font-size:11px;margin-top:5px;'
            f'text-align:center;">{color} RISK ({severity}%)</div>'
            f'{img_html}'
            '<div style="font-size:12px;line-height:1.5;">'
            f'<b>Source:</b> {source}<br/>'
            f'<b>Event:</b> {event}<br/>'
            f'<b>Location:</b> {lat:.6f}, {lon:.6f}<br/>'
            f'<b>Category:</b> {category}<br/>'
            f'<b>Severity:</b> {severity}%<br/>'
            '</div>'
            f'<div style="margin-top:8px;text-align:center;">{nav_button}</div>'
            '</div>'
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{filename} | {category} | {severity}%",
            icon=folium.Icon(color=pin_color, icon="info-sign"),
        ).add_to(m)

    # =========================================================================
    # SOS SIGNALS (RESCUE TEAM TAB ONLY)
    # =========================================================================

    if sos_list:

        for sig in sos_list:

            try:
                sos_lat = float(sig["lat"])
                sos_lon = float(sig["lon"])
            except (ValueError, TypeError, KeyError):
                continue

            if sos_lat == 0.0 and sos_lon == 0.0:
                continue

            sos_nav_url = (
                f"https://www.google.com/maps/dir/?api=1"
                f"&destination={sos_lat},{sos_lon}"
            )

            sos_nav_button = (
                f'<a href="{sos_nav_url}" target="_blank" '
                'style="background-color:#0d6efd;color:white;'
                'padding:5px 8px;text-decoration:none;'
                'border-radius:4px;font-size:11px;'
                'font-weight:bold;">📍 Launch Navigation Route</a>'
            )

            sos_popup_html = (
                '<div style="font-family:Arial,sans-serif;width:220px;">'
                '<h4 style="margin:0;padding-bottom:4px;border-bottom:1px solid #ccc;">🆘 SOS Signal</h4>'
                '<div style="background-color:#000000;color:white;padding:4px 6px;'
                'border-radius:4px;font-weight:bold;font-size:11px;margin-top:5px;'
                'text-align:center;">EMERGENCY SOS</div>'
                '<div style="font-size:12px;line-height:1.6;margin-top:6px;">'
                f"<b>ID:</b> {sig.get('id', 'N/A')}<br/>"
                f"<b>Time:</b> {sig.get('timestamp', 'N/A')}<br/>"
                f"<b>Location:</b> {sos_lat:.6f}, {sos_lon:.6f}<br/>"
                '</div>'
                f'<div style="margin-top:8px;text-align:center;">{sos_nav_button}</div>'
                '</div>'
            )

            folium.Marker(
                location=[sos_lat, sos_lon],
                popup=folium.Popup(sos_popup_html, max_width=240),
                tooltip=f"SOS Signal {sig.get('id', '')}",
                icon=folium.Icon(color="black", icon="exclamation-sign"),
            ).add_to(m)

    # =========================================================================
    # RENDER
    # =========================================================================

    st_folium(
        m,
        use_container_width=True,
        height=500,
        key=f"map_{tab_key}",
        returned_objects=[],
    )