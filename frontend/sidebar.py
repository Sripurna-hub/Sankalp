import pandas as pd
import streamlit as st

from backend.database import save_db


def render_sidebar(db_data, drone_info):
    st.sidebar.title("Control Panel")

    if st.sidebar.button("Switch Operating Mode"):
        st.session_state.app_mode = None
        st.rerun()

    st.sidebar.subheader("Active Drone Recon Zone")
    st.sidebar.caption(f"**Zone:** {drone_info['location_name']}")
    st.sidebar.caption(
        f"**Center:** `{drone_info['lat']}, {drone_info['lon']}`"
    )
    st.sidebar.caption(
        f"**Radius:** `{drone_info['radius_km']} km`"
    )

    st.sidebar.divider()

    with st.sidebar.expander(
        "📍 Manually Set Disaster Location (Optional)",
        expanded=False
    ):

        st.caption(
            "Use this before the drone is deployed. Setting a "
            "manual location makes it the new active disaster "
            "zone center (75 km radius kept the same), so "
            "citizen/infra uploads near it verify correctly "
            "instead of being rejected as outside the zone."
        )

        manual_lat = st.number_input(
            "Disaster Location Lat",
            format="%.4f",
            value=float(drone_info["lat"]),
            key="manual_zone_lat_in"
        )

        manual_lon = st.number_input(
            "Disaster Location Lon",
            format="%.4f",
            value=float(drone_info["lon"]),
            key="manual_zone_lon_in"
        )

        set_manual_zone = st.button(
            "Set as Active Disaster Zone",
            type="primary",
            use_container_width=True,
            key="set_manual_zone_btn"
        )

        if set_manual_zone:

            drone_info["lat"] = float(manual_lat)
            drone_info["lon"] = float(manual_lon)
            drone_info["location_name"] = "Manually Set Disaster Zone"
            drone_info["source"] = "manual"

            save_db(db_data)

            st.sidebar.success(
                "Active disaster zone updated to manual location!"
            )

            st.rerun()

        if drone_info.get("source") == "manual":

            reset_to_drone = st.button(
                "Reset to Drone Deployment Zone",
                use_container_width=True,
                key="reset_manual_zone_btn"
            )

            if reset_to_drone:

                drone_info["lat"] = 11.5324
                drone_info["lon"] = 76.1512
                drone_info["radius_km"] = 75.0
                drone_info["location_name"] = "Wayanad Landslide Zone"
                drone_info["source"] = "drone"

                save_db(db_data)

                st.sidebar.success(
                    "Reverted to drone deployment zone."
                )

                st.rerun()

    st.sidebar.divider()

    st.sidebar.subheader("1. Citizen CSV Registry")

    public_csv = st.sidebar.file_uploader(
        "Citizen CSV (INITIAL DATA.csv)",
        type=["csv"],
        key="sidebar_public_csv"
    )

    if public_csv is not None:
        try:
            df_pub = pd.read_csv(public_csv)
            df_pub.columns = df_pub.columns.str.strip()
            st.session_state.public_csv_metadata = df_pub
            st.sidebar.success(
                f"Loaded {len(df_pub)} Citizen Records!"
            )
        except Exception as e:
            st.sidebar.error(f"Error loading Citizen CSV: {e}")

    st.sidebar.divider()

    st.sidebar.subheader("2. Infrastructure CSV Registry")

    infra_csvs = st.sidebar.file_uploader(
        "Infrastructure CSVs (Multi-Sensor)",
        type=["csv"],
        accept_multiple_files=True,
        key="sidebar_infra_csvs"
    )

    if infra_csvs:
        try:
            combined_dfs = []
            for file in infra_csvs:
                df_temp = pd.read_csv(file)
                df_temp.columns = df_temp.columns.str.strip()
                combined_dfs.append(df_temp)
            if combined_dfs:
                st.session_state.infra_csv_metadata = pd.concat(
                    combined_dfs,
                    ignore_index=True
                )
                st.sidebar.success(
                    f"Merged {len(infra_csvs)} CSVs "
                    f"({len(st.session_state.infra_csv_metadata)} rows)!"
                )
        except Exception as e:
            st.sidebar.error(f"Error loading Infra CSVs: {e}")
