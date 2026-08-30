import pandas as pd
import streamlit as st

def render_sidebar(drone_info):
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
