import streamlit as st

from backend.database import load_db
from frontend.home import render_home
from frontend.sidebar import render_sidebar
from frontend.public_tab import render_public_tab
from frontend.infrastructure_tab import render_infrastructure_tab
from frontend.rescue_tab import render_rescue_tab

st.set_page_config(layout="wide", page_title="Autonomous Disaster Response Engine")

db_data = load_db()

if "public_csv_metadata" not in st.session_state:
    st.session_state.public_csv_metadata = None
if "infra_csv_metadata" not in st.session_state:
    st.session_state.infra_csv_metadata = None
if "last_processed_cit_file" not in st.session_state:
    st.session_state.last_processed_cit_file = None
if "last_processed_infra_file" not in st.session_state:
    st.session_state.last_processed_infra_file = None
if "app_mode" not in st.session_state:
    st.session_state.app_mode = None

if st.session_state.app_mode is None:
    render_home()
else:
    drone_info = db_data["drone_active_area"]
    render_sidebar(db_data, drone_info)

    if st.session_state.app_mode == "active":
        st.title("Active Disaster Command Center")
        st.error(f"LIVE DISASTER ACTIVE IN: {drone_info['location_name'].upper()}")
    else:
        st.title("Hackathon Demo Platform")
        tab_public, tab_infra, tab_rescue = st.tabs([
            "Public Tab",
            "Infrastructure Assessment Tab",
            "Rescue Team Tab"
        ])
        with tab_public:
            render_public_tab(db_data, drone_info)
        with tab_infra:
            render_infrastructure_tab(db_data, drone_info)
        with tab_rescue:
            render_rescue_tab(db_data, drone_info)
